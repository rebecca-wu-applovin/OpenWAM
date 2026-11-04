"""Checkpoint save / load / management utilities."""

import logging
import os
import re

logger = logging.getLogger(__name__)


# --- Deploy assets (write-once) ---


def save_config(output_dir: str, cfg) -> None:
    """Save the run config as ``config.yaml`` in a fresh run dir (create_run_dir refused an existing one).

    Args:
        output_dir: str, the run dir.
        cfg: DictConfig, the resolved run config.
    """
    config_path = os.path.join(output_dir, "config.yaml")
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, config_path)
    logger.info("Saved config to %s", config_path)


def save_normalization_stats(output_dir: str, stats_path: str) -> None:
    """Copy the dataset's stats file into the run dir as ``normalization_stats.npy`` (the deploy artifact).

    The caller decides from ``dataloader.normalize_mode`` that normalization is on; the file keeps its
    ``{<action_mode>: ..., <action_mode>_state: ...}`` blocks, which deploy reads by the saved config's action_mode.

    Args:
        output_dir: str, the run dir.
        stats_path: str, the dataset's pooled stats file (``dataset.normalization_stats_path``).
    """
    import shutil

    dst = os.path.join(output_dir, "normalization_stats.npy")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(stats_path, dst)
    logger.info("[normalizer] Copied the stats into the run dir:\n  src: %s\n  dst: %s", stats_path, dst)


def _normalization_stats_values_equal(left, right) -> bool:
    """Exact deep equality of two stats payloads (nested dicts / lists of arrays and scalars).

    Args:
        left, right: the two loaded payloads.

    Returns:
        bool, True iff they have the same keys, shapes and values (floats bit-equal, NaN == NaN).
    """
    import numpy as np

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_normalization_stats_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_normalization_stats_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_arr, right_arr = np.asarray(left), np.asarray(right)
        if left_arr.shape != right_arr.shape:
            return False
        # Strict resume must preserve the exact transform: the restored weights and optimizer state were trained in
        # the original coordinates, so no tolerance.
        floating = np.issubdtype(left_arr.dtype, np.floating) or np.issubdtype(right_arr.dtype, np.floating)
        return bool(np.array_equal(left_arr, right_arr, equal_nan=floating))
    return left == right


def verify_resume_normalization_stats(output_dir: str, stats_path: str) -> None:
    """Fail fast when a resumed run's stats copy differs from the dataset's stats file in any way.

    A resume keeps the run dir's ``normalization_stats.npy``; if it differs from what the dataset normalizes with,
    training would continue in other coordinates than the restored weights. Old runs whose copy differs (other
    fields, other layout) are not special-cased: rebuild their copy or warm-start with ``finetune_ckpt_path``.

    Args:
        output_dir: str, the resumed run dir.
        stats_path: str, the dataset's pooled stats file (normalization on; the caller decides from the config).
    """
    import pickle

    import numpy as np

    dst = os.path.join(output_dir, "normalization_stats.npy")
    if not os.path.isfile(dst):
        raise FileNotFoundError(
            f"resume blocked: {dst} is missing while the dataset normalizes with {stats_path}. Use "
            "finetune_ckpt_path to warm-start with freshly copied stats, or restore the run's normalization_stats.npy."
        )
    try:
        ckpt_payload = np.load(dst, allow_pickle=True).item()
        data_payload = np.load(stats_path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(
            f"resume blocked: failed to read normalization stats for comparison "
            f"(checkpoint={dst}, dataset={stats_path}): {exc}. "
            "Use finetune_ckpt_path to warm-start instead of resume_ckpt_path."
        ) from exc

    if not _normalization_stats_values_equal(ckpt_payload, data_payload):
        raise ValueError(
            "resume blocked: checkpoint normalization_stats.npy does not match the "
            f"dataset stats at {stats_path}. Restored weights/optimizer were trained under "
            "different normalization coordinates than the current dataset transform. "
            "Use finetune_ckpt_path to warm-start instead of resume_ckpt_path.\n"
            f"  checkpoint: {dst}\n"
            f"  dataset:    {stats_path}"
        )


# --- Checkpoint I/O (read/write training state) ---


def save_weights(accelerator, architecture, output_path: str, global_step: int, *, final: bool) -> None:
    """Write the weights safetensors (the deploy artifact).

    Only rank-0 writes. ZeRO-1/2 (this repo's only stages, see
    ``training.zero_stage``) replicates the bf16 params on every rank, so
    ``get_state_dict`` is rank-local: non-writers return immediately —
    otherwise every rank can clone the full state dict to host memory at once
    and exhaust the node during checkpointing. Pruning is the caller's job, run
    after the full state is also written so the two lines stay in lockstep.
    """
    from tqdm import tqdm

    if not accelerator.is_main_process:
        return
    ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
    msg = f"[checkpoint] Saving {'final ' if final else ''}step {global_step} -> {ckpt_path}"
    logger.info(msg)
    tqdm.write(msg)
    state_dict = accelerator.get_state_dict(architecture)
    accelerator.unwrap_model(architecture).save_checkpoint(ckpt_path, state_dict=state_dict)
    msg = f"[checkpoint] Saved{' final' if final else ''}: {ckpt_path}"
    logger.info(msg)
    tqdm.write(msg)


def save_full_state(accelerator, output_path: str, global_step: int, opt_step: int, epoch: int) -> None:
    """Write full Accelerate state to ``accel_state_step_N/`` for resume.

    ALL ranks enter (DeepSpeed shards optimizer state per-rank). rank-0 writes the
    ``trainer_state.json`` marker last (atomic) so a half-written dir is never picked
    by ``find_latest_accel_state``.
    """
    import json

    state_dir = os.path.join(output_path, f"accel_state_step_{global_step}")
    if accelerator.is_main_process:
        os.makedirs(state_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {"global_step": int(global_step), "opt_step": int(opt_step), "epoch": int(epoch)}
        meta_path = os.path.join(state_dir, "trainer_state.json")
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)


def load_full_state(accelerator, state_dir: str) -> dict:
    """Restore optimizer/scheduler/RNG/model from ``state_dir`` (call AFTER prepare).

    Returns the ``trainer_state.json`` contents (global_step / opt_step / epoch); a missing file raises.
    """
    import json

    # trainer_state.json is the state's completion marker (written last): no marker, no resume.
    meta_path = os.path.join(state_dir, "trainer_state.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{meta_path} is missing: {state_dir} is not a complete training state")
    accelerator.load_state(state_dir)
    with open(meta_path) as f:
        return json.load(f)


# --- Discovery (locate latest step) ---

# One name rule per kind: weights files and resume-state dirs, ``<prefix><step>`` with an int step.
WEIGHTS_NAME = re.compile(r"^checkpoint_step_(\d+)\.safetensors$")
STATE_NAME = re.compile(r"^accel_state_step_(\d+)$")


def list_steps(run_dir: str, name: re.Pattern) -> list[tuple[int, str]]:
    """Every entry of ``run_dir`` whose name matches ``name``, by step (the one checkpoint-name parser).

    Args:
        run_dir: str, a run dir (a missing dir has no entries).
        name: compiled pattern with the step as group 1 (:data:`WEIGHTS_NAME` or :data:`STATE_NAME`).

    Returns:
        list of ``(step, path)`` sorted by step ascending; names that do not match (malformed steps) are skipped.
    """
    if not os.path.isdir(run_dir):
        return []
    found = []
    for entry in os.listdir(run_dir):
        m = name.match(entry)
        if m is not None:
            found.append((int(m.group(1)), os.path.join(run_dir, entry)))
    return sorted(found)


def find_latest_weights(run_dir: str) -> str:
    """Return the highest-step ``checkpoint_step_N.safetensors`` in *run_dir* (the finetune source).

    Args:
        run_dir: str, a run dir.

    Returns:
        str, the file's path. FileNotFoundError when there is none; a step-0-only dir logs a warning (likely a
        crash before the first real save).
    """
    weights = list_steps(run_dir, WEIGHTS_NAME)
    if not weights:
        raise FileNotFoundError(f"No file in {run_dir} matches checkpoint_step_<int>.safetensors")
    latest_step, latest_path = weights[-1]
    if latest_step == 0:
        logger.warning("Latest checkpoint in %s is step 0 (%s); verify before finetune.", run_dir, latest_path)
    return latest_path


def find_latest_accel_state(run_dir: str) -> str | None:
    """Return the highest-step *usable* ``accel_state_step_N/`` in *run_dir*, or None.

    Usable = ``trainer_state.json`` present. ``save_full_state`` writes that marker
    atomically AFTER ``accelerator.save_state`` returns, so its presence proves the
    (possibly large / sharded) state finished writing. DeepSpeed's ``save_state``
    writes a ``pytorch_model/`` dir and no ``random_states_*.pkl``, so the marker —
    not RNG files — is the completion signal. Half-written dirs lack it and are skipped.

    Args:
        run_dir: str, a run dir.

    Returns:
        str path of the state dir, or None when no usable state exists.
    """
    usable = [
        path for _, path in list_steps(run_dir, STATE_NAME) if os.path.isfile(os.path.join(path, "trainer_state.json"))
    ]
    return usable[-1] if usable else None


# --- Resume position (pure math) ---


def compute_resume_position(global_step: int, batches_per_epoch: int, grad_accum: int) -> tuple[int, int, int]:
    """Map a resumed ``global_step`` to ``(start_epoch, skip_first_batches, aligned_global_step)``.

    ``skip`` is floored to a grad_accum boundary so the first optimizer step after
    resume sees a full accumulation cycle; ``aligned_global_step`` pulls ``global_step``
    back to that same boundary so the floored-off batches are not re-trained and the
    per-step seed (keyed on global_step) stays matched. No-op at grad_accum=1.
    """
    batches_per_epoch = max(batches_per_epoch, 1)
    start_epoch = global_step // batches_per_epoch
    skip = global_step % batches_per_epoch
    if grad_accum > 1 and skip % grad_accum != 0:
        skip = (skip // grad_accum) * grad_accum
    aligned_global_step = start_epoch * batches_per_epoch + skip
    return start_epoch, skip, aligned_global_step


# --- Retention (prune) ---


def prune_checkpoints(run_dir: str, keep_weights: int, keep_states: int) -> None:
    """Keep the newest ``keep_weights`` weights files and ``keep_states`` resume-state dirs; delete the rest.

    Rank-0 only (the caller guards). Per-step saves prune both lines with the same k, so a kept weights file keeps
    its sibling state dir; the finished run passes ``keep_states=0`` (no resume state survives a finished run).

    Args:
        run_dir: str, the run dir.
        keep_weights: int, how many ``checkpoint_step_*.safetensors`` to keep (highest steps).
        keep_states: int, how many ``accel_state_step_*`` dirs to keep (highest steps).
    """
    import shutil

    weights = list_steps(run_dir, WEIGHTS_NAME)
    for _, path in weights[: max(0, len(weights) - keep_weights)]:
        os.remove(path)
        logger.info("Removed old checkpoint: %s", path)
    states = list_steps(run_dir, STATE_NAME)
    for _, path in states[: max(0, len(states) - keep_states)]:
        # A failed state delete costs disk only; it must not kill the run.
        try:
            shutil.rmtree(path)
            logger.info("Removed accelerate state dir: %s", path)
        except OSError as e:
            logger.warning("Failed to remove accelerate state dir %s: %s", path, e)


# --- Run dir (create / reuse, per-step save, finish) ---


def create_run_dir(accelerator, base_output_path: str, run_dir_name: str | None, write_assets) -> str:
    """Create a fresh run dir on rank 0 (writing its deploy assets there) and broadcast its path to every rank.

    Args:
        accelerator: the Accelerator (``is_main_process``).
        base_output_path: str, ``training.output_path``.
        run_dir_name: str, ``training.run_dir_name``; None = the launch timestamp.
        write_assets: callable(run_dir: str) -> None, run on rank 0 in the new dir (config, stats, backbone assets).

    Returns:
        str, the run dir (same on every rank). FileExistsError (on every rank) when a fixed name already holds a run.
    """
    import torch.distributed as dist

    if run_dir_name is not None:
        # Checked on every rank (shared FS) so a clash raises everywhere instead of deadlocking the broadcast.
        existing = os.path.join(base_output_path, str(run_dir_name), "config.yaml")
        if os.path.exists(existing):
            raise FileExistsError(
                f"training.run_dir_name={run_dir_name}: {existing} already exists; pick a new run name "
                f"(or use training.resume_ckpt_path to continue that run)."
            )
    # Every rank finishes the clash check before rank 0 writes config.yaml below; otherwise a slower
    # rank sees rank 0's fresh config.yaml and raises (Modal runs 88/90, 2026-10-16).
    dist.barrier()

    path_list = [None]
    if accelerator.is_main_process:
        from datetime import datetime

        name = str(run_dir_name) if run_dir_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_list[0] = os.path.join(base_output_path, name)
        os.makedirs(path_list[0], exist_ok=True)
        write_assets(path_list[0])
    dist.broadcast_object_list(path_list, src=0)
    logger.info("Checkpoints will be saved to %s", path_list[0])
    return path_list[0]


def latest_resume_state(run_dir: str) -> str:
    """The state a resume continues from: the run dir's latest usable ``accel_state_step_*``.

    Args:
        run_dir: str, the resumed run dir (every rank resolves it independently on the shared FS).

    Returns:
        str, the state dir. FileNotFoundError when the run has no usable state.
    """
    state_dir = find_latest_accel_state(run_dir)
    if state_dir is None:
        raise FileNotFoundError(
            f"resume_ckpt_path={run_dir} has no usable accel_state_step_*; full states "
            f"are only written when training.save_full_states_for_resume=true (and a finished run drops "
            f"them) — use finetune_ckpt_path to warm-start from weights instead."
        )
    logger.info("[resume] reusing run dir %s (state=%s)", run_dir, os.path.basename(state_dir))
    return state_dir


class RunDir:
    """One run's output dir and its save policy: periodic weights (+ resume state) with pruning, final weights.

    Args:
        path: str, the run dir (from :func:`create_run_dir`, or the resumed run dir).
        save_steps: int, save every N micro steps; None = never save (no final weights either).
        keep_last_k: int, weights files (and resume states) kept by the rolling prune.
        save_full_states: bool, also write the resumable accel state at each save.
    """

    def __init__(self, path: str, *, save_steps: int | None, keep_last_k: int, save_full_states: bool):
        self.path = path
        self.save_steps = save_steps
        self.keep_last_k = keep_last_k
        self.save_full_states = save_full_states

    def save_step(self, accelerator, architecture, global_step: int, opt_step: int, epoch: int) -> None:
        """At every ``save_steps`` boundary: weights, then the full state (if on), then prune both lines in lockstep.

        Args:
            accelerator: the Accelerator (every rank enters; full-state saving is collective).
            architecture: the prepared (wrapped) architecture.
            global_step: int, micro steps done (> 0).
            opt_step: int, optimizer steps done.
            epoch: int, current epoch.
        """
        if self.save_steps is None or global_step % self.save_steps != 0:
            return
        save_weights(accelerator, architecture, self.path, global_step, final=False)
        if self.save_full_states:
            save_full_state(accelerator, self.path, global_step, opt_step, epoch)
        if accelerator.is_main_process:
            prune_checkpoints(self.path, self.keep_last_k, self.keep_last_k)

    def finish(self, accelerator, architecture, global_step: int) -> None:
        """Every exit of a finished run: final weights (unless saving is off), drop all resume state, keep k weights.

        Args:
            accelerator: the Accelerator (every rank enters; it barriers before the prune).
            architecture: the prepared (wrapped) architecture.
            global_step: int, the last micro step.
        """
        if self.save_steps is not None:
            save_weights(accelerator, architecture, self.path, global_step, final=True)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            prune_checkpoints(self.path, self.keep_last_k, 0)
