"""Architecture construction for finetune / resume.

When ``training.finetune_ckpt_path`` or ``training.resume_ckpt_path`` is set,
the trainer builds the architecture from that checkpoint dir, with the same
:func:`openwam.model.build_architecture_from_ckpt_dir` the deploy loader uses:
the module skeletons come from the model config (yaml geometry) and the
non-weight assets (tokenizers / processors / third-party ``config.json``) the
backbones copied into the checkpoint dir. The original pretrained backbone
directory (``assets.video_backbone.model_path``, not compared) need not exist on the
training host.

Both paths build the shell from the launch config, after comparing it with the
checkpoint's ``config.yaml``; nothing is merged or written back:

  - :func:`build_for_finetune` (a new run from these weights): the launch's
    ``model`` + ``common`` (minus the derived ``common.layout``) must equal the
    checkpoint's key by key (:func:`check_finetune_cfg`); ``training`` /
    ``dataloader`` may differ. Then a strict ``load_checkpoint`` of the
    requested / latest safetensors.
  - :func:`build_for_resume` (the same run continued): ``scripts/train.py``
    already runs the run's own config (:func:`load_resume_cfg`), and the whole
    config is compared again here (:func:`check_resume_cfg`), so the launch
    config IS the saved one. Storage is given to the meta shell
    (``materialize_meta``); accelerate's ``load_state`` fills it after ``prepare``.
"""

import logging
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from openwam.config import cfg_get
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.train.utils.checkpointing import find_latest_weights

logger = logging.getLogger(__name__)


def _resolve_ckpt_path(source: str) -> tuple[str, str | None]:
    """Split a finetune / resume source.

    Args:
        source: str, a checkpoint dir or one of its ``checkpoint_step_*.safetensors`` files.

    Returns:
        (ckpt_dir, explicit_weights): the dir, and the file when one was named (else None).
    """
    source = os.fspath(source)
    if source.endswith(".safetensors"):
        # Fail fast on a typo'd filename — otherwise safetensors only errors
        # after minutes of architecture-skeleton construction.
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Explicit checkpoint weights file does not exist: {source}")
        return os.path.dirname(os.path.abspath(source)), os.path.abspath(source)
    return source, None


# ----------------------------------------------------------------------------
# config comparison (finetune: model + common; resume: the whole config)
# ----------------------------------------------------------------------------

# Cap on a single value's printed length in the diff report: long lists (e.g. the
# cosmos3 VAE latents_mean) would bury the keys that actually changed.
_DIFF_VALUE_MAXLEN = 110
# Placeholder for the side of a difference that lacks the key.
ABSENT = "<absent>"
# Keys a resume launch legitimately sets differently from the run's config.yaml: where to resume
# from, and the fresh-run dir fields a resume launch drops (the run dir is resume_ckpt_path's).
RESUME_LAUNCH_KEYS = ("training.resume_ckpt_path", "training.output_path", "training.run_dir_name")
# Subtrees only the saved config.yaml carries: resolve_config derives the layout and writes it before save_config.
# Derived, so finetune does not compare it either (its sources are compared, or may differ on purpose).
SAVED_ONLY_KEYS = ("common.layout",)


def _flatten_cfg(node, prefix: str = "") -> dict:
    """Flatten a plain-container config into ``{dotted.key: leaf}``.

    Lists stay leaves: per-index keys would bury the real diff.

    Args:
        node: dict (or None), a plain container from ``OmegaConf.to_container``.
        prefix: str prepended to every key (``"model."`` etc.).

    Returns:
        dict ``{dotted.key: leaf}``; an empty dict value is a leaf.
    """
    flat: dict = {}
    for key, val in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            flat.update(_flatten_cfg(val, prefix=f"{path}."))
        else:
            flat[path] = val
    return flat


def _short(val) -> str:
    """``repr(val)`` cut to ``_DIFF_VALUE_MAXLEN`` characters (for diff reports).

    Args:
        val: any config leaf.

    Returns:
        str, the (possibly truncated) repr.
    """
    text = repr(val)
    return text if len(text) <= _DIFF_VALUE_MAXLEN else text[: _DIFF_VALUE_MAXLEN - 3] + "..."


def _under(key: str, prefixes: tuple) -> bool:
    """True if dotted ``key`` is one of ``prefixes`` or lies below one of them."""
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def _diff_flat(saved: dict, live: dict) -> list[tuple[str, Any, Any]]:
    """Every key where two flat configs disagree.

    Args:
        saved: dict ``{dotted.key: leaf}``, the checkpoint's side.
        live: dict ``{dotted.key: leaf}``, the launch's side.

    Returns:
        list of ``(dotted_key, saved_value, live_value)`` sorted by key; a side that lacks the key holds ``ABSENT``.
    """
    diffs = []
    for key in sorted(saved.keys() | live.keys()):
        s, v = saved.get(key, ABSENT), live.get(key, ABSENT)
        if s != v:
            diffs.append((key, s, v))
    return diffs


def model_identity(cfg: DictConfig) -> dict:
    """The keys that decide what the model is: ``model`` and ``common`` minus the derived ``common.layout``.

    Args:
        cfg: DictConfig, a whole run config (a checkpoint's config.yaml or the launch config).

    Returns:
        dict ``{"model.<dotted>": leaf, "common.<dotted>": leaf}`` with interpolations evaluated (compare what the
        values are, not how the yaml spells them); lists stay leaves. KeyError when ``model`` / ``common`` is missing.
    """
    flat: dict = {}
    for section in ("model", "common"):
        flat.update(_flatten_cfg(OmegaConf.to_container(cfg_get(cfg, section), resolve=True), prefix=f"{section}."))
    # The layout is derived from model / dataloader keys: not a decision of its own.
    return {k: v for k, v in flat.items() if not _under(k, SAVED_ONLY_KEYS)}


def check_finetune_cfg(ckpt_cfg: DictConfig, live_cfg: DictConfig, path: str) -> None:
    """Raise if a finetune launch's model / common config differs from the checkpoint's.

    The shell is built from the launch config and the checkpoint's weights loaded strictly into it, so the two
    must describe the same model; any intended model change is a new config for a new pretraining, not a finetune.

    Args:
        ckpt_cfg: DictConfig, the checkpoint's ``config.yaml``.
        live_cfg: DictConfig, the launch config (resolved).
        path: str, path of the checkpoint's ``config.yaml`` (for the message).

    Returns:
        None; raises ValueError listing every differing key as ``  <key>: ckpt=<v> launch=<v>``.
    """
    diffs = _diff_flat(model_identity(ckpt_cfg), model_identity(live_cfg))
    if not diffs:
        return
    lines = "\n".join(f"  {k}: ckpt={_short(s)} launch={_short(v)}" for k, s, v in diffs)
    raise ValueError(
        f"[finetune] this launch's model / common config differs from {path} on {len(diffs)} key(s); the "
        f"checkpoint's weights only fit the model they were trained as, so set these keys to the checkpoint's "
        f"values:\n{lines}"
    )


def diff_resume_cfg(saved_cfg: DictConfig, live_cfg: DictConfig) -> list:
    """Every key where a resume launch's config disagrees with the run's saved config.yaml.

    Args:
        saved_cfg: DictConfig, the run's ``config.yaml``.
        live_cfg: DictConfig, the config this launch composed (yaml defaults + CLI).

    Returns:
        sorted list of ``(dotted_key, saved_value, live_value)``; a side that lacks the key holds
        ``ABSENT``. ``RESUME_LAUNCH_KEYS`` and saved-only ``SAVED_ONLY_KEYS`` are not differences.
    """
    # Unresolved: compare what the yaml says, not what interpolations evaluate to on this host.
    saved = _flatten_cfg(OmegaConf.to_container(saved_cfg, resolve=False))
    live = _flatten_cfg(OmegaConf.to_container(live_cfg, resolve=False))
    return [
        (k, s, v)
        for k, s, v in _diff_flat(saved, live)
        if not (_under(k, RESUME_LAUNCH_KEYS) or (v == ABSENT and _under(k, SAVED_ONLY_KEYS)))
    ]


def check_resume_cfg(saved_cfg: DictConfig, live_cfg: DictConfig, path: str) -> None:
    """Raise if a resume launch's config differs from the run's saved config.yaml.

    Args:
        saved_cfg: DictConfig, the run's ``config.yaml``.
        live_cfg: DictConfig, the config this launch composed.
        path: str, path of ``config.yaml`` (for the message).

    Returns:
        None; raises ValueError listing every differing key.
    """
    diffs = diff_resume_cfg(saved_cfg, live_cfg)
    if not diffs:
        return
    lines = "\n".join(f"  {k}: saved={_short(s)} launch={_short(v)}" for k, s, v in diffs)
    raise ValueError(
        f"[resume] this launch's config differs from {path} on {len(diffs)} key(s); a resume must run "
        f"the run's own config (use training.finetune_ckpt_path to start a new run from these weights "
        f"with a different config):\n{lines}"
    )


def load_resume_cfg(live_cfg: DictConfig, resume_path: str) -> DictConfig:
    """The config a resume runs with: the run's saved ``config.yaml``, which the launch must match.

    A resume continues ONE run, so every section (model, training, dataloader, project) comes from
    the file the run started with. Any other difference between this launch's config and that file
    (changed code defaults, a new CLI override) raises instead of silently changing the run.

    Args:
        live_cfg: DictConfig, the config this launch composed.
        resume_path: str, ``training.resume_ckpt_path`` (run dir, or a safetensors file inside it).

    Returns:
        DictConfig, the saved config with the ``RESUME_LAUNCH_KEYS`` taken from ``live_cfg``.
    """
    ckpt_dir, _ = _resolve_ckpt_path(resume_path)
    path = os.path.join(ckpt_dir, "config.yaml")
    saved = OmegaConf.load(path)
    check_resume_cfg(saved, live_cfg, path)
    # Only the resume-location keys come from the launch (written into the just-loaded file, before resolve_config).
    with open_dict(saved):
        for key in RESUME_LAUNCH_KEYS:
            OmegaConf.update(saved, key, cfg_get(live_cfg, key), force_add=True)
    return saved


def _load_ckpt_cfg(ckpt_dir: str) -> DictConfig:
    """Read a run dir's ``config.yaml`` as written (not resolved).

    Args:
        ckpt_dir: str, checkpoint (run) dir.

    Returns:
        DictConfig. Raises ValueError when it has no ``model`` section (not a dir written by this trainer).
    """
    path = os.path.join(ckpt_dir, "config.yaml")
    ckpt_cfg = OmegaConf.load(path)
    if "model" not in ckpt_cfg:
        raise ValueError(
            f"{path} has no `model:` section, so there is nothing to build the architecture from. Point "
            "training.finetune_ckpt_path / resume_ckpt_path at a run directory written by this trainer."
        )
    return ckpt_cfg


def build_for_finetune(source: str, cfg: DictConfig, train_options: TrainOptions) -> BaseWAMArchitecture:
    """Warm-start: the launch config's architecture (checked equal to the checkpoint's), weights loaded strictly.

    Args:
        source: str, ``training.finetune_ckpt_path`` (a run dir, or one ``checkpoint_step_*.safetensors`` in it).
        cfg: the launch run config, resolved and read-only; its ``model`` + ``common`` must equal the
            checkpoint's (:func:`check_finetune_cfg`); ``merge(cfg.common, cfg.model)`` builds the shell.
        train_options: TrainOptions the trainer decided (passed to the architecture constructor).

    Returns:
        BaseWAMArchitecture with every weight loaded from the requested / latest safetensors.
    """
    from openwam.model import build_architecture_from_ckpt_dir

    ckpt_dir, explicit_weights = _resolve_ckpt_path(source)
    # Compare before building: a differing model would fail the strict load only after minutes of construction.
    check_finetune_cfg(_load_ckpt_cfg(ckpt_dir), cfg, os.path.join(ckpt_dir, "config.yaml"))
    logger.info("[finetune] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, OmegaConf.merge(cfg.common, cfg.model), train_options)
    weights = explicit_weights or find_latest_weights(ckpt_dir)
    # Plain print so the warm-start is visible on the launch terminal even
    # when logger output is drowned out; non-main ranks have print disabled.
    print(f"[finetune] loading pretrained weights: {weights}", flush=True)
    architecture.load_checkpoint(weights)
    print("[finetune] pretrained weights loaded OK", flush=True)
    return architecture


def build_for_resume(source: str, cfg: DictConfig, train_options: TrainOptions) -> BaseWAMArchitecture:
    """Resume: the run's own architecture, with zeroed storage for accelerate ``load_state`` to fill.

    The safetensors are not read: ``load_state`` restores a strict superset (module weights incl. frozen
    params) after ``prepare``, and the latest safetensors may be truncated by a mid-save kill even though the
    run is still recoverable from its accel state.

    Args:
        source: str, ``training.resume_ckpt_path`` (the run dir).
        cfg: the run config, resolved and read-only; any difference from the saved config.yaml outside the
            resume-location keys raises (:func:`check_resume_cfg`), so ``merge(cfg.common, cfg.model)`` is the
            run's own and builds the shell.
        train_options: TrainOptions the trainer decided (passed to the architecture constructor).

    Returns:
        BaseWAMArchitecture with real (zero) storage everywhere and the shell's real buffers kept.
    """
    from openwam.model import build_architecture_from_ckpt_dir
    from openwam.model.video_backbone.common.weights import materialize_meta

    ckpt_dir, _explicit_weights = _resolve_ckpt_path(source)
    # scripts/train.py already swapped in the saved config (load_resume_cfg); this guards direct callers.
    check_resume_cfg(_load_ckpt_cfg(ckpt_dir), cfg, os.path.join(ckpt_dir, "config.yaml"))
    logger.info("[resume] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, OmegaConf.merge(cfg.common, cfg.model), train_options)
    # set_dtype_device runs before load_state and cannot copy out of meta tensors.
    materialize_meta(architecture, "cpu")
    return architecture
