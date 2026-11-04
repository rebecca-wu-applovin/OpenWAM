"""Architecture construction for finetune / resume.

When ``training.finetune_ckpt_path`` or ``training.resume_ckpt_path`` is set,
the trainer builds the architecture from that checkpoint alone, with the same
:func:`openwam.model.build_architecture_from_ckpt_dir` the deploy loader uses:
the module skeletons come from the model config (yaml geometry) and the
non-weight assets (tokenizers / processors / third-party ``config.json``) the
backbones copied into the checkpoint dir. The original pretrained backbone
directory (``model.video_backbone.model_path``) does not need to exist on the
training host. The two entry points differ only in which config builds the
shell and in how its weights are filled:

  - :func:`build_for_finetune`: ckpt ``model`` section + live overrides, then a
    strict ``load_checkpoint`` of the requested / latest safetensors;
  - :func:`build_for_resume`: the ckpt ``model`` section verbatim, storage given
    to the meta shell (``materialize_meta``); accelerate's ``load_state`` fills
    it after ``prepare``.

The checkpoint's ``config.yaml`` is the RECONSTRUCTION base, not the run's
authority. On the finetune path the ckpt's ``model`` section is layered under
the live ``cfg.model``: every value the operator set in ``configs/`` (or on the
CLI) wins. The merged result is written back into ``cfg.model``, so one config
both builds the architecture and is written by ``save_config`` into the new run
dir. Resume continues one run whose ``config.yaml`` is already on disk, so the
ckpt config stays authoritative there and any live divergence is reported by
:func:`warn_live_model_cfg_ignored` instead.
"""

import logging
import os

from omegaconf import DictConfig, OmegaConf, open_dict

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
# ckpt model config (base) + live model config (override)
# ----------------------------------------------------------------------------

# Keys whose live-vs-ckpt disagreement an override cannot express: each one
# selects WHICH module classes the checkpoint's weights belong to, so overriding
# one of these would build a skeleton of a different backbone / architecture
# class — a state_dict topology no safetensors file matches. Value is the CLI
# hint for the fix.
_ARCH_IDENTITY_KEYS = (
    ("video_backbone.name", "model/video_backbone=<ckpt value>"),
    ("architecture.framework", "model=<ckpt value>"),
)

# Cap on a single value's printed length in the diff report: long lists (e.g. the
# cosmos3 VAE latents_mean) would bury the keys that actually changed.
_DIFF_VALUE_MAXLEN = 110


def _flatten_cfg(node, prefix: str = "") -> dict:
    """Flatten a plain-container config into ``{dotted.key: leaf}``.

    Lists stay leaves: per-index keys would bury the real diff.
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
    text = repr(val)
    return text if len(text) <= _DIFF_VALUE_MAXLEN else text[: _DIFF_VALUE_MAXLEN - 3] + "..."


def diff_model_cfgs(ckpt_model, live_model) -> tuple[list, list, list]:
    """Compare two ``model`` configs key-by-key.

    Returns ``(overridden, live_only, ckpt_only)`` where ``overridden`` holds
    ``(dotted_key, ckpt_value, live_value)`` for keys both sides define with
    different values, and the other two hold ``(dotted_key, value)``.
    """
    base = _flatten_cfg(OmegaConf.to_container(ckpt_model, resolve=True) or {})
    over = _flatten_cfg(OmegaConf.to_container(live_model, resolve=True) or {})
    # Keys are unique, so tuple ordering never has to compare the values.
    overridden = sorted((k, base[k], over[k]) for k in over if k in base and base[k] != over[k])
    live_only = sorted((k, over[k]) for k in over if k not in base)
    ckpt_only = sorted((k, base[k]) for k in base if k not in over)
    return overridden, live_only, ckpt_only


def _check_arch_identity(ckpt_model, live_model, ckpt_dir: str) -> None:
    """Reject a live/ckpt disagreement on the keys that pick the module classes."""
    for key, hint in _ARCH_IDENTITY_KEYS:
        ckpt_val = OmegaConf.select(ckpt_model, key, default=None)
        live_val = OmegaConf.select(live_model, key, default=None)
        if ckpt_val is None or live_val is None or ckpt_val == live_val:
            continue
        raise ValueError(
            f"finetune config mismatch on model.{key}: live config says {live_val!r}, "
            f"{os.path.join(ckpt_dir, 'config.yaml')} says {ckpt_val!r}. This key selects which "
            "module classes the checkpoint's weights belong to, so it is the one thing the live "
            "config cannot override — the merged skeleton would match no safetensors file. Set "
            f"`{hint}` to the checkpoint's value, or point training.finetune_ckpt_path at a "
            "checkpoint of the architecture you want to train."
        )


def _log_model_cfg_diff(ckpt_dir: str, overridden: list, live_only: list, ckpt_only: list) -> None:
    """Report the merge on the launch terminal — an override must never be silent."""
    header = (
        f"[finetune] model config = {os.path.join(ckpt_dir, 'config.yaml')} (base) "
        f"+ live cfg.model (override): {len(overridden)} overridden, "
        f"{len(live_only)} live-only, {len(ckpt_only)} ckpt-only"
    )
    logger.info(header)
    # Plain print (rank-0 only; train.py silences print on other ranks) so the
    # merge is visible on the launch terminal even when logs are drowned out.
    print(header, flush=True)
    for key, ckpt_val, live_val in overridden:
        line = f"[finetune]   override  model.{key}: {_short(ckpt_val)} (ckpt) -> {_short(live_val)} (live)"
        logger.info(line)
        print(line, flush=True)
    # The other two buckets are counted in the header but only spelled out at
    # DEBUG: they are rarely actionable, and logging is not rank-gated here, so at
    # INFO they would repeat once per rank on every launch.
    for key, live_val in live_only:
        logger.debug("[finetune]   live-only model.%s = %s", key, _short(live_val))
    for key, ckpt_val in ckpt_only:
        logger.debug("[finetune]   from-ckpt model.%s = %s", key, _short(ckpt_val))


def merge_ckpt_model_cfg(ckpt_cfg: DictConfig, cfg: DictConfig, ckpt_dir: str) -> DictConfig:
    """Layer the live run's ``model`` config on top of the checkpoint's.

    The ckpt's ``model`` section is the BASE (the geometry the checkpoint's weights
    were trained with). The live ``cfg.model`` is the OVERRIDE, so every value the
    operator set wins — including an explicit ``null``, which is how
    ``bridge_layers: null`` clears a ckpt's explicit list.
    Keys only the ckpt has survive; keys only the live config has are added.

    The merged result is written back into ``cfg.model`` so that exactly one
    config both builds the architecture here and is written by ``save_config``
    into the new run dir. Without the write-back the new checkpoint's
    ``config.yaml`` would advertise values the model was never built with, and
    deploying it would rebuild a different model than the one that trained.

    Returns the merged node as it now lives in ``cfg``.
    """
    ckpt_model = ckpt_cfg.model
    live_model = OmegaConf.select(cfg, "model", default=None)
    if live_model is None:
        return ckpt_model

    _check_arch_identity(ckpt_model, live_model, ckpt_dir)
    _log_model_cfg_diff(ckpt_dir, *diff_model_cfgs(ckpt_model, live_model))

    merged = OmegaConf.merge(ckpt_model, live_model)
    with open_dict(cfg):
        cfg.model = merged
    return cfg.model


def warn_live_model_cfg_ignored(ckpt_cfg: DictConfig, cfg: DictConfig, ckpt_dir: str) -> None:
    """Report live ``model`` keys the resume path discards.

    Resume continues ONE run: its ``config.yaml`` is already on disk and
    ``setup_output_dir`` reuses the dir without rewriting it, so the ckpt config
    stays authoritative and a live override would make that file lie about the
    model being trained. It must still not be silent — a resumed run whose live
    yaml says something else is an operator error worth seeing.
    """
    ckpt_model = OmegaConf.select(ckpt_cfg, "model", default=None)
    live_model = OmegaConf.select(cfg, "model", default=None)
    if ckpt_model is None or live_model is None:
        return
    overridden, _live_only, _ckpt_only = diff_model_cfgs(ckpt_model, live_model)
    if not overridden:
        return
    msg = (
        f"[resume] live cfg.model disagrees with {os.path.join(ckpt_dir, 'config.yaml')} on "
        f"{len(overridden)} key(s). Resume continues the checkpoint's run, so the CKPT values win "
        "and the live ones are IGNORED (use training.finetune_ckpt_path to start a new run whose "
        "config overrides the checkpoint's):"
    )
    logger.warning(msg)
    print(msg, flush=True)
    for key, ckpt_val, live_val in overridden:
        line = f"[resume]   model.{key}: using {_short(ckpt_val)} (ckpt), ignoring {_short(live_val)} (live)"
        logger.warning(line)
        print(line, flush=True)


def _load_ckpt_cfg(ckpt_dir: str) -> DictConfig:
    """Read a run dir's ``config.yaml``.

    Args:
        ckpt_dir: str, checkpoint (run) dir.

    Returns:
        DictConfig. Raises ValueError when it has no ``model`` section (not a dir written by this trainer).
    """
    path = os.path.join(ckpt_dir, "config.yaml")
    ckpt_cfg = OmegaConf.load(path)
    if OmegaConf.select(ckpt_cfg, "model", default=None) is None:
        raise ValueError(
            f"{path} has no `model:` section, so there is nothing to build the architecture from. Point "
            "training.finetune_ckpt_path / resume_ckpt_path at a run directory written by this trainer."
        )
    return ckpt_cfg


def build_for_finetune(source: str, cfg: DictConfig):
    """Warm-start: the checkpoint's architecture with the live model config on top, weights loaded strictly.

    Args:
        source: str, ``training.finetune_ckpt_path`` (a run dir, or one ``checkpoint_step_*.safetensors`` in it).
        cfg: the live run config; ``cfg.model`` is replaced by the merged model config (what save_config writes).

    Returns:
        BaseWAMArchitecture with every weight loaded from the requested / latest safetensors.
    """
    from openwam.model import build_architecture_from_ckpt_dir

    ckpt_dir, explicit_weights = _resolve_ckpt_path(source)
    model_cfg = merge_ckpt_model_cfg(_load_ckpt_cfg(ckpt_dir), cfg, ckpt_dir)
    logger.info("[finetune] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, model_cfg)
    weights = explicit_weights or find_latest_weights(ckpt_dir)
    # Plain print so the warm-start is visible on the launch terminal even
    # when logger output is drowned out; non-main ranks have print disabled.
    print(f"[finetune] loading pretrained weights: {weights}", flush=True)
    architecture.load_checkpoint(weights)
    print("[finetune] pretrained weights loaded OK", flush=True)
    return architecture


def build_for_resume(source: str, cfg: DictConfig):
    """Resume: the checkpoint's own architecture, with zeroed storage for accelerate ``load_state`` to fill.

    The safetensors are not read: ``load_state`` restores a strict superset (module weights incl. frozen
    params) after ``prepare``, and the latest safetensors may be truncated by a mid-save kill even though the
    run is still recoverable from its accel state.

    Args:
        source: str, ``training.resume_ckpt_path`` (the run dir).
        cfg: the live run config; only compared with the ckpt's (the ckpt config wins, divergence is logged).

    Returns:
        BaseWAMArchitecture with real (zero) storage everywhere and the shell's real buffers kept.
    """
    from openwam.model import build_architecture_from_ckpt_dir
    from openwam.model.video_backbone.common.weights import materialize_meta

    ckpt_dir, _explicit_weights = _resolve_ckpt_path(source)
    ckpt_cfg = _load_ckpt_cfg(ckpt_dir)
    warn_live_model_cfg_ignored(ckpt_cfg, cfg, ckpt_dir)
    logger.info("[resume] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, ckpt_cfg.model)
    # set_dtype_device runs before load_state and cannot copy out of meta tensors.
    materialize_meta(architecture, "cpu")
    return architecture
