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
ckpt config stays authoritative there and any live divergence raises
(:func:`check_resume_cfg`).
"""

import logging
import os

from omegaconf import DictConfig, OmegaConf, open_dict

from openwam.model.architectures.base import TrainOptions
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


# Keys a resume launch legitimately sets differently from the run's config.yaml: where to resume
# from, and the fresh-run dir fields a resume launch drops (the run dir is resume_ckpt_path's).
RESUME_LAUNCH_KEYS = ("training.resume_ckpt_path", "training.output_path", "training.run_dir_name")
# Subtrees only the saved config.yaml carries: save_assets_for_deployment writes the specs and resolve_config the
# layout before save_config (resolve_config re-derives and checks the layout of the loaded file).
SAVED_ONLY_KEYS = ("model.video_backbone.components", "model.video_backbone.tokenizer", "common.layout")


def _under(key: str, prefixes: tuple) -> bool:
    """True if dotted ``key`` is one of ``prefixes`` or lies below one of them."""
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def diff_resume_cfg(saved_cfg: DictConfig, live_cfg: DictConfig) -> list:
    """Every key where a resume launch's config disagrees with the run's saved config.yaml.

    Args:
        saved_cfg: DictConfig, the run's ``config.yaml``.
        live_cfg: DictConfig, the config this launch composed (yaml defaults + CLI).

    Returns:
        sorted list of ``(dotted_key, saved_value, live_value)``; a side that lacks the key holds
        ``"<absent>"``. ``RESUME_LAUNCH_KEYS`` and saved-only ``SAVED_ONLY_KEYS`` are not differences.
    """
    # Unresolved: compare what the yaml says, not what interpolations evaluate to on this host.
    saved = _flatten_cfg(OmegaConf.to_container(saved_cfg, resolve=False) or {})
    live = _flatten_cfg(OmegaConf.to_container(live_cfg, resolve=False) or {})
    absent = "<absent>"
    diffs = []
    for key in sorted(saved.keys() | live.keys()):
        if _under(key, RESUME_LAUNCH_KEYS) or (key not in live and _under(key, SAVED_ONLY_KEYS)):
            continue
        s, v = saved.get(key, absent), live.get(key, absent)
        if s != v:
            diffs.append((key, s, v))
    return diffs


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
    # Only the resume-location keys come from the launch.
    with open_dict(saved):
        for key in RESUME_LAUNCH_KEYS:
            OmegaConf.update(saved, key, OmegaConf.select(live_cfg, key, default=None), force_add=True)
    return saved


_LAYOUT_SOURCE_KEYS = ("history_offsets", "future_offsets", "num_frames", "history_sampling", "history_window")


def check_resume_layout(cfg: DictConfig, ckpt_cfg: DictConfig, ckpt_dir: str) -> None:
    """Reject a resume whose live sample layout differs from the checkpoint's.

    The resumed model is built from ``ckpt_cfg.common.layout`` while the training dataset is built from the live
    ``cfg.dataloader``; a disagreement in the layout or in the values it comes from would train the checkpoint on
    other samples without any error.

    Args:
        cfg: the live run config, resolved (``cfg.common.layout`` set by the entry point).
        ckpt_cfg: the checkpoint's config, resolved.
        ckpt_dir: str, the checkpoint dir (error message only).

    Returns:
        None. Raises ValueError naming every differing layout field and every differing ``_LAYOUT_SOURCE_KEYS``
        dataloader key, each with its live and ckpt value.
    """
    live = OmegaConf.to_container(cfg.common.layout)
    ckpt = OmegaConf.to_container(ckpt_cfg.common.layout)
    # Both come from the same latent_layout(), so they have the same fields.
    diffs = [f"common.layout.{k}: live {live[k]!r} vs ckpt {ckpt[k]!r}" for k in sorted(live) if live[k] != ckpt[k]]
    for key in _LAYOUT_SOURCE_KEYS:
        # Plain containers, so list values compare by content.
        live_val = OmegaConf.to_container(cfg.dataloader)[key]
        ckpt_val = OmegaConf.to_container(ckpt_cfg.dataloader)[key]
        if live_val != ckpt_val:
            diffs.append(f"dataloader.{key}: live {live_val!r} vs ckpt {ckpt_val!r}")
    if not diffs:
        return
    raise ValueError(
        f"[resume] the live sample layout differs from {os.path.join(ckpt_dir, 'config.yaml')}'s: "
        f"{'; '.join(diffs)}. The model and its training so far follow the checkpoint's layout but the dataset is "
        "built from the live config; set these live keys to the checkpoint's values, or start a new run with "
        "training.finetune_ckpt_path."
    )


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


def build_for_finetune(source: str, cfg: DictConfig, train_options: TrainOptions):
    """Warm-start: the checkpoint's architecture with the live model config on top, weights loaded strictly.

    Args:
        source: str, ``training.finetune_ckpt_path`` (a run dir, or one ``checkpoint_step_*.safetensors`` in it).
        cfg: the live run config, already resolved (``resolve_config``); ``cfg.model`` is replaced by the merged
            model config (what save_config writes) and the layout is re-checked against it.
        train_options: TrainOptions the trainer decided (passed to the architecture constructor).

    Returns:
        BaseWAMArchitecture with every weight loaded from the requested / latest safetensors.
    """
    from openwam.config import resolve_config
    from openwam.model import build_architecture_from_ckpt_dir

    ckpt_dir, explicit_weights = _resolve_ckpt_path(source)
    model_cfg = merge_ckpt_model_cfg(_load_ckpt_cfg(ckpt_dir), cfg, ckpt_dir)
    # The merge may change model keys the layout derives from: re-resolve (checks cfg.common.layout still holds).
    resolve_config(cfg)
    logger.info("[finetune] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, model_cfg, cfg.common.layout, train_options)
    weights = explicit_weights or find_latest_weights(ckpt_dir)
    # Plain print so the warm-start is visible on the launch terminal even
    # when logger output is drowned out; non-main ranks have print disabled.
    print(f"[finetune] loading pretrained weights: {weights}", flush=True)
    architecture.load_checkpoint(weights)
    print("[finetune] pretrained weights loaded OK", flush=True)
    return architecture


def build_for_resume(source: str, cfg: DictConfig, train_options: TrainOptions):
    """Resume: the checkpoint's own architecture, with zeroed storage for accelerate ``load_state`` to fill.

    The safetensors are not read: ``load_state`` restores a strict superset (module weights incl. frozen
    params) after ``prepare``, and the latest safetensors may be truncated by a mid-save kill even though the
    run is still recoverable from its accel state.

    Args:
        source: str, ``training.resume_ckpt_path`` (the run dir).
        cfg: the run config, resolved; only compared with the ckpt's (any difference outside the resume-location
            keys raises, see :func:`check_resume_cfg`).
        train_options: TrainOptions the trainer decided (passed to the architecture constructor).

    Returns:
        (architecture, ckpt_cfg): the BaseWAMArchitecture with real (zero) storage everywhere and the shell's real
        buffers kept, and the resolved checkpoint config it was built from (the run's effective config: every
        model-side decision of the resumed run reads it, never the live one).
    """
    from openwam.config import resolve_config
    from openwam.model import build_architecture_from_ckpt_dir
    from openwam.model.video_backbone.common.weights import materialize_meta

    ckpt_dir, _explicit_weights = _resolve_ckpt_path(source)
    ckpt_cfg = resolve_config(_load_ckpt_cfg(ckpt_dir))
    check_resume_layout(cfg, ckpt_cfg, ckpt_dir)
    # scripts/train.py already swapped in the saved config (load_resume_cfg); this guards direct callers.
    check_resume_cfg(ckpt_cfg, cfg, os.path.join(ckpt_dir, "config.yaml"))
    logger.info("[resume] building architecture from ckpt dir: %s", ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, ckpt_cfg.model, ckpt_cfg.common.layout, train_options)
    # set_dtype_device runs before load_state and cannot copy out of meta tensors.
    materialize_meta(architecture, "cpu")
    return architecture, ckpt_cfg
