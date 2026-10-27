"""Load one checkpoint for deploy: the architecture with its weights, and the normalizer of its action space.

``load_architecture`` rebuilds the empty architecture from the checkpoint's model config (each video backbone family
has its own source of pretrained structure, see below), loads every weight from the checkpoint file, then moves it to
the requested dtype / device in eval mode. ``build_normalizer`` builds the raw-space normalizer the dataset trained
with from the run's ``normalization_stats.npy``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

# training.mixed_precision -> the dtype the weights are served in
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


def load_architecture(model_cfg: DictConfig, ckpt_dir: str, ckpt_name: str, *, dtype: torch.dtype,
                      device: str) -> BaseWAMArchitecture:
    """Build the checkpoint's architecture and load its weights.

    Args:
        model_cfg: the checkpoint's ``model`` config block.
        ckpt_dir: run dir (holds the self-contained component artifacts some backbones need, e.g. ``reason1/``).
        ckpt_name: ``checkpoint_step_N.safetensors`` inside ``ckpt_dir``.
        dtype: weight dtype (``DTYPES[training.mixed_precision]``).
        device: e.g. ``"cuda:0"``.
    Returns:
        the architecture on ``device`` in ``dtype``, eval mode, every weight from the checkpoint.
    """
    from openwam.model import build_architecture, resolve_architecture_config

    # Build the empty architecture (the video backbone is created inside it from its source).
    resolved_arch = resolve_architecture_config(model_cfg)
    params = dict(resolved_arch.params)

    # ``resolve_architecture_config`` may hand back ``params["video_backbone"]``
    # as a ``DictConfig``. Subsequent ``vb_params["_source"] = <dict>``
    # assignments would then be auto-promoted by OmegaConf to a ``DictConfig``,
    # which causes ``WanVideoBackbone.from_pretrained`` to mis-dispatch the
    # deploy source into the training path (``build_training_pipeline``
    # requires ``cfg.training``, which is absent from a video_backbone-only
    # subtree). Pinning to a plain ``dict`` here keeps the deploy dispatch
    # honest.
    vb_params = params.get("video_backbone")
    if vb_params is None:
        vb_params = {}
    elif not isinstance(vb_params, dict) or isinstance(vb_params, DictConfig):
        vb_params = OmegaConf.to_container(vb_params, resolve=True) or {}
    params["video_backbone"] = vb_params

    # Provide video backbone source from config components or model_path.
    vb_components = OmegaConf.select(model_cfg, "video_backbone.components", default=None)
    if vb_components is not None:
        logger.info("Using config-embedded component specs for video-backbone construction")
        vb_cfg_dict = OmegaConf.to_container(model_cfg.video_backbone, resolve=True)
        # CosmosPredict25 Reason1 self-containment: the ckpt's Reason1 weights
        # live in the unified safetensors and the small structural artifacts
        # (config.json + tokenizer.json) under ``<ckpt_dir>/reason1/`` —
        # ``save_deploy_assets`` writes the component marker and the artifact
        # dir together. Clearing ``text_encoder_path`` makes
        # ``build_cosmos_predict25_pipeline`` take its deploy/empty-shell path;
        # a missing ``reason1/`` dir then hard-errors inside ``from_empty``.
        # Iterate the plain-dict copy. ``vb_components`` is an OmegaConf
        # ``ListConfig`` whose entries are ``DictConfig`` (NOT a ``dict``
        # subclass) — so ``isinstance(c, dict)`` would always be False
        # against the real saved config, silently skipping the marker.
        # ``source: state_dict`` distinguishes the Reason1 marker from Wan's
        # umt5 ``text_encoder`` component (a ``model_class`` instantiation spec).
        has_reason1_state_component = any(
            isinstance(c, dict) and c.get("attr") == "text_encoder" and c.get("source") == "state_dict"
            for c in (vb_cfg_dict.get("components") or [])
        )
        if has_reason1_state_component:
            prev_path = vb_cfg_dict.get("text_encoder_path")
            vb_cfg_dict["text_encoder_path"] = None
            logger.info(
                "Using self-contained Reason1 artifacts from %s (clearing external text_encoder_path=%r)",
                os.path.join(ckpt_dir, "reason1"),
                prev_path,
            )
        # Symmetric with the Reason1 clearing above: when the ckpt carries a VAE
        # state component, its weights live in the unified safetensors (via
        # ``vae``), so any training-time ``vae_path`` must be cleared.
        # Otherwise ``build_cosmos_predict25_pipeline``'s deploy guard
        # (``ckpt_dir is not None and vae_path_override is None``) stays False and
        # ``_resolve_vae_path`` raises FileNotFoundError on a host lacking the
        # original path, before weights ever load.
        has_vae_state_component = any(
            isinstance(c, dict) and c.get("attr") == "vae" for c in (vb_cfg_dict.get("components") or [])
        )
        if has_vae_state_component and vb_cfg_dict.get("vae_path") is not None:
            prev_vae_path = vb_cfg_dict.get("vae_path")
            vb_cfg_dict["vae_path"] = None
            logger.info(
                "Using self-contained VAE weights from checkpoint (clearing external vae_path=%r)",
                prev_vae_path,
            )
        vb_params["_source"] = vb_cfg_dict
        vb_params["_ckpt_dir"] = ckpt_dir
    else:
        model_path = OmegaConf.select(model_cfg, "video_backbone.model_path", default=None)
        if model_path is not None:
            vb_name = str(OmegaConf.select(model_cfg, "video_backbone.name", default=""))
            if vb_name.startswith(("cosmos_predict25_", "cosmos3_")):
                # Cosmos families carry fields the path-only string source would
                # lose (predict2.5: `shift_video` / `model_variant` /
                # `text_encoder_path`; cosmos3: `shift_video` / prompt knobs).
                # `from_pretrained` accepts a dict natively (via
                # `_video_backbone_cfg`). Wan never hits this branch.
                vb_params["_source"] = {k: v for k, v in vb_params.items() if not str(k).startswith("_")}
            elif vb_name in ("ge_base_fast", "flux3_action_base"):
                # No self-contained components: rebuild the pretrained skeleton from model_path the way the
                # trainer's resume does (ckpt_model_loader: the run's video_backbone cfg + its ckpt dir); the
                # load_checkpoint below then overwrites every weight.
                vb_params["_source"] = OmegaConf.to_container(model_cfg.video_backbone, resolve=True)
                vb_params["_ckpt_dir"] = ckpt_dir
            else:
                vb_params["_source"] = str(model_path)
        else:
            logger.warning(
                "No components or model_path in config; architecture __init__ will attempt to build from config."
            )

    # tri_system: use VLM checkpoint saved in the checkpoint dir instead of
    # the external training-time path. The trainer copies the VLM directory
    # to ``<ckpt_dir>/vlm_backbone/`` so deploy is self-contained.
    if resolved_arch.canonical.framework == "tri_system":
        vlm_cfg = params.get("vlm_backbone", {})
        if not isinstance(vlm_cfg, dict):
            vlm_cfg = OmegaConf.to_container(vlm_cfg, resolve=True) or {}
            params["vlm_backbone"] = vlm_cfg
        vlm_dir = os.path.join(ckpt_dir, "vlm_backbone")
        if os.path.isdir(vlm_dir):
            vlm_cfg["checkpoint_path"] = vlm_dir
            logger.info("Using self-contained VLM checkpoint from %s", vlm_dir)

    architecture = build_architecture(resolved_arch.registry_name, params)
    logger.info("Architecture: %s (framework=%s variant=%s)", resolved_arch.registry_name,
                resolved_arch.canonical.framework, resolved_arch.canonical.variant)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger.info("Loading checkpoint: %s (dtype=%s, device=%s)", ckpt_path, dtype, device)
    architecture.load_checkpoint(ckpt_path)
    # Top-down: architecture -> video backbone -> submodules.
    architecture.set_dtype_device(dtype, torch.device(device))
    architecture.eval()
    return architecture


def build_normalizer(dataloader_cfg: DictConfig, ckpt_dir: str):
    """The raw-space normalizer the checkpoint trained with (yam_sharpa: per ``action_mode``).

    Args:
        dataloader_cfg: the checkpoint's ``dataloader`` block (``normalize_mode``, ``action_mode``).
        ckpt_dir: run dir holding ``normalization_stats.npy``.
    Returns:
        object with ``normalize(raw proprio (..., D))`` and ``unnormalize(model action (..., A))``: a ``Normalizer``,
        wrapped in :class:`_UnifyAwareNormalizer` for ``unify_*`` (the model's 80-D unified space <-> raw D).
    """
    from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE, Normalizer, load_mode_stats
    from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec
    from openwam.dataloader.utils.yam_sharpa_repr import is_unify, unify_action_map

    action_mode, norm_mode = str(dataloader_cfg.action_mode), str(dataloader_cfg.normalize_mode)
    stats_path = os.path.join(ckpt_dir, "normalization_stats.npy")
    stats = load_mode_stats(stats_path, action_mode)
    if stats is None:
        raise KeyError(f"{stats_path} has no {action_mode!r} entry; regenerate it for this checkpoint's action_mode")
    if norm_mode not in YAML_TO_NORM_MODE:
        raise ValueError(f"normalize_mode={norm_mode!r} is not one of {sorted(YAML_TO_NORM_MODE)}")
    normalizer = Normalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=stats)
    logger.info("[normalizer] mode=%s action_mode=%s dim=%d (%s)", norm_mode, action_mode, len(stats["mean"]), stats_path)
    if not is_unify(action_mode):
        return normalizer
    # unify_*: the model works in the unified 80-D space; stats are in the raw (pre-scatter) space.
    dst_index = parse_unify_spec(unify_action_map(action_mode), UNIFY_DIM)
    return _UnifyAwareNormalizer(normalizer, dst_index, UNIFY_DIM)


class _UnifyAwareNormalizer:
    """Inverts the train-time ``normalize -> map_to_unify`` of ``unify_*`` checkpoints.

    * ``unnormalize`` (model action out): gather unified (..., unify_dim) -> raw (..., D), then unnormalize.
    * ``normalize`` (proprio in): normalize raw (..., D), then scatter to unified (..., unify_dim).

    Args:
        inner: the raw-space ``Normalizer``.
        dst_index: (D,) int, unified slot of every raw dim (actions and proprio share it).
        unify_dim: int, width of the unified space (80).
    """

    def __init__(self, inner, dst_index: np.ndarray, unify_dim: int):
        from openwam.dataloader.utils.unify_action import map_to_unify, unmap_from_unify

        self._inner = inner
        self._dst_index = np.asarray(dst_index, dtype=np.int64)
        self._unify_dim = int(unify_dim)
        self._map_to_unify, self._unmap_from_unify = map_to_unify, unmap_from_unify

    def unnormalize(self, x):
        """x: (..., unify_dim) normalized unified action. Returns: (..., D) physical raw action."""
        arr = np.asarray(x)
        if arr.shape[-1] != self._unify_dim:
            raise ValueError(f"unify normalizer: expected (..., {self._unify_dim}) actions, got {arr.shape}")
        return self._inner.unnormalize(self._unmap_from_unify(arr, self._dst_index))

    def normalize(self, x):
        """x: (..., D) raw proprio. Returns: (..., unify_dim) normalized unified proprio."""
        unified, _mask = self._map_to_unify(self._inner.normalize(np.asarray(x)), self._dst_index, self._unify_dim)
        return unified
