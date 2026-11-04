"""Abstract base class for WAM (World-Action Model) architectures.

Supported architecture families:

1. **Dual-System** (`framework=dual_system`)
   A separate ActionDiT consumes features from the video DiT. Variants:
   `joint_cross_attn` (bridge cross-attention after a full video forward)
   / `joint_self_attn` (MMDiT-style mixed attention at every layer, driven
   by :class:`DualSystemMoTDriver`).

Each architecture composes the backbones it owns and implements its own
``forward()``.
"""

import dataclasses
import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from openwam.model.compile_options import compile_enabled
from openwam.model.video_backbone.registry import LoadMode


def _wrap_single_forward(module: nn.Module) -> None:
    """Wrap a single module's ``forward`` in ``torch.no_grad``. Idempotent."""
    if getattr(module, "_openwam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    module._openwam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """Wrap ``forward`` of ``module`` AND every submodule in its subtree in ``torch.no_grad``.

    Recursion matters because callers commonly bypass the root forward and call a
    nested submodule directly (e.g. a frozen HF model whose inner ``.model`` is called
    to skip the LM head) — wrapping only the root ``forward`` would leave that path
    grad-tracking. Recursively wrapping every descendant makes the semantic complete:
    any entry point into the frozen subtree is in ``no_grad``.

    Idempotent — a marker attribute on each module prevents double-wrapping if
    ``freeze_modules`` runs more than once. ``nn.Module.modules()`` deduplicates
    via its internal memo, so cyclic registrations (e.g. test fakes with
    ``self.model = self``) are visited once.

    Safe to apply to any module: if the caller already wraps the call in a
    ``no_grad`` context (e.g. ``prepare_inputs``), the inner ``no_grad`` is a
    no-op; if the caller is inside a grad-tracking forward, it short-circuits
    activation saving.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen base,
    or training only the head of an otherwise frozen model), do NOT pass the
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want
    frozen instead. The current freeze list in ``configs/model/*.yaml``
    only names complete subtrees, so this limitation does not bite today.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openwam.model.action_backbone.base import ActionDiTBackbone
    from openwam.model.video_backbone.base import TextCondition, VideoBackbone


@dataclass
class ActionState:
    """Mutable state container used by the joint self-attention path.

    Only ``DualSystemSelfAttnArchitecture`` needs this — its action stream is
    threaded through ``DualSystemMoTDriver``, which mutates the payload across
    layers. DualSystem cross-attn doesn't go through this
    container.

    Fields:
        action_latents: (B, T_action, action_dim) noisy actions (input to forward).
        timestep: action diffusion timestep (raw shape preserved for the action
            backbone's internal use).
        payload: backbone-specific per-forward state (typically
            ``ActionDiTState``).
    """

    action_latents: Optional[Tensor] = None
    timestep: Optional[Tensor] = None
    payload: Optional[Any] = None


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Composes a ``video_backbone`` and an ``action_backbone`` plus optional
    extra backbones. Subclasses instantiate the appropriate action backbone
    subclass in ``__init__`` and own the complete ``forward()`` control flow.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
        mode: LoadMode the entry point chose for the video backbone (PRETRAINED: upstream weights;
            CHECKPOINT: an empty shell of ``ckpt_dir`` the caller fills). The entry points (trainer, deploy
            loader, via ``build_architecture``) always pass it; the default only serves direct test builds.
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.
    """

    # --- Construction & config resolution ---

    def __init__(self, cfg=None, *, mode: LoadMode = LoadMode.PRETRAINED, ckpt_dir: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["ActionDiTBackbone"] = None
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Forward-time training runtime flags. Trainer calls
        # ``set_training_runtime`` once during construction so ``prepare_inputs``
        # can read these without the trainer having to thread them through.
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        # Optional action normalizer for deployment. ``generate`` uses it to
        # return real-scale actions; deploy-side proprio preprocessing uses it
        # to normalize raw robot state into the model's training space.
        self.normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg, mode, ckpt_dir)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg, mode: LoadMode, ckpt_dir: Optional[str]) -> None:
        """Build ``self.video_backbone`` from ``cfg.video_backbone`` with the constructor ``mode`` selects.

        Args:
            cfg: architecture params; ``cfg["video_backbone"]`` is the backbone's DictConfig slice (absent: no
                backbone, e.g. tests that attach a mock).
            mode: LoadMode decided by the entry point. ckpt_dir: str for CHECKPOINT, None for PRETRAINED.

        Returns:
            None. Raises ValueError when the yaml's temporal contract or required mask mode disagrees.
        """
        from openwam.model.video_backbone import build_video_backbone

        vb_cfg = cfg.get("video_backbone") if isinstance(cfg, dict) else getattr(cfg, "video_backbone", None)
        if vb_cfg is None:
            return

        # A backbone yaml may pin the architecture's video mask (LongLive: the Long-WAM history mask).
        # Checked before the build so a wrong launch fails before the weight load.
        required_mask = self._cfg_get(vb_cfg, "required_video_attention_mask_mode", None)
        actual_mask = self._cfg_get(cfg, "video_attention_mask_mode", None)
        if required_mask is not None and actual_mask != required_mask:
            raise ValueError(
                f"video_backbone {vb_cfg.name!r} requires model.architecture.video_attention_mask_mode="
                f"{required_mask!r}, got {actual_mask!r}"
            )
        self.video_backbone = build_video_backbone(vb_cfg, mode, ckpt_dir)

        # Cross-check: yaml-declared temporal contract must match what the
        # backbone actually exposes. Drift here would let the dataloader
        # enforce the wrong divisibility rule and let the mask-downsampler
        # produce a wrong-length tail, so we fail-fast at backbone init.
        declared_tc = self._cfg_get(vb_cfg, "temporal_compression", 4)
        declared_causal = self._cfg_get(vb_cfg, "causal_temporal", True)
        actual_tc = self.video_backbone.temporal_compression
        actual_causal = self.video_backbone.causal_temporal
        if (declared_tc, declared_causal) != (actual_tc, actual_causal):
            raise ValueError(
                f"video_backbone.temporal_compression / causal_temporal yaml "
                f"({declared_tc}, {declared_causal}) does not match native VAE "
                f"({actual_tc}, {actual_causal}). Update the yaml fields to match."
            )

    def _resolve_video_dim(self, cfg) -> int:
        """Resolve video_dim from config or video_backbone; raise if neither provides it."""
        dim = int(cfg.get("video_dim", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "video_dim", 0))
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError("video_dim must be specified in config or inferred from video_backbone")
        return dim

    # --- Backbone composition ---

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones should override this to include
        them. The returned dict
        is used by ``init_training_schedulers``, ``set_dtype_device``,
        ``move_frozen_to_device``, and ``save_assets_for_deployment`` to iterate
        over all backbones generically.
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """Flow-matching scheduler for the action stream (owned by action_backbone)."""
        if self.action_backbone is None:
            raise RuntimeError("action_backbone is not initialized")
        return self.action_backbone.scheduler

    @property
    def video_scheduler(self):
        """Flow-matching scheduler for the video stream (owned by video_backbone)."""
        if self.video_backbone is None:
            raise RuntimeError("video_backbone is not initialized")
        return self.video_backbone.scheduler

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        return self.action_backbone.bridge_layers if self.action_backbone is not None else ()

    @property
    def uses_proprioception(self) -> bool:
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None and self.action_backbone.uses_proprioception
        )

    # --- Proprio-as-context conditioning ---

    def _init_proprio_context(self, cfg, *, text_dim: int = 4096) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning."""
        enabled = bool(self._cfg_get(cfg, "use_proprioception", False))
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_slot_emb: Optional[nn.Embedding] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        # N > 1 proprio tokens (one per history frame, oldest first; sparse history).
        self.num_proprio_tokens = int(self._cfg_get(cfg, "num_proprio_tokens", 1) or 1)
        if not enabled:
            return
        state_dim = int(self._cfg_get(cfg, "state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError("use_proprioception=True requires explicit state_dim for context-token proprio.")
        self.proprio_dim = state_dim
        self.proprio_encoder = nn.Linear(state_dim, self.context_dim)
        if self.num_proprio_tokens > 1:
            # Learned absolute slot PE: proprio is only read through cross-attn /
            # prefix KV, which carry no RoPE. Zero-init = starts as N plain tokens.
            # A child module (not a bare Parameter) so the optimizer's top-level
            # module scan picks it up.
            self.proprio_slot_emb = nn.Embedding(self.num_proprio_tokens, self.context_dim)
            nn.init.zeros_(self.proprio_slot_emb.weight)

    def _append_proprio_context_token(
        self, text: "TextCondition", proprio: Optional[Tensor], sample_mask: Optional[Tensor]
    ) -> "TextCondition":
        """Append the proprio token(s) to the text context and extend its mask.

        Args:
            text: TextCondition, context (B, L, D_text) and context_mask (B, L).
            proprio: (B, D), (B, 1, D) or (B, N, D) normalized proprio (N = num_proprio_tokens); unused when
                proprio-as-context is off.
            sample_mask: optional per-sample proprio mask, (B,), (B, 1) or (B, 1, D) bool (any dim True =
                the sample has proprio). Masked samples get a zero token and a False mask entry, which also
                isolates the proprio_encoder gradient on them (input * 0 -> weight grad = 0).

        Returns:
            TextCondition with context (B, L + N, D_text) and context_mask (B, L + N) (``extra`` shared);
            ``text`` itself when proprio-as-context is off.
        """
        if not bool(getattr(self, "_use_proprioception_context", False)):
            return text
        if self.proprio_encoder is None:
            raise RuntimeError("proprio context is enabled but proprio_encoder is not initialized.")
        if proprio is None:
            raise ValueError("use_proprioception=True requires `proprio` from sample['proprio'] or obs['state'].")
        n_tok = int(getattr(self, "num_proprio_tokens", 1) or 1)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0, :]
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)  # (B, 1, D)
        if proprio.ndim != 3 or proprio.shape[1] != n_tok:
            raise ValueError(
                f"proprio must be [B, D] / [B, 1, D] (1 token) or [B, {n_tok}, D] ({n_tok} tokens), "
                f"got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[-1]}")

        context = text.context
        if context.shape[0] != proprio.shape[0]:
            if proprio.shape[0] == 1 and context.shape[0] > 1:
                proprio = proprio.expand(context.shape[0], -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch between context and proprio: {context.shape[0]} vs {proprio.shape[0]}"
                )

        # Normalize sample_mask to (B, 1) bool on context's device. Accepted shapes (after stacking in
        # prepare_inputs): (B,) enable-per-sample, (B, 1), or (B, 1, D) per-dim mask whose sample-level
        # enable is ``mask.any(dim=-1)``.
        if sample_mask is None:
            sample_mask = torch.ones(
                (proprio.shape[0], 1),
                dtype=torch.bool,
                device=context.device,
            )
        else:
            sample_mask = sample_mask.to(device=context.device, dtype=torch.bool)
            if sample_mask.ndim == 3:
                # (B, 1, D) -> (B, 1): True if any per-dim slot is real
                sample_mask = sample_mask.any(dim=-1)
            elif sample_mask.ndim == 1:
                sample_mask = sample_mask.unsqueeze(-1)
            if sample_mask.shape != (proprio.shape[0], 1):
                raise ValueError(
                    f"_proprio_sample_mask shape {tuple(sample_mask.shape)} must be ({proprio.shape[0]}, 1)"
                )

        # (B, N, context_dim): one token per proprio frame.
        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=self.proprio_encoder.weight.dtype)
        )
        if getattr(self, "proprio_slot_emb", None) is not None:
            proprio_token = proprio_token + self.proprio_slot_emb.weight.to(proprio_token.dtype)
        proprio_token = proprio_token.to(dtype=context.dtype)
        # Token * mask cuts the weight grad for mask=False samples
        # (input goes to zero so the encoder weight gradient contribution is zero).
        sample_mask_f = sample_mask.to(proprio_token.dtype).unsqueeze(-1)  # (B, 1, 1)
        proprio_token = proprio_token * sample_mask_f
        sample_mask = sample_mask.expand(-1, proprio_token.shape[1])  # (B, N)

        # The proprio tokens sit after the (possibly padded) text, so the valid tokens are not a contiguous
        # prefix any more: the mask is authoritative.
        context_mask = text.context_mask.to(device=context.device, dtype=torch.bool)
        return dataclasses.replace(
            text,
            context=torch.cat([context, proprio_token], dim=1),
            context_mask=torch.cat([context_mask, sample_mask], dim=1),
        )

    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Dispatch to each backbone — they own their own dtype/device handling."""
        self._dtype = dtype
        self._device = device
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        slot_emb = getattr(self, "proprio_slot_emb", None)
        if slot_emb is not None:
            slot_emb.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    # --- Normalizer (deployment) ---

    def attach_normalizer(self, normalizer) -> None:
        """Attach (or clear) an action normalizer used by ``generate``.

        Deployment paths build the same normalizer used by training from
        ``normalization_stats.npy``. ``generate`` uses it to return real-scale actions,
        while server-side proprio preprocessing uses it to normalize raw robot
        state into the model's training space. Pass ``None`` to clear.
        """
        self.normalizer = normalizer

    def normalize_deploy_proprio(self, proprio):
        """Normalize raw deploy proprio (array-like) into a float32 tensor; ``None`` passes through.

        The denoising loop re-casts to the model device/dtype, so a CPU tensor is fine.
        """
        if proprio is None:
            return None

        import numpy as np
        import torch

        arr = np.asarray(proprio, dtype=np.float32)
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            arr = normalizer.normalize(arr)
        return torch.from_numpy(arr)

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str, *, state_dict: dict | None = None) -> None:
        """Save architecture state to safetensors.

        ``state_dict`` defaults to ``self.state_dict()`` (deploy export); the
        trainer passes a gathered state_dict (ZeRO/DDP all-gather) instead.
        """
        from safetensors.torch import save_file

        if state_dict is None:
            state_dict = self.state_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load architecture state from a safetensors checkpoint.

        Missing or unexpected keys raise under ``strict=True``.

        Meta-device sub-modules (self-contained deploy empty shells built
        via ``from_empty`` / ``init_empty_weights``) need
        ``load_state_dict(..., assign=True)`` — the default in-place copy is
        a silent no-op against meta tensors and leaves the shells
        unpopulated. ``assign=True`` rebinds the parameter slot to the
        safetensors tensor instead. We only flip the flag when meta params
        actually exist so the training-resume path (real-device params,
        in-place copy preserves identity) is unchanged.
        """
        from safetensors.torch import load_file

        from openwam.model.video_backbone.common.weights import retie, tied_groups

        state_dict = load_file(path)
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        # assign=True would untie tied weights (frozen text encoders' embeddings); re-tie them after.
        groups = tied_groups(self)
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=has_meta)
        if strict and (missing or unexpected):
            raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        retie(self, groups)

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training.

        Single source of truth for each stream's α-shift:
        ``video_backbone.shift_video`` and ``action_backbone.shift_action``.
        The same properties are read by the deploy schedule at inference time,
        so the discrete training sigma buffer and the inference denoising
        trajectory are sampled from the same shifted schedule — train/inference
        cannot drift, and the shift is owned by the checkpoint config (not a
        separate deploy-time knob).

        ``shift is None`` (the default for configs without an explicit override)
        falls back to each scheduler's template default (Wan/action = 5.0),
        i.e. bit-identical pre-shift behavior.
        """
        # ``getattr`` (rather than direct attribute access) so test doubles
        # / mocks that don't carry the shift property still work — they fall
        # back to the scheduler's template default, matching the production
        # no-override path.
        for name, bb in self.backbones.items():
            if not hasattr(bb, "scheduler"):
                continue
            kwargs = {"training": True}
            shift = getattr(bb, "shift_video" if name == "video_backbone" else "shift_action", None)
            if shift is not None:
                kwargs["shift"] = float(shift)
            bb.scheduler.set_timesteps(num_timesteps, **kwargs)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``prepare_inputs`` decorator, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph, the wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like
        ``video_backbone.text_encoder`` work naturally; unknown names
        are silently skipped, so a freeze list mentioning modules absent on a
        given architecture (a module that exists only on another backbone)
        is harmless.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                # Set eval mode on the frozen subtree. Use modules() instead
                # of .eval() to avoid infinite recursion when a submodule has
                # self-referential aliases (e.g. HF model.model = self).
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
                frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list: list[str] = ()) -> dict[str, nn.Module]:
        """Return top-level trainable sub-modules.

        Walks ``self.named_children()`` and returns modules that have at
        least one parameter with ``requires_grad=True``, excluding those
        in *freeze_list*. Used by ``optimizer_groups.build_trainable_parameters``
        to source the param groups for the optimizer.
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """Move named frozen modules to device.

        Searches via ``get_submodule`` on self first, then on each backbone.
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = None
                    try:
                        found = bb.get_submodule(name)
                    except (AttributeError, KeyError):
                        found = None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def save_assets_for_deployment(self, output_dir: str) -> None:
        """Make the checkpoint directory self-contained for deploy.

        The trainer calls it once when it creates the run dir. Each backbone's
        :meth:`VideoBackbone.save_deploy_assets` copies its non-weight files (tokenizer / processor /
        third-party ``config.json``) into ``output_dir``; the module geometry is already in ``config.yaml``.
        Every backbone base declares the hook (default no-op), so no probing here.

        Args:
            output_dir: str, the run dir.

        Returns:
            None.
        """
        for bb in self.backbones.values():
            bb.save_deploy_assets(output_dir)

    # --- Training: preprocessing ---

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """Set forward-time training flags consumed by ``prepare_inputs``.

        Trainers call this once during construction. Keeping these on the
        architecture keeps ``prepare_inputs(batch)`` self-contained — the
        trainer no longer needs to thread these flags through every loss call.
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Aggregate a list of dataset samples into a batched inputs dict.

        The returned dict is designed to be
        unpacked directly into ``compute_loss`` via ``**inputs``.

        Args:
            batch: List of dataset samples (each a dict). A single dict is
                accepted as well and treated as a one-sample batch.

        Returns:
            Dict ready for ``compute_loss``: ``clean_latents`` (B, z, N_clean, h, w) the clean prefix,
            ``input_latents`` (B, z, T_lat, h, w) the full clean clip, ``text`` (TextCondition), the action /
            proprio tensors and masks, and the forward-time flags.
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
        from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

        if isinstance(batch, dict):
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        all_frames: list = []
        all_prompts: list = []
        all_actions: list = []
        all_proprios: list = []
        all_proprio_masks: list = []
        all_action_masks: list = []
        all_video_masks: list = []

        for sample in samples:
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])

            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            # Carry proprio whenever the sample provides it — the main-stream
            # proprio-context path consumes it downstream, so the bridge into
            # ``inputs`` is not gated on a single consumer's flag.
            proprio = sample.get("proprio")
            if proprio is not None:
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                elif proprio.ndim == 2:
                    pass  # (N, D): one proprio per history frame (sparse history)
                else:
                    raise ValueError(
                        f"sample['proprio'] must be [D], [1, D] or [N, D], got shape {tuple(proprio.shape)}"
                    )
            all_proprios.append(proprio)

            # Collect per-sample proprio_mask. Two accepted shapes:
            #   * 1D ``(1,) bool`` — legacy "is the proprio token enabled".
            #   * 2D ``(1, D) bool`` — per-dim mask; sample-level enable is
            #     ``pmask.any(dim=-1)``. Used by RoboCOIN/EgoDex/RoboTwin/OXE
            #     after the 2D mask migration.
            # Default for readers that don't emit the field: all True (1,).
            # (Mixed 1-D / 2-D ranks across a batch are reconciled just before
            # the stack below, so the bare (1,) default is safe.)
            pmask = sample.get("proprio_mask")
            if pmask is None:
                pmask = torch.ones(1, dtype=torch.bool)
            else:
                if isinstance(pmask, np.ndarray):
                    pmask = torch.from_numpy(pmask)
                pmask = pmask.to(dtype=torch.bool)
                if pmask.ndim == 0:
                    pmask = pmask.unsqueeze(0)
            all_proprio_masks.append(pmask)

            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_action_masks.append(amask)
            all_video_masks.append(vmask)

        # Sparse history: the first H frames of every clip are single-frame history (0 = first-frame layout).
        # Transitional: read from the samples until Task 8 puts the layout in cfg.
        n_hist = {int(s.get("num_history_frames", 0) or 0) for s in samples}
        if len(n_hist) != 1:
            raise ValueError(f"Mixed num_history_frames in batch: {sorted(n_hist)}")
        clean_latents, input_latents, text = self.video_backbone.preprocess_train(
            all_frames, all_prompts, num_hist_frames=n_hist.pop()
        )

        action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

        inputs = {
            "clean_latents": clean_latents,
            "input_latents": input_latents,
            "text": text,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": action_data,
        }

        # Bridge proprio into inputs whenever the batch carries it (not gated on
        # uses_proprioception): the main-stream proprio-context path reads
        # inputs["proprio"]. Consumers that don't need it simply ignore it.
        if all_proprios[0] is not None:
            inputs["proprio"] = torch.stack(all_proprios, dim=0).contiguous()
            # Reconcile mixed 1-D (1,) / 2-D (1, D) proprio_masks before stacking:
            # promote any 1-D enable-flag to (1, D) (broadcasts the sample-level
            # flag across all dims) so a batch mixing a 2-D reader mask with a
            # 1-D default/external mask doesn't raise on rank mismatch. All-1-D
            # and all-2-D batches are left untouched.
            if len({m.ndim for m in all_proprio_masks}) > 1:
                pdim = max((m.shape[-1] for m in all_proprio_masks if m.ndim == 2), default=1)
                all_proprio_masks = [
                    m if m.ndim == 2 else m.reshape(m.shape[0], 1).expand(m.shape[0], pdim) for m in all_proprio_masks
                ]
            inputs["proprio_mask"] = torch.stack(all_proprio_masks, dim=0).contiguous()

        if all_action_masks[0] is not None:
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)
        if all_video_masks[0] is not None:
            # ``latent[0]`` is always clean conditioning (every layout has clean_latents), so the mask skips it.
            skip_first = True
            # Pass the backbone's temporal_compression so the tail-grouping
            # divisor matches the actual latent-T produced by the encoder.
            # The default 4 in ``downsample_video_mask_to_latent`` is the Wan
            # VAE legacy; for V-JEPA / other encoders it would silently emit
            # a wrong-length mask. See VideoBackbone.temporal_compression for
            # the source-of-truth contract.
            temporal_factor = int(self.video_backbone.temporal_compression)
            latent_masks = [
                downsample_video_mask_to_latent(~m, temporal_factor=temporal_factor, skip_first=skip_first)
                for m in all_video_masks
            ]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        return inputs

    # --- Training: loss computation ---

    def compute_loss(
        self,
        *,
        actions: Optional[torch.Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        **inputs,
    ) -> dict:
        """Compute joint video-action flow matching loss.

        This is the single entry point for training loss computation.
        Handles timestep sampling, noise injection, forward pass, and
        loss calculation internally.

        Callers should produce ``inputs`` via ``self.prepare_inputs(batch)``
        (preferred) or assemble it manually with the same keys: ``clean_latents`` /
        ``input_latents`` / ``text`` plus any of ``actions / proprio /
        action_is_pad / video_is_pad / use_gradient_checkpointing[_offload] /
        max_timestep_boundary / min_timestep_boundary``.

        Args:
            actions: (B, T_action, action_dim) ground truth actions. May also
                be passed via ``inputs["actions"]``.
            lambda_video: Weight for video loss term.
            lambda_action: Weight for action loss term.
            **inputs: Preprocessed video/text tensors plus forward-time flags.

        Returns:
            dict with keys: loss, loss_video, loss_action.
        """
        vb = self.video_backbone
        action_scheduler = self.action_backbone.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        B = inputs["input_latents"].shape[0]

        # --- Sample video timesteps ---
        # Backbones with a native training noise distribution (07: GE / flux)
        # expose ``sample_training_noise``; the rest use the shared scheduler grid.
        native_sampler = getattr(vb, "sample_training_noise", None)
        video_loss_weights = None
        if native_sampler is not None:
            video_timesteps, video_sigmas, video_loss_weights = native_sampler(B, device=_device, dtype=_dtype)
            video_timestep_ids = None
        else:
            video_timestep_ids = torch.randint(min_tb, max_tb, (B,))
            video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
            video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]

        # The clean prefix (1 first-frame latent or H history latents) stays clean.
        n_clean = inputs["clean_latents"].shape[2]
        inputs["latents"][:, :, :n_clean] = inputs["clean_latents"]

        # --- Prepare action noise ---
        noisy_actions, action_target, action_timesteps, action_timestep_ids, action_sigmas, a_sigma_bc = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if lambda_action > 0 and actions is not None:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            action_noise = torch.randn_like(actions)
            if action_sigmas.dim() == 1:
                a_sigma_bc = action_sigmas.view(B, 1, 1)
            else:
                a_sigma_bc = action_sigmas.unsqueeze(-1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # --- Joint forward pass ---
        forward_inputs = dict(inputs)
        proprio = forward_inputs.pop("proprio", None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        # Padding masks are kept in `inputs` for loss-side masking but dropped
        # from `forward_inputs` so they don't leak into vb.prepare(). Per FastWAM
        # MoT design, attention itself does not consume sample-level padding.
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)

        # Route per-sample proprio_mask through pipeline_inputs to
        # ``_append_proprio_context_token``. Internal-only key; pop'd there.
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask

        # Use ``self(...)`` (not ``self.forward(...)``) so ``nn.Module.__call__``
        # is invoked and any architecture-level forward-pre-hooks fire.
        video_noise_pred, action_noise_pred = self(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio=proprio,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            **forward_inputs,
            timestep=video_timesteps,
        )

        # --- Video loss ---
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            inputs,
            _device,
            timestep_weights=video_loss_weights,
        )

        if lambda_action == 0 or action_noise_pred is None:
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # --- Action loss ---
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        result = {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

        return result

    def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device, timestep_weights=None):
        """Per-sample weighted video MSE loss.

        ``timestep_weights`` (B,) overrides the scheduler's per-timestep weights
        (backbones with a native noise sampler, see ``compute_loss``).
        """
        import torch.nn.functional as F

        video_is_pad = inputs.get("video_is_pad")

        # The clean prefix is not a denoising target. Transitional: its length is read off clean_latents
        # until Task 8 puts it in cfg (layout.num_hist_latents).
        n_skip = int(inputs["clean_latents"].shape[2])
        noise_pred = noise_pred[:, :, n_skip:]
        target = target[:, :, n_skip:]

        vb = self.video_backbone
        if timestep_weights is not None:
            tw = timestep_weights.to(dtype=torch.float32, device=device)
        else:
            tw = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_frame = per_element.mean(dim=(1, 3, 4))

        if video_is_pad is not None:
            if video_is_pad.shape[-1] != noise_pred.shape[2]:
                raise ValueError(
                    f"video_is_pad length {video_is_pad.shape[-1]} does not match "
                    f"trimmed noise_pred T={noise_pred.shape[2]} (n_skip={n_skip}). "
                    "Expected mask sized to T_lat minus leading conditioning latents."
                )
            video_is_pad = video_is_pad.to(device=per_frame.device, dtype=torch.bool)
            valid_mask = ~video_is_pad
            per_frame = per_frame * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_frame.sum(dim=1) / valid_count
        else:
            per_sample = per_frame.mean(dim=1)

        return (per_sample * tw).mean()

    def _compute_action_loss(self, noise_pred, target, timestep_ids, scheduler, inputs, device):
        """Per-sample weighted action MSE loss.

        Supports two action_is_pad shapes:
          * **(B, T) bool** — legacy per-timestep mask (pre-2D migration).
            Each masked timestep contributes 0 to per-sample loss; per_sample =
            sum_t(loss_t) / N_valid_t, where loss_t = mean over D dims.
          * **(B, T, D) bool** — 2-D mask covering both time AND per-dim
            validity (new default for RoboCOIN/EgoDex/RoboTwin/OXE). Each
            (t, d) cell contributes only when mask[t, d] is True; per_sample =
            sum_{t,d}(loss_{t,d}) / N_valid_cells.

        The two paths are *mathematically equivalent* whenever the 2D mask
        is a broadcast of the 1D time mask across all D dims (i.e. every
        valid timestep has every dim valid): both reduce to
        sum_{t,d}(loss_{t,d}) / (N_valid_t * D). This equivalence is covered by
        the 2-D action-mask regression tests.

        A (T, 2) per-hand mask is handled by the legacy 2D path
        (rank-3 mask, per-element broadcasting), unchanged by this migration.
        """
        import torch.nn.functional as F

        tw = scheduler.training_weight(timestep_ids).to(dtype=torch.float32, device=device)
        if tw.ndim != 1:
            raise ValueError(f"action loss weights must be per-sample [B], got shape {tuple(tw.shape)}")
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

        action_is_pad = inputs.get("action_is_pad")

        if action_is_pad is None:
            per_sample = per_element.mean(dim=(1, 2))
            return (per_sample * tw).mean()

        action_is_pad = action_is_pad.to(device=per_element.device, dtype=torch.bool)
        valid_mask_f = (~action_is_pad).float()

        # New default path: (B, T, D) per-element mask that exactly matches
        # per_element's shape. Used by RoboCOIN/EgoDex/RoboTwin/OXE after the
        # 2D mask migration.
        if valid_mask_f.shape == per_element.shape:
            weighted = per_element * valid_mask_f
            per_sample = weighted.sum(dim=(1, 2)) / valid_mask_f.sum(dim=(1, 2)).clamp(min=1)
            return (per_sample * tw).mean()

        # Legacy fallback path. Handles:
        #   (a) 1D ``(B, T)`` per-timestep mask (pre-migration contract; still
        #       valid for any reader that didn't migrate).
        #   (b) ``(B, T, K)`` mask whose K dim does NOT match per_element's D
        #       (e.g. predictor stub used in tests where pred ∈ R^{T×2} but
        #       reader emits 14-dim mask; OR a per-hand mask). Collapse
        #       to per-step via ``any(dim=-1)`` so "any dim valid → timestep
        #       contributes" — preserves legacy semantics.
        if valid_mask_f.ndim == 3:
            valid_mask_f = (valid_mask_f > 0).any(dim=-1).float()
        per_step = per_element.mean(dim=2)
        per_step = per_step * valid_mask_f
        valid_count = valid_mask_f.sum(dim=1).clamp(min=1)
        per_sample = per_step.sum(dim=1) / valid_count
        return (per_sample * tw).mean()

    # --- Inference: generation ---

    def _resolve_inactive_action_dims(
        self, active_action_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Resolve which unified-action dims must ride the analytic noise path.

        An explicit ``active_action_mask`` wins; otherwise the active indices
        are inferred from the attached unify normalizer's scatter map (its
        absence, or a width mismatch with ``action_dim``, disables pinning).
        Returns a bool ``(action_dim,)`` mask of INACTIVE dims, or ``None``
        when every dim is active.
        """
        if active_action_mask is None:
            normalizer = getattr(self, "normalizer", None)
            active_action_indices = getattr(normalizer, "_dst_index", None)
            unified_action_dim = getattr(normalizer, "_unify_dim", None)
            if (
                active_action_indices is not None
                and unified_action_dim is not None
                and int(unified_action_dim) == self.action_dim
            ):
                active_action_indices = torch.as_tensor(active_action_indices, device=device, dtype=torch.long)
                if active_action_indices.numel() and (
                    int(active_action_indices.min()) < 0 or int(active_action_indices.max()) >= self.action_dim
                ):
                    raise ValueError(
                        f"Unified action indices must be within [0, {self.action_dim}); "
                        f"got {active_action_indices.tolist()}."
                    )
                active_action_mask = torch.zeros(self.action_dim, device=device, dtype=torch.bool)
                active_action_mask[active_action_indices] = True

        if active_action_mask is None:
            return None
        active_action_mask = torch.as_tensor(active_action_mask, device=device, dtype=torch.bool)
        if active_action_mask.shape != (self.action_dim,):
            raise ValueError(
                f"active_action_mask must have shape ({self.action_dim},); got {tuple(active_action_mask.shape)}."
            )
        inactive_action_dims = ~active_action_mask
        if not bool(inactive_action_dims.any()):
            return None
        return inactive_action_dims

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        first_frame_image=None,
        num_frames: int = 49,
        action_num_frames: Optional[int] = None,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        dit_cache=None,
        decode_video: bool = True,
        profile: bool = False,
        prompt_embed_cache: Optional[dict] = None,
        proprio: Optional[Tensor] = None,
        cfg_scale: float = 1.0,
        cfg_merge: bool = False,
        active_action_mask: Optional[Tensor] = None,
        **extra_pipeline_inputs: Any,
    ) -> dict:
        """Execute joint video-action denoising driven by a schedule.

        This is the single entry point for inference. External code
        (engine) should call this instead of touching video_backbone directly.

        The condition is the backbone's deploy encode (``preprocess_condition``) of the first frame alone, the
        same encode the sparse layout uses for its history; the video noise is built here.

        Args:
            first_frame_image: the clean first frame, one PIL image (or a list holding exactly one).
            num_frames: Video frame count T (1 + temporal_compression * k) of the denoised clip.
            height / width / tiled / tile_size / tile_stride / prompt_embed_cache: ignored (the grid comes from
                the encoded first frame; decode is the backbone's; the text goes through the backbone's
                ``text_lookup``). Kept until Task 8 deletes ``generate``.
            action_num_frames: Raw state/action window length. Generated
                action chunk length is ``action_num_frames - 1``. Defaults to
                ``num_frames`` for datasets whose video/action rates match.
            active_action_mask: Optional ``(action_dim,)`` boolean mask for the
                benchmark being generated. Inactive unified-action dimensions
                stay on their analytic zero-padding noise path. When omitted,
                the mask is inferred from the attached unified normalizer.

        Returns:
            dict with ``video`` (list of PIL images or None) and
            ``actions`` ((T, action_dim) numpy array).
        """
        import time

        # Defensive: deploy/model_loader.py:161 already flips eval at load,
        # but ad-hoc callers (notebooks, mid-training eval callbacks) might
        # invoke `generate()` without going through that path. Idempotent
        # — guards CFG dropout and any other
        # training-only behavior from firing during inference.
        self.eval()

        vb = self.video_backbone
        device = self.device
        dtype = self.dtype

        t0 = time.time()

        # §15 — validate CFG up front. CFG vs dit_cache (§15.D6): cond/uncond
        # use the same cache key, so re-running uncond off a cond-tagged hit
        # would silently corrupt the velocity prediction. MVP keeps it simple
        # — disable the cache whenever CFG is on; future work can add a
        # (cond, uncond) slot. CFG is applied by the denoising loop below, not
        # forwarded to the backbone preprocess (Wan does no CFG at inference).
        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")
        if cfg_scale_f > 1.0:
            dit_cache = None

        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

        # Condition: the first frame encoded alone + the text of the T-frame clip.
        images = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        if len(images) != 1 or images[0] is None:
            raise ValueError("generate is first-frame conditioned: pass exactly one first_frame_image.")
        num_latents = vb.num_future_latents(num_frames)  # 1 + (T-1)/tc, validates T
        clean_latents, text = vb.preprocess_condition(
            [images], [prompt], num_frames=int(num_frames), num_latents=num_latents
        )
        vb.scheduler.set_timesteps(num_inference_steps, shift=shift)

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] pipeline_prep: %.3fs", time.time() - t0)

        # Video noise: seeded CPU generator, float32, then the model dtype / device.
        _, z, _, h_lat, w_lat = clean_latents.shape
        generator = torch.Generator("cpu").manual_seed(int(seed))
        noise = torch.randn((1, z, num_latents, h_lat, w_lat), generator=generator, device="cpu", dtype=torch.float32)
        inputs_shared = {"latents": noise.to(dtype=dtype, device=device), "clean_latents": clean_latents, "text": text}
        if input_video_latents is not None:
            inputs_shared["latents"] = input_video_latents
        # Architecture-specific pipeline inputs are forwarded as-is. Subclasses
        # extract what they recognize in their forward(); unrelated architectures
        # never see these keys because callers only pass them via super().generate.
        for key, value in extra_pipeline_inputs.items():
            if value is not None:
                inputs_shared[key] = value
        latents = inputs_shared["latents"].clone()
        latents[:, :, : clean_latents.shape[2]] = clean_latents
        inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio is None:
                raise ValueError("use_proprioception=True requires `proprio` during generation.")
            inputs_shared["proprio"] = proprio.to(device=device, dtype=dtype)

        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        # Unified-action checkpoints scatter raw actions into a larger zero-padded
        # space.  The inactive dimensions may be excluded from the training loss,
        # so their predicted flow is unconstrained.  Keep those dimensions on the
        # analytic forward-noise path instead of feeding unconstrained updates back
        # into the next denoising step.
        inactive_action_dims = self._resolve_inactive_action_dims(active_action_mask, device)

        t_loop = time.time()
        action_latents = self._denoise_loop(
            schedule,
            inputs_shared,
            action_latents,
            inactive_action_dims=inactive_action_dims,
            dit_cache=dit_cache,
            cfg_scale=cfg_scale_f,
            cfg_merge=bool(cfg_merge),
        )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] denoising_loop: %.3fs", time.time() - t_loop)

        # VAE decode (only when pixels were requested).
        if decode_video:
            video_frames = vb.decode_video(inputs_shared["latents"])
        else:
            video_frames = None

        actions = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)

        return {"video": video_frames, "actions": actions}

    def _denoise_loop(
        self,
        schedule,
        inputs_shared: dict,
        action_latents: Tensor,
        *,
        inactive_action_dims: Optional[Tensor] = None,
        dit_cache=None,
        cfg_scale: float = 1.0,
        cfg_merge: bool = False,
    ) -> Tensor:
        """Joint video-action Euler loop over a ``(t_video, t_action)`` schedule (shared by deploy and vis).

        Args:
            schedule: list of (t_video, t_action) pairs ending with the (0, 0) sentinel (``make_schedule``).
            inputs_shared: forward kwargs; ``latents`` (B, C, T, h, w) start noisy and are updated in place
                in the dict, the ``clean_latents`` prefix (B, C, H, h, w) is re-imposed each step.
            action_latents: (B, K, A) initial action noise.
            inactive_action_dims: (A,) index of unified-action dims kept on the analytic noise path, or None.
            dit_cache / cfg_scale / cfg_merge: as in :meth:`generate`.

        Returns: (B, K, A) denoised actions (normalized space); the final video latents are in
            ``inputs_shared["latents"]``.
        """
        from tqdm import tqdm

        dtype, device = self.dtype, self.device
        cfg_scale_f = float(cfg_scale)
        inactive_action_noise = None

        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_v = t_v / num_train_ts_v
            sigma_a = t_a / num_train_ts_a
            sigma_v_next = t_v_next / num_train_ts_v
            sigma_a_next = t_a_next / num_train_ts_a

            video_stepping = sigma_v != sigma_v_next
            action_stepping = sigma_a != sigma_a_next

            if not video_stepping and not action_stepping:
                continue

            # A frozen stream (sigma plateau, e.g. linear_offset delay) still rides
            # the forward context: under a v→a-visible attention mask, dropping
            # its tokens would change the other stream's prediction. Freezing
            # only gates the *update* below, never the forward inputs.
            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)

            if (
                dit_cache is not None
                and video_stepping
                and not dit_cache.should_recompute(sigma_v, require_action=action_stepping)
            ):
                # Reuse cached predictions from the previous joint forward.
                # Joint denoising can only skip the whole DiT call when the
                # action prediction was cached together with video prediction.
                noise_pred = dit_cache.get_cached()
                action_noise_pred = dit_cache.get_cached_action() if action_stepping else None
            else:
                if cfg_scale_f > 1.0:
                    # _forward_with_cfg runs its own cudagraph_mark_step_begin()
                    # before each inner forward (1 for cfg_merge, 2 for sequential).
                    noise_pred, action_noise_pred = self._forward_with_cfg(
                        action_latents=action_latents,
                        a_timestep=a_timestep,
                        inputs_shared=inputs_shared,
                        v_timestep=v_timestep,
                        cfg_scale=cfg_scale_f,
                        cfg_merge=bool(cfg_merge),
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred, action_noise_pred = self.forward(
                        action_latents,
                        a_timestep,
                        **inputs_shared,
                        timestep=v_timestep,
                    )
                if dit_cache is not None and video_stepping:
                    dit_cache.update(noise_pred, sigma_v, action_noise_pred)

            if video_stepping:
                new_latents = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
                clean = inputs_shared["clean_latents"]
                new_latents = new_latents.clone()
                new_latents[:, :, : clean.shape[2]] = clean
                inputs_shared["latents"] = new_latents

            if action_stepping and action_noise_pred is not None:
                if inactive_action_dims is not None and inactive_action_noise is None:
                    sigma_a_f = float(sigma_a)
                    if sigma_a_f <= 0.0:
                        raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                    inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )
                if inactive_action_dims is not None:
                    action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)

        return action_latents

    @torch.no_grad()
    def prepare_sparse_condition(self, history_frames, prompts, proprio, *, num_future_frames: int) -> dict:
        """Forward conditioning of the sparse-history layout from raw observations (deploy).

        The video backbone encodes the history + text exactly as its training path does
        (``preprocess_condition`` is the history half of ``preprocess_train``).

        Args:
            history_frames: list of B clips, each H PIL canvases (oldest first, frame t last), training layout.
            prompts: list of B str.
            proprio: (B, H, D) normalized proprio, one row per history frame (oldest first).
            num_future_frames: int F, future frames the denoised latents cover.

        Returns: forward kwargs ``clean_latents`` (B, z, H, h, w), ``text`` (TextCondition for the H + T_fut
            latent grid) and ``proprio`` (B, H, D) on the model device / dtype.
        """
        vb = self.video_backbone
        num_latents = len(history_frames[0]) + vb.num_future_latents(num_future_frames)
        clean_latents, text = vb.preprocess_condition(
            history_frames,
            list(prompts),
            # The text describes the equivalent contiguous clip of these latents.
            num_frames=1 + int(vb.temporal_compression) * (num_latents - 1),
            num_latents=num_latents,
        )
        proprio = torch.as_tensor(proprio).to(device=self.device, dtype=self.dtype)
        return {"clean_latents": clean_latents, "text": text, "proprio": proprio}

    @torch.no_grad()
    def sample(
        self,
        cond_inputs: dict,
        schedule,
        *,
        latent_shape,
        action_shape,
        seed: int,
        active_action_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Denoise video latents + an action chunk from seeded noise, given forward conditioning.

        Args:
            cond_inputs: forward kwargs without ``latents`` (``prepare_sparse_condition``, or ``prepare_inputs``
                with the GT / training-only entries removed); the ``clean_latents`` (B, z, N, h, w) prefix is
                written over the first N latent frames and re-imposed every step.
            schedule: (t_video, t_action) list from ``make_schedule``.
            latent_shape: (B, z, T, h, w) of the full latent clip (clean prefix + denoised part).
            action_shape: (B, K, A) of the action chunk.
            seed: int, seeds one generator that draws the video noise, then the action noise.
            active_action_mask: optional (A,) bool, see :meth:`generate`.

        Returns: (latents (B, z, T, h, w), actions (B, K, A) in normalized space).
        """
        device, dtype = self.device, self.dtype
        inputs = dict(cond_inputs)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn(tuple(latent_shape), generator=gen, device=device, dtype=dtype)
        clean = inputs["clean_latents"]
        latents[:, :, : clean.shape[2]] = clean
        inputs["latents"] = latents
        action_latents = torch.randn(tuple(action_shape), generator=gen, device=device, dtype=dtype)
        action_latents = self._denoise_loop(
            schedule,
            inputs,
            action_latents,
            inactive_action_dims=self._resolve_inactive_action_dims(active_action_mask, device),
        )
        return inputs["latents"], action_latents

    # --- §15: Classifier-Free Guidance helpers (inference-time) ---

    def _forward_with_cfg(
        self,
        *,
        action_latents: Optional[Tensor],
        a_timestep: Optional[Tensor],
        inputs_shared: dict,
        v_timestep: Tensor,
        cfg_scale: float,
        cfg_merge: bool,
    ) -> tuple:
        """Run cond + uncond forwards and combine via ``pred = uncond + s·(cond - uncond)``.

        Two paths gated on ``cfg_merge``:

        - ``cfg_merge=False`` (default): two sequential forwards. The cond
          branch consumes ``inputs_shared['context']`` unchanged; the uncond
          branch temporarily swaps in ``inputs_shared['uncond_context']`` and
          swaps back via ``try/finally``.
        - ``cfg_merge=True``: stack ``[uncond, cond]`` along batch axis 0 for
          all batch-shaped tensors in ``inputs_shared`` plus ``action_latents``
          and the timesteps, then chunk the merged output. One forward, ~B=2
          memory peak.

        Action and video streams share the same text context, so a single
        ``cfg_scale`` is applied to both ``noise_pred`` and (when present)
        ``action_noise_pred``.
        """
        uncond_context = inputs_shared.get("uncond_context")
        if not isinstance(uncond_context, Tensor):
            raise RuntimeError(
                "CFG combine requested but `inputs_shared['uncond_context']` is missing "
                "or not a tensor; no video backbone produces an unconditional context (see followups)."
            )

        if cfg_merge:
            expanded, exp_al, exp_vt, exp_at = _expand_inputs_for_cfg(
                inputs_shared,
                action_latents=action_latents,
                v_timestep=v_timestep,
                a_timestep=a_timestep,
            )
            torch.compiler.cudagraph_mark_step_begin()
            merged_noise, merged_action = self.forward(exp_al, exp_at, **expanded, timestep=exp_vt)
            uncond_noise, cond_noise = merged_noise.chunk(2, dim=0)
            noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
            if isinstance(merged_action, Tensor):
                uncond_action, cond_action = merged_action.chunk(2, dim=0)
                action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
            else:
                action_noise_pred = None
            return noise_pred, action_noise_pred

        # Sequential path: cond → uncond → combine. Mark before each forward
        # so CUDA Graph tree sees both as distinct dispatch sites.
        torch.compiler.cudagraph_mark_step_begin()
        cond_noise, cond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        saved_context = inputs_shared["context"]
        inputs_shared["context"] = uncond_context
        try:
            torch.compiler.cudagraph_mark_step_begin()
            uncond_noise, uncond_action = self.forward(action_latents, a_timestep, **inputs_shared, timestep=v_timestep)
        finally:
            inputs_shared["context"] = saved_context

        noise_pred = _combine_cfg(uncond_noise, cond_noise, cfg_scale)
        if isinstance(cond_action, Tensor) and isinstance(uncond_action, Tensor):
            action_noise_pred = _combine_cfg(uncond_action, cond_action, cfg_scale)
        else:
            # One side dropped the action stream; keep cond as-is.
            action_noise_pred = cond_action
        return noise_pred, action_noise_pred

    # --- Deploy helpers (combine action module + video backbone) ---

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply architecture-specific deploy-time compile optimizations."""
        _ = compile_enabled(compile_cfg, default=False, strict=True)
        vb_compile = getattr(getattr(self, "video_backbone", None), "apply_compile_optimizations", None)
        if callable(vb_compile):
            vb_compile(compile_cfg)

    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        latents: Tensor,
        timestep: Tensor,
        text: "TextCondition",
        clean_latents: Tensor,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the joint video + action forward.

        Each concrete architecture implements its own forward end-to-end —
        runs the video DiT block loop, captures or interleaves with the
        action stream as appropriate, and returns
        ``(video_noise_pred, action_noise_pred)``. When ``noisy_actions`` is
        None (CFG nega pass / video-only generation) the action term is
        None.

        Args:
            noisy_actions: (B, K, A) or None. action_timestep: (B,) or None.
            latents: (B, z, T_lat, h, w) noisy video latents (clean prefix already written in).
            timestep: (B,) video timestep. text: TextCondition from the video backbone.
            clean_latents: (B, z, N_clean, h, w) clean prefix; only its length N_clean is read here.
            proprio: normalized proprio for the proprio-as-context token(s).
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool.
            pipeline_inputs: ``_proprio_sample_mask`` (per-sample proprio mask) and training-only entries
                (``input_latents``) that the forward ignores.
        """
        ...


# ----------------------------------------------------------------------
# §15 — Classifier-Free Guidance helpers (module-level so they stay
# stateless / testable without a full architecture instance).
# ----------------------------------------------------------------------


def _combine_cfg(uncond: Tensor, cond: Tensor, scale: float) -> Tensor:
    """Linear CFG combine: ``uncond + scale·(cond - uncond)``.

    Matches upstream Cosmos formula in
    ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:484-488``.
    """
    return uncond + float(scale) * (cond - uncond)


# Keys in ``inputs_shared`` that carry a leading batch axis and therefore
# need duplication when stacking ``[uncond, cond]`` for cfg_merge=True.
_CFG_BATCH_AXIS_KEYS: tuple = (
    "latents",
    "input_latents",
    "proprio",
    "clean_latents",
)


def _expand_inputs_for_cfg(
    inputs_shared: dict,
    *,
    action_latents: Optional[Tensor],
    v_timestep: Tensor,
    a_timestep: Optional[Tensor],
) -> Tuple[dict, Optional[Tensor], Tensor, Optional[Tensor]]:
    """Stack ``[uncond, cond]`` along batch axis for the cfg_merge=True path.

    Returns ``(expanded_inputs_shared, action_latents, v_timestep, a_timestep)``
    where the inputs_shared copy has:

    - ``context`` replaced by ``cat([uncond_context, cond_context], dim=0)``
    - ``uncond_context`` cleared (downstream forwards don't read it)
    - every other batch-axis tensor in ``_CFG_BATCH_AXIS_KEYS`` duplicated

    Scalar / None entries are passed through unchanged.
    """
    uncond_context = inputs_shared["uncond_context"]
    cond_context = inputs_shared["context"]
    expanded = dict(inputs_shared)
    expanded["context"] = torch.cat([uncond_context, cond_context], dim=0)
    expanded["uncond_context"] = None

    # proprio can come in raw 1D ``(D,)`` shape (the architecture's
    # ``_compute_proprio`` normalises inside forward); cfg_merge
    # stacks BEFORE forward so we must normalise to ``(B, D)`` first,
    # otherwise ``cat([(D,), (D,)], dim=0)`` lands on ``(2·D,)`` and the
    # last-dim check downstream raises. Mirrors the (B, 1, D) → (B, D)
    # squeeze the architecture itself does.
    proprio = expanded.get("proprio")
    if isinstance(proprio, Tensor):
        if proprio.ndim == 1:
            expanded["proprio"] = proprio.unsqueeze(0)
        elif proprio.ndim == 3 and proprio.shape[1] == 1:
            expanded["proprio"] = proprio[:, 0, :]

    for key in _CFG_BATCH_AXIS_KEYS:
        v = expanded.get(key)
        if isinstance(v, Tensor):
            expanded[key] = torch.cat([v, v], dim=0)
    al = torch.cat([action_latents, action_latents], dim=0) if isinstance(action_latents, Tensor) else None
    vt = torch.cat([v_timestep, v_timestep], dim=0) if isinstance(v_timestep, Tensor) else v_timestep
    at = torch.cat([a_timestep, a_timestep], dim=0) if isinstance(a_timestep, Tensor) else None
    return expanded, al, vt, at
