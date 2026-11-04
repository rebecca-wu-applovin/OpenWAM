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

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from openwam.model.video_backbone.registry import LoadMode, build_video_backbone


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
    ``no_grad`` context (e.g. ``preprocess_train``), the inner ``no_grad`` is a
    no-op; if the caller is inside a grad-tracking forward, it short-circuits
    activation saving.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen base,
    or training only the head of an otherwise frozen model), do NOT pass the
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want frozen instead.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openwam.dataloader.collate import Batch
    from openwam.model.action_backbone.base import ActionDiTBackbone
    from openwam.model.video_backbone.base import TextCondition, VideoBackbone


@dataclass(frozen=True)
class TrainOptions:
    """Training knobs the trainer decides once (from ``cfg.training``) and passes to the architecture constructor.

    Fields:
        use_gradient_checkpointing: bool, recompute block activations in the training backward.
        use_gradient_checkpointing_offload: bool, also offload the checkpointed activations to CPU.
        lambda_video: float, weight of the video flow-matching loss (0 = the term is left out of the loss).
        lambda_action: float, weight of the action flow-matching loss (0 = the term is left out of the loss).
        mot_checkpoint_mixed_attn: bool, joint_self_attn: grad-checkpoint the mixed attention in training.
    """

    use_gradient_checkpointing: bool
    use_gradient_checkpointing_offload: bool
    lambda_video: float
    lambda_action: float
    mot_checkpoint_mixed_attn: bool


# Video backbone properties the action stream reads from ``model.video_backbone.geometry``.
GEOMETRY_KEYS = ("dim", "num_heads", "head_dim", "num_layers", "text_dim")

# Deploy builds: no loss is ever computed, so no checkpointing and both loss weights unused.
INFERENCE_ONLY = TrainOptions(
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    lambda_video=0.0,
    lambda_action=0.0,
    mot_checkpoint_mixed_attn=False,
)


@dataclass
class ActionState:
    """Mutable action-stream state that ``DualSystemMoTDriver`` threads across layers (joint_self_attn only).

    Fields:
        action_latents: (B, T_action, action_dim) noisy actions (input to forward).
        timestep: action diffusion timestep (raw shape preserved for the action backbone's internal use).
        payload: backbone-specific per-forward state (typically ``ActionDiTState``).
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
        cfg: ``OmegaConf.merge(cfg.common, cfg.model)``: ``layout`` (the latent layout ``resolve_config`` derived;
            the preprocessing, the attention mask, the loss and deploy read it, never re-derived here),
            ``action_dim`` / ``state_dim``, ``architecture`` (``clean_attention`` / ``attention_mask_mode`` pick
            the frame-level mask, ``proprio_input`` the proprio routing; validated by ``resolve_config``),
            ``video_backbone`` (the backbone's slice) and ``action_backbone``.
        mode: LoadMode the entry point chose for the video backbone (PRETRAINED: upstream weights;
            CHECKPOINT: an empty shell of ``ckpt_dir`` the caller fills).
        assets: ``cfg.assets.video_backbone`` (upstream paths) for PRETRAINED, None for CHECKPOINT.
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.
        train_options: TrainOptions (gradient checkpointing, loss weights) the trainer decided; deploy passes
            ``INFERENCE_ONLY``.
    """

    # Built by the subclass ``__init__`` (every architecture has an action stream).
    action_backbone: "ActionDiTBackbone"

    # --- Construction & config resolution ---

    def __init__(
        self, cfg: DictConfig, *, mode: LoadMode, assets: Optional[DictConfig], ckpt_dir: Optional[str],
        train_options: TrainOptions,
    ):
        from openwam.model.architectures.utils.mask_modes import build_frame_mask

        super().__init__()
        self.cfg = cfg
        self.layout = cfg.layout
        self.train_options = train_options
        arch = cfg.architecture
        # Frame-level [latents, action] visibility, built once; backbones expand it to tokens per forward.
        self.frame_mask = build_frame_mask(self.layout, str(arch.clean_attention), str(arch.attention_mask_mode))
        # How the video backbone gets the proprio (validated by resolve_config); build_contexts applies it.
        self.proprio_input = str(arch.proprio_input)
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        self.video_backbone: "VideoBackbone" = self._build_video_backbone(cfg.video_backbone, mode, assets, ckpt_dir)

    @staticmethod
    def _build_video_backbone(
        vb_cfg, mode: LoadMode, assets: Optional[DictConfig], ckpt_dir: Optional[str]
    ) -> "VideoBackbone":
        """Build the video backbone with the load mode the entry point chose, and check the yaml's contracts.

        Args:
            vb_cfg: ``cfg.model.video_backbone`` (DictConfig slice; ``name`` picks the registered class).
            mode: LoadMode decided by the entry point.
            assets: ``cfg.assets.video_backbone`` (DictConfig) for PRETRAINED, None for CHECKPOINT.
            ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.

        Returns:
            VideoBackbone. Raises ValueError when the yaml's temporal contract or ``geometry`` block disagrees
            with the built backbone.
        """
        video_backbone = build_video_backbone(vb_cfg, mode, assets, ckpt_dir)
        # The layout and the dataloader read the yaml's temporal_compression: it must be the VAE's.
        declared = (int(vb_cfg["temporal_compression"]), bool(vb_cfg["causal_temporal"]))
        actual = (video_backbone.temporal_compression, video_backbone.causal_temporal)
        if declared != actual:
            raise ValueError(
                f"video_backbone.temporal_compression / causal_temporal yaml {declared} does not match native VAE "
                f"{actual}. Update the yaml fields to match."
            )
        BaseWAMArchitecture._check_geometry(vb_cfg.geometry, video_backbone)
        return video_backbone

    @staticmethod
    def _check_geometry(geometry: DictConfig, video_backbone: "VideoBackbone") -> None:
        """Raise ValueError listing every ``geometry`` key whose yaml value differs from the built module.

        Args: geometry: ``model.video_backbone.geometry`` (``GEOMETRY_KEYS`` ints the action stream reads).
        video_backbone: the built VideoBackbone (a meta-device CHECKPOINT shell exposes the same properties).
        """
        # The action stream sizes itself from the yaml; a stale block (e.g. a dit override) must fail here.
        diff = {k: (int(geometry[k]), getattr(video_backbone, k)) for k in GEOMETRY_KEYS}
        diff = {k: v for k, v in diff.items() if v[0] != v[1]}
        if diff:
            raise ValueError(f"video_backbone.geometry (yaml, built) differs from the built module: {diff}")

    # --- Backbone composition ---

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones should override this to include
        them. The returned dict
        is used by ``set_dtype_device``,
        ``move_frozen_to_device``, and ``save_assets_for_deployment`` to iterate
        over all backbones generically.
        """
        return {"video_backbone": self.video_backbone, "action_backbone": self.action_backbone}

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """Flow-matching scheduler for the action stream (owned by action_backbone)."""
        return self.action_backbone.scheduler

    @property
    def video_scheduler(self):
        """Flow-matching scheduler for the video stream (owned by video_backbone)."""
        return self.video_backbone.scheduler

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim

    @property
    def bridge_layers(self) -> tuple:
        return self.action_backbone.bridge_layers

    @property
    def uses_proprioception(self) -> bool:
        return self._use_proprioception_context or self.action_backbone.uses_proprioception

    # --- Proprio-as-context conditioning ---

    def _init_proprio_context(self, *, enabled: bool, state_dim: int, text_dim: int) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning.

        Args:
            enabled: bool, ``architecture.use_proprioception``.
            state_dim: int proprio width D (``common.state_dim``); only read when enabled.
            text_dim: int, width of the text context the proprio tokens are appended to.

        Returns:
            None. Declares ``proprio_encoder`` / ``proprio_slot_emb`` (None when unused).
        """
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_slot_emb: Optional[nn.Embedding] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        # One proprio token per history frame (oldest first), from the layout.
        self.num_proprio_tokens = int(self.layout.num_proprio_tokens)
        if not enabled:
            return
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

    def _proprio_tokens(self, proprio: Tensor, proprio_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the proprio rows into context tokens (one per history frame).

        Args:
            proprio: (B, N, D) normalized proprio, N = ``layout.num_proprio_tokens`` (oldest first), D = state_dim.
            proprio_mask: (B, N, D) bool, True = real entry; a row with no real entry gets a zero token and a
                False mask entry, which also isolates the proprio_encoder gradient on it (input * 0 -> grad 0).

        Returns:
            (tokens (B, N, context_dim) in the encoder dtype, token_mask (B, N) bool).
        """
        expect = (self.num_proprio_tokens, self.proprio_dim)
        if proprio.ndim != 3 or tuple(proprio.shape[1:]) != expect or proprio_mask.shape != proprio.shape:
            raise ValueError(
                f"proprio must be (B, {expect[0]}, {expect[1]}) with a mask of the same shape, got "
                f"{tuple(proprio.shape)} / {tuple(proprio_mask.shape)}"
            )
        weight = self.proprio_encoder.weight
        tokens = self.proprio_encoder(proprio.to(device=weight.device, dtype=weight.dtype))
        if self.proprio_slot_emb is not None:
            # Learned absolute slot PE (one per history frame).
            tokens = tokens + self.proprio_slot_emb.weight.to(tokens.dtype)
        # A row is real when any of its dims is; masked rows become zero tokens.
        token_mask = proprio_mask.to(device=tokens.device, dtype=torch.bool).any(dim=-1)
        return tokens * token_mask.to(tokens.dtype).unsqueeze(-1), token_mask

    def build_contexts(
        self, text: "TextCondition", proprio: Tensor, proprio_mask: Tensor
    ) -> tuple["TextCondition", "TextCondition"]:
        """The text context of each stream; the one place ``proprio_input`` is applied.

        ActionDiT always reads text + the proprio tokens (when ``use_proprioception``). The video backbone reads
        the same context under ``proprio_input=context_token`` and the text alone under ``backbone_stream`` (it
        embeds ``proprio`` in its own stream, see ``VideoBackbone.prepare``).

        Args:
            text: TextCondition, context (B, L, D_text), context_mask (B, L).
            proprio: (B, N, D) normalized proprio, N = ``layout.num_proprio_tokens``.
            proprio_mask: (B, N, D) bool, True = real entry.

        Returns:
            (video_text, action_text): TextConditions; with proprio tokens their context is (B, L + N, D_text) and
            context_mask (B, L + N) (``extra`` shared with ``text``).
        """
        if not self._use_proprioception_context:
            return text, text
        tokens, token_mask = self._proprio_tokens(proprio, proprio_mask)
        context = text.context
        # The proprio tokens sit after the (possibly padded) text: the mask, not a prefix length, marks valid tokens.
        action_text = dataclasses.replace(
            text,
            context=torch.cat([context, tokens.to(dtype=context.dtype)], dim=1),
            context_mask=torch.cat([text.context_mask.to(device=context.device, dtype=torch.bool), token_mask], dim=1),
        )
        video_text = action_text if self.proprio_input == "context_token" else text
        return video_text, action_text

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
        # The proprio modules exist only under use_proprioception (declared None otherwise).
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(dtype=dtype, device=device)
        if self.proprio_slot_emb is not None:
            self.proprio_slot_emb.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

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

    def load_checkpoint(self, path: str) -> None:
        """Fill the architecture from one of our safetensors checkpoints (deploy / finetune), strictly.

        The one strict rule (``load_strict``): missing / unexpected keys, a shape mismatch, or a tensor still on
        meta after the load raise. A CHECKPOINT shell holds meta tensors, which only ``assign=True`` can fill (an
        in-place copy into meta is a silent no-op); an architecture with real storage everywhere is copied into
        in place. Tied weights stay tied either way.

        Args:
            path: str, a ``checkpoint_step_*.safetensors`` file holding exactly ``self.state_dict()``'s keys.

        Returns:
            None. Raises ValueError on any mismatch (see ``load_strict``).
        """
        from safetensors.torch import load_file

        from openwam.model.video_backbone.common.weights import load_strict

        # Whether this is a meta shell is a property of the module itself, not of the caller's mode.
        has_meta = any(t.is_meta for t in [*self.parameters(), *self.buffers()])
        load_strict(self, load_file(path), assign=has_meta)

    # --- Training: module management ---

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``preprocess_train`` / ``preprocess_condition``, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph, the wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like ``video_backbone.text_encoder`` work.

        Args:
            names: list[str] dotted submodule paths (``training.freeze``); each must exist on this model.

        Returns:
            list[str], ``names`` (all frozen). ValueError naming ``training.freeze`` and the path when one is not
            a submodule.
        """
        for name in names:
            try:
                module = self.get_submodule(name)
            except AttributeError as err:
                raise ValueError(f"training.freeze: {name!r} is not a submodule of {type(self).__name__}") from err
            module.requires_grad_(False)
            # Set eval mode on the frozen subtree. Use modules() instead
            # of .eval() to avoid infinite recursion when a submodule has
            # self-referential aliases (e.g. HF model.model = self).
            for sub in module.modules():
                sub.training = False
            _wrap_forward_in_no_grad(module)
        return list(names)

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

    # --- Training: loss computation ---

    def compute_loss(self, batch: "Batch") -> dict:
        """Joint video-action flow-matching loss of one batch (the single training entry point).

        Encodes the clips and prompts with the frozen video backbone (``preprocess_train``, no grad), samples
        the video / action timesteps and noise, runs the joint forward and weights the two losses with
        ``train_options.lambda_video`` / ``lambda_action``.

        Args:
            batch: Batch on the model device / dtype (``Batch.to``): video, prompt, action (B, K, A),
                action_is_pad (B, K, A), proprio (B, H, D), proprio_mask (B, H, D).

        Returns:
            dict of 0-dim tensors: loss (weighted sum of the terms with a nonzero lambda, carries the graph),
            loss_video, loss_action (weighted, detached).
        """
        video_backbone = self.video_backbone
        video_scheduler, action_scheduler = self.video_scheduler, self.action_scheduler
        lambda_video, lambda_action = self.train_options.lambda_video, self.train_options.lambda_action
        _dtype = self.dtype
        _device = self.device

        # Clean latent clip (B, z, num_latents, h, w); the first layout.num_hist_latents are the condition.
        clean_latents, text = video_backbone.preprocess_train(batch.video, batch.prompt, self.layout)
        B = clean_latents.shape[0]

        # --- Sample video timesteps ---
        # The scheduler owns the training noise law (yaml train_timestep): grid draw or a family's native one.
        video_timesteps, video_sigmas, video_loss_weights = video_scheduler.sample_train(
            B, device=_device, dtype=_dtype
        )

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        video_noise = torch.randn_like(clean_latents)
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        noisy_latents = video_scheduler.add_noise(clean_latents, video_noise, sigma_bc)
        video_target = video_scheduler.training_target(clean_latents, video_noise)

        # The clean prefix (the layout's history latents) stays clean.
        n_clean = int(self.layout.num_hist_latents)
        noisy_latents[:, :, :n_clean] = clean_latents[:, :, :n_clean]

        # --- Prepare action noise ---
        # Drawn after the video noise, so the global RNG sequence is video timesteps, video noise, action ones.
        # The action stream always runs (one forward path); lambda_action only weights its loss.
        action_timesteps, action_sigmas, action_loss_weights = action_scheduler.sample_train(
            B, device=_device, dtype=_dtype
        )
        action_noise = torch.randn_like(batch.action)
        noisy_actions = action_scheduler.add_noise(batch.action, action_noise, action_sigmas.view(B, 1, 1))
        action_target = action_scheduler.training_target(batch.action, action_noise)

        # --- Joint forward pass ---
        # ``self(...)`` (not ``self.forward``) so nn.Module.__call__ fires architecture-level hooks.
        video_noise_pred, action_noise_pred = self(
            noisy_actions,
            action_timesteps,
            latents=noisy_latents,
            timestep=video_timesteps,
            text=text,
            proprio=batch.proprio,
            proprio_mask=batch.proprio_mask,
            use_gradient_checkpointing=self.train_options.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.train_options.use_gradient_checkpointing_offload,
        )

        # --- Video loss ---
        loss_video = self._compute_video_loss(video_noise_pred, video_target, video_loss_weights)

        # --- Action loss ---
        loss_action = self._compute_action_loss(action_noise_pred, action_target, action_loss_weights, batch.action_is_pad)

        # Same rule for both terms: a 0 weight leaves that term out of ``loss`` (the trainer rejects both 0).
        if lambda_video == 0:
            loss = lambda_action * loss_action
        elif lambda_action == 0:
            loss = lambda_video * loss_video
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        return {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

    def _compute_video_loss(self, noise_pred: Tensor, target: Tensor, loss_weights: Tensor) -> Tensor:
        """Per-sample weighted video MSE loss over the denoised (future) latents.

        Args:
            noise_pred / target: (B, z, num_latents, h, w); the first ``layout.num_hist_latents`` are skipped.
            loss_weights: (B,) float32 per-sample weights from ``scheduler.sample_train``, on the loss device.

        Returns:
            0-dim float32 loss.
        """
        import torch.nn.functional as F

        # The clean prefix is not a denoising target.
        n_skip = int(self.layout.num_hist_latents)
        noise_pred = noise_pred[:, :, n_skip:]
        target = target[:, :, n_skip:]

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        # Every frame is real (the dataset never pads): plain mean over frames.
        per_sample = per_element.mean(dim=(1, 3, 4)).mean(dim=1)
        return (per_sample * loss_weights).mean()

    def _compute_action_loss(
        self, noise_pred: Tensor, target: Tensor, loss_weights: Tensor, action_is_pad: Tensor
    ) -> Tensor:
        """Per-sample weighted action MSE loss over the supervised (step, dim) cells.

        Args:
            noise_pred / target: (B, K, A) predicted / target velocity.
            loss_weights: (B,) float32 per-sample weights from ``scheduler.sample_train``, on the loss device.
            action_is_pad: (B, K, A) bool, True = no supervision for that cell.

        Returns:
            0-dim float32 loss: per sample sum over valid cells / number of valid cells, weighted, batch mean.
        """
        import torch.nn.functional as F

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        if action_is_pad.shape != per_element.shape:
            raise ValueError(f"action_is_pad {tuple(action_is_pad.shape)} must match the action {tuple(per_element.shape)}")
        valid_mask_f = (~action_is_pad.to(device=per_element.device, dtype=torch.bool)).float()
        per_sample = (per_element * valid_mask_f).sum(dim=(1, 2)) / valid_mask_f.sum(dim=(1, 2)).clamp(min=1)
        return (per_sample * loss_weights).mean()

    # --- Inference: generation ---

    def _resolve_inactive_action_dims(
        self, active_action_mask: Optional[Tensor], device: torch.device
    ) -> Optional[Tensor]:
        """Which unified-action dims ride the analytic noise path, from the caller's explicit mask.

        Args:
            active_action_mask: (action_dim,) bool of the active dims, or None (every dim active). The deploy engine
                builds it from ``action_mode`` (the unify scatter map); nothing is inferred here.
            device: device of the returned mask.

        Returns: (action_dim,) bool mask of the INACTIVE dims, or None when every dim is active.
        """
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

    def _denoise_loop(
        self,
        schedule,
        latents: Tensor,
        action_latents: Tensor,
        *,
        hist_latents: Tensor,
        text: "TextCondition",
        proprio: Tensor,
        proprio_mask: Tensor,
        inactive_action_dims: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Joint video-action Euler loop over a ``(t_video, t_action)`` schedule (``sample``: deploy and vis).

        Args:
            schedule: list of (t_video, t_action) pairs ending with the (0, 0) sentinel (``make_schedule``).
            latents: (B, C, T, h, w) initial latents (noise, clean prefix written in).
            action_latents: (B, K, A) initial action noise.
            hist_latents: (B, C, num_hist_latents, h, w) clean prefix, re-imposed after every video step.
            text: TextCondition. proprio: (B, H, D) normalized proprio. proprio_mask: (B, H, D) bool.
            inactive_action_dims: (A,) index of unified-action dims kept on the analytic noise path, or None.

        Returns: (latents (B, C, T, h, w) denoised video latents, actions (B, K, A) in normalized space).
        """
        from tqdm import tqdm

        dtype, device = self.dtype, self.device
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

            # Inference never checkpoints.
            noise_pred, action_noise_pred = self.forward(
                action_latents,
                a_timestep,
                latents=latents,
                timestep=v_timestep,
                text=text,
                proprio=proprio,
                proprio_mask=proprio_mask,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

            if video_stepping:
                new_latents = self.video_scheduler.step(noise_pred, sigma_v, sigma_v_next, latents)
                new_latents = new_latents.clone()
                new_latents[:, :, : hist_latents.shape[2]] = hist_latents
                latents = new_latents

            if action_stepping and action_noise_pred is not None:
                if inactive_action_dims is not None and inactive_action_noise is None:
                    sigma_a_f = float(sigma_a)
                    if sigma_a_f <= 0.0:
                        raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                    inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
                action_latents = self.action_scheduler.step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )
                if inactive_action_dims is not None:
                    action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)

        return latents, action_latents

    @torch.no_grad()
    def sample(
        self,
        hist_latents: Tensor,
        text: "TextCondition",
        proprio: Tensor,
        schedule,
        *,
        seed: int,
        active_action_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Denoise the future video latents + an action chunk from seeded noise, given the condition.

        Args:
            hist_latents: (B, z, num_hist_latents, h, w) clean history latents (``preprocess_condition``), written
                over the first latents and re-imposed every step.
            text: TextCondition for the (num_latents, h, w) grid (``preprocess_condition``).
            proprio: (B, H, D) normalized proprio on the model device / dtype, one row per history frame (all real).
            schedule: (t_video, t_action) list from ``make_schedule``.
            seed: int, seeds one generator that draws the video noise, then the action noise.
            active_action_mask: optional (A,) bool of the active unified-action dims; None = every dim active.

        Returns: (latents (B, z, layout.num_latents, h, w), actions (B, layout.action_horizon, A) in normalized
            space). The shapes come from the layout; h, w are the history latents' grid.
        """
        device, dtype = self.device, self.dtype
        b, z, _, h, w = hist_latents.shape
        gen = torch.Generator(device=device).manual_seed(int(seed))
        latents = torch.randn((b, z, int(self.layout.num_latents), h, w), generator=gen, device=device, dtype=dtype)
        latents[:, :, : hist_latents.shape[2]] = hist_latents
        action_shape = (b, int(self.layout.action_horizon), self.action_dim)
        action_latents = torch.randn(action_shape, generator=gen, device=device, dtype=dtype)
        return self._denoise_loop(
            schedule,
            latents,
            action_latents,
            hist_latents=hist_latents,
            text=text,
            proprio=proprio,
            proprio_mask=torch.ones(proprio.shape, dtype=torch.bool, device=proprio.device),
            inactive_action_dims=self._resolve_inactive_action_dims(active_action_mask, device),
        )

    @torch.no_grad()
    def rollout(
        self,
        history: list,
        prompts: list,
        proprio: Tensor,
        schedule,
        *,
        seed: int,
        active_action_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Encode the history + prompts and denoise the future from seeded noise: deploy's and training vis's rollout.

        Args:
            history: B lists of H PIL frames each (the history canvases, oldest first).
            prompts: B prompt strings.
            proprio: (B, H, D) normalized proprio on the model device / dtype.
            schedule: (t_video, t_action) list from ``make_schedule``.
            seed: int, seeds the video and action noise.
            active_action_mask: (A,) bool of the active unified-action dims, or None (every dim active); required so
                each caller states its choice.

        Returns: (latents (B, z, layout.num_latents, h, w), actions (B, layout.action_horizon, A) normalized), as
            :meth:`sample`. Callers that want pixels decode with ``video_backbone.decode_future(latents, layout)``.
        """
        hist_latents, text = self.video_backbone.preprocess_condition(history, prompts, self.layout)
        return self.sample(hist_latents, text, proprio, schedule, seed=seed, active_action_mask=active_action_mask)

    @abstractmethod
    def forward(
        self,
        noisy_actions: Tensor,
        action_timestep: Tensor,
        *,
        latents: Tensor,
        timestep: Tensor,
        text: "TextCondition",
        proprio: Tensor,
        proprio_mask: Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Run the joint video + action forward.

        Each concrete architecture implements its own forward end-to-end —
        runs the video DiT block loop, captures or interleaves with the
        action stream as appropriate, and returns
        ``(video_noise_pred, action_noise_pred)``. Both streams always run (training weights the action loss
        with ``lambda_action``, possibly 0).

        Args:
            noisy_actions: (B, K, A) noisy actions. action_timestep: (B,) action timestep.
            latents: (B, z, num_latents, h, w) noisy video latents (clean prefix already written in; its length is
                ``layout.num_hist_latents``).
            timestep: (B,) video timestep. text: TextCondition from the video backbone.
            proprio: (B, H, D) normalized proprio (``build_contexts`` turns it into context tokens; a
                ``backbone_stream`` family also embeds it in its own stream).
            proprio_mask: (B, H, D) bool, True = real entry.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool (training: ``train_options``;
                inference: False).
        """
        ...
