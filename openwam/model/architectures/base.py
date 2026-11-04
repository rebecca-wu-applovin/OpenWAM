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
from torch import Tensor, nn

from openwam.model.compile_options import compile_enabled
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
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want
    frozen instead. The current freeze list in ``configs/model/*.yaml``
    only names complete subtrees, so this limitation does not bite today.
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
    """

    use_gradient_checkpointing: bool
    use_gradient_checkpointing_offload: bool
    lambda_video: float
    lambda_action: float


# Deploy builds: no loss is ever computed, so no checkpointing and both loss weights unused.
INFERENCE_ONLY = TrainOptions(
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, lambda_video=0.0, lambda_action=0.0
)


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
        cfg: dict of architecture params (``resolve_architecture_config(...).params``): ``clean_attention`` and
            ``attention_mask_mode`` pick the frame-level mask, ``proprio_input`` the proprio routing (all validated by
            ``resolve_config``), ``video_backbone`` is the backbone's cfg slice.
        layout: ``cfg.common.layout`` from the entry point's ``resolve_config`` (DictConfig): the latent layout
            the preprocessing, the attention mask, the loss and deploy read. Never re-derived here.
        mode: LoadMode the entry point chose for the video backbone (PRETRAINED: upstream weights;
            CHECKPOINT: an empty shell of ``ckpt_dir`` the caller fills).
        ckpt_dir: str checkpoint dir for CHECKPOINT, None for PRETRAINED.
        train_options: TrainOptions (gradient checkpointing, loss weights) the trainer decided; deploy passes
            ``INFERENCE_ONLY``.
    """

    # Built by the subclass ``__init__`` (every architecture has an action stream).
    action_backbone: "ActionDiTBackbone"

    # --- Construction & config resolution ---

    def __init__(self, cfg: dict, *, layout, mode: LoadMode, ckpt_dir: Optional[str], train_options: TrainOptions):
        from openwam.model.architectures.utils.mask_modes import build_frame_mask

        super().__init__()
        self.cfg = cfg
        self.layout = layout
        self.train_options = train_options
        # Frame-level [latents, action] visibility, built once; backbones expand it to tokens per forward.
        self.frame_mask = build_frame_mask(layout, str(cfg["clean_attention"]), str(cfg["attention_mask_mode"]))
        # How the video backbone gets the proprio (validated by resolve_config); build_contexts applies it.
        self.proprio_input = str(cfg["proprio_input"])
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Optional action normalizer for deployment: the deploy engine uses it to
        # return real-scale actions and to normalize raw robot state into the model's training space.
        self.normalizer = None
        # Optional deploy post-unnormalize hook (relative action modes), set by the deploy model loader.
        self.deploy_action_decoder = None

        self.video_backbone: "VideoBackbone" = self._build_video_backbone(cfg["video_backbone"], mode, ckpt_dir)

    @staticmethod
    def _build_video_backbone(vb_cfg, mode: LoadMode, ckpt_dir: Optional[str]) -> "VideoBackbone":
        """Build the video backbone with the load mode the entry point chose, and check the yaml's VAE contract.

        Args:
            vb_cfg: ``cfg.model.video_backbone`` (DictConfig slice; ``name`` picks the registered class).
            mode: LoadMode decided by the entry point. ckpt_dir: str for CHECKPOINT, None for PRETRAINED.

        Returns:
            VideoBackbone. Raises ValueError when the yaml's temporal contract disagrees with the built backbone.
        """
        video_backbone = build_video_backbone(vb_cfg, mode, ckpt_dir)
        # The layout and the dataloader read the yaml's temporal_compression: it must be the VAE's.
        declared = (int(vb_cfg["temporal_compression"]), bool(vb_cfg["causal_temporal"]))
        actual = (video_backbone.temporal_compression, video_backbone.causal_temporal)
        if declared != actual:
            raise ValueError(
                f"video_backbone.temporal_compression / causal_temporal yaml {declared} does not match native VAE "
                f"{actual}. Update the yaml fields to match."
            )
        return video_backbone

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

    def _init_proprio_context(self, cfg: dict, *, text_dim: int) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning.

        Args:
            cfg: dict of architecture params (``use_proprioception``, ``state_dim``).
            text_dim: int, width of the text context the proprio tokens are appended to.

        Returns:
            None. Declares ``proprio_encoder`` / ``proprio_slot_emb`` (None when unused).
        """
        enabled = bool(cfg["use_proprioception"])
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_slot_emb: Optional[nn.Embedding] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        # One proprio token per history frame (oldest first), from the layout.
        self.num_proprio_tokens = int(self.layout.num_proprio_tokens)
        if not enabled:
            return
        # Required by the yaml; only read when the proprio tokens exist.
        state_dim = int(cfg["state_dim"])
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

    # --- Normalizer (deployment) ---

    def attach_normalizer(self, normalizer) -> None:
        """Attach (or clear) the deploy action normalizer.

        Deployment paths build the same normalizer used by training from
        ``normalization_stats.npy``. The deploy engine uses it to return real-scale actions
        and to normalize raw robot state into the model's training space. Pass ``None`` to clear.
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
        if self.normalizer is not None:
            arr = self.normalizer.normalize(arr)
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

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training.

        Single source of truth for each stream's α-shift:
        ``video_backbone.shift_video`` and ``action_backbone.shift_action``.
        The same properties are read by the deploy schedule at inference time,
        so the discrete training sigma buffer and the inference denoising
        trajectory are sampled from the same shifted schedule — train/inference
        cannot drift, and the shift is owned by the checkpoint config (not a
        separate deploy-time knob).

        Args:
            num_timesteps: int, size of the discrete training grid.

        Returns:
            None.
        """
        video_backbone, action_backbone = self.video_backbone, self.action_backbone
        video_backbone.scheduler.set_timesteps(num_timesteps, training=True, shift=video_backbone.shift_video)
        action_backbone.scheduler.set_timesteps(num_timesteps, training=True, shift=action_backbone.shift_action)

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
        action_scheduler = self.action_backbone.scheduler
        lambda_video, lambda_action = self.train_options.lambda_video, self.train_options.lambda_action
        _dtype = self.dtype
        _device = self.device

        # Clean latent clip (B, z, num_latents, h, w); the first layout.num_hist_latents are the condition.
        clean_latents, text = video_backbone.preprocess_train(batch.video, batch.prompt, self.layout)
        B = clean_latents.shape[0]

        # --- Sample video timesteps ---
        # The scheduler owns the training noise law: the grid draw (Wan / cosmos3) or a native one (GE / flux).
        video_timesteps, video_sigmas, video_loss_weights = video_backbone.scheduler.sample_train(
            B, device=_device, dtype=_dtype
        )

        # --- Add video noise (flow-matching: linear interp + velocity target) ---
        video_noise = torch.randn_like(clean_latents)
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        noisy_latents = (1 - sigma_bc) * clean_latents + sigma_bc * video_noise
        video_target = video_noise - clean_latents

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
        """Resolve which unified-action dims must ride the analytic noise path.

        An explicit ``active_action_mask`` wins; otherwise the active indices
        are inferred from the attached unify normalizer's scatter map (its
        absence, or a width mismatch with ``action_dim``, disables pinning).
        Returns a bool ``(action_dim,)`` mask of INACTIVE dims, or ``None``
        when every dim is active.
        """
        if active_action_mask is None:
            normalizer = self.normalizer
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
        dit_cache=None,
        cfg_scale: float = 1.0,
        cfg_merge: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Joint video-action Euler loop over a ``(t_video, t_action)`` schedule (``sample``: deploy and vis).

        Args:
            schedule: list of (t_video, t_action) pairs ending with the (0, 0) sentinel (``make_schedule``).
            latents: (B, C, T, h, w) initial latents (noise, clean prefix written in).
            action_latents: (B, K, A) initial action noise.
            hist_latents: (B, C, num_hist_latents, h, w) clean prefix, re-imposed after every video step.
            text: TextCondition. proprio: (B, H, D) normalized proprio. proprio_mask: (B, H, D) bool.
            inactive_action_dims: (A,) index of unified-action dims kept on the analytic noise path, or None.
            dit_cache: optional DiT velocity cache. cfg_scale / cfg_merge: classifier-free guidance (scale 1 = off).

        Returns: (latents (B, C, T, h, w) denoised video latents, actions (B, K, A) in normalized space).
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
                        inputs_shared={"latents": latents, "text": text, "proprio": proprio},
                        v_timestep=v_timestep,
                        cfg_scale=cfg_scale_f,
                        cfg_merge=bool(cfg_merge),
                    )
                else:
                    torch.compiler.cudagraph_mark_step_begin()
                    # Inference never checkpoints (it would also bypass the compiled loops).
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
                if dit_cache is not None and video_stepping:
                    dit_cache.update(noise_pred, sigma_v, action_noise_pred)

            if video_stepping:
                new_latents = latents + noise_pred * (sigma_v_next - sigma_v)
                new_latents = new_latents.clone()
                new_latents[:, :, : hist_latents.shape[2]] = hist_latents
                latents = new_latents

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
            active_action_mask: optional (A,) bool of the active unified-action dims (None: from the normalizer).

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
        self.video_backbone.apply_compile_optimizations(compile_cfg)

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
    "proprio",
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

    for key in _CFG_BATCH_AXIS_KEYS:
        v = expanded.get(key)
        if isinstance(v, Tensor):
            expanded[key] = torch.cat([v, v], dim=0)
    al = torch.cat([action_latents, action_latents], dim=0) if isinstance(action_latents, Tensor) else None
    vt = torch.cat([v_timestep, v_timestep], dim=0) if isinstance(v_timestep, Tensor) else v_timestep
    at = torch.cat([a_timestep, a_timestep], dim=0) if isinstance(a_timestep, Tensor) else None
    return expanded, al, vt, at
