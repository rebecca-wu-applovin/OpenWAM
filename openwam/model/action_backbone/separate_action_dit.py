"""ActionDiT: the action stream of the dual-system architectures, one class per coupling.

:class:`ActionDiT` is the shared base: action encoder, action-owned text/proprio context embedding, action
timestep conditioning, RoPE table, action decoder, scheduler, and the input preparation
(:meth:`ActionDiT.prepare_state`). Each coupling is its own subclass:

- :class:`CrossAttnActionDiT` (``joint_cross_attn``): a stack of :class:`CrossAttnActionDiTBlock` blocks
  (self-attn over action tokens + cross-attn to one video bridge feature + cross-attn to the text/proprio context
  + FFN). :meth:`CrossAttnActionDiT.forward` runs once, after the video backbone ran to completion, on the list of
  bridge features (one per block).
- :class:`MoTActionDiT` (``joint_self_attn``): a stack of :class:`SelfAttnActionDiTBlock` blocks with the same
  Q/K/V split layout as the video DiT. :class:`DualSystemMoTDriver` drives the per-layer loop through
  :meth:`MoTActionDiT.pre_attn_at_layer` / :meth:`MoTActionDiT.post_attn_at_layer`, concatenating Q/K/V across
  modalities and running one mixed attention.

Inspired by:
- CoVAR: Bridge attention between video and action streams (cross_attn).
- UWM: Independent diffusion timesteps for video and action.
- HunyuanVideo-Foley: Dual-stream MMDiT.
- SD3 MMDiT / FastWAM MoT: True joint self-attention with independent
  per-modality projections (self_attn).
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from openwam.model.action_backbone.components import (
    RMSNorm,
    TimestepEmbedding,
    TimestepModulation,
    precompute_freqs_cis_1d,
    rope_apply_1d,
)
from openwam.model.common.attention import flash_attention
from openwam.model.common.flow_scheduler import FlowScheduler
from openwam.model.grad_checkpoint import gradient_checkpoint_forward


@dataclass
class ActionDiTState:
    """Per-forward action-stream state: built by :meth:`ActionDiT.prepare_state`, read by the block loop.

    The MoT driver threads it through :meth:`MoTActionDiT.pre_attn_at_layer` /
    :meth:`MoTActionDiT.post_attn_at_layer` (only ``x_action`` changes per layer) and
    :meth:`MoTActionDiT.extract_prediction` decodes it.
    """

    x_action: torch.Tensor  # (B, T_action, dim) action hidden state
    t_mod: torch.Tensor  # (B, n_mod, dim) timestep modulation (n_mod = 9 cross-attn, 6 MoT)
    action_freqs: torch.Tensor  # (T_action, head_dim // 2) complex RoPE frequencies of the action positions
    context: torch.Tensor  # (B, L, dim) action-projected text/proprio context
    context_mask: torch.Tensor  # (B, T_action, L) bool, True = attend


class ActionSelfAttention(nn.Module):
    """Self-attention over action tokens with FastWAM-style heterogeneous projection.

    Q/K/V project from the residual ``hidden_dim`` to a separate attention
    space ``num_heads * attn_head_dim``; ``o`` projects back. This is the
    layout that lets two MoT experts share a single mixed attention even
    when their residual streams have different widths
    ([wan_video_dit.py:179-184](references/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L179)).
    For ``attn_head_dim == hidden_dim // num_heads`` this collapses back to
    the homogeneous ``Linear(dim, dim)`` form. 1D RoPE is applied to Q/K.
    """

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, eps: float):
        super().__init__()
        # attn_head_dim (positive, even for RoPE) is validated once, by ActionDiT.__init__.
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, S, hidden_dim) action tokens. freqs: (S, attn_head_dim // 2) complex RoPE frequencies.

        Returns: (B, S, hidden_dim) attention output, projected back by ``o``.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # RoPE rotates per head: split heads, rotate Q/K.
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        q = rope_apply_1d(q, freqs)
        k = rope_apply_1d(k, freqs)
        # the shared dispatcher takes packed heads (B, S, H*D) and returns the same layout
        q, k, v = (rearrange(t, "b n s d -> b s (n d)") for t in (q, k, v))
        x = flash_attention(q, k, v, num_heads=self.num_heads)
        return self.o(x)


class BridgeCrossAttention(nn.Module):
    """Cross-attention from action tokens to a key/value sequence (a video bridge feature or the context).

    Q is projected from the action ``hidden_dim``, K/V from ``kv_hidden_dim``; both land in the shared
    ``num_heads * attn_head_dim`` attention space. Inspired by CoVAR's Bridge Attention and mimic-video's
    cross-attention to intermediate video features.
    """

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, eps: float, kv_hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kv_hidden_dim = kv_hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(kv_hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(kv_hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x_action: torch.Tensor, x_kv: torch.Tensor, ctx_mask: torch.Tensor | None) -> torch.Tensor:
        """Args:
            x_action: (B, S_q, hidden_dim) action tokens.
            x_kv: (B, S_kv, kv_hidden_dim) keys/values (a projected video bridge, or the action context).
            ctx_mask: (B, S_q, S_kv) bool, True = attend; None = every key (each call site states it; the video
                bridge passes None).

        Returns: (B, S_q, hidden_dim) attention output.
        """
        q = self.norm_q(self.q(x_action))
        k = self.norm_k(self.k(x_kv))
        v = self.v(x_kv)

        # (B, Sq, Skv) mask -> (B, 1, Sq, Skv) so it broadcasts over heads; a mask forces SDPA in flash_attention
        if ctx_mask is not None:
            ctx_mask = ctx_mask.unsqueeze(1)
        x = flash_attention(q, k, v, num_heads=self.num_heads, attn_mask=ctx_mask)
        return self.o(x)


class CrossAttnActionDiTBlock(nn.Module):
    """One :class:`CrossAttnActionDiT` block.

    Layout: self-attn → cross-attn (to the video bridge) → cross-attn (to the text/proprio context) → FFN.
    Self-attn, bridge cross-attn and FFN are AdaLN-modulated by the action diffusion timestep (9 params);
    context cross-attn mirrors Wan/FastWAM text conditioning and is not timestep-gated.
    """

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, ffn_dim: int, eps: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        self.self_attn = ActionSelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        self.self_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        # The bridge is projected to the action width by CrossAttnActionDiT.video_projs before it gets here.
        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=hidden_dim)
        self.bridge_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.context_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=hidden_dim)
        self.context_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.modulation = nn.Parameter(torch.randn(1, 9, hidden_dim) / hidden_dim**0.5)

    def forward(
        self,
        x_action: torch.Tensor,
        x_video: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x_action: (B, T_action, dim) action hidden state. x_video: (B, T_video, dim) projected bridge feature.
            t_mod: (B, 9, dim) timestep modulation. freqs: (T_action, head_dim // 2) complex RoPE frequencies.
            context: (B, L, dim) projected context. context_mask: (B, T_action, L) bool, True = attend.

        Returns: (B, T_action, dim) action hidden state after the block.
        """
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff) = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(9, dim=1)

        h = self.self_attn_norm(x_action) * (1 + scale_sa) + shift_sa
        x_action = x_action + gate_sa * self.self_attn(h, freqs=freqs)

        h = self.bridge_attn_norm(x_action) * (1 + scale_ca) + shift_ca
        # The video bridge has no padded tokens: no mask.
        x_action = x_action + gate_ca * self.cross_attn(h, x_video, ctx_mask=None)

        x_action = x_action + self.context_attn(self.context_attn_norm(x_action), context, ctx_mask=context_mask)

        h = self.ffn_norm(x_action) * (1 + scale_ff) + shift_ff
        x_action = x_action + gate_ff * self.ffn(h)

        return x_action


class SelfAttnActionDiTBlock(nn.Module):
    """One :class:`MoTActionDiT` block, driven by the MoT joint self-attention loop.

    The block is split around self-attention so :class:`DualSystemMoTDriver` can
    concatenate video/action Q/K/V, run one mixed attention, and return the
    action slice. After that mixed attention, the action stream cross-attends
    to its own text/proprio context embedding and then runs the FFN. The
    mixed self-attention and FFN are AdaLN-modulated by the action diffusion
    timestep (6 params); context cross-attention is not timestep-gated.

    There is no ``forward``: :class:`MoTActionDiT` (``pre_attn_at_layer`` / ``post_attn_at_layer``) calls the
    sub-modules directly.
    """

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, ffn_dim: int, eps: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        # Self-attention with RoPE-aware Q/K — Q/K/V live in the shared
        # num_heads * attn_head_dim attention space; o() projects back to
        # ``hidden_dim`` so the residual stream can have its own width.
        self.self_attn = ActionSelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        # Cross-attention to the action expert's own text/proprio context
        # embedding. KV therefore lives in the action residual width.
        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=hidden_dim)

        self.self_attn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.context_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)


class ActionDiT(nn.Module):
    """Shared base of the two ActionDiT couplings (not built on its own).

    Owns the scheduler, the action encoder, the action-owned text/proprio context embedding, the RoPE table and
    the input preparation. Each subclass builds the rest (``time_embedding``, ``time_projection``, ``blocks``,
    ``action_decoder``, plus its coupling modules) in its own ``__init__``, keeping the parameter creation order
    (and so the init RNG draws) of the released checkpoints.

    Args:
        action_dim: raw action width (``common.action_dim``).
        dim: action residual width. May differ from the video hidden dim: Q/K/V project into the shared
            ``num_heads * attn_head_dim`` attention space (FastWAM-Joint pattern).
        ffn_dim: FFN intermediate width.
        num_heads / attn_head_dim: attention heads and per-head dim (yaml ``num_heads`` / ``head_dim``, the video
            geometry's; the MoT driver checks the parity).
        num_layers: number of action blocks.
        text_dim: raw context width before the action expert's own text/proprio projection.
        freq_dim / max_action_len / eps: sinusoidal timestep width, action RoPE table length, norm epsilon.
        shift_action: α-shift of the action scheduler's sigma grid (yaml ``shift_action``, train + deploy).
        train_timestep: DictConfig, the action scheduler's training law (yaml ``train_timestep``).
    """

    # Built by each subclass __init__ (between / after its coupling modules, in checkpoint creation order).
    time_embedding: TimestepEmbedding
    time_projection: TimestepModulation
    blocks: nn.ModuleList
    action_decoder: nn.Linear

    @staticmethod
    def _shared_kwargs(ab: DictConfig, action_dim: int, num_layers: int) -> dict:
        """The one place the ``action_backbone`` yaml keys map to the shared constructor args.

        Args:
            ab: ``merge(common, model).action_backbone`` (DictConfig): dim / ffn_dim / num_heads / head_dim /
                text_dim / freq_dim / max_action_len / eps / shift_action / train_timestep.
            action_dim: int, raw action width (``common.action_dim``).
            num_layers: int, number of action blocks (the architecture decides it).

        Returns: dict of :class:`ActionDiT` constructor kwargs.
        """
        # yaml head_dim is the constructor's attn_head_dim.
        return dict(
            action_dim=int(action_dim), dim=int(ab.dim), ffn_dim=int(ab.ffn_dim), num_heads=int(ab.num_heads),
            attn_head_dim=int(ab.head_dim), num_layers=int(num_layers), text_dim=int(ab.text_dim),
            freq_dim=int(ab.freq_dim), max_action_len=int(ab.max_action_len), eps=float(ab.eps),
            shift_action=float(ab.shift_action), train_timestep=ab.train_timestep,
        )

    def __init__(
        self,
        *,
        action_dim: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        text_dim: int,
        freq_dim: int,
        max_action_len: int,
        eps: float,
        shift_action: float,
        train_timestep: DictConfig,
    ):
        super().__init__()
        # The only copy of the action α-shift, read by training and inference alike.
        self.scheduler = FlowScheduler(shift=float(shift_action), train_timestep=train_timestep)
        # The one head_dim check: RoPE rotates pairs of channels.
        if attn_head_dim <= 0 or attn_head_dim % 2 != 0:
            raise ValueError(f"attn_head_dim must be a positive even int, got {attn_head_dim}")

        self.action_dim = action_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.num_layers = num_layers
        self.text_dim = int(text_dim)
        self.freq_dim = freq_dim
        self.max_action_len = max_action_len
        self.eps = eps

        # FastWAM-compatible action token embedding: raw action -> hidden.
        self.action_encoder = nn.Linear(action_dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        # Both couplings rely on RoPE inside attention — no learned absolute PE.
        # Keep this as a plain attribute, not a buffer: Module.to(dtype=bf16)
        # casts complex buffers to real and drops the imaginary part. _apply()
        # below moves it across devices while preserving complex dtype.
        # RoPE freqs are sized by attn_head_dim (the per-head attention dim that
        # both modalities share), not by hidden_dim/num_heads.
        self.freqs = precompute_freqs_cis_1d(attn_head_dim, max_action_len)

    def _apply(self, fn, recurse=True):
        """Module ``_apply`` (``.to`` / ``.cuda`` / ...), then move the RoPE table to the parameters' device.

        ``freqs`` is a plain complex attribute, not a buffer (a dtype cast must not touch it), so it follows the
        device here. Args / Returns: as :meth:`torch.nn.Module._apply`.
        """
        module = super()._apply(fn, recurse=recurse)
        self.freqs = self.freqs.to(device=self.action_decoder.weight.device)
        return module

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move the parameters / buffers to (dtype, device); the RoPE table follows the device (``_apply``)."""
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str) -> None:
        """No-op: the action weights are fully in the safetensors checkpoint (the architecture's deploy hook).

        Args: output_dir: str, the checkpoint dir (unused). Returns: None.
        """

    def _embed_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Args: action_tokens: (B, T_action, action_dim) noisy actions. Returns: (B, T_action, dim) tokens."""
        # The one length check: the RoPE table has max_action_len rows.
        T = action_tokens.shape[1]
        if T > self.max_action_len:
            raise ValueError(
                f"Action sequence length {T} exceeds max_action_len {self.max_action_len} (the RoPE table length)."
            )
        return self.action_encoder(action_tokens)

    def _prepare_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the raw context to the action width and expand its mask to one row per action token.

        Args:
            context: (B, L, text_dim) raw text/proprio context. context_mask: (B, L) bool, True = attend.
            batch_size: int B of the actions. seq_len: int T_action. dtype / device: of the action tokens.

        Returns: (context_emb (B, L, dim), context_attn_mask (B, T_action, L) bool).
        """
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B, L, text_dim], got {tuple(context.shape)}")
        if context.shape[0] != batch_size:
            raise ValueError(f"context batch size ({context.shape[0]}) must match actions batch size ({batch_size})")
        if context.shape[-1] != self.text_dim:
            raise ValueError(f"context last dim must match text_dim={self.text_dim}, got {context.shape[-1]}")
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"context_mask must have shape {tuple(context.shape[:2])}, got {tuple(context_mask.shape)}"
            )

        context = context.to(device=device, dtype=dtype)
        context_emb = self.text_embedding(context)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        # Every action token sees the same context keys.
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        return context_emb, context_attn_mask

    def _prepare_timestep(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Args: timestep: (B,) or (1,) action timestep. batch_size: int B.

        Returns: (B,) timestep; a (1,) one is broadcast, which only inference may do.
        """
        if timestep.ndim != 1:
            raise ValueError(f"action timestep must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"action timestep length must be 1 or batch size ({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        return timestep

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> ActionDiTState:
        """Embed the actions and build the per-forward inputs of the block loop.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy action sequence.
            timestep: (B,) or (1,) action diffusion timestep.
            context: (B, L, text_dim) raw text/proprio context.
            context_mask: (B, L) bool, True = attend.

        Returns:
            ActionDiTState: x_action (B, T_action, dim), its modulation, RoPE frequencies and projected context.
        """
        x = self._embed_actions(noisy_actions)
        timestep = self._prepare_timestep(timestep, noisy_actions.shape[0])
        t_mod = self.time_projection(self.time_embedding(timestep))
        # _embed_actions checked T_action <= max_action_len, the table length.
        action_freqs = self.freqs[: x.shape[1]].to(device=x.device)
        action_context, action_context_mask = self._prepare_context(
            context,
            context_mask,
            batch_size=noisy_actions.shape[0],
            seq_len=x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
        return ActionDiTState(
            x_action=x,
            t_mod=t_mod,
            action_freqs=action_freqs,
            context=action_context,
            context_mask=action_context_mask,
        )


class CrossAttnActionDiT(ActionDiT):
    """``joint_cross_attn`` ActionDiT: runs once on the video bridge features, one bridge per block.

    Args: as :class:`ActionDiT`, plus ``video_dim``: int width of the bridge features (the video hidden dim),
    projected to ``dim`` per block on entry.
    """

    @classmethod
    def from_cfg(cls, ab: DictConfig, *, action_dim: int, num_layers: int) -> "CrossAttnActionDiT":
        """Build from the ``action_backbone`` slice (keys of :meth:`ActionDiT._shared_kwargs` + ``video_dim``).

        Args: ab: the slice. action_dim: int raw action width. num_layers: int, one block per bridged video layer.

        Returns: CrossAttnActionDiT, freshly initialised on CPU.
        """
        return cls(**cls._shared_kwargs(ab, action_dim, num_layers), video_dim=int(ab.video_dim))

    def __init__(self, *, video_dim: int, **shared):
        super().__init__(**shared)
        self.video_dim = int(video_dim)
        dim, num_layers = self.dim, self.num_layers
        # Creation order (= init RNG draws) is the checkpoints': video_projs come before the time embedding.
        if video_dim != dim:
            self.video_projs = nn.ModuleList([nn.Linear(video_dim, dim) for _ in range(num_layers)])
        else:
            self.video_projs = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.time_embedding = TimestepEmbedding(self.freq_dim, dim)
        # 9 modulation params per block: 3 self-attn + 3 bridge cross-attn + 3 FFN.
        self.time_projection = TimestepModulation(dim, 9)
        self.blocks = nn.ModuleList(
            [
                CrossAttnActionDiTBlock(dim, self.num_heads, self.head_dim, self.ffn_dim, self.eps)
                for _ in range(num_layers)
            ]
        )
        # FastWAM-compatible simple action decoder; no AdaLN and no zero init.
        self.action_decoder = nn.Linear(dim, self.action_dim)

    def _validate_bridges(self, x: torch.Tensor, bridges: list[torch.Tensor]) -> None:
        """Raise unless there is one (B, T_video, video_dim) bridge per block on the actions' device / dtype.

        Args: x: (B, T_action, dim) embedded actions. bridges: list of bridge features, block order.
        """
        if len(bridges) != self.num_layers:
            raise ValueError(f"got {len(bridges)} bridge features for {self.num_layers} action blocks (one each)")
        for i, bridge in enumerate(bridges):
            if bridge.ndim != 3:
                raise ValueError(
                    f"bridge for action block {i} must have shape [B, T_video, C], got {tuple(bridge.shape)}"
                )
            if bridge.shape[0] != x.shape[0]:
                raise ValueError(
                    f"bridge batch size ({bridge.shape[0]}) must match actions batch size ({x.shape[0]})."
                )
            if bridge.device != x.device:
                raise RuntimeError(
                    f"bridge device mismatch at action block {i}: bridge={bridge.device}, actions={x.device}."
                )
            if bridge.dtype != x.dtype:
                raise RuntimeError(
                    f"bridge dtype mismatch at action block {i}: bridge={bridge.dtype}, actions={x.dtype}."
                )
            if bridge.shape[-1] != self.video_dim:
                raise ValueError(
                    f"bridge last dim ({bridge.shape[-1]}) must match video_dim={self.video_dim} for action block {i}."
                )

    def forward(
        self,
        action_tokens: torch.Tensor,
        bridges: list[torch.Tensor],
        timestep: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        """Action prediction from the per-block video bridge features (the video backbone already ran).

        Args:
            action_tokens: (B, T_action, action_dim) noisy action sequence.
            bridges: list of num_layers (B, T_video, video_dim) video hidden states, block i's bridge at index i.
            timestep: (B,) or (1,) action diffusion timestep.
            context: (B, L, text_dim) raw text/proprio context. context_mask: (B, L) bool, True = attend.
            use_gradient_checkpointing / use_gradient_checkpointing_offload: bool, per-block checkpointing
                (only while training).

        Returns:
            (B, T_action, action_dim) predicted action velocity.
        """
        state = self.prepare_state(action_tokens, timestep, context=context, context_mask=context_mask)
        x = state.x_action
        self._validate_bridges(x, bridges)

        for i, block in enumerate(self.blocks):
            x_video_i = self.video_projs[i](bridges[i])
            x = gradient_checkpoint_forward(
                block,
                self.training and use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
                x_video_i,
                state.t_mod,
                state.action_freqs,
                state.context,
                state.context_mask,
            )

        return self.action_decoder(x)


class MoTActionDiT(ActionDiT):
    """``joint_self_attn`` ActionDiT: one block per video layer, driven by :class:`DualSystemMoTDriver`.

    Args: as :class:`ActionDiT`; ``num_layers`` is the video backbone's layer count.
    """

    @classmethod
    def from_cfg(cls, ab: DictConfig, *, action_dim: int, num_layers: int) -> "MoTActionDiT":
        """Build from the ``action_backbone`` slice (keys of :meth:`ActionDiT._shared_kwargs`).

        Args: ab: the slice. action_dim: int raw action width. num_layers: int, the video backbone's layer count.

        Returns: MoTActionDiT, freshly initialised on CPU.
        """
        return cls(**cls._shared_kwargs(ab, action_dim, num_layers))

    def __init__(self, **shared):
        super().__init__(**shared)
        dim = self.dim
        self.time_embedding = TimestepEmbedding(self.freq_dim, dim)
        # 6 modulation params per block: 3 joint self-attn + 3 FFN (text cross-attn has no modulation in the
        # MoT block, mirroring FastWAM's Wan-aligned DiTBlock).
        self.time_projection = TimestepModulation(dim, 6)
        self.blocks = nn.ModuleList(
            [
                SelfAttnActionDiTBlock(dim, self.num_heads, self.head_dim, self.ffn_dim, self.eps)
                for _ in range(self.num_layers)
            ]
        )
        # FastWAM-compatible simple action decoder; no AdaLN and no zero init.
        self.action_decoder = nn.Linear(dim, self.action_dim)

    def pre_attn_at_layer(
        self, layer_id: int, astate: ActionDiTState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """First half of an action block: self-attn norm + AdaLN + Q/K/V + RoPE.

        Args:
            layer_id: int, action block index. astate: ActionDiTState from :meth:`prepare_state` (or the
                previous layer).

        Returns: (q, k, v, post_state): Q/K/V shaped ``(B, T_action, num_heads * head_dim)`` (matching
        the layout produced by Wan ``self_attn.q/k/v`` after RMSNorm and RoPE
        — ready to concatenate with the video Q/K/V); ``post_state`` the dict of block tensors
        :meth:`post_attn_at_layer` consumes.
        """
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]

        chunks = (block.modulation.to(dtype=astate.t_mod.dtype, device=astate.t_mod.device) + astate.t_mod).chunk(
            6, dim=1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = astate.x_action
        attn_input = block.self_attn_norm(residual_x) * (1 + scale_msa) + shift_msa

        sa = block.self_attn
        q = sa.norm_q(sa.q(attn_input))
        k = sa.norm_k(sa.k(attn_input))
        v = sa.v(attn_input)

        # Apply RoPE in head-split layout to match flash_attn / Wan rope_apply.
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        q = rope_apply_1d(q, astate.action_freqs)
        k = rope_apply_1d(k, astate.action_freqs)
        # Driver consumes (B, S, H*D) — keep modality streams in matching layout.
        q_out = rearrange(q, "b n s d -> b s (n d)", n=self.num_heads)
        k_out = rearrange(k, "b n s d -> b s (n d)", n=self.num_heads)
        v_out = rearrange(v, "b n s d -> b s (n d)", n=self.num_heads)

        # Tensors the block's second half (post_attn_at_layer) consumes.
        post_state = {
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q_out, k_out, v_out, post_state

    def post_attn_at_layer(
        self,
        layer_id: int,
        astate: ActionDiTState,
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> ActionDiTState:
        """Second half of an action block: gate(residual, self_attn.o(attn_out))
        → action-owned text/proprio cross-attn → FFN.

        ``attn_out`` is the unprojected attention output for the action slice
        of the joint mixed attention; ``self_attn.o`` is applied here.
        Language/proprio conditioning uses the action expert's independent
        ``text_embedding`` stored in :class:`ActionDiTState`.

        Args:
            layer_id: int, action block index. astate: ActionDiTState. attn_out: (B, T_action, num_heads * head_dim)
                attention output of the action slice. post_state: dict from :meth:`pre_attn_at_layer`.

        Returns: ``astate`` with ``x_action`` (B, T_action, dim) advanced by one block.
        """
        block: SelfAttnActionDiTBlock = self.blocks[layer_id]
        residual_x = post_state["residual_x"]
        gate_msa, shift_mlp = post_state["gate_msa"], post_state["shift_mlp"]
        scale_mlp, gate_mlp = post_state["scale_mlp"], post_state["gate_mlp"]

        # Gated residual of the joint attention, then the (ungated) context cross-attn, then the gated FFN.
        x = residual_x + gate_msa * block.self_attn.o(attn_out)
        x = x + block.cross_attn(block.context_attn_norm(x), astate.context, ctx_mask=astate.context_mask)
        mlp_input = block.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * block.ffn(mlp_input)

        astate.x_action = x
        return astate

    def extract_prediction(self, astate: ActionDiTState) -> torch.Tensor:
        """astate: ActionDiTState after the last block. Returns: (B, T_action, action_dim) decoded prediction."""
        return self.action_decoder(astate.x_action)


__all__ = ["ActionDiT", "ActionDiTState", "CrossAttnActionDiT", "MoTActionDiT"]
