# Vendored from Genie-Envisioner (AgibotTech/Genie-Envisioner @ d41358f,
# models/ltx_models/transformer_ltx_multiview.py), itself modified from Diffusers /
# LTX-Video. Apache License 2.0 (see the upstream LICENSE).
#
# Changes vs upstream: the action expert is dropped (OpenWAM brings its own
# ActionDiT), diffusers' stock ``Attention`` replaces GE's verbatim copy of it
# (same parameters / state-dict keys), and dead branches (gradient checkpointing,
# video buffers) are removed. Parameter names are unchanged so
# ``GE_base_fast_v0.1.safetensors`` loads strictly. The upstream ``forward``s of the
# model and the block are removed: OpenWAM drives the blocks through ``ge/backbone.py``
# (pre / post attention halves) using the same modules.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from einops import rearrange


class LTXVideoAttentionProcessor2_0:
    """GE's multi-view attention processor (upstream verbatim, comments trimmed).

    Self-attn (``image_rotary_emb`` given): per-view on a ``(b v)`` batch, or all views
    concatenated when ``cross_view_attn``. Cross-attn: every view's queries of one sample
    attend the shared text of that sample.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        n_view: int = 1,
        cross_view_attn: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if cross_view_attn:
            if attention_mask is not None:
                batch_size = batch_size // n_view
                attention_mask = attention_mask.repeat(batch_size, n_view, n_view)
                sequence_length = attention_mask.shape[-1]
        elif image_rotary_emb is not None and attention_mask is not None:
            attention_mask = attention_mask.repeat(batch_size, 1, 1)
            sequence_length = attention_mask.shape[-1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(encoder_hidden_states))
        value = attn.to_v(encoder_hidden_states)

        if image_rotary_emb is not None:
            # Self-attn: RoPE per view, then optionally merge the views into one sequence.
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            if cross_view_attn:
                query = rearrange(query, "(b v) l c -> b (v l) c", v=n_view)
                key = rearrange(key, "(b v) l c -> b (v l) c", v=n_view)
                value = rearrange(value, "(b v) l c -> b (v l) c", v=n_view)
        else:
            # Cross-attn: all views of a sample share one text sequence.
            query = rearrange(query, "(b v) l c -> b (v l) c", v=n_view)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if cross_view_attn or image_rotary_emb is None:
            hidden_states = rearrange(hidden_states, "b (v l) c -> (b v) l c", v=n_view)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class LTXVideoRotaryPosEmbed(nn.Module):
    """LTX 3D RoPE over a ``(num_frames, height, width)`` grid, upstream verbatim."""

    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def forward(self, hidden_states, rope_interpolation_scale=None, num_frames=None, height=None, width=None, **kwargs):
        """Returns ``(cos, sin)``, each ``(B, F*H*W, dim)`` fp32, token order (f, h, w)."""
        batch_size = hidden_states.shape[0]
        # Always compute rope in fp32.
        grid_h = torch.arange(height, dtype=torch.float32, device=hidden_states.device)
        grid_w = torch.arange(width, dtype=torch.float32, device=hidden_states.device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=hidden_states.device)
        grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        if rope_interpolation_scale is not None:
            # (1/latent_frame_rate, spatial ratio, spatial ratio) rescale into the base grid.
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width
        grid = grid.flatten(2, 4).transpose(1, 2)

        freqs = self.theta ** torch.linspace(
            math.log(1.0, self.theta), math.log(self.theta, self.theta), self.dim // 6,
            device=hidden_states.device, dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)
        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)
        if self.dim % 6 != 0:
            # Identity rotation on the leftover channels.
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        return cos_freqs, sin_freqs


class LTXVideoTransformerBlock(nn.Module):
    """LTX block modules: RMSNorm + AdaLN self-attn (attn1), un-normed text cross-attn (attn2), FFN.

    No ``forward``: ``ge/backbone.py`` runs them (``pre_attn_at_layer`` / ``post_attn_at_layer``).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim, heads=num_attention_heads, kv_heads=num_attention_heads, dim_head=attention_head_dim,
            bias=attention_bias, cross_attention_dim=None, out_bias=attention_out_bias, qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )
        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads,
            kv_heads=num_attention_heads, dim_head=attention_head_dim, bias=attention_bias,
            out_bias=attention_out_bias, qk_norm=qk_norm, processor=LTXVideoAttentionProcessor2_0(),
        )
        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)


class LTXVideoTransformer3DModel(ModelMixin, ConfigMixin):
    """GE multi-view LTX transformer (video stream only)."""

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        use_view_embed: bool = True,
        max_view: int = 3,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)
        self.use_view_embed = use_view_embed
        if self.use_view_embed:
            # Learned per-view embedding + its AdaLN modulation (added to temb).
            self.view_embed = nn.Parameter(torch.randn(max_view, inner_dim))
            self.view_ada = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim, base_num_frames=20, base_height=2048, base_width=2048,
            patch_size=patch_size, patch_size_t=patch_size_t, theta=10000.0,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim, qk_norm=qk_norm, activation_fn=activation_fn,
                    attention_bias=attention_bias, attention_out_bias=attention_out_bias, eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)


def apply_rotary_emb(x, freqs):
    """Rotate ``x`` (B, S, dim) by ``freqs = (cos, sin)`` each (1 or B, S, dim); upstream verbatim."""
    cos, sin = freqs
    batch_size = x.shape[0]
    if cos.shape[0] == 1 and batch_size > 1:
        cos = cos.repeat(batch_size, 1, 1)
    if sin.shape[0] == 1 and batch_size > 1:
        sin = sin.repeat(batch_size, 1, 1)
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
