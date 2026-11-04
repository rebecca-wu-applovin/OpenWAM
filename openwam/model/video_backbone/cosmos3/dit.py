"""Batched block-loop engine for the Cosmos3-Edge backbone.

Reimplements ``Cosmos3OmniTransformer.forward``'s math with a batch dimension,
driving the vendored module's own submodules (projections, norms, MLPs,
``rotary_emb``, ``time_embedder``) so the weights and numerics are upstream's.
The upstream forward is a flat packed sequence with no batch axis and no
attention mask; OpenWAM batches are uniform-shaped, so instead of NVIDIA's
multi-sample packing we run ``(B, S, ·)`` tensors with:

- und stream (text): causal self-attention with a right-padding key mask,
  computed **once per forward** for the whole tower (the und stream never reads
  gen), caching per layer the gen-facing K (``k_norm_und_for_gen`` + rotary
  applied) and V — mirroring the processor's ``k_und_for_gen`` exactly.
- gen stream (video): per layer, bidirectional attention over
  ``[cached und K/V ; gen K/V]`` with GQA (``enable_gqa=True``), then the
  gen-side MLP half. Timestep conditioning is the upstream additive scatter:
  embeddings are added only to noisy-frame tokens at ``prepare()`` time.

Deviation from upstream (documented): ``finalize`` projects **all** gen tokens
through ``proj_out`` and returns the full ``(B, C, T, H, W)`` grid; upstream
decodes only noisy-frame tokens and zero-fills the rest. OpenWAM's loss masks
clean prefix frames via ``num_clean_prefix_frames``, so the extra frames are
never trained on; parity tests compare noisy frames only.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.common.checkpoint import (
    gradient_checkpoint_forward,
)

__all__ = [
    "patchify_latents",
    "unpatchify_tokens",
    "compute_rotary",
    "rotary_inv_freq",
    "run_und_tower",
    "prepare_block_loop",
    "run_block",
    "finalize_block_loop",
    "pre_self_attn",
    "post_self_attn",
]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x_bshd: Tensor, cos_bsd: Tensor, sin_bsd: Tensor) -> Tensor:
    """Upstream rotation, batched: ``x * cos + rotate_half(x) * sin`` with the
    head axis broadcast (upstream unsqueezes the head dim into cos/sin)."""
    cos = cos_bsd.unsqueeze(2)
    sin = sin_bsd.unsqueeze(2)
    return x_bshd * cos + _rotate_half(x_bshd) * sin


def patchify_latents(latents: Tensor, patch_size: int) -> Tuple[Tensor, Tuple[int, int, int], Tuple[int, int]]:
    """``(B, C, T, H, W)`` → ``(B, T·Hp·Wp, p·p·C)`` tokens (T-major raster).

    H/W are zero-padded up to a multiple of ``patch_size`` (upstream behavior).
    Returns ``(tokens, (T, Hp, Wp), (H_orig, W_orig))``.
    """
    b, c, t, h, w = latents.shape
    p = patch_size
    h_pad = (h + p - 1) // p * p
    w_pad = (w + p - 1) // p * p
    if h_pad != h or w_pad != w:
        latents = F.pad(latents, (0, w_pad - w, 0, h_pad - h))
    hp, wp = h_pad // p, w_pad // p
    x = latents.reshape(b, c, t, hp, p, wp, p)
    x = torch.einsum("bcthpwq->bthwpqc", x).reshape(b, t * hp * wp, p * p * c)
    return x, (t, hp, wp), (h, w)


def unpatchify_tokens(
    tokens: Tensor, grid: Tuple[int, int, int], patch_size: int, latent_channel: int, orig_hw: Tuple[int, int]
) -> Tensor:
    """Inverse of :func:`patchify_latents`: ``(B, T·Hp·Wp, p·p·C)`` → ``(B, C, T, H, W)``."""
    b = tokens.shape[0]
    t, hp, wp = grid
    p = patch_size
    x = tokens.reshape(b, t, hp, wp, p, p, latent_channel)
    x = torch.einsum("bthwpqc->bcthpwq", x).reshape(b, latent_channel, t, hp * p, wp * p)
    h, w = orig_hw
    return x[:, :, :, :h, :w]


def compute_rotary(net, position_ids_3bn: Tensor, device, dtype) -> Tuple[Tensor, Tensor]:
    """cos/sin ``(B, N, head_dim)`` from the vendored rotary module (fp32 matmul inside)."""
    cos, sin = net.rotary_emb(position_ids_3bn.to(device), device=device, dtype=dtype)
    return cos, sin


def rotary_inv_freq(head_dim: int, rope_theta: float) -> Tensor:
    """The rotary frequency table, recomputed in fp32 from config alone.

    Single source for the two places that need it without a trustworthy tensor
    to copy: the deploy meta shell (``inv_freq`` is non-persistent, so it is
    absent from the state_dict) and the fp32 repair in ``set_dtype_device``
    when no pristine copy was captured. Mirrors
    ``Cosmos3VLTextRotaryEmbedding.__init__`` exactly.
    """
    return 1.0 / (float(rope_theta) ** (torch.arange(0, int(head_dim), 2, dtype=torch.float32) / int(head_dim)))


def _und_attention_mask(und_mask: Tensor) -> Tensor:
    """``(B, L)`` real-token mask → ``(B, 1, L, L)`` bool: causal ∧ key-is-real."""
    b, length = und_mask.shape
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=und_mask.device))
    return causal.unsqueeze(0).unsqueeze(0) & und_mask.view(b, 1, 1, length)


# ``und_mask is None`` is the canonical "every und token is real" signal. It is
# decided host-side at pack time (all prompts the same length ⇒ no padding), and
# it matters for speed, not just tidiness: an explicit bool ``attn_mask``
# disqualifies both fused SDPA kernels — flash rejects any non-null mask, and
# the memory-efficient kernel rejects unequal q/kv head counts under
# ``enable_gqa`` — so a mask that carries no information costs an order of
# magnitude in time and memory. Passing ``None`` also keeps
# ``widen_mask_for_prefix_kv`` free of a ``.all()`` device sync, which would
# otherwise break the compiled MoT graph.


@torch.no_grad()
def run_und_tower(
    net,
    input_ids: Tensor,
    und_mask: Optional[Tensor],
    cos_und: Tensor,
    sin_und: Tensor,
) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
    """Run the full und (text) tower once, batched.

    The und pathway is frozen and computationally independent of gen, so it runs
    under ``no_grad`` — its per-layer K/V enter gen attention as constants.

    ``und_mask`` is the ``(B, L)`` real-token mask, or ``None`` when no prompt in
    the batch is padded — the unpadded case then runs as plain causal attention
    (``is_causal=True``, no explicit mask), which keeps the fused kernels.

    Returns ``(context, und_kv)`` where ``context`` is ``net.norm``-finalized
    hidden states ``(B, L, D)`` and ``und_kv[i] = (k_for_gen, v)`` with shapes
    ``(B, L, kv_heads, head_dim)``; ``k_for_gen`` already has
    ``k_norm_und_for_gen`` (when present) and rotary applied, matching the
    processor's ``k_und_for_gen``.
    """
    b, length = input_ids.shape
    n_heads = int(net.config.num_attention_heads)
    kv_heads = int(net.config.num_key_value_heads)
    head_dim = int(net.config.head_dim)

    und_seq = net.embed_tokens(input_ids)
    attn_mask = None if und_mask is None else _und_attention_mask(und_mask)
    is_causal = und_mask is None

    und_kv: List[Tuple[Tensor, Tensor]] = []
    for layer in net.layers:
        attn = layer.self_attn
        normed = layer.input_layernorm(und_seq)
        q = attn.to_q(normed).view(b, length, n_heads, head_dim)
        k = attn.to_k(normed).view(b, length, kv_heads, head_dim)
        v = attn.to_v(normed).view(b, length, kv_heads, head_dim)
        q = attn.norm_q(q)
        k = attn.norm_k(k)
        k_for_gen = attn.k_norm_und_for_gen(k) if attn.k_norm_und_for_gen is not None else k
        q = _apply_rope(q, cos_und, sin_und)
        k = _apply_rope(k, cos_und, sin_und)
        k_for_gen = _apply_rope(k_for_gen, cos_und, sin_und)
        und_kv.append((k_for_gen, v))

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=True,
        )
        attn_out = attn.to_out(out.transpose(1, 2).reshape(b, length, n_heads * head_dim))
        residual = und_seq + attn_out
        und_seq = residual + layer.mlp(layer.post_attention_layernorm(residual))

    context = net.norm(und_seq)
    return context, und_kv


def _noisy_frames(grid_t: int, num_clean_prefix_frames: int, device) -> Tensor:
    ncp = max(0, min(int(num_clean_prefix_frames), grid_t))
    return torch.arange(ncp, grid_t, device=device, dtype=torch.long)


def prepare_block_loop(
    net,
    *,
    latents: Optional[Tensor] = None,
    input_latents: Optional[Tensor] = None,
    timestep: Tensor,
    context: Tensor,
    und_mask: Optional[Tensor] = None,
    context_mask: Optional[Tensor] = None,
    und_kv: Optional[List[Tuple[Tensor, Tensor]]] = None,
    vision_positions: Optional[Tensor] = None,
    num_clean_prefix_frames: int = 0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **_ignored,
) -> BlockLoopState:
    """Pack the gen stream and assemble the loop state.

    ``context`` / ``und_mask`` / ``und_kv`` / ``vision_positions`` come from the
    backbone's preprocess (und tower already run). ``und_mask`` is the und
    padding mask consumed by the gen-attention prefix; it rides its own key
    because the architecture may extend ``context`` with a proprio token (the
    gen blocks never consume ``context`` — there is no cross-attention).
    ``timestep`` is ``(B,)`` or per-frame ``(B, T)``; the additive timestep
    embedding is applied here to noisy frames only (upstream contract).
    """
    x_in = latents if latents is not None else input_latents
    if x_in is None:
        raise ValueError("cosmos3 prepare_block_loop needs `latents` or `input_latents`.")
    if und_kv is None or vision_positions is None:
        raise ValueError("cosmos3 prepare_block_loop needs `und_kv` and `vision_positions` from preprocess.")

    b, _c, t_lat, _h, _w = x_in.shape
    p = int(net.config.latent_patch_size)
    tokens, grid, orig_hw = patchify_latents(x_in, p)
    grid_t, grid_h, grid_w = grid
    tokens = net.proj_in(tokens.to(next(net.proj_in.parameters()).dtype))
    target_dtype = tokens.dtype

    # Additive timestep embedding on noisy frames only.
    noisy = _noisy_frames(grid_t, num_clean_prefix_frames, tokens.device)
    if noisy.numel() > 0:
        ts = timestep.to(torch.float32)
        if ts.ndim == 0:
            ts = ts.view(1).expand(b)
        if ts.ndim == 1:
            ts_frames = ts.view(b, 1).expand(b, noisy.numel())
        else:  # (B, T) per-frame
            ts_frames = ts[:, noisy]
        ts_eff = ts_frames.reshape(-1) * float(net.config.timestep_scale)
        te_dtype = next(net.time_embedder.parameters()).dtype
        emb = net.time_embedder(net.time_proj(ts_eff).to(te_dtype)).to(target_dtype)
        emb = emb.view(b, noisy.numel(), 1, -1)
        tokens_btf = tokens.view(b, grid_t, grid_h * grid_w, -1)
        tokens_btf[:, noisy] = tokens_btf[:, noisy] + emb
        tokens = tokens_btf.view(b, grid_t * grid_h * grid_w, -1)

    cos_gen, sin_gen = compute_rotary(net, vision_positions, tokens.device, target_dtype)
    if cos_gen.shape[0] == 1 and b > 1:
        cos_gen = cos_gen.expand(b, -1, -1)
        sin_gen = sin_gen.expand(b, -1, -1)

    # ``und_mask is None`` means "no und padding" and is carried through as
    # None all the way to SDPA (see the note above _und_attention_mask); it is
    # NOT normalized into an all-True tensor here.
    #
    # ``context_mask`` is deliberately NOT derived from ``und_mask``: it is the
    # caller's mask of ``context`` (text + the proprio tokens the architecture
    # appends for the action stream), while ``und_mask`` covers the und tokens
    # only. Nothing in the cosmos3 gen path consumes ``state.context_mask``
    # (there is no cross-attention), so it stays whatever the caller passed.

    zero = torch.zeros((), dtype=target_dtype, device=tokens.device)
    return BlockLoopState(
        hidden_states=tokens,
        time_mod=zero,
        rope_freqs=torch.zeros((), dtype=torch.complex64, device=tokens.device),
        context=context,
        context_mask=context_mask,
        prefix_kv_len=int(und_kv[0][0].shape[1]),
        prefix_kv_mask=und_mask,
        grid_frames=grid_t,
        grid_height=grid_h,
        grid_width=grid_w,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        extras={
            "und_kv": und_kv,
            "und_mask": und_mask,
            "cos_gen": cos_gen,
            "sin_gen": sin_gen,
            "orig_hw": orig_hw,
            "noisy_frames": noisy,
        },
    )


def _gen_block_forward(
    layer,
    gen_seq: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    und_mask: Optional[Tensor],
    cos_gen: Tensor,
    sin_gen: Tensor,
    gen_mask: Optional[Tensor] = None,
) -> Tensor:
    """One decoder layer's gen half: bidirectional attention over
    ``[und K/V ; gen K/V]`` (GQA) + gen MLP. Mirrors the upstream layer body.

    ``gen_mask`` is an optional ``(S, S)`` (or ``(B, 1, S, S)``) bool mask over
    the gen block of the key axis, read from ``extras["shared_attention_mask"]``
    by :func:`run_block`. ``None`` keeps the gen block fully visible (the plain
    video path).
    """
    b, s, _ = gen_seq.shape
    attn = layer.self_attn
    n_heads = q_heads = attn.num_attention_heads
    kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim

    normed = layer.input_layernorm_moe_gen(gen_seq)
    q = attn.add_q_proj(normed).view(b, s, q_heads, head_dim)
    k = attn.add_k_proj(normed).view(b, s, kv_heads, head_dim)
    v = attn.add_v_proj(normed).view(b, s, kv_heads, head_dim)
    q = attn.norm_added_q(q)
    k = attn.norm_added_k(k)
    q = _apply_rope(q, cos_gen, sin_gen)
    k = _apply_rope(k, cos_gen, sin_gen)

    all_k = torch.cat([k_und.to(k.dtype), k], dim=1)
    all_v = torch.cat([v_und.to(v.dtype), v], dim=1)
    l_und = k_und.shape[1]
    if und_mask is None and gen_mask is None:
        # Nothing to express: every und key is real and the gen block is fully
        # visible. Passing None here is what lets SDPA reach a fused kernel.
        mask = None
    else:
        # (B, 1, S_q, L_und + S): und columns gated by the padding mask; the gen
        # block is fully visible unless the caller supplied a cross-modal mask.
        mask = torch.ones((b, 1, s, l_und + s), dtype=torch.bool, device=gen_seq.device)
        if und_mask is not None:
            mask[:, :, :, :l_und] = und_mask.view(b, 1, 1, l_und)
        if gen_mask is not None:
            gm = gen_mask if gen_mask.dim() == 4 else gen_mask.view(1, 1, s, s)
            mask[:, :, :, l_und:] = gm.to(device=mask.device, dtype=torch.bool)

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), all_k.transpose(1, 2), all_v.transpose(1, 2), attn_mask=mask, enable_gqa=True
    )
    attn_out = attn.to_add_out(out.transpose(1, 2).reshape(b, s, n_heads * head_dim))
    residual = gen_seq + attn_out
    return residual + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(residual))


def run_block(
    net,
    block_id: int,
    state: BlockLoopState,
    *,
    block_forward: Callable[..., Tensor] = _gen_block_forward,
) -> BlockLoopState:
    layer = net.layers[block_id]
    k_und, v_und = state.extras["und_kv"][block_id]
    state.hidden_states = gradient_checkpoint_forward(
        block_forward,
        state.use_gradient_checkpointing,
        state.use_gradient_checkpointing_offload,
        layer,
        state.hidden_states,
        k_und,
        v_und,
        state.extras["und_mask"],
        state.extras["cos_gen"],
        state.extras["sin_gen"],
        state.extras.get("shared_attention_mask"),
    )
    return state


def finalize_block_loop(net, state: BlockLoopState) -> Tensor:
    gen_out = net.norm_moe_gen(state.hidden_states)
    tokens = net.proj_out(gen_out)
    grid = (int(state.grid_frames), int(state.grid_height), int(state.grid_width))
    return unpatchify_tokens(
        tokens, grid, int(net.config.latent_patch_size), int(net.config.latent_channel), state.extras["orig_hw"]
    )


# ======================================================================
# MoT pre/post attention split for the Cosmos3-Edge gen stream.
#
# Splits one decoder layer's gen half at the attention op so the dual-system MoT
# driver can mix video and action Q/K/V in a single SDPA call:
#
# - :func:`pre_self_attn` runs norm → GQA projections → per-head norms → rotary,
#   prepends the cached und-stream K/V (the layer's ``k_norm_und_for_gen`` + rope
#   already applied by ``run_und_tower``), and **expands the 8 KV heads to the 16
#   Q heads** via ``repeat_interleave`` — mathematically identical to SDPA's
#   ``enable_gqa`` grouping (query head ``h`` reads kv head ``h // groups``), and
#   it keeps the driver's flat ``(B, S, H·D)`` single-``num_heads`` contract
#   untouched. Keys are therefore ``prefix_kv_len`` tokens longer than queries;
#   the driver widens the joint mask accordingly (``BlockLoopState.prefix_kv_*``).
# - :func:`post_self_attn` applies the output projection, residual add, and the
#   gen MLP half. It mutates nothing but the returned hidden states, satisfying
#   the driver's ``_step_checkpointed`` invariant.
#
# Parity contract: ``post(pre(x) → SDPA) == dit._gen_block_forward(x)``
# at fp32 tolerance (see tests/test_cosmos3_joint_self_attn.py).
# ======================================================================


def pre_self_attn(
    layer,
    gen_seq: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    cos_gen: Tensor,
    sin_gen: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Gen-half first stage. Returns ``(q, k, v, post_state)`` with
    ``q (B, S, H·D)`` and ``k/v (B, L_und + S, H·D)`` (KV heads pre-expanded)."""
    b, s, _ = gen_seq.shape
    attn = layer.self_attn
    n_heads = attn.num_attention_heads
    kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim
    groups = attn.num_key_value_groups

    normed = layer.input_layernorm_moe_gen(gen_seq)
    q = attn.norm_added_q(attn.add_q_proj(normed).view(b, s, n_heads, head_dim))
    k = attn.norm_added_k(attn.add_k_proj(normed).view(b, s, kv_heads, head_dim))
    v = attn.add_v_proj(normed).view(b, s, kv_heads, head_dim)
    q = _apply_rope(q, cos_gen, sin_gen)
    k = _apply_rope(k, cos_gen, sin_gen)

    all_k = torch.cat([k_und.to(k.dtype), k], dim=1)
    all_v = torch.cat([v_und.to(v.dtype), v], dim=1)
    if groups > 1:
        all_k = all_k.repeat_interleave(groups, dim=2)
        all_v = all_v.repeat_interleave(groups, dim=2)

    post_state = {"layer": layer, "residual": gen_seq}
    return (
        q.reshape(b, s, n_heads * head_dim),
        all_k.reshape(b, all_k.shape[1], n_heads * head_dim),
        all_v.reshape(b, all_v.shape[1], n_heads * head_dim),
        post_state,
    )


def post_self_attn(attn_out: Tensor, post_state: dict) -> Tensor:
    """Gen-half second stage: output projection → residual → gen MLP."""
    layer = post_state["layer"]
    residual = post_state["residual"]
    x = residual + layer.self_attn.to_add_out(attn_out)
    return x + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(x))


def state_pre_attn(net, block_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Adapter from BlockLoopState to :func:`pre_self_attn`."""
    layer = net.layers[block_id]
    k_und, v_und = state.extras["und_kv"][block_id]
    return pre_self_attn(
        layer,
        state.hidden_states,
        k_und,
        v_und,
        state.extras["cos_gen"],
        state.extras["sin_gen"],
    )


def state_post_attn(state: BlockLoopState, attn_out: Tensor, post_state: dict) -> BlockLoopState:
    """Adapter writing :func:`post_self_attn`'s result back onto the state."""
    state.hidden_states = post_self_attn(attn_out, post_state)
    return state
