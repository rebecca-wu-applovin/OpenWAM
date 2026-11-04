"""Hard merge gate: ``pre_attn_at_layer + attention + post_attn_at_layer`` must
produce numerically identical output to ``DiTBlock.forward`` / ``SelfAttnActionDiTBlock.forward``.

The split is the foundation of the joint self-attention path. If it diverges
from the plain block forward, every test that uses ``run_block`` silently keeps working while the new path quietly
produces wrong outputs. This file pins the equivalence at zero atol.
"""

from __future__ import annotations

import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT, SelfAttnActionDiTBlock
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.dit import DiTBlock, precompute_freqs_cis_3d


def _attention_single_stream(q, k, v, num_heads):
    """Standalone attention matching what DiTBlock.self_attn.attn does internally."""
    from openwam.model.video_backbone.wan.dit import flash_attention

    return flash_attention(q, k, v, num_heads=num_heads)


def _attention_action(q_bsd, k_bsd, v_bsd, num_heads):
    """Standalone attention for action stream — Q/K/V already in (B, S, H*D)."""
    from einops import rearrange

    n = num_heads
    q = rearrange(q_bsd, "b s (n d) -> b n s d", n=n)
    k = rearrange(k_bsd, "b s (n d) -> b n s d", n=n)
    v = rearrange(v_bsd, "b s (n d) -> b n s d", n=n)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return rearrange(out, "b n s d -> b s (n d)", n=n)


def _make_wan_dit_block(dim=64, num_heads=4, ffn_dim=128):
    block = DiTBlock(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
    block.eval()
    return block


def _wan_freqs(num_tokens: int, dim: int, num_heads: int) -> torch.Tensor:
    """Build a 3D RoPE freqs tensor matching what ``WanVideoBackbone.prepare`` produces."""
    head_dim = dim // num_heads
    f, h, w = num_tokens, 1, 1
    f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim, end=max(num_tokens, 1024))
    freqs = torch.cat(
        [
            f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs


def _masked_action_context(dit: ActionDiT, batch_size: int):
    context = torch.zeros(batch_size, 1, dit.text_dim)
    context_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    return context, context_mask


# ---------------------------------------------------------------------------
# Video DiTBlock equivalence: DiTBlock.forward is pre_attn -> attention -> post_attn; pin it against the
# block math written out inline (the one-piece forward it replaced), so the halves cannot drift.
# ---------------------------------------------------------------------------


def _wan_block_inline(block: DiTBlock, state: BlockLoopState, context_mask: torch.Tensor) -> torch.Tensor:
    """One Wan DiT block written out in one piece (the pre-split ``DiTBlock.forward``).

    Args:
        block: DiTBlock. state: BlockLoopState (hidden_states (B, S, dim), time_mod (B, S, 6, dim), rope_freqs,
            context (B, L, dim)). context_mask: (B, 1, S, L) bool.

    Returns:
        (B, S, dim) block output.
    """
    from openwam.model.video_backbone.wan.dit import modulate, rope_apply

    t_mod = state.time_mod
    chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (c.squeeze(2) for c in chunks)

    residual_x = state.hidden_states
    attn_input = modulate(block.norm1(state.hidden_states), shift_msa, scale_msa)

    sa = block.self_attn
    q = rope_apply(sa.norm_q(sa.q(attn_input)), state.rope_freqs, sa.num_heads)
    k = rope_apply(sa.norm_k(sa.k(attn_input)), state.rope_freqs, sa.num_heads)
    v = sa.v(attn_input)
    attn_out = _attention_single_stream(q, k, v, sa.num_heads)

    x = block.gate(residual_x, gate_msa, sa.o(attn_out))
    x = x + block.cross_attn(block.norm3(x), state.context, ctx_mask=context_mask)
    mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
    return block.gate(x, gate_mlp, block.ffn(mlp_input))


def test_wan_dit_block_split_equivalence():
    """DiTBlock.forward (pre_attn -> attention -> post_attn) == the inline block math, exact (atol=0)."""
    torch.manual_seed(0)
    # head_dim must be divisible by 6 so precompute_freqs_cis_3d's f/h/w split
    # produces full-coverage (matches Wan2.x production heads which are 64/96/128).
    dim, num_heads, ffn_dim = 96, 4, 192
    B, S = 2, 7

    block = _make_wan_dit_block(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
    x = torch.randn(B, S, dim)
    t_mod = torch.randn(B, S, 6, dim)  # per-token modulation, the only layout build_time_modulation emits
    context = torch.randn(B, 4, dim)
    context_mask = torch.ones(B, 1, S, 4, dtype=torch.bool)
    context_mask[1, :, :, -1] = False  # one padded text key
    attn_mask = torch.ones(S, S, dtype=torch.bool)
    freqs = _wan_freqs(S, dim, num_heads)

    with torch.no_grad():
        out_block = block(x, context, t_mod, freqs, context_mask, attn_mask)
    state = BlockLoopState(
        hidden_states=x, time_mod=t_mod, rope_freqs=freqs, context=context, grid_frames=S, grid_height=1, grid_width=1
    )
    with torch.no_grad():
        out_inline = _wan_block_inline(block, state, context_mask)

    # Hard equivalence — bf16 wouldn't allow atol=0 but we're in fp32.
    assert torch.equal(out_block, out_inline), (
        f"DiTBlock.forward disagrees with the inline math; max diff = {(out_block - out_inline).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Action SelfAttnActionDiTBlock equivalence — drives the ActionDiT(joint_self_attn) path.
# ---------------------------------------------------------------------------


def test_action_mot_block_split_equivalence():
    """ActionDiT pre_attn + attention + post_attn ≡ SelfAttnActionDiTBlock.forward (no context)."""
    torch.manual_seed(0)
    action_dim, dim, num_heads, ffn_dim = 7, 32, 4, 64
    B, S = 2, 5
    bridge_layers = (0, 1)

    dit = ActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=action_dim,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        num_layers=len(bridge_layers),
        video_dim=dim,
        bridge_layers=bridge_layers,
        variant="joint_self_attn",
        attn_head_dim=dim // num_heads,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    dit.eval()

    actions = torch.randn(B, S, action_dim)
    timestep = torch.randn(B)

    # Reference: drive the block manually
    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        payload = astate.payload
        payload.context = None
        payload.context_mask = None
        x_init = payload.x_action.clone()
        t_mod = payload.t_mod.clone()
        freqs = payload.action_freqs.clone()

        block: SelfAttnActionDiTBlock = dit.blocks[0]
        # SelfAttnActionDiTBlock.forward with context=None: self-attn norm/proj + RoPE + FFN.
        out_ref = block(x_init, context=None, t_mod=t_mod, freqs=freqs)

    # Split: ActionDiT pre/post_attn_at_layer round-trip with single-stream attention
    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate2 = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        astate2.payload.context = None
        astate2.payload.context_mask = None
        q, k, v, post = dit.pre_attn_at_layer(0, astate2)
        attn_out = _attention_action(q, k, v, num_heads)
        astate2 = dit.post_attn_at_layer(0, astate2, attn_out, post)
        out_split = astate2.payload.x_action

    assert torch.allclose(out_ref, out_split, atol=0, rtol=0), (
        f"ActionDiT split disagrees with SelfAttnActionDiTBlock.forward; max diff = {(out_ref - out_split).abs().max().item()}"
    )


def test_action_mot_block_heterogeneous_hidden_dim():
    """FastWAM-Joint-style: action hidden_dim=32 but Q/K/V live in num_heads*attn_head_dim=64.

    The split (pre_attn + SDPA + post_attn) must still equal the standalone
    SelfAttnActionDiTBlock.forward at atol=0. This is the regression gate for the
    heterogeneous-hidden refactor.
    """
    torch.manual_seed(0)
    # Heterogeneous: action hidden_dim=32, attn_hidden_dim = 4*16 = 64.
    action_dim, dim, num_heads, attn_head_dim, ffn_dim = 7, 32, 4, 16, 64
    B, S = 2, 5
    bridge_layers = (0, 1)

    dit = ActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=action_dim,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        num_layers=len(bridge_layers),
        video_dim=128,  # video residual width — irrelevant for the action-only check
        bridge_layers=bridge_layers,
        variant="joint_self_attn",
        attn_head_dim=attn_head_dim,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    dit.eval()

    block: SelfAttnActionDiTBlock = dit.blocks[0]
    # Sanity: q outputs the attention space (4*16=64), o brings it back to 32.
    assert block.self_attn.q.weight.shape == (num_heads * attn_head_dim, dim)
    assert block.self_attn.o.weight.shape == (dim, num_heads * attn_head_dim)

    actions = torch.randn(B, S, action_dim)
    timestep = torch.randn(B)

    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        payload = astate.payload
        payload.context = None
        payload.context_mask = None
        x_init = payload.x_action.clone()
        t_mod = payload.t_mod.clone()
        freqs = payload.action_freqs.clone()
        out_ref = block(x_init, context=None, t_mod=t_mod, freqs=freqs)

        context, context_mask = _masked_action_context(dit, B)
        astate2 = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        astate2.payload.context = None
        astate2.payload.context_mask = None
        q, k, v, post = dit.pre_attn_at_layer(0, astate2)
        # Q/K/V are already in (B, S, num_heads*attn_head_dim) — driver-ready.
        assert q.shape == (B, S, num_heads * attn_head_dim)
        attn_out = _attention_action(q, k, v, num_heads)
        astate2 = dit.post_attn_at_layer(0, astate2, attn_out, post)
        out_split = astate2.payload.x_action

    assert torch.allclose(out_ref, out_split, atol=0, rtol=0), (
        f"heterogeneous-hidden split disagrees with SelfAttnActionDiTBlock.forward; "
        f"max diff = {(out_ref - out_split).abs().max().item()}"
    )
