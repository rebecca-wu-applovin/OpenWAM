"""Cosmos3 ``proprio_in_und_prefix``: proprio tokens run through the und tower after the text.

1. ``run_und_suffix`` == running the whole und tower on ``[text ; suffix]`` (padded text, masked suffix slots).
2. Backbone ``prepare``: und prefix grows by N, the video stream reads the proprio tokens, gradients reach
   them, and a sample whose proprio mask is False is untouched by its tokens.
3. Without the tokens, ``proprio_in_und_prefix=true`` raises.
"""

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward, text_pack  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import Cosmos3OmniTransformer  # noqa: E402

MINI = dict(
    attention_bias=False,
    head_dim=6,
    hidden_size=12,
    intermediate_size=24,
    latent_channel=2,
    latent_patch_size=1,
    num_attention_heads=2,
    num_hidden_layers=2,
    num_key_value_heads=1,
    patch_latent_dim=2,
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    hidden_act="relu2",
    rms_norm_eps=1e-5,
    rope_axes_dim=[1, 1, 1],
    rope_theta=1e8,
    vocab_size=32,
)
# Nano-flavoured twin (SiLU, und QK-norm, no k_norm_und_for_gen).
NANO_MINI = dict(MINI, qk_norm_for_text=True, use_und_k_norm_for_gen=False, hidden_act="silu", rms_norm_eps=1e-6)


def _rotary(net, start, length):
    """Rotary cos/sin (1, length, head_dim) of und positions start..start+length-1."""
    pos = text_pack.text_mrope_positions(start + length, float_positions=True)[:, start:]
    return dit_forward.compute_rotary(net, pos.unsqueeze(1), torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("cfg", [MINI, NANO_MINI], ids=["edge", "nano"])
def test_suffix_matches_full_tower(cfg):
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**cfg).eval()
    # Sample 0: 3 real text tokens + 2 pads; sample 1: 5 real. Sample 1's second suffix slot is masked.
    text = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    text_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    suffix_ids = torch.tensor([[9, 10], [11, 12]])
    suffix_mask = torch.tensor([[True, True], [True, False]])

    with torch.no_grad():
        # Reference: one pass over [text ; suffix].
        cos, sin = _rotary(net, 0, 7)
        _, ref_kv, _ = dit_forward.run_und_tower_cached(
            net, torch.cat([text, suffix_ids], 1), torch.cat([text_mask, suffix_mask], 1), cos, sin
        )
        # Cached text, then the suffix as embeddings.
        cos_t, sin_t = _rotary(net, 0, 5)
        _, text_kv, text_k = dit_forward.run_und_tower_cached(net, text, text_mask, cos_t, sin_t)
        cos_s, sin_s = _rotary(net, 5, 2)
        suffix_kv = dit_forward.run_und_suffix(
            net, net.embed_tokens(suffix_ids), suffix_mask, text_k, text_kv, text_mask, cos_s, sin_s
        )

    for (rk, rv), (sk, sv) in zip(ref_kv, suffix_kv):
        assert torch.allclose(sk, rk[:, 5:], atol=1e-5)
        assert torch.allclose(sv, rv[:, 5:], atol=1e-5)


def _backbone(flag):
    """Mini Cosmos3 backbone on CPU fp32 with ``proprio_in_und_prefix=flag``."""
    torch.manual_seed(0)
    vb = Cosmos3EdgeVideoBackbone(
        net=Cosmos3OmniTransformer(**MINI),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        prompt_templates=False,
        proprio_in_und_prefix=flag,
    ).eval()
    vb.set_dtype_device(torch.float32, torch.device("cpu"))
    return vb


def _inputs(vb, proprio, proprio_mask):
    """prepare() kwargs for B=2 latents (2, 2, 3, 2, 2), 4 text tokens (sample 0 padded), N=3 proprio tokens."""
    net = vb.dit
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    und_mask = torch.tensor([[True, True, True, False], [True] * 4])
    cos, sin = _rotary(net, 0, 4)
    with torch.no_grad():
        context, und_kv, und_k = dit_forward.run_und_tower_cached(net, ids, und_mask, cos, sin)
    grid = text_pack.patch_grid(3, 2, 2, 1)
    _, vis_pos = text_pack.build_joint_positions(4, grid, modality_margin=15000, fps=24.0)
    torch.manual_seed(1)
    return dict(
        latents=torch.randn(2, 2, 3, 2, 2),
        timestep=torch.full((2,), 500.0),
        context=context,
        und_mask=und_mask,
        und_kv=und_kv,
        und_k=und_k,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=1,
        proprio_tokens=proprio,
        proprio_token_mask=proprio_mask,
    )


def _video(vb, inputs):
    """Video-only block loop. Returns (2, 2, 3, 2, 2) prediction."""
    state = vb.prepare(**inputs)
    for i in range(vb.num_layers):
        state = vb.run_block(i, state)
    return vb.finalize(state)


def test_video_reads_proprio_through_the_und_prefix():
    vb = _backbone(True)
    assert vb.reads_proprio_tokens
    mask = torch.tensor([[True] * 3, [False] * 3])  # sample 1 has no proprio
    proprio = torch.randn(2, 3, MINI["hidden_size"], requires_grad=True)

    state = vb.prepare(**_inputs(vb, proprio, mask))
    # Prefix = 4 text + 3 proprio keys; the mask carries both.
    assert state.prefix_kv_len == 7
    assert state.prefix_kv_mask.tolist() == [[True, True, True, False, True, True, True], [True] * 4 + [False] * 3]

    out = _video(vb, _inputs(vb, proprio, mask))
    other = _video(vb, _inputs(vb, proprio.detach() + 1.0, mask))
    # Sample 0 reads its proprio; sample 1 (mask False) does not.
    assert (out[0] - other[0]).abs().max() > 1e-4
    assert torch.equal(out[1], other[1])
    # Gradient flows from the video output back into the proprio tokens of sample 0 only.
    out[:, :, 1:].sum().backward()
    assert proprio.grad[0].abs().sum() > 0
    assert proprio.grad[1].abs().sum() == 0


def test_flag_on_without_tokens_raises():
    vb = _backbone(True)
    inputs = _inputs(vb, torch.zeros(2, 3, MINI["hidden_size"]), torch.ones(2, 3, dtype=torch.bool))
    del inputs["proprio_tokens"], inputs["proprio_token_mask"]
    with pytest.raises(ValueError, match="proprio_in_und_prefix"):
        vb.prepare(**inputs)


def test_flag_off_keeps_the_text_prefix():
    vb = _backbone(False)
    assert not vb.reads_proprio_tokens
    inputs = _inputs(vb, None, None)
    del inputs["proprio_tokens"], inputs["proprio_token_mask"]
    assert vb.prepare(**inputs).prefix_kv_len == 4
