"""Cosmos3 ``proprio_in_und_prefix`` and ``text_pad_length`` (ported from yjw/1008_our_data c4ba32c / d88b5a7).

1. ``run_und_suffix`` == running the whole und tower on ``[text ; suffix]`` (padded text, masked suffix slots).
2. Backbone ``prepare``: the und prefix grows by N, the video stream reads the proprio tokens, gradients reach
   them, and a sample whose proprio mask is False is untouched by its tokens.
3. The flag on without proprio tokens in the context raises; the flag off keeps the text-only prefix.
4. A fixed ``text_pad_length`` makes a prompt's und K/V and vision positions independent of the batch.
"""

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("diffusers")

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.base import TextCondition  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit as cosmos3_dit  # noqa: E402
from openwam.model.video_backbone.cosmos3 import text as cosmos3_text  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import Cosmos3OmniTransformer  # noqa: E402
from tests._cosmos3_shell import edge_scalar_kwargs, unused_io_kwargs  # noqa: E402

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
LAYOUT = OmegaConf.create({"num_hist_latents": 1})


def _rotary(net, start, length):
    """Rotary cos / sin (1, length, head_dim) of und positions start .. start+length-1."""
    pos = cosmos3_text.text_mrope_positions(start + length, float_positions=True)[:, start:]
    return cosmos3_dit.compute_rotary(net, pos.unsqueeze(1), torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("cfg", [MINI, NANO_MINI], ids=["edge", "nano"])
def test_suffix_matches_full_tower(cfg):
    """Cached text + run_und_suffix gives the suffix K/V a single tower pass over [text ; suffix] gives."""
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
        _, ref_kv, _ = cosmos3_dit.run_und_tower(
            net, torch.cat([text, suffix_ids], 1), torch.cat([text_mask, suffix_mask], 1), cos, sin
        )
        # Cached text, then the suffix as embeddings.
        cos_t, sin_t = _rotary(net, 0, 5)
        _, text_kv, text_k = cosmos3_dit.run_und_tower(net, text, text_mask, cos_t, sin_t)
        cos_s, sin_s = _rotary(net, 5, 2)
        suffix_kv = cosmos3_dit.run_und_suffix(
            net, net.embed_tokens(suffix_ids), suffix_mask, text_k, text_kv, text_mask, cos_s, sin_s
        )

    for (rk, rv), (sk, sv) in zip(ref_kv, suffix_kv):
        torch.testing.assert_close(sk, rk[:, 5:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sv, rv[:, 5:], atol=1e-5, rtol=1e-5)


def _backbone(flag):
    """Mini Cosmos3 backbone on CPU fp32 with ``proprio_in_und_prefix=flag``. Returns: Cosmos3EdgeVideoBackbone."""
    torch.manual_seed(0)
    vb = Cosmos3EdgeVideoBackbone(
        net=Cosmos3OmniTransformer(**MINI),
        **unused_io_kwargs(),
        **dict(edge_scalar_kwargs(), proprio_in_und_prefix=flag),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        clip_fps=24.0,
    ).eval()
    vb.set_dtype_device(torch.float32, torch.device("cpu"))
    return vb


def _text(vb, proprio, proprio_mask):
    """TextCondition as the architecture hands it to prepare: 4 text tokens (sample 0 padded) + N proprio tokens.

    Args:
        vb: the mini backbone. proprio: (2, N, 12) proprio tokens, or None for a text-only context.
        proprio_mask: (2, N) bool, or None.

    Returns:
        TextCondition with context (2, 4 + N, 12), context_mask (2, 4 + N) and the und bundle in ``extra``.
    """
    net = vb.dit
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    und_mask = torch.tensor([[True, True, True, False], [True] * 4])
    cos, sin = _rotary(net, 0, 4)
    with torch.no_grad():
        context, und_kv, und_k = cosmos3_dit.run_und_tower(net, ids, und_mask, cos, sin)
    grid = cosmos3_text.patch_grid(3, 2, 2, 1)
    _, vis_pos = cosmos3_text.build_joint_positions(4, grid, modality_margin=15000, fps=24.0)
    extra = {"und_kv": und_kv, "und_mask": und_mask, "vision_positions": vis_pos.unsqueeze(1), "und_k": und_k}
    if proprio is None:
        return TextCondition(context=context, context_mask=und_mask, extra=extra)
    return TextCondition(
        context=torch.cat([context, proprio], 1), context_mask=torch.cat([und_mask, proprio_mask], 1), extra=extra
    )


def _prepare(vb, text):
    """prepare() on fixed B=2 latents (2, 2, 3, 2, 2). Returns: BlockLoopState."""
    torch.manual_seed(1)
    return vb.prepare(
        latents=torch.randn(2, 2, 3, 2, 2),
        timestep=torch.full((2,), 500.0),
        text=text,
        layout=LAYOUT,
        proprio=torch.zeros(2, 3, 4),
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )


def _video(vb, text):
    """Video-only block loop, every gen token visible. Returns: (2, 2, 3, 2, 2) prediction."""
    state = _prepare(vb, text)
    s = state.hidden_states.shape[1]
    for i in range(vb.num_layers):
        full = torch.ones(s, state.prefix_kv_len + s, dtype=torch.bool)
        state = vb.run_block(i, state, full)
    return vb.finalize(state)


def test_video_reads_proprio_through_the_und_prefix():
    vb = _backbone(True)
    mask = torch.tensor([[True] * 3, [False] * 3])  # sample 1 has no proprio
    proprio = torch.randn(2, 3, MINI["hidden_size"], requires_grad=True)

    state = _prepare(vb, _text(vb, proprio, mask))
    # Prefix = 4 text + 3 proprio keys; the mask carries both.
    assert state.prefix_kv_len == 7
    assert state.prefix_kv_mask.tolist() == [[True, True, True, False, True, True, True], [True] * 4 + [False] * 3]

    out = _video(vb, _text(vb, proprio, mask))
    other = _video(vb, _text(vb, proprio.detach() + 1.0, mask))
    # Sample 0 reads its proprio; sample 1 (mask False) does not.
    assert (out[0] - other[0]).abs().max() > 1e-4
    assert torch.equal(out[1], other[1])
    # Gradient flows from the video output back into the proprio tokens of sample 0 only.
    out[:, :, 1:].sum().backward()
    assert proprio.grad[0].abs().sum() > 0
    assert proprio.grad[1].abs().sum() == 0


def test_flag_on_without_tokens_raises():
    vb = _backbone(True)
    with pytest.raises(ValueError, match="proprio_in_und_prefix"):
        _prepare(vb, _text(vb, None, None))


def test_flag_off_keeps_the_text_prefix():
    """Off: the proprio tokens stay in the context (the ActionDiT reads them), the und prefix is the text alone."""
    vb = _backbone(False)
    proprio = torch.randn(2, 3, MINI["hidden_size"])
    state = _prepare(vb, _text(vb, proprio, torch.ones(2, 3, dtype=torch.bool)))
    assert state.prefix_kv_len == 4
    # The video output does not depend on the proprio tokens.
    ones = torch.ones(2, 3, dtype=torch.bool)
    assert torch.equal(_video(vb, _text(vb, proprio, ones)), _video(vb, _text(vb, proprio + 1.0, ones)))


@pytest.mark.parametrize("pad", [None, 16], ids=["batch_max", "fixed"])
def test_fixed_pad_makes_a_prompt_independent_of_its_batch(pad, monkeypatch):
    """A prompt alone vs batched with a longer one: with a fixed pad its und K/V and the vision positions are
    identical; the batch-max pad shifts the positions (the vision t-offset is the padded und length + margin)."""
    vb = _backbone(False)
    vb._text_pad_length = pad
    vb._prompt_templates = False
    vb._tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 0})()
    # Tokenization is not under test: each prompt's ids are given ("short" = 3 tokens, "long" = 6).
    ids = {"short": [1, 2, 3], "long": [4, 5, 6, 7, 8, 9]}
    monkeypatch.setattr(cosmos3_text, "tokenize_prompt", lambda tok, text, **kw: ids[text])
    alone = vb.encode_text(["short"], grid=(3, 2, 2), num_frames=9)
    batched = vb.encode_text(["short", "long"], grid=(3, 2, 2), num_frames=9)
    same_pos = torch.equal(alone.extra["vision_positions"], batched.extra["vision_positions"])
    if pad is None:
        assert not same_pos
        return
    assert same_pos
    assert alone.context.shape[1] == batched.context.shape[1] == 16
    for (k1, v1), (k2, v2) in zip(alone.extra["und_kv"], batched.extra["und_kv"]):
        torch.testing.assert_close(k1[0], k2[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v1[0], v2[0], atol=1e-6, rtol=1e-6)
