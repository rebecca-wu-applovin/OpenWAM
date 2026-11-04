"""Smoke tests for the video backbone module.

Verifies that the video backbone can be imported and basic model
instantiation works without requiring GPU or model weights.
"""

import torch


def test_import_wan_model():
    """WanModel (video DiT) should be importable."""
    from openwam.model.video_backbone.wan.dit import WanModel

    assert WanModel is not None


def test_wan_model_has_dim():
    """WanModel instance should expose .dim attribute for video_dim derivation."""
    from openwam.model.video_backbone.wan.dit import WanModel

    # Tiny model for testing (not real weights)
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
    )
    assert model.dim == 64


def test_import_vae():
    """WanVideoVAE should be importable."""
    from openwam.model.video_backbone.wan.vae import WanVideoVAE

    assert WanVideoVAE is not None


def test_import_text_encoder():
    """WanTextEncoder should be importable."""
    from openwam.model.video_backbone.wan.text_encoder import WanTextEncoder

    assert WanTextEncoder is not None


def _build_tiny_wan_backbone(*, ti2v: bool):
    """Construct a minimal real WanVideoBackbone wrapping a CPU WanModel.

    ``ti2v=True`` flips the ``seperated_timestep`` + ``fuse_vae_embedding_in_latents``
    pair so the TI2V branch of ``prepare()`` fires (independent of
    ``force_per_token_t_mod``). ``ti2v=False`` exercises the non-TI2V path that
    falls through to the ``force_per_token_t_mod`` elif.
    """
    from openwam.model.video_backbone.wan.backbone import WanBackbone
    from openwam.model.video_backbone.wan.dit import WanModel

    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        seperated_timestep=ti2v,
        fuse_vae_embedding_in_latents=ti2v,
    )
    backbone = WanBackbone(
        dit=model,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        tokenizer_dir="",
        shift_video=5.0,
        text_max_length=512,
    )
    return backbone, model


def _time_mod(model, latents, timestep, **flags):
    """Run ``build_time_modulation`` with every flag off unless given. Returns: the (B, L, 6, dim) / (B, 6, dim) t_mod."""
    from openwam.model.video_backbone.wan.dit_forward import build_time_modulation

    kw = dict(
        fuse_vae_embedding_in_latents=False,
        force_per_token_t_mod=False,
        num_clean_prefix_frames=0,
        zero_clean_prefix_t_mod=False,
        has_first_frame_latents=False,
    )
    kw.update(flags)
    return build_time_modulation(model, timestep, latents, patch_size=(1, 2, 2), **kw)[1]


def test_force_per_token_t_mod_broadcast_shape():
    """`force_per_token_t_mod=True` on a non-TI2V DiT must produce 4D t_mod with
    every token sharing the broadcasted ``time_embedding(timestep)`` value."""
    _, model = _build_tiny_wan_backbone(ti2v=False)
    B, F, H, W = 1, 2, 4, 4
    latents = torch.randn(B, model.in_dim, F, H, W)
    timestep = torch.tensor([500.0])

    t_mod = _time_mod(model, latents, timestep, force_per_token_t_mod=True)
    assert t_mod.dim() == 4, "force_per_token_t_mod must yield a 4D t_mod"

    # Every token's row in t_mod should equal the broadcasted single-timestep
    # projection — proves the broadcast path didn't accidentally vary by token.
    # ``time_projection`` is a Linear over an (B, L, dim) tensor whose L rows
    # are identical inputs, so the outputs are mathematically equal; we still
    # need a small atol because batched matmul (CPU MKL in particular) is not
    # guaranteed to use the same reduction order across rows (~3e-8 in float32).
    first = t_mod[:, 0]
    assert torch.allclose(t_mod, first.unsqueeze(1).expand_as(t_mod), atol=1e-6, rtol=1e-6)


def test_zero_clean_prefix_t_mod_overwrites_first_frame():
    """`zero_clean_prefix_t_mod=True` + a clean ref must overwrite the first
    frame's t_mod with the ``time_embedding(timestep=0)`` projection, leaving
    later frames at the sampled timestep — mirrors TI2V's first-frame zeroing."""
    _, model = _build_tiny_wan_backbone(ti2v=False)
    B, F, H, W = 1, 3, 4, 4
    latents = torch.randn(B, model.in_dim, F, H, W)
    timestep = torch.tensor([500.0])

    t_mod = _time_mod(
        model, latents, timestep, force_per_token_t_mod=True, zero_clean_prefix_t_mod=True, has_first_frame_latents=True
    )
    assert t_mod.dim() == 4

    # ``tokens_per_frame`` reflects the after-patchify token layout
    # (H//patch_h * W//patch_w with patch=(1,2,2) → 2*2=4 tokens/frame).
    tokens_per_frame = (H * W) // 4

    # First-frame tokens must all share the same t_mod row (the t=0 projection),
    # different from the sampled-t row used by later frames. atol/rtol cover
    # the same ~3e-8 batched-matmul roundoff documented above.
    first_frame_rows = t_mod[:, :tokens_per_frame]
    later_frame_rows = t_mod[:, tokens_per_frame:]
    assert torch.allclose(first_frame_rows, first_frame_rows[:, :1].expand_as(first_frame_rows), atol=1e-6, rtol=1e-6)
    assert torch.allclose(later_frame_rows, later_frame_rows[:, :1].expand_as(later_frame_rows), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first_frame_rows[:, 0], later_frame_rows[:, 0]), (
        "first-frame t_mod row should differ from the sampled-timestep row"
    )

    # Sanity: the first-frame row must match the manually computed t=0 projection.
    from openwam.model.video_backbone.wan.dit import sinusoidal_embedding_1d

    zero_ts = torch.zeros_like(timestep)
    t_zero = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, zero_ts).to(latents.dtype))
    expected = model.time_projection(t_zero).unflatten(1, (6, model.dim))  # (B, 6, dim)
    assert torch.allclose(first_frame_rows[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_zero_clean_prefix_t_mod_inactive_without_trigger():
    """Without a clean reference AND ``num_clean_prefix_frames=0``, the
    ``zero_clean_prefix_t_mod=True`` flag is a no-op — every token retains the
    sampled-timestep projection."""
    _, model = _build_tiny_wan_backbone(ti2v=False)
    latents = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([500.0])

    t_mod = _time_mod(model, latents, timestep, force_per_token_t_mod=True, zero_clean_prefix_t_mod=True)
    assert t_mod.dim() == 4
    first = t_mod[:, 0]
    # Same batched-matmul roundoff caveat as
    # ``test_force_per_token_t_mod_broadcast_shape``.
    assert torch.allclose(t_mod, first.unsqueeze(1).expand_as(t_mod), atol=1e-6, rtol=1e-6)


def test_ti2v_branch_ignores_zero_clean_prefix_t_mod_kwarg():
    """TI2V's ``seperated_timestep + fuse_vae_embedding_in_latents`` branch
    fires before the ``force_per_token_t_mod`` elif and handles first-frame
    zeroing on its own. Passing ``zero_clean_prefix_t_mod=True`` to a TI2V
    backbone must be a structurally inert no-op."""
    _, model = _build_tiny_wan_backbone(ti2v=True)
    latents = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([500.0])

    a = _time_mod(model, latents, timestep, fuse_vae_embedding_in_latents=True, num_clean_prefix_frames=1)
    b = _time_mod(
        model,
        latents,
        timestep,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
    )
    assert a.dim() == 4 and b.dim() == 4
    assert torch.allclose(a, b), "TI2V branch must be unaffected by the broadcast-path kwargs"


def test_wan_prepare_takes_text_condition_and_layout():
    """``WanBackbone.prepare`` reads context / context_mask from the TextCondition and gives the first
    ``layout.num_hist_latents`` latent frames t = 0 (TI2V per-token timestep)."""
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.base import TextCondition

    backbone, model = _build_tiny_wan_backbone(ti2v=True)
    latents = torch.randn(1, 4, 3, 4, 4)
    timestep = torch.tensor([500.0])
    mask = torch.tensor([[True, True, False]])
    text = TextCondition(context=torch.randn(1, 3, 32), context_mask=mask)

    layout = OmegaConf.create({"num_hist_latents": 2, "num_latents": 3})
    state = backbone.prepare(
        latents=latents, timestep=timestep, text=text, layout=layout, proprio=torch.zeros(1, 2, 3),
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    )
    assert torch.equal(state.context_mask, mask)
    # Frames 0-1 (4 tokens each) carry the t = 0 modulation, frame 2 the sampled one.
    expected = _time_mod(model, latents, timestep, fuse_vae_embedding_in_latents=True, num_clean_prefix_frames=2)
    assert torch.equal(state.time_mod, expected)
    assert torch.allclose(state.time_mod[:, 0], state.time_mod[:, 7], atol=1e-6)
    assert not torch.allclose(state.time_mod[:, 0], state.time_mod[:, 8])


def test_license_exists():
    """Apache 2.0 LICENSE file must exist in the extracted Wan license directory."""
    from pathlib import Path

    license_path = (
        Path(__file__).resolve().parents[1] / "openwam" / "model" / "video_backbone" / "wan" / "license" / "LICENSE"
    )
    assert license_path.exists(), f"LICENSE not found at {license_path}"
    content = license_path.read_text()
    assert "Apache License" in content


if __name__ == "__main__":
    test_import_wan_model()
    test_wan_model_has_dim()
    test_import_vae()
    test_import_text_encoder()
    test_license_exists()
    print("All smoke tests passed.")
