"""GPU real-weights load test for the cosmos3 backbones (predict2.5 real_load parity).

Gated on CUDA + the bundle + diffusers; parametrized over Edge and Nano (each
skips on its own when its bundle is absent). Asserts the released geometry,
runs preprocess_input_for_train end-to-end on synthetic PIL frames, drives the
full block loop shape-conservingly, and round-trips the VAE.
"""

from pathlib import Path

import pytest
import torch

from tests._assets import asset_path

pytestmark = pytest.mark.gpu

# registry name -> (asset key, expected (dim, num_layers, num_heads, text_dim))
VARIANTS = {
    "cosmos3_edge": ("OPENWAM_COSMOS3_EDGE", (2048, 28, 16, 2048)),
    "cosmos3_nano": ("OPENWAM_COSMOS3_NANO", (4096, 36, 32, 4096)),
}


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def variant(request):
    """Yields (name, backbone, expected geometry); builds one variant at a time on GPU."""
    name = request.param
    asset_key, geometry = VARIANTS[name]
    asset = Path(asset_path(asset_key))
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not asset.exists():
        pytest.skip(f"{name} bundle missing at {asset}.")
    pytest.importorskip("diffusers")
    from openwam.model.video_backbone import build_video_backbone

    cfg = {"model": {"video_backbone": {"name": name, "model_path": str(asset)}}}
    backbone = build_video_backbone(name, cfg)
    backbone.set_dtype_device(torch.bfloat16, torch.device("cuda"))
    yield name, backbone, geometry
    del backbone  # free the GPU before the next variant builds
    torch.cuda.empty_cache()


def test_released_geometry(variant):
    _, vb, (dim, layers, heads, text_dim) = variant
    assert vb.dim == dim
    assert vb.num_layers == layers
    assert vb.num_heads == heads
    assert vb.head_dim == 128
    assert vb.text_dim == text_dim
    assert vb.temporal_compression == 4
    assert vb.causal_temporal is True
    assert getattr(vb.dit, "lm_head", None) is None


def test_preprocess_and_block_loop_shapes(variant):
    _, vb, (_, layers, _, text_dim) = variant
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(9)]
    with torch.no_grad():
        inputs = vb.preprocess_input_for_train(frames=[clip], text=["pick up the bottle"])
        lat = inputs["input_latents"]
        assert lat.shape[0] == 1 and lat.shape[1] == 48
        assert lat.shape[2] == 1 + (9 - 1) // 4  # 3 latent frames
        assert lat.shape[3] == 96 // 16 and lat.shape[4] == 160 // 16
        assert inputs["context"].shape[-1] == text_dim
        assert len(inputs["und_kv"]) == layers
        assert inputs["num_clean_prefix_frames"] == 1

        state = vb.prepare(
            latents=lat.to(vb.dtype),
            timestep=torch.tensor([500.0], device=lat.device),
            context=inputs["context"],
            und_mask=inputs["und_mask"],
            und_kv=inputs["und_kv"],
            vision_positions=inputs["vision_positions"],
            num_clean_prefix_frames=1,
        )
        assert state.prefix_kv_len == inputs["context"].shape[1]
        for i in range(vb.num_layers):
            state = vb.run_block(i, state)
        out = vb.finalize(state)
    assert out.shape == lat.shape
    assert torch.isfinite(out.float()).all()


def test_vae_roundtrip(variant):
    vb = variant[1]
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(5)]
    with torch.no_grad():
        inputs = vb.preprocess_input_for_train(frames=[clip], text=["x"])
        frames = vb.decode_video(inputs["input_latents"].to(vb.dtype))
    assert len(frames) == 5
    assert frames[0].size == (160, 96)
