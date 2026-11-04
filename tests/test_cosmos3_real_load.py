"""GPU real-weights load test for the cosmos3 backbones (predict2.5 real_load parity).

Gated on CUDA + the bundle + diffusers; parametrized over Edge and Nano (each
skips on its own when its bundle is absent). Asserts the released geometry,
runs preprocess_train end-to-end on synthetic PIL frames, drives the
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
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from openwam.model.video_backbone import LoadMode, build_video_backbone

    repo = str(Path(__file__).resolve().parent.parent)
    with initialize_config_dir(config_dir=f"{repo}/configs", version_base=None):
        cfg = compose("train", overrides=[f"model/video_backbone={name}", f"model.video_backbone.model_path={asset}"])
    vb_cfg = cfg.model.video_backbone
    OmegaConf.set_readonly(vb_cfg, True)
    backbone = build_video_backbone(vb_cfg, LoadMode.PRETRAINED, None)
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
    assert vb.dit.lm_head is None


def test_preprocess_and_block_loop_shapes(variant):
    _, vb, (_, layers, _, text_dim) = variant
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(9)]
    from openwam.model.architectures.utils.mask_modes import build_frame_mask
    from tests._layout import make_layout

    layout = make_layout(h=1, f=8)  # the first-frame layout: 9 frames -> 1 + 2 latents
    with torch.no_grad():
        lat, text = vb.preprocess_train([clip], ["pick up the bottle"], layout)
        assert lat.shape[0] == 1 and lat.shape[1] == 48
        assert lat.shape[2] == 1 + (9 - 1) // 4  # 3 latent frames
        assert lat.shape[3] == 96 // 16 and lat.shape[4] == 160 // 16
        assert text.context.shape[-1] == text_dim
        assert len(text.extra["und_kv"]) == layers

        state = vb.prepare(
            latents=lat.to(vb.dtype),
            timestep=torch.tensor([500.0], device=lat.device),
            text=text,
            layout=layout,
            proprio=torch.zeros(1, int(layout.num_proprio_tokens), 48, device=lat.device),  # unused by cosmos3
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
        assert state.prefix_kv_len == text.context.shape[1]
        masks = vb.build_joint_attention_masks(build_frame_mask(layout, "bidirectional", "mutual"), state, 0)
        for i in range(vb.num_layers):
            state = vb.run_block(i, state, masks[i])
        out = vb.finalize(state)
    assert out.shape == lat.shape
    assert torch.isfinite(out.float()).all()


def test_vae_roundtrip(variant):
    vb = variant[1]
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    from tests._layout import make_layout

    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(5)]
    layout = make_layout(h=1, f=4)  # frame t + 4 future frames -> 1 + 1 latents
    with torch.no_grad():
        lat, _ = vb.preprocess_train([clip], ["x"], layout)
        frames = vb.decode_future(lat.to(vb.dtype), layout)
    assert len(frames) == 4  # the 4 future frames (the decoded current frame is dropped)
    assert frames[0].size == (160, 96)
