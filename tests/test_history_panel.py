"""history_layout: panel — the 4 past canvases tiled into one half-resolution 2x2 canvas (no dataset needed)."""

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.transforms.multiview import compose_history_panel


def _solid(value, size=(224, 384)):
    """A (W, H) = size RGB canvas filled with `value`."""
    return Image.new("RGB", size, (value, value, value))


def test_panel_tiles_oldest_first_row_major():
    """Tile i holds past frame i: top-left oldest, bottom-right newest; output keeps the canvas size."""
    panel = compose_history_panel([_solid(v) for v in (10, 20, 30, 40)])
    assert panel.size == (224, 384)
    px = np.asarray(panel)
    w2, h2 = 112, 192
    for i, value in enumerate((10, 20, 30, 40)):
        r, c = divmod(i, 2)
        assert (px[r * h2:(r + 1) * h2, c * w2:(c + 1) * w2] == value).all()


def test_panel_tile_is_the_half_resolution_frame():
    """Each tile equals the whole canvas LANCZOS-resized to half height and width."""
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (384, 224, 3), dtype=np.uint8)) for _ in range(4)]
    px = np.asarray(compose_history_panel(frames))
    half = np.asarray(frames[3].resize((112, 192), Image.LANCZOS))
    assert (px[192:, 112:] == half).all()


def test_panel_rejects_wrong_inputs():
    with pytest.raises(ValueError):
        compose_history_panel([_solid(0)] * 5)  # exactly 4 past frames
    with pytest.raises(ValueError):
        compose_history_panel([_solid(0)] * 3 + [_solid(0, (226, 384))])  # one size
    with pytest.raises(ValueError):
        compose_history_panel([_solid(0, (223, 384))] * 4)  # even sizes


def test_history_layout_validation():
    from openwam.dataloader.yam_sharpa import _validate_history_layout

    five = np.asarray([-80, -60, -40, -20, 0])
    assert _validate_history_layout("frames", None) == "frames"
    assert _validate_history_layout("panel", five) == "panel"
    with pytest.raises(ValueError):
        _validate_history_layout("grid", five)  # unknown layout
    with pytest.raises(ValueError):
        _validate_history_layout("panel", five[1:])  # 4 rows = 3 past + t: not a 2x2 panel
    with pytest.raises(ValueError):
        _validate_history_layout("panel", None)  # needs the sparse layout


def test_trainer_rejects_model_history_count_mismatch():
    """A panel dataset emits 2 history frames; a model configured for 5 would mask the wrong frames."""
    from omegaconf import OmegaConf

    from openwam.train.openwam_trainer import OpenWAMTrainer

    class _DS:
        sparse_history, num_model_history_frames = True, 2

    class _VB:  # history_encoding=frames: one latent per history frame
        def num_history_latents(self, n):
            return n

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.dataset = type("_Multi", (), {"datasets": [_DS()]})()
    trainer.architecture = type("_Arch", (), {"video_backbone": _VB()})()
    trainer._check_history_frames(OmegaConf.create({"architecture": {"num_history_frames": 2}}))
    with pytest.raises(ValueError, match="clean history latents"):
        trainer._check_history_frames(OmegaConf.create({"architecture": {"num_history_frames": 5}}))
