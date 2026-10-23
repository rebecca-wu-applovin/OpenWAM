"""history_encoding=clip latent-count rule (no weights needed); the encode itself is checked on GPU in
outputs/investigations/25_history_clip (independent VAE reference + train/deploy parity)."""

import pytest

from openwam.model.video_backbone.base import VideoBackbone
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v


def _wan(mode):
    """A Wan22Ti2v shell with only what num_history_latents reads (4x temporal VAE)."""
    vb = Wan22Ti2v.__new__(Wan22Ti2v)
    vb._time_division_factor = 4
    vb.__dict__["_history_encoding"] = mode  # bypass nn.Module.__setattr__ (no __init__ ran)
    return vb


def test_wan_history_latents():
    assert _wan("frames").num_history_latents(6) == 6  # one latent per frame
    assert _wan("clip").num_history_latents(6) == 3  # 5 past = 1+4 -> 2 latents, + frame t
    assert _wan("clip").num_history_latents(2) == 2  # 1 past frame -> 1 latent, + frame t
    with pytest.raises(ValueError):
        _wan("clip").num_history_latents(5)  # 4 past frames are not 1+4k


def test_only_wan_accepts_clip():
    with pytest.raises(ValueError):
        Wan22Ti2v.history_encoding.fset(_wan("frames"), "panel")
    with pytest.raises(NotImplementedError):
        VideoBackbone.history_encoding.fset(object(), "clip")  # every non-Wan backbone keeps frames
    VideoBackbone.history_encoding.fset(object(), "frames")
