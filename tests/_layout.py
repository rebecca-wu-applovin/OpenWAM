"""Latent layouts for tests that build architectures / backbones directly (the entry point's ``resolve_config``
derives the real one into ``cfg.common.layout``)."""

from omegaconf import OmegaConf

from openwam.config import latent_layout


def make_layout(h: int = 1, f: int = 8, *, tc: int = 4, encoding: str = "per_frame", num_frames: int = 33):
    """Layout of H history frames (the last one the current frame) and F future frames.

    Args:
        h: int H >= 1. f: int F = tc * k. tc: int VAE temporal compression. encoding: per_frame | clip.
        num_frames: int, the action window (action chunk = num_frames - 1).

    Returns:
        read-only DictConfig shaped like ``cfg.common.layout``.
    """
    layout = OmegaConf.create(latent_layout(list(range(-h + 1, 1)), list(range(1, f + 1)), num_frames, encoding, tc))
    OmegaConf.set_readonly(layout, True)
    return layout
