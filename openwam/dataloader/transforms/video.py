"""Video color jitter: one set of random factors applied to every frame of a clip (temporal consistency)."""

import numpy as np
from PIL import ImageEnhance


def color_jitter_enabled(color_jitter) -> bool:
    """Whether a yaml ``color_jitter`` value asks for jitter.

    ``None`` / ``False`` / ``{}`` → disabled. A dict may also disable itself
    explicitly with ``enabled: false`` while keeping its strength knobs.
    """
    if not color_jitter:
        return False
    if hasattr(color_jitter, "get") and not color_jitter.get("enabled", True):
        return False
    return True


class VideoColorJitter:
    """Random brightness / contrast / saturation jitter, the same factors for all frames of a clip.

    The readers build it for the train split only. Factors come from the sample's own RNG (the mixture's
    ``(seed, epoch, idx)`` generator), so a sample does not depend on the worker or the global RNG state.

    Args:
        brightness: Max brightness change factor.
        contrast: Max contrast change factor.
        saturation: Max saturation change factor.
        hue: Max hue change (not applied in PIL-only mode).
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, frames: list, rng: np.random.Generator) -> list:
        """Jitter one clip.

        Args:
            frames: list of T PIL images (RGB).
            rng: numpy Generator of the sample; three draws (brightness, contrast, saturation factor).

        Returns:
            list of T jittered PIL images.
        """
        # One factor per enhancement, the same for all frames (temporal consistency).
        b_factor = 1.0 + rng.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + rng.uniform(-self.contrast, self.contrast)
        s_factor = 1.0 + rng.uniform(-self.saturation, self.saturation)

        augmented = []
        for frame in frames:
            frame = ImageEnhance.Brightness(frame).enhance(b_factor)
            frame = ImageEnhance.Contrast(frame).enhance(c_factor)
            augmented.append(ImageEnhance.Color(frame).enhance(s_factor))
        return augmented
