"""Video color jitter: one set of random factors applied to every frame of a clip (temporal consistency)."""

import random

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

    The readers build it for the train split only. Factors come from python's ``random`` (seeded per worker).

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

    def apply(self, frames: list) -> list:
        """Jitter one clip.

        Args:
            frames: list of T PIL images (RGB).

        Returns:
            list of T jittered PIL images.
        """
        # Sample random factors (same for all frames)
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        s_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        augmented = []
        for frame in frames:
            frame = ImageEnhance.Brightness(frame).enhance(b_factor)
            frame = ImageEnhance.Contrast(frame).enhance(c_factor)
            augmented.append(ImageEnhance.Color(frame).enhance(s_factor))
        return augmented
