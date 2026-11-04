"""Multi-view image composition: the camera tiles stacked into one canvas.

Dataset-neutral; the yam_sharpa reader and the deploy engine (``InferenceEngine.observation``) both use it, so deploy
builds the training canvas. Kept in a transform module to avoid deploy → dataset coupling.
"""

from typing import Tuple, Union

from PIL import Image


def resize_camera_tile(image: Image.Image, height: int, width: int) -> Image.Image:
    """Resize one camera frame to its tile, exactly as the vstack training decode does.

    Direct LANCZOS stretch, no crop (``video_io.decode_video_frames``); the deploy server calls it too, so
    deploy sees the training pixels.

    Args:
        image: PIL.Image, the native camera frame.
        height, width: int, tile size in pixels.

    Returns:
        PIL.Image of size ``(width, height)``; the input itself when it already has that size.
    """
    # PIL returns a copy for a same-size resize; skip it (identical pixels).
    return image if image.size == (width, height) else image.resize((width, height), Image.LANCZOS)


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Direct BILINEAR resize without aspect-ratio preservation.

    Matches FastWAM's per-camera ``torchvision.transforms.functional.resize(...,
    BILINEAR, antialias=True)`` step in its multi-view composition.
    """
    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_vstack_layout(
    frames_by_camera: dict,
    camera_layout: list,
    out_h: int,
    out_w: int,
    return_missing_mask: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    """Stack N cameras top-to-bottom, each resized to ``(out_h / N, out_w)``.

    Each camera is BILINEAR-resized directly to its tile (e.g. 480x640 → 240x320
    keeps the 4:3 aspect for two views on a 480x320 canvas).

    Args:
        frames_by_camera: dict ``{camera_name: PIL.Image}`` (any size); a missing key → black tile.
        camera_layout: list of N camera names, ``camera_layout[0]`` is the top tile.
        out_h, out_w: int, canvas size in pixels; ``out_h`` must be divisible by N.
        return_missing_mask: bool, also return an ``L`` mask (255 = black-filled tile),
            so augmentation can tell padding from real black pixels.

    Returns:
        PIL.Image of size ``(out_w, out_h)``, or ``(image, missing_mask)``.
    """
    n = len(camera_layout)
    if n < 1:
        raise ValueError("vstack layout expects at least one camera")
    if out_h % n:
        raise ValueError(f"vstack layout: out_h={out_h} must be divisible by the camera count {n}")
    tile_h = out_h // n

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    missing_mask = Image.new("L", (out_w, out_h), 0) if return_missing_mask else None
    for i, cam in enumerate(camera_layout):
        frame = frames_by_camera.get(cam)
        top = i * tile_h  # y offset of tile i
        if frame is not None:
            canvas.paste(_stretch_resize(frame, tile_h, out_w), (0, top))
        elif missing_mask is not None:
            missing_mask.paste(255, (0, top, out_w, top + tile_h))

    if missing_mask is not None:
        return canvas, missing_mask
    return canvas


__all__ = ["assemble_vstack_layout", "resize_camera_tile"]
