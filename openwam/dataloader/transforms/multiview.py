"""Multi-view image composition: the camera tiles stacked into one canvas.

Dataset-neutral; the yam_sharpa reader and the deploy engine (``InferenceEngine.observation``) both use it, so deploy
builds the training canvas. Kept in a transform module to avoid deploy → dataset coupling.
"""

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


def assemble_vstack_layout(frames_by_camera: dict, camera_layout: list, out_h: int, out_w: int) -> Image.Image:
    """Stack N camera tiles top-to-bottom into one canvas.

    Every tile is already ``(out_h / N, out_w)``: the training decode and the deploy server both resize each camera
    with :func:`resize_camera_tile` first, so the tiles are pasted as they are.

    Args:
        frames_by_camera: dict ``{camera_name: PIL.Image}`` of tiles (width ``out_w``, height ``out_h / N``); a
            missing key → black tile.
        camera_layout: list of N camera names, ``camera_layout[0]`` is the top tile.
        out_h, out_w: int, canvas size in pixels; ``out_h`` must be divisible by N.

    Returns:
        PIL.Image of size ``(out_w, out_h)``.
    """
    n = len(camera_layout)
    if n < 1:
        raise ValueError("vstack layout expects at least one camera")
    if out_h % n:
        raise ValueError(f"vstack layout: out_h={out_h} must be divisible by the camera count {n}")
    tile_h = out_h // n

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    for i, cam in enumerate(camera_layout):
        frame = frames_by_camera.get(cam)
        if frame is None:
            continue  # black tile
        # A tile of another size would be cropped / leave black rows: resize_camera_tile was skipped.
        if frame.size != (out_w, tile_h):
            raise ValueError(f"vstack layout: tile {cam!r} is {frame.size}, expected {(out_w, tile_h)} (w, h)")
        canvas.paste(frame, (0, i * tile_h))  # tile i starts at y = i * tile_h
    return canvas


__all__ = ["assemble_vstack_layout", "resize_camera_tile"]
