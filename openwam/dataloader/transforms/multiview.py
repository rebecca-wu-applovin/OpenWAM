"""Multi-view image composition and RoboTwin prompt formatting.

These helpers used to live inside ``robotwin.py``. The image utilities
(``crop_and_resize`` / ``assemble_multiview_layout``) are dataset-neutral and
reused by both the RoboTwin dataloader and the deployment ``policy_server``.
``format_prompt_for_inference`` is RoboTwin's training-time prompt template —
deploy no longer uses it (the server is prompt-agnostic; the RoboTwin eval
client owns its own copy in ``benchmarks/robotwin/prompt_template.py``).
Keeping these in a transform module avoids deploy → dataset hard-coupling.
"""

from typing import Tuple, Union

from PIL import Image

# Single source of truth for the L-shape multiview layout used by RoboTwin
# (and by the deploy server as a fallback when ``cfg.dataloader.camera_layout``
# is missing). The matching yaml key in ``configs/dataloader/robotwin.yaml`` is
# ``camera_layout`` and should mirror this list byte-for-byte.
DEFAULT_MULTIVIEW_CAMERA_LAYOUT = ("head_camera", "left_camera", "right_camera")


def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Center-crop and resize an image to ``(target_width, target_height)``.

    Scales the image so the shorter side matches the target, then center-crops
    to the exact target resolution. Preserves aspect ratio.
    """
    img_w, img_h = image.size
    scale = max(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


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


def compose_history_panel(frames: list) -> Image.Image:
    """Tile 4 past canvases into one canvas of the same size: a 2x2 panel, each at half resolution.

    Sparse-history ``history_layout: panel``: the past frames are one history latent (motion), frame t keeps
    full resolution (space). Row-major, oldest first (top-left = oldest, bottom-right = newest past frame); each
    tile is the whole canvas (all cameras) LANCZOS-resized to half height and width. Shared by the training
    dataset and the deploy engine, so both feed the model the same pixels.

    Args:
        frames: list of 4 PIL.Image canvases of one size (W, H), oldest first; H and W even.

    Returns:
        PIL.Image of size (W, H).
    """
    if len(frames) != 4:
        raise ValueError(f"history panel tiles exactly 4 past frames, got {len(frames)}")
    w, h = frames[0].size
    if w % 2 or h % 2 or any(f.size != (w, h) for f in frames):
        raise ValueError(f"history panel needs 4 canvases of one even size, got {[f.size for f in frames]}")
    panel = Image.new(frames[0].mode, (w, h))
    for i, frame in enumerate(frames):
        # i -> (row, col) = (i // 2, i % 2)
        panel.paste(frame.resize((w // 2, h // 2), Image.LANCZOS), ((i % 2) * (w // 2), (i // 2) * (h // 2)))
    return panel


def _stretch_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Direct BILINEAR resize without aspect-ratio preservation.

    Matches FastWAM's per-camera ``torchvision.transforms.functional.resize(...,
    BILINEAR, antialias=True)`` step in its multi-view composition.
    """
    return image.resize((target_width, target_height), Image.BILINEAR)


def assemble_multiview_layout(
    frames_by_camera: dict,
    camera_layout: list,
    out_h: int,
    out_w: int,
    top_height_ratio: float = 2.0 / 3.0,
    return_missing_mask: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    """3-camera L-shape composition (FastWAM / RoboTwin compatible).

    Layout with the default ratio:
        top    -> (out_h * 2/3, out_w)          full width, 2/3 of height
        bot-L  -> (out_h * 1/3, out_w / 2)      half width, 1/3 of height
        bot-R  -> (out_h * 1/3, out_w / 2)      half width, 1/3 of height

    Each camera is BILINEAR-resized directly to its slot with no aspect-ratio
    preservation (slight horizontal/vertical stretch is accepted), then pasted
    without gaps. At ``out_h=384, out_w=320`` this produces a canvas with
    top=256x320 and each bottom=128x160, matching FastWAM exactly.

    Args:
        frames_by_camera: ``{camera_name: PIL.Image}``. Missing key -> black region.
        camera_layout: ordered list of 3 camera names (top, bot-left, bot-right).
        out_h, out_w: final canvas size in pixels.
        top_height_ratio: fraction of height allocated to the top camera.
        return_missing_mask: Also return an ``L``-mode mask whose white pixels
            identify slots with no source frame. This lets image augmentation
            exclude structural black padding without treating real black pixels
            inside a valid camera image as padding.

    Returns:
        PIL Image of size ``(out_w, out_h)``. If ``return_missing_mask`` is
        true, returns ``(image, missing_mask)`` instead.
    """
    if len(camera_layout) != 3:
        raise ValueError(f"multiview layout expects 3 cameras, got {len(camera_layout)}: {camera_layout}")

    top_h = int(round(out_h * top_height_ratio))
    bottom_h = out_h - top_h
    half_w = out_w // 2
    right_w = out_w - half_w

    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    missing_mask = Image.new("L", (out_w, out_h), 0) if return_missing_mask else None

    top_frame = frames_by_camera.get(camera_layout[0])
    if top_frame is not None:
        canvas.paste(_stretch_resize(top_frame, top_h, out_w), (0, 0))
    elif missing_mask is not None:
        missing_mask.paste(255, (0, 0, out_w, top_h))

    bl_frame = frames_by_camera.get(camera_layout[1])
    if bl_frame is not None:
        canvas.paste(_stretch_resize(bl_frame, bottom_h, half_w), (0, top_h))
    elif missing_mask is not None:
        missing_mask.paste(255, (0, top_h, half_w, out_h))

    br_frame = frames_by_camera.get(camera_layout[2])
    if br_frame is not None:
        canvas.paste(_stretch_resize(br_frame, bottom_h, right_w), (half_w, top_h))
    elif missing_mask is not None:
        missing_mask.paste(255, (half_w, top_h, out_w, out_h))

    if missing_mask is not None:
        return canvas, missing_mask
    return canvas


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
            same contract as :func:`assemble_multiview_layout`.

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


def format_prompt_for_inference(base_prompt: str) -> str:
    """RoboTwin training-time prompt template.

    The RoboTwin dataloader wraps every instruction with this before the model
    sees it. Deploy is prompt-agnostic on the server side, so the RoboTwin
    benchmark client re-implements this byte-for-byte in
    ``benchmarks/robotwin/prompt_template.py`` (pinned by a regression test) to
    keep eval prompts in-distribution with training.
    """
    return "A video recorded from a robot's point of view executing the following instruction: " + base_prompt


__all__ = [
    "DEFAULT_MULTIVIEW_CAMERA_LAYOUT",
    "crop_and_resize",
    "assemble_multiview_layout",
    "assemble_vstack_layout",
    "compose_history_panel",
    "format_prompt_for_inference",
]
