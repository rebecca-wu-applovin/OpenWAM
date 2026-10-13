"""Shared PyAV video-frame decoding helper.

Hosts ``decode_video_frames`` — the seek-to-keyframe PyAV decode path used by
every LeRobot v3 reader (the ``LeRobotV3Reader`` family). Kept as a leaf utility
module so the shared base reader and the concrete readers depend on it instead
of reaching into a sibling reader.

This module is dependency-light (PIL + PyAV only, no project imports) so any
reader can import it without triggering heavy module side-effects or import cycles.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# Silence libav stderr spam ("Unknown OBU type 0", per-frame hwaccel probe
# warnings). Bypasses Python logging — must use av.logging directly.
try:
    import av as _av

    _av.logging.set_level(_av.logging.FATAL)
except Exception:
    pass


_seek_fallback_count = 0
_seek_fallback_log_threshold = 20  # log first N events verbosely, then every 100th


def _warn_seek_fallback(video_path: str, min_idx: int, max_idx: int, reason: str) -> None:
    """Record + log a seek-fast-path fallback to seek(0) sequential decode.

    Why: seek-to-keyframe is a perf optimization; a fallback means we paid
    decode cost from frame 0 (slow). If this fires often it's a real signal
    (bad pts metadata, broken keyframe interval) — make it visible.
    """
    global _seek_fallback_count
    _seek_fallback_count += 1
    n = _seek_fallback_count
    if n <= _seek_fallback_log_threshold or n % 100 == 0:
        logger.warning(
            "video_io seek-fallback #%d: %s frames=[%d..%d] reason=%s",
            n,
            video_path,
            min_idx,
            max_idx,
            reason,
        )


# One decoder thread per open file. The codec default (auto) spins up a pool sized for every core on each
# open (~0.3-0.5 s on a 208-core box) and oversubscribes the CPU; DataLoader workers already decode in parallel.
_DECODER_THREADS = 1

# Per-thread swscale context, reused across frames: PyAV's frame.to_image() builds a fresh one per frame
# (~3 ms init on 640x480, more than the conversion). Same conversion parameters, so identical pixels.
_tls = threading.local()
_inherited_reformatters = []  # created by a parent before fork; kept alive, never freed (see _reformatter)


def _reformatter():
    """Return this thread's PyAV VideoReformatter, recreated after a fork.

    A reformatter made in the parent (e.g. vis-sample decoding before the DataLoader forks) owns swscale slice
    threads that do not exist in the child, so using it there blocks forever, and freeing it joins those
    threads (also blocks). The child therefore makes its own and keeps the inherited one referenced.
    Returns: av.video.reformatter.VideoReformatter.
    """
    r = getattr(_tls, "reformatter", None)
    if r is None or getattr(_tls, "pid", None) != os.getpid():
        from av.video.reformatter import VideoReformatter

        if r is not None:
            _inherited_reformatters.append(r)
        r = _tls.reformatter = VideoReformatter()
        _tls.pid = os.getpid()
    return r


def _to_image(frame) -> Image.Image:
    """Bit-identical, cheaper ``frame.to_image()``: shared swscale context + one copy into PIL.

    frame: av.VideoFrame. Returns: (H, W) RGB PIL image at the frame's native size.
    """
    return Image.fromarray(_reformatter().reformat(frame, format="rgb24").to_ndarray(), "RGB")


# Per file (immutable while training): (keyframe gap, B-frame seek delay), both in frames.
_keyframe_probe_cache: Dict[str, Tuple[int, int]] = {}


def _keyframe_probe(container, stream, video_path: str, pts_per_frame: float) -> Tuple[int, int]:
    """Probe keyframe spacing and B-frame delay from the first few packet headers (no decoding), cached per file.

    The mp4 demuxer seeks on dts, so with B-frames (keyframe dts = pts - delay) a seek to t can land on the
    keyframe *after* t; seeking to t - delay lands on the last keyframe <= t.
    container, stream: open PyAV container and its video stream (callers seek afterwards).
    video_path: cache key. pts_per_frame: stream pts units per frame.
    Returns: (gap, delay): largest gap between the first <=4 keyframes (1 << 30 if fewer than 2 found),
        and the largest keyframe pts - dts (0 without B-frames).
    """
    probe = _keyframe_probe_cache.get(video_path)
    if probe is None:
        keys: List[int] = []
        delay = 0
        for i, packet in enumerate(container.demux(stream)):
            if packet.pts is not None and packet.is_keyframe:
                keys.append(int(round(packet.pts / pts_per_frame)))
                if packet.dts is not None:
                    delay = max(delay, int(math.ceil((packet.pts - packet.dts) / pts_per_frame - 1e-6)))
            if len(keys) >= 4 or i >= 512:
                break
        gap = max(b - a for a, b in zip(keys, keys[1:])) if len(keys) >= 2 else 1 << 30
        probe = _keyframe_probe_cache[video_path] = (gap, delay)
    return probe


def _decode_targets(container, stream, video_path: str, frame_indices: List[int]) -> Optional[Dict[int, Image.Image]]:
    """Decode only what the targets need, seeking to the keyframe before each target when that is cheaper.

    For each target (ascending) keep decoding forward if it is within one keyframe gap of the last decoded
    frame, else seek. With the Sharpa videos (keyframe every 2 frames) a 9-frame window spread over 33
    frames decodes ~9-18 frames instead of 33; with long gaps this degrades to one seek + forward decode.
    Some streams start output one frame after the keyframe a seek lands on, so a missed target retries
    1-3 gaps earlier.
    container, stream: open PyAV container / video stream. video_path: probe cache key.
    frame_indices: wanted absolute frame indices.
    Returns: {index: PIL image (native size)}, or None (no pts metadata, or a target was not found).
    """
    if not (stream.frames and stream.duration and stream.frames > 0):
        return None
    pts_per_frame = stream.duration / stream.frames
    gap, delay = _keyframe_probe(container, stream, video_path, pts_per_frame)
    step = gap if gap < 1 << 30 else 2
    want = set(frame_indices)
    got: Dict[int, Image.Image] = {}
    frames = None  # decode generator positioned after frame `pos`
    pos = -1
    for t in sorted(want):
        if t in got:
            continue
        for back in range(4):
            if back or frames is None or t <= pos or t - pos > gap:
                seek_frame = max(0, t - back * step - delay)
                container.seek(int(seek_frame * pts_per_frame), stream=stream, backward=True, any_frame=False)
                frames = container.decode(stream)
            for frame in frames:
                if frame.pts is None:
                    continue
                pos = int(round(frame.pts / pts_per_frame))
                if pos in want and pos not in got:
                    got[pos] = _to_image(frame)
                if pos >= t:
                    break
            else:
                frames = None  # end of stream
            if t in got:
                break
        else:
            return None
    return got


def decode_video_frames(video_path: str, frame_indices: List[int], height: int, width: int) -> List[Image.Image]:
    """Decode requested frames via PyAV, seeking to the keyframe before each target (see ``_decode_targets``).

    Repacked file-NNN.mp4 can hold many episodes, so decoding sequentially from
    frame zero can dominate worker time.

    Falls back to seek(0) + sequential when PTS rounding misses a target.

    No fallback chain (decord / cv2): mp4s are byte-exact stream copy from
    validated upstream encodes; pyav failures are real bugs we want to see.

    video_path: mp4 path. frame_indices: absolute frame indices (any order, repeats allowed).
    height, width: output size (LANCZOS resize when it differs from the video).
    Returns: list of len(frame_indices) RGB PIL images, (height, width), in frame_indices order.
    """
    if not frame_indices:
        return []
    import av

    target = set(frame_indices)
    min_idx = min(frame_indices)
    max_idx = max(frame_indices)
    container = av.open(video_path, options={"hwaccel": "none"})
    try:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = _DECODER_THREADS
        idx_map: Optional[Dict[int, Image.Image]] = None
        try:
            idx_map = _decode_targets(container, stream, video_path, frame_indices)
            reason = "missed_targets"
        except av.AVError as e:
            reason = f"seek_raised: {type(e).__name__}: {e}"
        if idx_map is None:
            # PTS rounding can drift ±1 or metadata is missing; decode sequentially from frame 0.
            _warn_seek_fallback(video_path, min_idx, max_idx, reason=reason)
            idx_map = {}
            container.seek(0, stream=stream, backward=True, any_frame=False)
            for i, frame in enumerate(container.decode(stream)):
                if i in target:
                    idx_map[i] = _to_image(frame)
                if i >= max_idx:
                    break
    finally:
        container.close()
    missing = set(frame_indices) - idx_map.keys()
    if missing:
        raise RuntimeError(f"missing frames {missing} in {video_path}")
    # PIL returns a copy for a same-size resize; skip it (identical pixels, fresh images either way).
    return [im if im.size == (width, height) else im.resize((width, height), Image.LANCZOS)
            for im in (idx_map[i] for i in frame_indices)]


__all__ = ["decode_video_frames"]
