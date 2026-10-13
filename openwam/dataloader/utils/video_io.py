"""Shared PyAV video-frame decoding helper.

Hosts ``decode_video_frames`` — the per-target seek-to-keyframe PyAV decode path used by
every LeRobot v3 reader (the ``LeRobotV3Reader`` family). Kept as a leaf utility
module so the shared base reader and the concrete readers depend on it instead
of reaching into a sibling reader.

This module is dependency-light (PIL + PyAV only, no project imports) so any
reader can import it without triggering heavy module side-effects or import cycles.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

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


# Decoder threads per open file. The codec default (0 = auto) starts a thread pool sized for every core on each
# open, which on a 208-core machine costs ~0.3-0.5 s per call and oversubscribes the CPU when many DataLoader
# workers decode at once; the workers already decode in parallel, so one thread each is fastest
# (Sharpa 640x480 AV1, 5-frame window: ~60-130 ms with 1 thread vs ~350-520 ms with auto).
DECODER_THREADS = int(os.environ.get("OPENWAM_VIDEO_DECODE_THREADS", "1"))


def set_decoder_threads(stream) -> None:
    if DECODER_THREADS > 0:
        stream.codec_context.thread_count = DECODER_THREADS


# Keyframe spacing per file, probed once per process (files are immutable while training).
_keyframe_interval_cache: Dict[str, int] = {}


def _keyframe_interval(container, stream, video_path: str, pts_per_frame: float) -> int:
    """Largest gap (in frames) between the first few keyframes; a very large value if fewer than 2 are found.

    Only reads packet headers (no decoding). Leaves the demuxer at the start of the file; callers seek next.
    """
    gap = _keyframe_interval_cache.get(video_path)
    if gap is None:
        keys: List[int] = []
        for i, packet in enumerate(container.demux(stream)):
            if packet.pts is not None and packet.is_keyframe:
                keys.append(int(round(packet.pts / pts_per_frame)))
            if len(keys) >= 4 or i >= 512:
                break
        gap = max(b - a for a, b in zip(keys, keys[1:])) if len(keys) >= 2 else 1 << 30
        _keyframe_interval_cache[video_path] = gap
    return gap


def decode_frames_at(container, stream, video_path: str, frame_indices: List[int], convert) -> Optional[Dict[int, object]]:
    """Decode only what the requested frames need: {index: convert(frame)}, or None if a target was missed.

    A frame between keyframes can only be decoded from the keyframe before it, so for each target (in order) this
    either keeps decoding forward from the last decoded frame, when the target is at most one keyframe interval
    ahead, or seeks to the keyframe before the target. With short keyframe intervals this skips the frames between
    targets (the Sharpa videos have a keyframe every 2 frames, so a 5-frame window spread over 33 frames decodes about
    5-10 frames instead of 33); with long intervals it decodes forward, as a single seek would.
    Returns None (caller falls back to decoding from frame 0) when the stream has no usable pts metadata or a target
    is not found where its pts says it should be.
    """
    set_decoder_threads(stream)
    if not (stream.frames and stream.duration and stream.frames > 0):
        return None
    pts_per_frame = stream.duration / stream.frames
    gap = _keyframe_interval(container, stream, video_path, pts_per_frame)
    step = gap if gap < 1 << 30 else 2
    want = set(frame_indices)
    got: Dict[int, object] = {}
    frames = None  # decode generator positioned after frame `pos`
    pos = -1
    for t in sorted(want):
        if t in got:
            continue
        # Some streams (the HEVC depth videos) start output one frame after the keyframe a seek lands on, so when
        # the target is not reached, seek again 1, 2, 3 keyframe intervals earlier.
        for back in range(4):
            if back or frames is None or t <= pos or t - pos > gap:
                container.seek(int(max(0, t - back * step) * pts_per_frame), stream=stream, backward=True, any_frame=False)
                frames = container.decode(stream)
            for frame in frames:
                if frame.pts is None:
                    continue
                pos = int(round(frame.pts / pts_per_frame))
                if pos in want and pos not in got:
                    got[pos] = convert(frame)
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
    """Decode requested frames via PyAV, seeking to the keyframe before each target (see ``decode_frames_at``).

    Repacked file-NNN.mp4 can hold many episodes, so decoding sequentially from
    frame zero can dominate worker time.

    Falls back to seek(0) + sequential when PTS rounding misses a target.

    No fallback chain (decord / cv2): mp4s are byte-exact stream copy from
    validated upstream encodes; pyav failures are real bugs we want to see.
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
        idx_map: Optional[Dict[int, Image.Image]] = None
        try:
            idx_map = decode_frames_at(container, stream, video_path, frame_indices, lambda f: f.to_image())
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
                    idx_map[i] = frame.to_image()
                if i >= max_idx:
                    break
    finally:
        container.close()
    missing = set(frame_indices) - idx_map.keys()
    if missing:
        raise RuntimeError(f"missing frames {missing} in {video_path}")
    return [idx_map[i].resize((width, height), Image.LANCZOS) for i in frame_indices]


__all__ = ["decode_frames_at", "decode_video_frames", "set_decoder_threads"]
