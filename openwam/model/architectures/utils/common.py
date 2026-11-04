"""Architecture-side helpers shared across WAM frameworks.

Bridge-layer resolution.
"""

from __future__ import annotations

from typing import Any, Optional


def resolve_bridge_layers(cfg: Any, *, num_layers: Optional[int] = None) -> tuple:
    """Parse ``bridge_layers`` indices from an architecture config.

    Two input modes:
      - Explicit: ``cfg.bridge_layers`` is a list / tuple / comma-separated str.
      - Interval: ``cfg.bridge_layers`` is None and ``cfg.bridge_interval`` is set;
        the indices are ``range(0, num_layers, bridge_interval)``.

    ``num_layers`` should be passed by the caller (typically the video backbone's
    ``num_layers``). It falls back to ``cfg.num_dit_layers`` for backward
    compatibility.

    The output is always sorted with no duplicates.
    """
    bl_raw = cfg.get("bridge_layers", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_layers", None)

    if bl_raw is None:
        interval_raw = (
            cfg.get("bridge_interval", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_interval", None)
        )
        if interval_raw is None:
            raise ValueError("bridge_layers is null but bridge_interval is not set")

        if num_layers is None:
            num_layers = (
                cfg.get("num_dit_layers", None) if isinstance(cfg, dict) else getattr(cfg, "num_dit_layers", None)
            )
        if num_layers is None:
            raise ValueError("num_layers must be provided (or num_dit_layers set on cfg) when bridge_layers is null")
        num_layers = int(num_layers)

        interval = int(interval_raw)
        assert interval >= 1, f"bridge_interval must be >= 1, got {interval}"
        bl = tuple(range(0, num_layers, interval))
    elif isinstance(bl_raw, str):
        bl = tuple(int(x) for x in bl_raw.split(","))
    elif isinstance(bl_raw, tuple):
        bl = bl_raw
    else:
        bl = tuple(bl_raw)

    bl = tuple(sorted(bl))
    assert len(set(bl)) == len(bl), f"bridge_layers must be unique, got {bl}"
    return bl


__all__ = [
    "resolve_bridge_layers",
]
