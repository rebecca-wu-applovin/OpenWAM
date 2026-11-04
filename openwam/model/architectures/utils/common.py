"""Architecture-side helpers shared across WAM frameworks: bridge-layer selection."""

from __future__ import annotations


def bridge_layers(num_video_layers: int, bridge_interval: int) -> tuple[int, ...]:
    """Video DiT layers that feed (cross_attn) or pair with (self_attn) the ActionDiT blocks.

    Args:
        num_video_layers: int, the video backbone's layer count (yaml ``geometry.num_layers``).
        bridge_interval: int >= 1, stride over the video layers (1 = every layer).

    Returns: tuple of ascending layer ids ``range(0, num_video_layers, bridge_interval)``; one ActionDiT block each.
    """
    # range() with a non-positive step is empty / raises; reject it with the config key's name instead.
    if bridge_interval < 1:
        raise ValueError(f"model.architecture.bridge_interval must be >= 1, got {bridge_interval}")
    return tuple(range(0, num_video_layers, bridge_interval))


__all__ = ["bridge_layers"]
