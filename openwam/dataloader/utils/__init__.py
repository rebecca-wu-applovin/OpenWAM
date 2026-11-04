"""Shared dataloader helpers: small, stateless, dependency-light modules (no heavy import side effects).

Members:
  - ``lerobotv3``         — LeRobot v3 per-bucket helpers (info.json, episodes parquet, offsets, splits)
  - ``video_io``          — mp4 frame decode + resize to the camera tile
  - ``normalization``     — stats file, forward / inverse normalization
  - ``unify_action``      — raw action dims <-> unified slots (scatter / gather + dim mask)
  - ``yam_sharpa_repr``   — the yam_sharpa action / proprio representations and parquet columns
  - ``sparse_history``    — history row selection (training + deploy)
  - ``rotation``          — rot6d <-> rotation matrix
  - ``stats_computation`` — offline pooled-stats scan (CLI)
"""
