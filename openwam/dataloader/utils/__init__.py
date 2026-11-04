"""Shared dataloader helpers.

This package gathers small, stateless utility code reused across the
dataloader readers (yam_sharpa, ...). It is
deliberately dependency-light so any reader can import it without
triggering heavy module side-effects.

Members:
  - ``utils.lerobotv3``   — LeRobot v3 per-bucket helpers (info.json,
                            episodes parquet, offsets, splits, video
                            sampling validation)
"""
