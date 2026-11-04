"""``LeRobotV3Reader.sample``: every failed attempt logs a warning (dataset id, idx, attempt, repr of the error), the
next window is returned, and the last attempt's error is raised."""

import logging

import numpy as np
import pytest

from openwam.dataloader.bases import lerobot_v3_reader as reader_mod


class _Reader(reader_mod.LeRobotV3Reader):
    """A reader without data: sample ``idx`` fails when ``idx`` is in ``bad``, else returns ``{"idx": idx}``."""

    DATASET_NAME = "fake"

    def __init__(self, bad):
        self._dataset_id = "ds0"
        self._bad = set(bad)

    def __len__(self):
        return 10

    def _getitem_impl(self, idx, rng):
        if idx in self._bad:
            raise ValueError(f"broken window {idx}")
        return {"idx": idx}


def test_each_failure_warns_and_the_next_window_is_returned(caplog):
    """Two bad windows -> two warnings carrying the error message; the third window is the sample."""
    caplog.set_level(logging.WARNING, logger=reader_mod.__name__)
    out = _Reader(bad={3, 4}).sample(3, np.random.default_rng(0))
    assert out == {"idx": 5}
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2
    assert "fake(ds0): sample idx=3 failed on attempt 1/" in warnings[0]
    assert "ValueError('broken window 4')" in warnings[1]


def test_last_attempt_raises(monkeypatch, caplog):
    """After _GETITEM_MAX_RETRIES failures the error propagates (and every attempt was logged)."""
    monkeypatch.setattr(reader_mod, "_GETITEM_MAX_RETRIES", 3)
    caplog.set_level(logging.WARNING, logger=reader_mod.__name__)
    with pytest.raises(ValueError, match="broken window 2"):
        _Reader(bad=range(10)).sample(0, np.random.default_rng(0))
    assert len(caplog.records) == 3
