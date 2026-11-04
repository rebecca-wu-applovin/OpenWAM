"""Tests for checkpoint discovery / resume-position / retention helpers.

Pure filesystem + integer logic — no GPU, no real Accelerator (a tiny fake covers
the save/load_full_state metadata round-trip and the atomic completion marker).
"""

from pathlib import Path

import pytest

from openwam.train import checkpointing
from openwam.train.checkpointing import (
    STATE_NAME,
    WEIGHTS_NAME,
    RunDir,
    compute_resume_position,
    find_latest_accel_state,
    find_latest_weights,
    latest_resume_state,
    list_steps,
    load_full_state,
    prune_checkpoints,
    save_full_state,
    save_normalization_stats,
    verify_resume_normalization_stats,
)


def _make_accel_state(root: Path, step: int, *, marker: bool) -> Path:
    d = root / f"accel_state_step_{step}"
    d.mkdir()
    if marker:
        (d / "trainer_state.json").write_text("{}")
    return d


# --- find_latest_weights ---


def test_find_latest_weights_picks_highest_step(tmp_path):
    for s in (100, 200, 300):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
    assert find_latest_weights(str(tmp_path)).endswith("checkpoint_step_300.safetensors")


def test_find_latest_weights_skips_malformed(tmp_path):
    (tmp_path / "checkpoint_step_50.safetensors").write_text("x")
    (tmp_path / "checkpoint_step_garbage.safetensors").write_text("x")
    assert find_latest_weights(str(tmp_path)).endswith("checkpoint_step_50.safetensors")


def test_find_latest_weights_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_weights(str(tmp_path))


# --- find_latest_accel_state (marker = trainer_state.json) ---


def test_find_latest_accel_state_requires_marker(tmp_path):
    _make_accel_state(tmp_path, 100, marker=True)
    _make_accel_state(tmp_path, 200, marker=False)  # half-written: no completion marker
    # 200 lacks the marker -> skipped; 100 is the latest usable.
    assert find_latest_accel_state(str(tmp_path)).endswith("accel_state_step_100")


def test_find_latest_accel_state_picks_highest_usable(tmp_path):
    _make_accel_state(tmp_path, 100, marker=True)
    _make_accel_state(tmp_path, 200, marker=True)
    assert find_latest_accel_state(str(tmp_path)).endswith("accel_state_step_200")


def test_find_latest_accel_state_none_when_empty(tmp_path):
    assert find_latest_accel_state(str(tmp_path)) is None


# --- compute_resume_position (grad_accum alignment / off-by fix) ---


def test_resume_position_grad_accum_1_is_identity():
    # grad_accum=1: aligned == global_step always (zero regression vs. pre-fix behaviour).
    assert compute_resume_position(25, 10, 1) == (2, 5, 25)


def test_resume_position_floors_skip_and_pulls_back_global_step():
    # gs=10, bpe=100, grad_accum=4: skip 10 -> 8, aligned 10 -> 8 (no re-train, step matched).
    assert compute_resume_position(10, 100, 4) == (0, 8, 8)


def test_resume_position_alignment_across_epoch():
    # gs=16, bpe=10, grad_accum=4: start=1, skip 6 -> 4, aligned = 1*10 + 4 = 14.
    assert compute_resume_position(16, 10, 4) == (1, 4, 14)


def test_resume_position_already_aligned_unchanged():
    # gs=12, bpe=100, grad_accum=4: skip 12 already a multiple of 4 -> unchanged.
    assert compute_resume_position(12, 100, 4) == (0, 12, 12)


# --- list_steps: the one checkpoint-name parser ---


def test_list_steps_parses_sorts_and_skips_malformed(tmp_path):
    """Numeric sort (not lexicographic), exact names only; a missing dir has no entries."""
    for name in ("checkpoint_step_900.safetensors", "checkpoint_step_1000.safetensors", "checkpoint_step_x.safetensors",
                 "checkpoint_step_5.safetensors.tmp", "accel_state_step_7"):
        (tmp_path / name).write_text("x")
    assert [step for step, _ in list_steps(str(tmp_path), WEIGHTS_NAME)] == [900, 1000]
    assert list_steps(str(tmp_path), STATE_NAME) == [(7, str(tmp_path / "accel_state_step_7"))]
    assert list_steps(str(tmp_path / "missing"), WEIGHTS_NAME) == []


def test_latest_resume_state_without_a_usable_state_raises(tmp_path):
    _make_accel_state(tmp_path, 100, marker=False)
    with pytest.raises(FileNotFoundError, match="save_full_states_for_resume"):
        latest_resume_state(str(tmp_path))


# --- prune_checkpoints: the finished run (keep_states=0) ---


def test_prune_finished_run_keeps_recent_weights_and_removes_all_state(tmp_path):
    for s in (100, 200, 300, 400):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
        _make_accel_state(tmp_path, s, marker=True)
    prune_checkpoints(str(tmp_path), keep_weights=3, keep_states=0)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == [
        "checkpoint_step_200.safetensors",
        "checkpoint_step_300.safetensors",
        "checkpoint_step_400.safetensors",
    ]


# --- prune_checkpoints: per-step save, weights + accel_state in lockstep ---


def test_prune_in_lockstep(tmp_path):
    for s in (100, 200, 300):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
        _make_accel_state(tmp_path, s, marker=True)
    prune_checkpoints(str(tmp_path), keep_weights=2, keep_states=2)
    weights = sorted(p.name for p in tmp_path.glob("checkpoint_step_*"))
    states = sorted(p.name for p in tmp_path.glob("accel_state_step_*"))
    assert weights == ["checkpoint_step_200.safetensors", "checkpoint_step_300.safetensors"]
    assert states == ["accel_state_step_200", "accel_state_step_300"]


def test_prune_orders_steps_numerically(tmp_path):
    """Step 1000 is newer than step 900 (a string sort would drop it)."""
    for s in (900, 1000):
        (tmp_path / f"checkpoint_step_{s}.safetensors").write_text("x")
    prune_checkpoints(str(tmp_path), keep_weights=1, keep_states=1)
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint_step_1000.safetensors"]


# --- RunDir: per-step save + finish ---


class _SaveRecorder:
    """Stands in for save_weights / save_full_state: writes the file / dir they would, records the call."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(checkpointing, "save_weights", self.weights)
        monkeypatch.setattr(checkpointing, "save_full_state", self.state)

    def weights(self, accelerator, architecture, output_path, global_step, *, final):
        """Same args as save_weights; writes an empty weights file."""
        self.calls.append(("weights", global_step, final))
        Path(output_path, f"checkpoint_step_{global_step}.safetensors").write_text("x")

    def state(self, accelerator, output_path, global_step, opt_step, epoch):
        """Same args as save_full_state; writes a complete (marked) state dir."""
        self.calls.append(("state", global_step, opt_step, epoch))
        _make_accel_state(Path(output_path), global_step, marker=True)


def test_run_dir_saves_on_the_save_steps_boundary_and_prunes_in_lockstep(tmp_path, monkeypatch):
    rec = _SaveRecorder(monkeypatch)
    run_dir = RunDir(str(tmp_path), save_steps=2, keep_last_k=2, save_full_states=True)
    for step in range(1, 8):
        run_dir.save_step(_FakeAccelerator(), None, step, step, 0)
    assert rec.calls == [(kind, s, *rest) for s in (2, 4, 6) for kind, *rest in (("weights", False), ("state", s, 0))]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "accel_state_step_4",
        "accel_state_step_6",
        "checkpoint_step_4.safetensors",
        "checkpoint_step_6.safetensors",
    ]
    # Finish: final weights, every resume state dropped, k weights kept.
    run_dir.finish(_FakeAccelerator(), None, 7)
    assert rec.calls[-1] == ("weights", 7, True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_step_6.safetensors", "checkpoint_step_7.safetensors"]


def test_run_dir_without_full_states_writes_weights_only(tmp_path, monkeypatch):
    rec = _SaveRecorder(monkeypatch)
    RunDir(str(tmp_path), save_steps=1, keep_last_k=1, save_full_states=False).save_step(_FakeAccelerator(), None, 3, 3, 0)
    assert rec.calls == [("weights", 3, False)]


def test_run_dir_with_saving_off_writes_nothing(tmp_path, monkeypatch):
    """save_steps null: no periodic save and no final weights."""
    rec = _SaveRecorder(monkeypatch)
    run_dir = RunDir(str(tmp_path), save_steps=None, keep_last_k=1, save_full_states=True)
    run_dir.save_step(_FakeAccelerator(), None, 5, 5, 0)
    run_dir.finish(_FakeAccelerator(), None, 5)
    assert rec.calls == [] and list(tmp_path.iterdir()) == []


# --- save/load_full_state metadata round-trip + atomic completion marker ---


class _FakeAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def save_state(self, state_dir):
        import os

        os.makedirs(state_dir, exist_ok=True)  # stand-in for the sharded state write

    def load_state(self, state_dir):
        pass


def test_full_state_meta_round_trip(tmp_path):
    acc = _FakeAccelerator()
    save_full_state(acc, str(tmp_path), global_step=42, opt_step=10, epoch=3)
    state_dir = tmp_path / "accel_state_step_42"
    assert (state_dir / "trainer_state.json").is_file()  # atomic completion marker present
    assert load_full_state(acc, str(state_dir)) == {"global_step": 42, "opt_step": 10, "epoch": 3}


def test_load_full_state_without_trainer_state_raises(tmp_path):
    """A state dir without its completion marker is not resumable: no step-0 default."""
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        load_full_state(_FakeAccelerator(), str(tmp_path))


# --- save / verify_resume_normalization_stats ---


def _transform(v: float = 0.0) -> dict:
    """v: offset of every vector. Returns: one stats block with all six (3,) float32 vectors."""
    import numpy as np

    return {k: np.full(3, v, dtype=np.float32) for k in ("mean", "std", "min", "max", "q01", "q99")}


def _stats_pair(tmp_path, data_payload, ckpt_payload):
    """Write the dataset's stats file and the run dir's copy. Returns: (run_dir, dataset stats path) as str."""
    import numpy as np

    src = tmp_path / "dataset_stats.npy"
    run = tmp_path / "run"
    run.mkdir()
    np.save(src, data_payload)
    if ckpt_payload is not None:
        np.save(run / "normalization_stats.npy", ckpt_payload)
    return str(run), str(src)


def test_save_normalization_stats_copies_the_file(tmp_path):
    """The run dir gets the dataset's stats file byte for byte."""
    run, src = _stats_pair(tmp_path, {"fingertips_rel": _transform()}, None)
    save_normalization_stats(run, src)
    assert (Path(run) / "normalization_stats.npy").read_bytes() == Path(src).read_bytes()


def test_verify_resume_normalization_stats_missing_checkpoint_artifact(tmp_path):
    run, src = _stats_pair(tmp_path, {"fingertips_rel": _transform()}, None)
    with pytest.raises(FileNotFoundError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(run, src)


def test_verify_resume_normalization_stats_mismatch(tmp_path):
    run, src = _stats_pair(tmp_path, {"fingertips_rel": _transform(0.0)}, {"fingertips_rel": _transform(1.0)})
    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(run, src)


def test_verify_resume_normalization_stats_match(tmp_path):
    payload = {"fingertips_rel": _transform(), "fingertips_rel_state": _transform(2.0)}
    run, src = _stats_pair(tmp_path, payload, payload)
    verify_resume_normalization_stats(run, src)


def test_verify_resume_normalization_stats_rejects_allclose_only_match(tmp_path):
    import numpy as np

    ckpt = {"fingertips_rel": _transform(1.0)}
    # np.allclose accepts this delta at values near 1, but strict resume must not.
    ckpt["fingertips_rel"]["min"] = ckpt["fingertips_rel"]["min"] + np.float32(1e-6)
    run, src = _stats_pair(tmp_path, {"fingertips_rel": _transform(1.0)}, ckpt)
    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(run, src)


def test_verify_resume_normalization_stats_rejects_any_extra_or_missing_field(tmp_path):
    """No legacy-payload tolerance: an old run whose copy differs in any field resumes nowhere (it adapts to us)."""
    full = {"fingertips_rel": {**_transform(), "yam_sharpa_meta": "x"}}
    run, src = _stats_pair(tmp_path, full, {"fingertips_rel": _transform()})
    with pytest.raises(ValueError, match="finetune_ckpt_path"):
        verify_resume_normalization_stats(run, src)


def _resume_trainer(tmp_path, monkeypatch, normalize_mode):
    """A trainer shell with a resumable run dir. Returns: (trainer, run_dir, recorded verify calls)."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    import openwam.train.trainer as trainer_module
    from openwam.train.run_mode import RunMode

    run_dir = tmp_path / "run"
    state_dir = run_dir / "accel_state_step_12"
    state_dir.mkdir(parents=True)
    (state_dir / "trainer_state.json").write_text("{}")

    calls = []
    monkeypatch.setattr(
        trainer_module,
        "verify_resume_normalization_stats",
        lambda output_dir, stats_path: calls.append((output_dir, stats_path)),
    )
    trainer = object.__new__(trainer_module.OpenWAMTrainer)
    trainer.run_mode, trainer.ckpt_path = RunMode.RESUME, str(run_dir)
    trainer.cfg = OmegaConf.create(
        {
            "training": {"save_steps": 5, "keep_last_k_ckpts": 2, "save_full_states_for_resume": True},
            "dataloader": {"normalize_mode": normalize_mode},
        }
    )
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.dataset = SimpleNamespace(normalization_stats_path="/stats/yam_sharpa_fingertips_rel.npy")
    return trainer, run_dir, calls


def test_open_run_dir_verifies_stats_before_reusing_resume_run(tmp_path, monkeypatch):
    trainer, run_dir, calls = _resume_trainer(tmp_path, monkeypatch, "min-max")

    opened, resume_state_dir = trainer.open_run_dir()

    assert opened.path == str(run_dir)
    assert (opened.save_steps, opened.keep_last_k, opened.save_full_states) == (5, 2, True)
    assert resume_state_dir == str(run_dir / "accel_state_step_12")
    assert calls == [(str(run_dir), "/stats/yam_sharpa_fingertips_rel.npy")]


def test_open_run_dir_skips_the_stats_check_when_the_config_turns_normalization_off(tmp_path, monkeypatch):
    """normalize_mode null decides it (not a missing dataset attribute)."""
    trainer, run_dir, calls = _resume_trainer(tmp_path, monkeypatch, None)
    trainer.open_run_dir()
    assert calls == []


@pytest.mark.parametrize("meta, err", [({"global_step": 0, "opt_step": 0, "epoch": 0}, ValueError), ({}, KeyError)])
def test_resume_needs_a_positive_global_step(tmp_path, monkeypatch, meta, err):
    """A resumed state at step 0 (or without a step) raises instead of silently restarting from step 0."""
    from types import SimpleNamespace

    import openwam.train.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "load_full_state", lambda accelerator, state_dir: meta)
    trainer = object.__new__(trainer_module.OpenWAMTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    with pytest.raises(err):
        trainer.load_resume_state(str(tmp_path), batches_per_epoch=10, grad_accum=1)


@pytest.mark.parametrize(
    "saved_step, bpe, grad_accum, want",
    [(25, 10, 1, (25, 25, 2, 5)), (10, 100, 4, (8, 2, 0, 8)), (16, 10, 4, (14, 3, 1, 4))],
)
def test_load_resume_state_maps_the_saved_step(tmp_path, monkeypatch, saved_step, bpe, grad_accum, want):
    """(global_step, opt_step, start_epoch, skip) from a saved step: aligned to grad_accum, opt_step derived from it."""
    from types import SimpleNamespace

    import openwam.train.trainer as trainer_module

    meta = {"global_step": saved_step, "opt_step": 999, "epoch": 999}  # only global_step is read back
    monkeypatch.setattr(trainer_module, "load_full_state", lambda accelerator, state_dir: meta)
    trainer = object.__new__(trainer_module.OpenWAMTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    assert trainer.load_resume_state(str(tmp_path), batches_per_epoch=bpe, grad_accum=grad_accum) == want
