"""scripts/modal/simeval_watch.py export + prune and RunMonitor on local folders (no Modal calls): a finished
checkpoint is copied into the run dir (eval_step_<N>[_<tag>]/, tag "smoke" for a smoke watch) and its rollouts folder
is then deleted from the results root; the monitor ends the watch on TRAINING_DONE or an idle run and logs each
summary to W&B once."""

import importlib.util
import json
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")


def _module():
    """Import simeval_watch.py as a module (it is a script, not a package member)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "modal" / "simeval_watch.py"
    spec = importlib.util.spec_from_file_location("simeval_watch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _results(root: Path, name: str, step: int) -> Path:
    """A results root with one DONE checkpoint of watch run ``name``: 2 rollouts (result.json, rollout.json, video,
    keyframes) and the set summary. Returns: the checkpoint's rollouts folder."""
    tag = f"step{step:06d}"
    rollouts = root / "v0.1-dev" / f"{name}_{tag}"
    for seed in (0, 1):
        d = rollouts / f"00001_sandwich__id__seed_{seed:04d}"
        (d / "keyframes").mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"base_scene_id": "00001_sandwich", "split": "id", "seed": seed,
                                                   "status": "success", "success": True, "steps": 316}))
        (d / "rollout.json").write_text("{}")
        (d / "video.mp4").write_bytes(b"mp4")
        (d / "keyframes" / "t00000.jpg").write_bytes(b"jpg")
    watch = root / "watch" / name / tag
    watch.mkdir(parents=True)
    (watch / "DONE_split=id").touch()
    (watch / "summary_split=id.json").write_text(json.dumps({"step": step, "success_rate": 1.0}))
    return rollouts


@pytest.mark.parametrize("tag", ["", "smoke"])
def test_export_then_prune_deletes_the_checkpoint_folder(tmp_path, tag):
    sw = _module()
    name = "0122_run" + (f"_{tag}" if tag else "")
    results, run_dir = tmp_path / "results", tmp_path / "runs" / "0122_run"
    run_dir.mkdir(parents=True)
    rollouts = _results(results, name, 93310)

    assert sw.export_checkpoints(name, run_dir, results, tag) == ["93310/split=id"]
    out = run_dir / sw.eval_dir_name(93310, tag)
    assert out.name == ("eval_step_93310_smoke" if tag else "eval_step_93310")
    assert sorted(p.name for p in (out / "00001_sandwich" / "seed_0001").iterdir()) == ["result.json", "video.mp4"]
    agg = json.loads((out / "result.json").read_text())
    assert agg["sets"]["split=id"]["summary"]["success_rate"] == 1.0 and len(agg["sets"]["split=id"]["rollouts"]) == 2

    assert sw.prune_exported(name, run_dir, results, tag) == [out]
    assert not rollouts.exists()                                               # the whole folder is gone
    assert (results / "watch" / name / "step093310" / "DONE_split=id").exists()  # what the watch resumes from
    assert (out / "00001_sandwich" / "seed_0000" / "video.mp4").exists()        # the run dir copy stays
    assert sw.export_checkpoints(name, run_dir, results, tag) == []            # exported once


@pytest.mark.parametrize("case", ["marker", "idle", "active"])
def test_monitor_ends_the_watch_when_training_is_over(tmp_path, case):
    """The done file appears after TRAINING_DONE or IDLE_STOP_S without a new checkpoint, not while the run trains."""
    import os
    import time

    sw = _module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = run_dir / "checkpoint_step_1000.safetensors"
    ckpt.write_bytes(b"w")
    if case == "marker":
        (run_dir / sw.DONE_MARKER).touch()
    if case == "idle":
        old = time.time() - sw.IDLE_STOP_S - 60
        os.utime(ckpt, (old, old))
    done = tmp_path / "watch" / "TRAINING_OVER"
    sw.RunMonitor(run_dir, tmp_path / "watch", done, None).check()
    assert done.exists() == (case != "active")


def test_monitor_logs_each_summary_once(tmp_path, monkeypatch):
    """Every summary_<set>.json is logged at its train_step once; the WANDB_<set> marker survives a respawn."""
    import sys
    import types

    sw = _module()
    logged = []

    class Run:
        def define_metric(self, *a, **k):
            pass

        def log(self, d):
            logged.append(d)

        def finish(self):
            pass

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **k: Run()))
    watch = tmp_path / "watch" / "0122_run"
    (watch / "step001000").mkdir(parents=True)
    (watch / "step001000" / "summary_split=id.json").write_text(
        json.dumps({"step": 1000, "success_rate": 0.5, "n": 580, "partial": True}))
    cfg = {"project": "p", "group": "g", "name": "0122_run_simeval"}
    for _ in range(2):  # the second monitor = a respawned coordinator
        sw.RunMonitor(tmp_path, watch, tmp_path / "done", cfg).log_summaries()
    assert logged == [{"train_step": 1000, "sim/split=id/success_rate": 0.5, "sim/split=id/n": 580}]
