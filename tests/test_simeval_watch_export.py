"""scripts/modal/simeval_watch.py export + prune on local folders (no Modal calls): a finished checkpoint is copied
into the run dir (eval_step_<N>[_<tag>]/) and its rollouts folder is then deleted from the results root."""

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


@pytest.mark.parametrize("tag", ["", "fix"])
def test_export_then_prune_deletes_the_checkpoint_folder(tmp_path, tag):
    sw = _module()
    name = "0122_run" + (f"_{tag}" if tag else "")
    results, run_dir = tmp_path / "results", tmp_path / "runs" / "0122_run"
    run_dir.mkdir(parents=True)
    rollouts = _results(results, name, 93310)

    assert sw.export_checkpoints(name, run_dir, results, tag) == ["93310/split=id"]
    out = run_dir / sw.eval_dir_name(93310, tag)
    assert out.name == ("eval_step_93310_fix" if tag else "eval_step_93310")
    assert sorted(p.name for p in (out / "00001_sandwich" / "seed_0001").iterdir()) == ["result.json", "video.mp4"]
    agg = json.loads((out / "result.json").read_text())
    assert agg["sets"]["split=id"]["summary"]["success_rate"] == 1.0 and len(agg["sets"]["split=id"]["rollouts"]) == 2

    assert sw.prune_exported(name, run_dir, results, tag) == [out]
    assert not rollouts.exists()                                               # the whole folder is gone
    assert (results / "watch" / name / "step093310" / "DONE_split=id").exists()  # what the watch resumes from
    assert (out / "00001_sandwich" / "seed_0000" / "video.mp4").exists()        # the run dir copy stays
    assert sw.export_checkpoints(name, run_dir, results, tag) == []            # exported once
