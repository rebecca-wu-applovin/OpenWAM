"""scripts/modal/train.py code shipping on local folders (no Modal calls): which checkout files a run ships, and a
failed first attempt moved aside by _auto_resume leaves the launch's code/ with the fresh attempt."""

import importlib.util
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")

REPO = Path(__file__).resolve().parents[1]


def _module():
    """Import scripts/modal/train.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("modal_train", REPO / "scripts" / "modal" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_code_files_ship_code_and_configs_only():
    files = {str(f) for f in _module()._code_files(REPO)}
    # what a container needs to train: the package, configs, launch scripts, pyproject
    for need in ("openwam/__init__.py", "configs/train.yaml", "scripts/train.py", "scripts/train.sh", "run_modal.sh",
                 "base.sh", "pyproject.toml", "benchmarks/simeval/server/serve.py"):
        assert need in files, need
    # never tests / docs / outputs / caches
    assert not [f for f in files if f.startswith(("tests/", "docs/", "outputs/")) or "__pycache__" in f]


def test_failed_attempt_moved_aside_keeps_the_code(tmp_path, monkeypatch):
    mt = _module()
    monkeypatch.setattr(mt, "REMOTE_REPO", str(tmp_path))
    run_dir = tmp_path / "outputs" / "g" / "0001_run"
    (run_dir / "code" / "openwam").mkdir(parents=True)
    (run_dir / "code" / "openwam" / "__init__.py").write_text("")
    (run_dir / "config.yaml").write_text("a: 1\n")   # died before its first resumable save

    overrides = ["training.output_path=outputs/g", "training.run_dir_name=0001_run"]
    kept, got = mt._auto_resume(overrides)
    assert kept == overrides and got == run_dir
    assert (run_dir / "code" / "openwam" / "__init__.py").exists() and not (run_dir / "config.yaml").exists()
    (aside,) = tmp_path.glob("outputs/g/0001_run.failed_*")
    assert (aside / "config.yaml").exists() and not (aside / "code").exists()
