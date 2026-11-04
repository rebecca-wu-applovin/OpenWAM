"""Resume smoke + text-cache ordering (train_debug: tiny random Wan, single GPU, DeepSpeed ZeRO-2).

Run 1 trains one step and saves weights + the resumable accel state. Run 2 resumes from it and trains a second
step. The trainer must build the text cache only after the resumed weights are in place
(prepare -> load_full_state -> build_text_cache): the resumed run starts from an empty (meta) checkpoint shell, so
a cache built before ``load_full_state`` would encode prompts with an uninitialized text encoder. The frozen
encoder is identical in both runs, so every cached prompt embedding must match bitwise.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from tests._assets import asset_path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cache_out: str, overrides: list[str], log: str, *, keep_state: bool) -> None:
    """Launch tests/_train_driver.py in a fresh single-rank process (env as scripts/train_debug.sh sets it).

    Args: cache_out: where the driver dumps the text cache. overrides: Hydra overrides. log: stdout/stderr file.
        keep_state: keep the accel_state dirs a finished run would drop.
    Returns: None. Fails the test (with the log tail) on a non-zero exit.
    """
    env = dict(
        os.environ,
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        LOCAL_WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=os.environ.get("MASTER_PORT", "29533"),
        WANDB_MODE="disabled",
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        sys.executable,
        os.path.join(REPO, "tests", "_train_driver.py"),
        cache_out,
        *(["--keep-resume-state"] if keep_state else []),
        *overrides,
    ]
    with open(log, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        with open(log) as f:
            pytest.fail(f"{' '.join(cmd)} failed:\n" + "".join(f.readlines()[-40:]))


@pytest.mark.gpu
def test_text_cache_after_resume(tmp_path):
    """Resume reaches the right step, and its text cache equals the fresh run's bitwise."""
    sources = {
        "simteleop": asset_path("OPENWAM_YAM_SHARPA_SIMTELEOP"),
        "gpt_rollout": asset_path("OPENWAM_YAM_SHARPA_GPT"),
    }
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not all(os.path.isdir(p) for p in sources.values()):
        pytest.skip(f"yam_sharpa datasets not found: {sources}")
    out = os.environ.get("OPENWAM_RESUME_SMOKE_DIR") or str(tmp_path)
    run_dir = os.path.join(out, "run")
    common = [f"dataloader.sources.{k}.dataset_dir={v}" for k, v in sources.items()] + [
        "dataloader.normalize_mode=null",  # no stats file needed; the cache does not depend on it
        f"training.output_path={out}",
        "training.run_dir_name=run",
        "training.save_steps=1",
        "training.save_full_states_for_resume=true",
        "++model.video_backbone.text_cache=true",
    ]
    _run(
        os.path.join(out, "cache_fresh.pt"),
        [*common, "training.max_steps=1"],
        os.path.join(out, "fresh.log"),
        keep_state=True,
    )
    assert os.path.isdir(os.path.join(run_dir, "accel_state_step_1"))
    _run(
        os.path.join(out, "cache_resume.pt"),
        [*common, "training.max_steps=2", f"training.resume_ckpt_path={run_dir}"],
        os.path.join(out, "resume.log"),
        keep_state=False,
    )

    # Step count: resumed at global_step 1, trained exactly one more step, saved step 2.
    with open(os.path.join(out, "resume.log")) as f:
        log = f.read()
    assert "[resume] resumed at global_step=1" in log
    assert os.path.isfile(os.path.join(run_dir, "checkpoint_step_2.safetensors"))
    assert not os.path.exists(os.path.join(run_dir, "checkpoint_step_3.safetensors"))

    fresh = torch.load(os.path.join(out, "cache_fresh.pt"))
    resumed = torch.load(os.path.join(out, "cache_resume.pt"))
    assert fresh.keys() == resumed.keys() and len(fresh) > 0
    bad = [p for p in fresh if not all(torch.equal(a, b) for a, b in zip(fresh[p], resumed[p], strict=True))]
    assert not bad, f"{len(bad)}/{len(fresh)} cached prompts differ after resume, e.g. {bad[:2]}"
