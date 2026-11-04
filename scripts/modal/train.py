"""OpenWAM training on Modal (workspace axon-foundational-research, env main).

Runs the unchanged ``scripts/train.sh`` inside a Modal container. The local
checkout is mounted at /opt/openwam on every run (code edits ship without an
image rebuild); datasets / weights / outputs come from shared Volumes.

Usage (from the repo root):
    # smoke test: tiny random DiT, real data, 1 GPU, 5 steps, no wandb
    OPENWAM_GPU_COUNT=1 modal run scripts/modal/train.py --debug

    # real run, 8 GPUs (OPENWAM_GPU_TYPE=H100|H200|B200, default H100), detached
    OPENWAM_GPU_TYPE=B200 modal run --detach scripts/modal/train.py \
        --overrides "dataloader.action_mode=fingertips_rel common.action_dim=48 ..."
    # (normally launched via run_modal.sh, which sets these env vars)

See scripts/modal/README.md for the volume layout, resume, and gotchas.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

REMOTE_REPO = "/opt/openwam"
# Inside the container this file is /root/train.py, so only resolve the checkout locally.
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)

# ── Shared Volumes (created by teammates; see README for the layout) ──
data_vol = modal.Volume.from_name("openwam-sharpa-data")  # LeRobot v3 datasets
assets_vol = modal.Volume.from_name("openwam-assets")  # backbone weights
runs_vol = modal.Volume.from_name("openwam-runs")  # outputs (we write under /runs/<RUNS_SUBDIR>)
RUNS_SUBDIR = "yjw"

# Paths the local configs hard-code (/root/projects/...) remapped to the mounts.
# Applied as Hydra overrides *before* user overrides; a user override with the
# same key replaces the default (Hydra rejects duplicate keys).
PATH_OVERRIDES = {
    "dataloader.sources.simteleop.dataset_dir": "/data/simteleop_1004_fleet",
    "dataloader.sources.gpt_rollout.dataset_dir": "/data/gpt_fleet_depth",
    "dataloader.normalization_stats_path": f"/runs/{RUNS_SUBDIR}/norm_stats_modal/yam_sharpa_${{.action_mode}}.npy",
    "assets.video_backbone.model_path": "/assets/video_backbone_ckpt/Wan2.2-TI2V-5B",
}

# ── Image: CUDA 12.8 devel (nvcc for DeepSpeed JIT ops) + the frozen local venv ──
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1", "build-essential", "ninja-build")
    .uv_pip_install(
        requirements=[str(REPO_ROOT / "scripts/modal/requirements-frozen.txt")],
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    # flux3_action_base's video VAE needs NATTEN; same build as the local .venv (no torch 2.7 wheel for 0.21.6).
    .uv_pip_install("natten==0.21.0+torch270cu128", find_links="https://whl.natten.org")
    .env(
        {
            "PYTHONPATH": REMOTE_REPO,  # replaces `pip install -e .`
            "PYTHONUNBUFFERED": "1",
            "NCCL_RAS_ENABLE": "0",  # avoids the NCCL RAS SIGSEGV at H100 shutdown
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
    # Source last so code edits never invalidate the dependency layers.
    .add_local_dir(
        REPO_ROOT,
        REMOTE_REPO,
        ignore=[".venv", ".git", "outputs", "assets", "wandb", "**/__pycache__", "docs", "**/*.mp4", "**/logs"],
    )
)

app = modal.App("openwam-yjw-train", image=image)


def _merge_overrides(user_overrides: list[str]) -> list[str]:
    """Prepend PATH_OVERRIDES unless the user already set the same key.

    user_overrides: list of Hydra CLI tokens, e.g. ["dataloader=yam_sharpa", "a.b=1"].
    Returns: list of Hydra CLI tokens (flags, then path defaults, then user key=value tokens).
    """
    # Hydra/argparse flags (e.g. --config-name=train_debug) must precede the key=value overrides.
    flags = [tok for tok in user_overrides if tok.startswith("-")]
    rest = [tok for tok in user_overrides if not tok.startswith("-")]
    user_keys = {tok.split("=", 1)[0].lstrip("+~") for tok in rest if "=" in tok}
    defaults = [f"{k}={v}" for k, v in PATH_OVERRIDES.items() if k not in user_keys]
    return flags + defaults + rest


def _periodic_commit(stop: threading.Event, every_s: int) -> None:
    """Commit the runs Volume every `every_s` seconds until `stop` is set.

    stop: event set by the main thread when training exits.
    every_s: commit interval in seconds.
    Returns: None. (Bounds checkpoint loss on preemption / timeout.)
    """
    while not stop.wait(every_s):
        try:
            runs_vol.commit()
        except Exception as e:  # a failed periodic commit must not kill training
            print(f"[modal] periodic volume commit failed: {e}", flush=True)


def _hydra_value(tokens: list[str], key: str) -> str | None:
    """Value of the last `key=value` Hydra override in `tokens`.

    tokens: Hydra CLI tokens.
    key: dotted config key, e.g. "training.run_dir_name".
    Returns: the value string, or None if `key` is not overridden.
    """
    vals = [t.split("=", 1)[1] for t in tokens if t.split("=", 1)[0] == key]
    return vals[-1] if vals else None


def _latest_state_step(run_dir: Path) -> int:
    """Highest step with a *complete* resume state in `run_dir` (the trainer's own find_latest_accel_state).

    run_dir: run directory that may hold accel_state_step_<N>/ dirs (runs in the container: repo on PYTHONPATH).
    Returns: N of the newest accel_state_step_<N>/ that has trainer_state.json, else -1.
    """
    from openwam.train.utils.checkpointing import STATE_NAME, find_latest_accel_state

    state_dir = find_latest_accel_state(str(run_dir))
    return -1 if state_dir is None else int(STATE_NAME.match(Path(state_dir).name).group(1))


def _auto_resume(overrides: list[str]) -> tuple[list[str], Path | None]:
    """Turn a (Modal-retried) fixed-name launch into a resume of its own run dir.

    A retry re-runs this call with the SAME overrides, so `training.run_dir_name`
    (<index>_<run_name>, fixed by run_modal.sh) points at the dir the earlier
    attempt created. No name matching across launches is involved.

    overrides: Hydra CLI tokens as sent by the caller.
    Returns: (overrides to actually run, run dir or None if it can't be determined).
    """
    resume = _hydra_value(overrides, "training.resume_ckpt_path")
    if resume:  # manual resume: trainer already reuses that dir
        return overrides, Path(REMOTE_REPO) / resume
    out, name = _hydra_value(overrides, "training.output_path"), _hydra_value(overrides, "training.run_dir_name")
    if not (out and name):  # timestamped-by-trainer run: dir unknown up front, no auto-resume
        return overrides, None
    run_dir = Path(REMOTE_REPO) / out / name
    step = _latest_state_step(run_dir)
    if step >= 0:
        print(f"[modal] auto-resume: {run_dir} has resume state at step {step}", flush=True)
        kept = [t for t in overrides if t.split("=", 1)[0] not in ("training.output_path", "training.run_dir_name")]
        return kept + [f"training.resume_ckpt_path={out}/{name}"], run_dir
    if any(run_dir.glob("checkpoint_step_*")):
        # Weights but no resume state = a finished run (finalize drops accel states) or one saved
        # without save_full_states_for_resume. Never touch it; the caller refuses to run.
        raise FileExistsError(f"{run_dir} holds weights but no resume state; refusing to restart over it")
    if (run_dir / "config.yaml").exists():
        # Earlier attempt died before its first resumable save: keep it for inspection, start fresh.
        aside = run_dir.with_name(f"{name}.failed_{time.strftime('%m_%d_%H_%M_%S')}")
        run_dir.rename(aside)
        print(f"[modal] auto-resume: no resume state yet; moved previous attempt to {aside}", flush=True)
    return overrides, run_dir


LOCAL_DATA = Path("/local_data")  # container-local disk (ephemeral_disk), wiped with the container


def _copy_tree_parallel(src: Path, dst: Path, workers: int = 32) -> int:
    """Copy every file under `src` to `dst` with a thread pool (Volume reads are latency-bound).

    src: source directory on the Volume.
    dst: destination directory on local disk (created).
    workers: number of concurrent file copies.
    Returns: int, total bytes copied.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    files = [f for f in src.rglob("*") if f.is_file()]
    for f in files:  # create the directory tree first so workers only copy
        (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)

    def _one(f: Path) -> int:
        shutil.copyfile(f, dst / f.relative_to(src))
        return f.stat().st_size

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(_one, files))


def _stage_data(overrides: list[str]) -> list[str]:
    """Copy every `*.dataset_dir=/data/...` source to local disk and point the overrides at the copy.

    Volume reads were the suspected cause of ~3.5x lower samples/s than the local box. Norm
    stats record their source paths, so staged runs use a separate norm_stats_local/ dir
    (rebuilt once per action mode, same values since the data is identical).

    overrides: Hydra CLI tokens (path defaults already merged).
    Returns: overrides with dataset_dir values under /data/ rewritten to /local_data/.
    """
    out = []
    for tok in overrides:
        key, _, val = tok.partition("=")
        if key.endswith(".dataset_dir") and val.startswith("/data/"):
            src, dst = Path(val), LOCAL_DATA / Path(val).relative_to("/data")
            t0 = time.time()
            nbytes = _copy_tree_parallel(src, dst)
            dt = time.time() - t0
            print(f"[modal] staged {src} -> {dst}: {nbytes / 1e9:.1f} GB in {dt / 60:.1f} min "
                  f"({nbytes / 1e9 / max(dt, 1e-6):.2f} GB/s)", flush=True)
            tok = f"{key}={dst}"
        elif key == "dataloader.normalization_stats_path":
            tok = tok.replace("/norm_stats_modal/", "/norm_stats_local/")
        out.append(tok)
    return out


def _kill_training(proc: subprocess.Popen) -> None:
    """SIGKILL the launched process group and any surviving scripts/train.py workers.

    proc: the Popen of the train.sh / train_debug.sh launch (own session, see start_new_session).
    Returns: None, once `proc` has been reaped.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # group already gone
    # torchrun workers may sit in their own sessions, so killpg alone can miss them.
    subprocess.run(["pkill", "-9", "-f", "scripts/train.py"], check=False)
    proc.wait()


def _run_training(
    cmd: list[str], overrides: list[str], test_kill_after_first_state: bool = False, stage_data: bool = False
) -> dict:
    """Run `cmd + overrides` in the repo, outputs/ on the runs Volume, resuming its own run dir on retry.

    cmd: argv prefix to execute from REMOTE_REPO (train.sh / train_debug.sh invocation).
    overrides: Hydra CLI tokens (path defaults already merged).
    test_kill_after_first_state: test hook; kill training once right after the first resume
        state is written (marker file prevents a second kill) to exercise retry + auto-resume.
    stage_data: copy the /data/ datasets to container-local disk before training (see _stage_data).
    Returns: {"ok": bool, "msg": str}. Raises (-> Modal retry) only if this attempt made
        progress, so a deterministic crash is not retried 10 times.
    """
    # Relative `outputs/...` paths in train.yaml (run dirs, checkpoints, wandb) land on the Volume.
    out_dir = Path(f"/runs/{RUNS_SUBDIR}/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    link = Path(REMOTE_REPO) / "outputs"
    if not link.exists():
        link.symlink_to(out_dir)
    runs_vol.reload()  # see everything earlier attempts committed

    try:
        overrides, run_dir = _auto_resume(overrides)
    except FileExistsError as e:
        print(f"[modal] FAILED: {e}", flush=True)
        return {"ok": False, "msg": str(e)}  # returned, not raised: no Modal retry
    env = dict(os.environ)
    if run_dir is not None:
        # One wandb run per run dir: retries / resumes append to it instead of creating a new run.
        env["WANDB_RUN_ID"] = run_dir.name
    start_step = _latest_state_step(run_dir) if run_dir else -1
    if stage_data:
        overrides = _stage_data(overrides)

    # Record which GPUs this container actually got (H100 vs B200) in the run log.
    subprocess.run(["nvidia-smi", "-L"], check=False)
    full = [*cmd, *overrides]
    print(f"[modal] {shlex.join(full)}", flush=True)
    stop = threading.Event()
    threading.Thread(target=_periodic_commit, args=(stop, 600), daemon=True).start()
    proc = None
    try:
        proc = subprocess.Popen(full, cwd=REMOTE_REPO, env=env, start_new_session=True)
        marker = run_dir / ".test_killed_once" if run_dir else None
        while proc.poll() is None:
            time.sleep(5)
            if test_kill_after_first_state and marker and not marker.exists() and _latest_state_step(run_dir) > start_step:
                marker.touch()
                _kill_training(proc)
                print("[modal] TEST: killed training after first resume state", flush=True)
        rc = proc.returncode
        if rc == 0 and run_dir is not None:
            # Training-finished signal for the sim-eval watcher (scripts/modal/simeval_watch.py: --done-marker).
            (run_dir / "TRAINING_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))
    finally:
        if proc is not None:
            _kill_training(proc)  # no orphaned rank may keep writing the run dir after we return / retry
        stop.set()
        runs_vol.commit()  # final commit, also on failure, so partial checkpoints/logs survive

    if rc == 0:
        return {"ok": True, "msg": "training finished"}
    end_step = _latest_state_step(run_dir) if run_dir else -1
    if end_step > start_step:
        # Progress was saved: raise so Modal retries; the retry auto-resumes from end_step.
        raise RuntimeError(f"training exited {rc} after saving resume state at step {end_step}; retrying")
    msg = f"training exited {rc} with no new resume state (start {start_step}, end {end_step}); not retrying"
    print(f"[modal] FAILED: {msg}", flush=True)
    return {"ok": False, "msg": msg}


VOLUMES = {"/data": data_vol, "/assets": assets_vol, "/runs": runs_vol}
COMMON = dict(
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb-yjw")],  # own key; teammate's wandb-secret can't write 1008_wam_training
    timeout=24 * 3600,  # Modal's per-attempt maximum; each retry gets a fresh 24h
    # Timeout / preemption / crash-after-progress -> rerun with the same args -> _auto_resume.
    retries=modal.Retries(max_retries=10, initial_delay=10.0, backoff_coefficient=1.0),
)


# GPU spec is fixed when the function is defined, so it comes from env vars read at import time
# (run_modal.sh sets them). The Modal UI then shows the GPUs actually requested. Inside the
# container the decorator values are not used, so the defaults there don't matter.
GPU_TYPES = ("H100", "H200", "B200")  # H200: same sm_90 as H100, 141 GB; B200: torch 2.7.1+cu128 ships sm_100 kernels
GPU_TYPE = os.environ.get("OPENWAM_GPU_TYPE", "H100")
GPU_COUNT = int(os.environ.get("OPENWAM_GPU_COUNT", "8"))
# Video decode in the dataloader is CPU-bound (64 cores: ~13.5 samples/s vs ~47 on the 208-core dev
# box), but Modal caps a container at 64 cores. Default 8 cores / 64 GB per GPU; override via env.
CPU_COUNT = int(os.environ.get("OPENWAM_CPU", str(min(64, 8 * GPU_COUNT))))
MEMORY_GB = int(os.environ.get("OPENWAM_MEMORY_GB", str(64 * GPU_COUNT)))
if modal.is_local() and (GPU_TYPE not in GPU_TYPES or GPU_COUNT not in (1, 8)):
    raise SystemExit(f"OPENWAM_GPU_TYPE must be one of {GPU_TYPES} and OPENWAM_GPU_COUNT 1 or 8, got {GPU_TYPE}:{GPU_COUNT}")


# ephemeral_disk (MiB): Modal accepts 524288..29300736 MiB.
@app.function(
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    cpu=CPU_COUNT,
    memory=MEMORY_GB * 1024,
    ephemeral_disk=512 * 1024,  # Modal minimum (524288 MiB); fits the ~137 GB staged datasets
    **COMMON,
)
def train(overrides: list[str], debug: bool = False, test_auto_resume: bool = False, stage_data: bool = True) -> dict:
    """Run training on the OPENWAM_GPU_TYPE x OPENWAM_GPU_COUNT GPUs this function was defined with.

    overrides: Hydra CLI tokens appended after the path defaults.
    debug: True -> train_debug.sh / configs/train_debug.yaml (tiny model, 1 process, no wandb).
    test_auto_resume: kill once after the first resume state (see _run_training).
    stage_data: copy datasets to local disk first; ignored (off) for debug runs.
    Returns: {"ok": bool, "msg": str} from _run_training.
    """
    if debug:
        cmd = ["env", "PYTHON=python", "bash", "scripts/train_debug.sh"]  # train_debug.sh defaults to .venv/bin/python
    else:
        cmd = ["bash", "scripts/train.sh"]  # torchrun on all visible GPUs (train.sh auto-detects the count)
    return _run_training(
        cmd, _merge_overrides(overrides), test_kill_after_first_state=test_auto_resume, stage_data=stage_data and not debug
    )


@app.local_entrypoint()
def main(overrides: str = "", debug: bool = False, test_auto_resume: bool = False) -> None:
    """CLI entry: `[OPENWAM_GPU_TYPE=B200] [OPENWAM_GPU_COUNT=1] modal run scripts/modal/train.py [--debug] [--overrides "..."]`.

    overrides: space-separated Hydra overrides (shell-quoted), e.g. "a.b=1 c=d".
    debug: run the tiny-model smoke test (use with OPENWAM_GPU_COUNT=1).
    test_auto_resume: test hook, kill once after the first resume state.
    Returns: None; exits non-zero if training failed without a retry.
    """
    stage = os.environ.get("OPENWAM_STAGE_DATA", "1") != "0"  # OPENWAM_STAGE_DATA=0 reads straight from the Volume
    print(f"[modal] requesting {GPU_TYPE}:{GPU_COUNT}, {CPU_COUNT} CPU, {MEMORY_GB} GB RAM, stage data to local disk: {stage and not debug}")
    res = train.remote(shlex.split(overrides), debug, test_auto_resume, stage)
    print(f"[modal] result: {res}")
    if not res["ok"]:
        raise SystemExit(1)
