"""Closed-loop sim eval of every checkpoint of an OpenWAM yam_sharpa training run, on Modal (16 x L40S per scene set).

``sim-policy-eval watch --model openwam`` (the sibling sim-policy-eval checkout) runs in a CPU coordinator: it waits
for each ``checkpoint_step_N.safetensors`` the trainer (scripts/modal/train.py) writes to the runs Volume, starts
OpenWAM's own policy server on it (benchmarks/simeval/serve.py, this checkout), evaluates the scene set (default:
every ID scene, i.e. the 540 the task verifier grades plus 40 ungradable) split over ``--shards`` ``shard_worker``
GPUs (default 16 L40S, each with its own server and ``--workers`` evaluate processes; ``--shards 0``: one container
does everything), and logs the success rate to the training
run's W&B project and group as a run ``<run>_simeval`` (x axis ``train_step``). After ``TRAINING_DONE`` (written by
train.py on success) it evaluates the final set and stops. Its supervisor (sim-policy-eval watch/supervisor.py)
reloads the runs Volume between evaluations, commits the results, and respawns the watch before Modal's 24 h cap.

Everything the watch needs from the run is read ONCE, here, from the run's ``config.yaml``: the action
representation (-> interface ``openwam_<action_mode>_v1``) and the W&B project / group / run name.

    modal deploy scripts/modal/simeval_watch.py                               # after every code change (both repos)
    modal run scripts/modal/simeval_watch.py::smoke --run <group>/<index>_<name> [--step N] [--shards 2]   # 3 scenes
    modal run scripts/modal/simeval_watch.py::launch --run <group>/<index>_<name> [--shards 16] [--intermediate split=id]
        (run_modal.sh calls this for every new run; the watch waits for the run's config.yaml if it is not there yet)
    modal app logs openwam-yjw-simeval

``--run`` is relative to openwam-runs:/yjw/outputs (run_modal.sh's run dir). Results: Volume openwam-yjw-simeval-results,
``watch/<run name>/`` (summary_<set>.tsv/json, per-checkpoint folders with result.json, videos, run report).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP = "openwam-yjw-simeval"
REMOTE_REPO = "/opt/openwam"                 # this checkout (policy server), same path as the training image
SIM_EVAL = "/opt/sim-policy-eval"            # the sibling sim-policy-eval checkout (the benchmark and watcher)
RUNS, ASSETS, SCENES, RESULTS = "/runs", "/assets", "/scenes", "/results"
RUNS_SUBDIR = "yjw"                          # train.py writes runs under /runs/yjw/outputs/<group>/<run>
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
SIM_EVAL_LOCAL = Path(os.environ.get("SIM_EVAL_LOCAL", REPO_ROOT.parent / "sim-policy-eval")) if modal.is_local() \
    else Path(SIM_EVAL)
GPU = os.environ.get("SIMEVAL_GPU", "L40S")
MAX_GPUS = int(os.environ.get("SIMEVAL_MAX_GPUS", "64"))   # shard machines alive at once, over ALL watched runs
VIDEO_PER_BASE = 10                          # a video for every ID scene (10 seeds per base scene)
SETTLE_S = 300                               # mtime age; with intact() it rejects files caught mid-commit (train.py commits every 600 s)
WATCH_BUDGET_S = 22 * 3600                   # ask the watch to stop between evaluations, then respawn
WATCH_HARD_S = 24 * 3600 - 40 * 60           # kill a running evaluation before Modal's 24 h cap does
DONE_MARKER = "TRAINING_DONE"                # train.py, on a successful finish
CONFIG_POLL_S = 60                           # a watch launched with its training run polls for the run's config.yaml
CONFIG_WAIT_S = 3 * 3600                     # ... this long (image build + queueing for GPUs), then gives up

runs_vol = modal.Volume.from_name("openwam-runs")
assets_vol = modal.Volume.from_name("openwam-assets")                 # backbone weights the run's config points at
scenes_vol = modal.Volume.from_name("sim-policy-eval-scenes")         # the benchmark's pinned scenes (warmed)
results_vol = modal.Volume.from_name("openwam-yjw-simeval-results", create_if_missing=True)
VOLUMES = {RUNS: runs_vol.read_only(), ASSETS: assets_vol.read_only(), SCENES: scenes_vol.read_only(),
           RESULTS: results_vol}
# export_eval only: writes eval_step_<N>/ into the run dirs (the trainer never touches those paths)
EXPORT_VOLUMES = {RUNS: runs_vol, RESULTS: results_vol}

# MuJoCo renders headless through EGL on the container's NVIDIA driver (glvnd needs the vendor file).
EGL_VENDOR = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}'

image = (
    # The training image's layers, unchanged (scripts/modal/train.py): the server runs in the training environment.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libgomp1", "build-essential", "ninja-build")
    .uv_pip_install(
        requirements=[str(REPO_ROOT / "scripts/modal/requirements-frozen.txt")],
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .uv_pip_install("natten==0.21.0+torch270cu128", find_links="https://whl.natten.org")
    # sim-policy-eval: EGL for MuJoCo, then its own Python 3.12 venv from the lock file (project code added below).
    .apt_install("libegl1", "libglvnd0", "libgles2")
    .run_commands("mkdir -p /usr/share/glvnd/egl_vendor.d",
                  f"echo '{EGL_VENDOR}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
                  "pip install uv==0.11.16")
    .add_local_file(SIM_EVAL_LOCAL / "pyproject.toml", f"{SIM_EVAL}/pyproject.toml", copy=True)
    .add_local_file(SIM_EVAL_LOCAL / "uv.lock", f"{SIM_EVAL}/uv.lock", copy=True)
    .run_commands(f"cd {SIM_EVAL} && uv sync --frozen --no-install-project --extra wandb --python 3.12")
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "NCCL_RAS_ENABLE": "0",
        "MUJOCO_GL": "egl",
        "SIM_POLICY_EVAL_CACHE": f"{SCENES}/blobs",                  # asset blobs on the scenes Volume
        "SIM_POLICY_EVAL_LOCK_DIR": "/tmp/sim-policy-eval-locks",    # flock on container disk, not the Volume
    })
    # Sources last, so code edits never invalidate the dependency layers. .git: the results record its commit.
    .add_local_dir(SIM_EVAL_LOCAL, SIM_EVAL, ignore=[".venv", "outputs", "**/__pycache__", "**/*.pyc", ".pytest_cache"])
    .add_local_dir(REPO_ROOT, REMOTE_REPO, ignore=[".venv", ".git", "outputs", "assets", "wandb", "**/__pycache__",
                                                   "docs", "**/*.mp4", "**/logs"])
)
app = modal.App(APP, image=image)


def run_settings(run: str) -> dict:
    """What the watch needs from a training run, read once from its config.yaml.

    Args:
        run: run dir relative to /runs/yjw/outputs, e.g. ``1013_action_repr_ablation/00012_unify_abs``.
    Returns:
        dict: run_dir (Path), name (the run dir's name = its W&B run id), interface (``openwam_<action_mode>_v1``),
        wandb_project, wandb_group.
    """
    from omegaconf import OmegaConf

    run_dir = Path(RUNS, RUNS_SUBDIR, "outputs", run)
    cfg_file = run_dir / "config.yaml"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"{cfg_file}: no such run (or the trainer has not written its config yet)")
    cfg = OmegaConf.load(cfg_file)
    if OmegaConf.select(cfg, "dataloader.type") != "yam_sharpa":
        raise ValueError(f"{run}: dataloader.type {OmegaConf.select(cfg, 'dataloader.type')!r}, not yam_sharpa")
    return {"run_dir": run_dir, "name": run_dir.name,
            "interface": f"openwam_{OmegaConf.select(cfg, 'dataloader.action_mode')}_v1",
            "wandb_project": OmegaConf.select(cfg, "project.wandb.project"),
            "wandb_group": OmegaConf.select(cfg, "project.wandb.group")}


def wait_for_config(run: str) -> None:
    """Block until the trainer has written (and committed) the run's config.yaml, so a watch can be launched together
    with its training run (run_modal.sh does) instead of after it.

    Args:
        run: run dir relative to /runs/yjw/outputs.
    Returns: None once the file is visible and parses; raises TimeoutError after CONFIG_WAIT_S.
    """
    import time

    from omegaconf import OmegaConf

    def ready(cfg_file: Path) -> bool:
        """True when the file exists and is valid YAML (a file caught mid-commit reads as NUL bytes)."""
        try:
            return cfg_file.is_file() and OmegaConf.load(cfg_file) is not None
        except Exception:  # noqa: BLE001  (partial file: try again after the next reload)
            return False

    cfg_file = Path(RUNS, RUNS_SUBDIR, "outputs", run, "config.yaml")
    deadline = time.monotonic() + CONFIG_WAIT_S
    while not ready(cfg_file):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{cfg_file}: not readable within {CONFIG_WAIT_S} s (did the training run start?)")
        print(f"[simeval] waiting for {cfg_file} (poll every {CONFIG_POLL_S} s)", flush=True)
        time.sleep(CONFIG_POLL_S)
        runs_vol.reload()   # the mount only shows files committed after the last reload


def watch_command(settings: dict, intermediate: str, final: str, workers: int, extra: list[str]) -> list[str]:
    """The ``sim-policy-eval watch`` argv for one run.

    Args:
        settings: run_settings(); intermediate / final: scene sets (``split=id``, ``none``...); workers: parallel
        evaluate processes; extra: more watch flags (e.g. ``--only-steps 5000 --once``).
    Returns:
        argv list.
    """
    cmd = [f"{SIM_EVAL}/.venv/bin/python", "-m", "sim_policy_eval.cli", "--configs", f"{SIM_EVAL}/configs", "watch",
           "--model", "openwam", "--interface", settings["interface"], "--run-dir", str(settings["run_dir"]),
           "--openwam-root", REMOTE_REPO, "--server-python", sys.executable, "--results", RESULTS,
           "--scenes-dir", f"{SCENES}/scenes", "--settle-s", str(SETTLE_S), "--done-marker", DONE_MARKER,
           "--intermediate", intermediate, "--final", final, "--workers", str(workers), "--judge", "off",
           "--video-per-base", str(VIDEO_PER_BASE)]
    if settings["wandb_project"]:
        cmd += ["--wandb-project", settings["wandb_project"], "--wandb-mode", "online",
                "--wandb-name", f"{settings['name']}_simeval"]
        if settings["wandb_group"]:
            cmd += ["--wandb-group", settings["wandb_group"]]
    return cmd + list(extra)


def _run_watch(run: str, intermediate: str, final: str, workers: int, shards: int, extra: list[str],
               attempt: int) -> int:
    """The watch under its supervisor (volume handshake, budgets, respawn); with ``shards`` > 0 every scene set runs
    on that many ``shard_worker`` GPUs, each with its own policy server, and this container only coordinates.

    Args:
        run: run dir relative to /runs/yjw/outputs; intermediate / final: scene sets; workers: evaluate processes
        per machine; shards: GPU machines per scene set (0: everything on this container's GPU); extra: more watch
        flags; attempt: respawn counter.
    Returns:
        the watch's exit code (0 = finished).
    """
    sys.path.insert(0, f"{SIM_EVAL}/src/sim_policy_eval/watch")   # stdlib only: importable from this python
    from supervisor import SYNC_FAILED, WATCH_STOP, supervise

    runs_vol.reload()
    wait_for_config(run)   # returns at once if the run exists (always true on a respawn)
    settings = run_settings(run)
    per_machine = ["--remote-shards", str(shards), "--shard-workers", str(workers)] if shards else []
    cmd = watch_command(settings, intermediate, final, workers, per_machine + list(extra))
    print(f"[simeval] attempt {attempt}: {' '.join(cmd)}", flush=True)

    def run_shards(payload: dict) -> list[dict]:
        calls = [shard_worker.spawn(spec) for spec in payload["specs"]]   # all shards at once
        outs = []
        for spec, call in zip(payload["specs"], calls):
            try:
                outs.append(call.get())
            except Exception as e:  # noqa: BLE001  (its scenes count as missing: the watch retries the set)
                outs.append({"rc": 1, "error": f"{type(e).__name__}: {e}", "rows": [], "shard": spec["shard"]})
        return outs

    # sim-policy-eval imports from its src; the policy server (started by the watch) adds its own repo paths.
    env = {**os.environ, "PYTHONPATH": f"{SIM_EVAL}/src"}
    rc = supervise(cmd, sync_dir=Path("/tmp/volsync"), reload_runs=runs_vol.reload, commit_results=results_vol.commit,
                   budget_s=WATCH_BUDGET_S, hard_s=WATCH_HARD_S, cwd=Path(SIM_EVAL), env=env,
                   **({"run_shards": run_shards, "on_reports": lambda: on_set_done(settings["name"], run)}
                      if shards else {}))
    export_eval.spawn(settings["name"], run)   # whatever finished last (also in single-container mode)
    if rc == WATCH_STOP:
        (watcher_sharded if shards else watcher).spawn(run, intermediate, final, workers, shards, extra, attempt + 1)
    elif rc == SYNC_FAILED:
        raise RuntimeError("openwam-runs could not be reloaded; the watch was stopped (relaunch it)")
    elif rc != 0:
        raise RuntimeError(f"sim-policy-eval watch exited rc={rc}")
    return rc


def on_set_done(name: str, run: str) -> None:
    """After every finished scene set (all shards committed): rebuild the reports and export into the run dir."""
    report_run.spawn(name)
    export_eval.spawn(name, run)


def export_rollouts(rollouts: Path, out: Path) -> list[dict]:
    """Copy one checkpoint's rollouts into ``out/<scene>/<seed>/{result.json, video.mp4}``.

    Args:
        rollouts: the runner's results folder of one checkpoint (``<benchmark_version>/<checkpoint id>``, holding
            ``<scene id>/result.json``); out: the export folder (``<run_dir>/eval_step_<N>``).
    Returns:
        one row per rollout (scene_id, base_scene_id, split, seed, status, success, verifier fields, steps).
    """
    import json
    import shutil

    rows = []
    for res in sorted(rollouts.glob("*/result.json")):
        row = json.loads(res.read_text())
        prefix, seed = res.parent.name.rsplit("__", 1)          # "<base>__id", "seed_0007"
        # <scene> = the base scene for ID, the scene id without its seed for OOD
        dest = out / (row.get("base_scene_id") if row.get("split") == "id" else prefix) / seed
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res, dest / "result.json")
        if (res.parent / "video.mp4").exists():
            shutil.copyfile(res.parent / "video.mp4", dest / "video.mp4")
        rows.append({k: row.get(k) for k in ("scene_id", "base_scene_id", "split", "seed", "status",
                                             "success", "task_verifier_graded", "task_verifier_stages",
                                             "ungradable", "steps")})
    return rows


def export_checkpoints(name: str, run_dir: Path, results: Path) -> list[str]:
    """Copy every finished (DONE) scene set of a watched run into the run dir, once per set.

    ``<run_dir>/eval_step_<N>/<scene>/<seed>/{result.json, video.mp4}`` per rollout (``<scene>`` = the base scene for
    ID, the scene id without its seed for OOD) and ``<run_dir>/eval_step_<N>/result.json``: per set the summary (the
    W&B numbers) and one row per rollout. ``EXPORTED_<set>`` marks a set as copied.

    Args:
        name: the watch's run name (= the run dir's name); run_dir: the training run dir (writable);
        results: the results root (holds watch/<name>/ and <benchmark_version>/<checkpoint id>/).
    Returns:
        "<step>/<set>" of the sets exported now.
    """
    import json

    done = []
    for step_dir in sorted((results / "watch" / name).glob("step*")):
        step = int(step_dir.name.removeprefix("step"))
        rollouts = next(iter(results.glob(f"*/{name}_{step_dir.name}")), None)   # <benchmark_version>/<checkpoint id>
        for marker in sorted(step_dir.glob("DONE_*")):
            set_name = marker.name.removeprefix("DONE_")
            out = run_dir / f"eval_step_{step}"
            summary_file = step_dir / f"summary_{set_name}.json"
            if (out / f"EXPORTED_{set_name}").exists() or rollouts is None or not summary_file.exists():
                continue
            rows = export_rollouts(rollouts, out)
            # result.json holds every exported set of this checkpoint
            agg_file = out / "result.json"
            agg = json.loads(agg_file.read_text()) if agg_file.exists() else {"run": name, "step": step, "sets": {}}
            agg["sets"][set_name] = {"summary": json.loads(summary_file.read_text()), "rollouts": rows}
            agg_file.write_text(json.dumps(agg, indent=1))
            (out / f"EXPORTED_{set_name}").touch()
            done.append(f"{step}/{set_name}")
    return done


@app.function(cpu=4, memory=16 * 1024, volumes=EXPORT_VOLUMES, timeout=2 * 3600)
def export_eval(name: str, run: str) -> list[str]:
    """export_checkpoints on the Volumes: reload both, copy, commit the runs Volume.

    Args: name: the watch's run name; run: run dir relative to /runs/yjw/outputs. Returns: the sets exported.
    """
    results_vol.reload()
    runs_vol.reload()
    done = export_checkpoints(name, Path(RUNS, RUNS_SUBDIR, "outputs", run), Path(RESULTS))
    if done:
        runs_vol.commit()
    print(f"[export] {run}: {done or 'nothing new'}", flush=True)
    return done


@app.function(gpu=GPU, cpu=24, memory=96 * 1024, volumes=VOLUMES, timeout=24 * 3600,
              secrets=[modal.Secret.from_name("wandb-yjw")])
def watcher(run: str, intermediate: str = "split=id", final: str = "split=id", workers: int = 12, shards: int = 0,
            extra: list[str] | None = None, attempt: int = 1) -> int:
    """One container does everything: checkpoint discovery, the policy server, the rollouts, videos, W&B."""
    return _run_watch(run, intermediate, final, workers, 0, extra or [], attempt)


@app.function(cpu=4, memory=16 * 1024, volumes=VOLUMES, timeout=24 * 3600,
              secrets=[modal.Secret.from_name("wandb-yjw")])
def watcher_sharded(run: str, intermediate: str = "split=id", final: str = "split=id", workers: int = 4,
                    shards: int = 16, extra: list[str] | None = None, attempt: int = 1) -> int:
    """The coordinator of ``--remote-shards`` (no GPU): checkpoint discovery, snapshots, summaries, markers, W&B."""
    return _run_watch(run, intermediate, final, workers, shards, extra or [], attempt)


@app.function(gpu=GPU, cpu=8, memory=64 * 1024, volumes=VOLUMES, timeout=12 * 3600, max_containers=MAX_GPUS)
def shard_worker(spec: dict) -> dict:
    """One shard of a checkpoint x scene set: its own policy server, check-server, the shard's rollouts and videos
    (sim-policy-eval watch/shard_worker.py). Results are committed every 5 min and at the end; the rows go back to
    the coordinator.

    Args:
        spec: the coordinator's shard spec (watch/shard_worker.py docstring).
    Returns:
        {"rc", "rows", "shard", ...} (the shard worker's out.json), or rc 1 with the error.
    """
    import json
    import subprocess
    import threading
    import time

    work = Path("/tmp/shard")
    work.mkdir(exist_ok=True)
    (work / "spec.json").write_text(json.dumps(spec))
    stop = threading.Event()

    def commits() -> None:
        while not stop.wait(300):
            try:
                results_vol.commit()
            except Exception as e:  # noqa: BLE001  (committed again later and at the end)
                print(f"[shard] commit failed: {e}", flush=True)

    threading.Thread(target=commits, daemon=True).start()
    try:
        rc = subprocess.run([f"{SIM_EVAL}/.venv/bin/python", "-m", "sim_policy_eval.watch.shard_worker",
                             str(work / "spec.json"), str(work / "out.json")], cwd=SIM_EVAL, check=False,
                            env={**os.environ, "PYTHONPATH": f"{SIM_EVAL}/src"}).returncode
    finally:
        stop.set()
        for _ in range(5):
            try:
                results_vol.commit()
                break
            except Exception as e:  # noqa: BLE001
                print(f"[shard] final commit failed: {e}; retrying", flush=True)
                time.sleep(15)
    if rc or not (work / "out.json").exists():
        return {"rc": 1, "error": f"shard worker exited rc={rc}", "rows": [], "shard": spec.get("shard")}
    return json.loads((work / "out.json").read_text())


@app.function(cpu=4, memory=16 * 1024, volumes={RESULTS: results_vol, SCENES: scenes_vol.read_only()}, timeout=3600)
def report_run(name: str) -> str:
    """Rebuild a watched run's report (curves, heatmap, best / worst) where every shard's files are visible.

    Args: name: the watch's run name (the run dir's name). Returns: the report folder.
    """
    import subprocess

    results_vol.reload()
    out = f"{RESULTS}/watch/{name}/report"
    for args in (["run-report", f"{RESULTS}/watch/{name}", "--out", out], ["report", "--results", RESULTS]):
        subprocess.run([f"{SIM_EVAL}/.venv/bin/python", "-m", "sim_policy_eval.cli", "--configs",
                        f"{SIM_EVAL}/configs", *args], cwd=SIM_EVAL, check=False,
                       env={**os.environ, "PYTHONPATH": f"{SIM_EVAL}/src"})
    results_vol.commit()
    return out


def _deployed(shards: int):
    """The deployed watcher (spawned calls outlive this terminal; a `modal run` app would kill them on exit)."""
    try:
        return modal.Function.from_name(APP, "watcher_sharded" if shards else "watcher")
    except modal.exception.NotFoundError:
        raise SystemExit(f"{APP} is not deployed: modal deploy scripts/modal/simeval_watch.py") from None


@app.local_entrypoint()
def launch(run: str, intermediate: str = "split=id", final: str = "split=id", shards: int = 16, workers: int = 4,
           extra: str = ""):
    """Start watching a run on the deployed app; returns at once (follow: modal app logs openwam-yjw-simeval).
    The run may not exist yet: the watch polls for its config.yaml (up to CONFIG_WAIT_S), so it can be launched
    together with the training run. ``shards`` GPUs per scene set (0: one container), ``workers`` evaluate processes
    per GPU."""
    call = _deployed(shards).spawn(run, intermediate, final, workers, shards, extra.split() if extra else None)
    print(f"[simeval] watching {run} on {shards or 1} x {GPU}: call {call.object_id}")


@app.local_entrypoint()
def smoke(run: str, step: int = 0, scenes: str = "split=id,per_base=1,limit=3", shards: int = 0, workers: int = 3):
    """One checkpoint (``step``, default: whatever exists, oldest first) on a few scenes, then exit."""
    # W&B: a separate project, so smoke points never land next to the training run's curves.
    extra = ["--once", "--video-per-base", "1", "--wandb-project", "openwam-simeval-smoke"] + (
        ["--only-steps", str(step), "--exact-steps"] if step else [])
    call = _deployed(shards).spawn(run, scenes, "none", workers, shards, extra)
    print(f"[simeval] smoke on {run} ({shards or 1} x {GPU}): call {call.object_id} (modal app logs {APP})")
