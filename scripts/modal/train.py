"""OpenWAM training on Modal (workspace axon-foundational-research, env main).

Runs the unchanged ``scripts/train.sh`` inside a Modal container. The local
checkout is mounted at /opt/openwam on every run (code edits ship without an
image rebuild); datasets / weights / outputs come from shared Volumes.

Usage (from the repo root):
    # smoke test: tiny random DiT, real data, 1 GPU, 5 steps, no wandb
    OPENWAM_GPU_COUNT=1 modal run scripts/modal/train.py --debug

    # real run, 8 GPUs (OPENWAM_GPU_TYPE=H100|H200|B200, default H100), detached
    OPENWAM_GPU_TYPE=B200 modal run --detach scripts/modal/train.py \
        --overrides "dataloader.action_mode=fingertips_rel model.architecture.action_dim=48 ..."
    # multi-node: OPENWAM_NODES=2 -> a Modal cluster of 2 x 8 GPUs (RDMA), one torchrun per node
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
    "model.video_backbone.model_path": "/assets/video_backbone_ckpt/Wan2.2-TI2V-5B",
}

# ── Image: CUDA 12.8 devel (nvcc for DeepSpeed JIT ops) + the frozen local venv ──
deps_image = (
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
            "NCCL_RAS_ENABLE": "0",  # same H100 shutdown-SIGSEGV workaround as docker/Dockerfile
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)
SOURCE_IGNORE = [".venv", ".git", "outputs", "assets", "wandb", "**/__pycache__", "third_party", "docs", "**/*.mp4", "**/logs"]
# Source last so code edits never invalidate the dependency layers.
image = deps_image.add_local_dir(REPO_ROOT, REMOTE_REPO, ignore=SOURCE_IGNORE)
# Multi-node (RDMA): NCCL's InfiniBand transport needs libibverbs.so.1 + libmlx5.so.1 (Modal multi-node guide).
# A separate layer on top of the deps, so the single-node image is unchanged.
multinode_image = deps_image.apt_install("libibverbs1", "ibverbs-providers").add_local_dir(
    REPO_ROOT, REMOTE_REPO, ignore=SOURCE_IGNORE
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


def _latest_state_step(run_dir: Path, require_all_shards: bool = True) -> int:
    """Highest step with a usable resume state in `run_dir` (the trainer's state_is_usable rule).

    run_dir: run directory that may hold accel_state_step_<N>/ dirs.
    require_all_shards: True = resumable (marker + one optimizer shard per rank: what the next attempt can load);
        False = marker only (rank 0's own save finished: the progress test, other nodes' shards are not visible here).
    Returns: N of the newest usable accel_state_step_<N>/, else -1.
    """
    from openwam.train.utils.checkpointing import state_is_usable  # container-only (torch env); same rule as resume

    steps = [
        int(d.name.rsplit("_", 1)[1])
        for d in run_dir.glob("accel_state_step_*")
        if state_is_usable(str(d), require_all_shards=require_all_shards)
    ]
    return max(steps, default=-1)


def _auto_resume(overrides: list[str], main_node: bool = True) -> tuple[list[str], Path | None]:
    """Turn a (Modal-retried) fixed-name launch into a resume of its own run dir.

    A retry re-runs this call with the SAME overrides, so `training.run_dir_name`
    (<index>_<run_name>, fixed by run_modal.sh) points at the dir the earlier
    attempt created. No name matching across launches is involved. In a multi-node
    cluster every node runs this on the same committed Volume (reloaded at start), so
    all reach the same decision; only the main node acts on the run dir (moving a
    failed first attempt aside), the others wait until that move is visible.

    overrides: Hydra CLI tokens as sent by the caller.
    main_node: False on the other nodes of a cluster (no side effects).
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
        if main_node:
            aside = run_dir.with_name(f"{name}.failed_{time.strftime('%m_%d_%H_%M_%S')}")
            run_dir.rename(aside)
            runs_vol.commit()  # other nodes wait for the move (the trainer refuses a dir holding config.yaml)
            print(f"[modal] auto-resume: no resume state yet; moved previous attempt to {aside}", flush=True)
        else:
            _wait_until_gone(run_dir / "config.yaml")
    return overrides, run_dir


def _wait_until_gone(path: Path, timeout_s: int = 900) -> None:
    """Poll (reloading the runs Volume) until `path` disappears: the main node moved a failed attempt aside.

    path: file another container is removing. timeout_s: give up after this long.
    Returns: None; raises TimeoutError if it is still there.
    """
    deadline = time.time() + timeout_s
    while path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"{path} still present after {timeout_s} s; the main node did not move it")
        time.sleep(10)
        runs_vol.reload()


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


class _PeerWatch:
    """Heartbeat between the nodes of a cluster, so one node's dead training stops the others.

    Without it a surviving node's ranks hang in NCCL after a peer dies, its container stays busy, and Modal's retry
    (re-run in the same containers) starts on the free node alone and times out in rendezvous. Every node listens on
    PEER_PORT and answers its training status (b"R" running, b"D" finished OK) plus a random token of this attempt.
    A peer that stops answering, or answers with a new token (Modal already restarted it), while this node still
    trains and that never reported b"D", means its training died: the caller kills its own.
    """

    def __init__(self, peers: list[str]):
        """peers: IPv4 addresses of the other nodes. Starts listening at once (before staging, so peers see it early)."""
        import socket

        global _LIVE_WATCH
        if _LIVE_WATCH is not None:  # an earlier input in this container raised before closing its listener
            _LIVE_WATCH._shutdown()
        _LIVE_WATCH = self
        self.peers, self.status, self.token = list(peers), b"R", os.urandom(4).hex().encode()
        self._misses = dict.fromkeys(self.peers, 0)
        self._tokens: dict[str, bytes] = {}  # first token seen from each peer = the attempt this node trains with
        self._done = set()  # peers that reported a successful finish: never a reason to kill
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("0.0.0.0", PEER_PORT))
        self._srv.listen(16)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        """Answer every connection with the current status byte until close(). Returns: None."""
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return  # closed
            with conn:
                try:
                    conn.sendall(self.status + self.token)
                except OSError:
                    pass

    def peer_died(self) -> str | None:
        """Probe every peer once.

        Returns: the IP of a peer unreachable for PEER_MISSES probes in a row, or running a different attempt, else None.
        """
        import socket

        for ip in self.peers:
            try:
                with socket.create_connection((ip, PEER_PORT), timeout=5) as c:
                    reply = c.recv(9)
                if len(reply) < 9:
                    raise OSError("short heartbeat reply")
                status, token = reply[:1], reply[1:]
                if status == b"D":
                    self._done.add(ip)
                if self._tokens.setdefault(ip, token) != token and ip not in self._done:
                    return ip  # the peer was restarted (a new attempt): this attempt is over
                self._misses[ip] = 0
            except OSError:
                self._misses[ip] += 1
            if ip not in self._done and self._misses[ip] >= PEER_MISSES:
                return ip
        return None

    def close(self, finished_ok: bool) -> None:
        """Stop answering. After a successful finish, keep answering b"D" for a while so slower peers see it first.

        finished_ok: this node's training exited 0. Returns: None.
        """
        if finished_ok:
            self.status = b"D"
            time.sleep(PEER_PROBE_S * (PEER_MISSES + 2))
        self._shutdown()

    def _shutdown(self) -> None:
        """Close the listener; shutdown() first so the blocked accept() returns and the port is free. Returns: None."""
        import socket

        try:
            self._srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._srv.close()


_LIVE_WATCH: _PeerWatch | None = None  # this container's current listener (containers serve several inputs)
PEER_PORT = 29501  # node heartbeat (MASTER_PORT + 1)
PEER_PROBE_S = 10  # probe interval
PEER_MISSES = 3  # consecutive failed probes = that node's training is gone


def _kill_leftovers(timeout_s: int = 120) -> None:
    """SIGKILL training processes left by an earlier (cancelled / killed) input in this container and wait until
    they are gone, so their open files no longer block runs_vol.reload().

    timeout_s: stop waiting after this long. Returns: None.
    """
    deadline = time.time() + timeout_s
    while subprocess.run(["pgrep", "-f", "scripts/train.py"], capture_output=True).returncode == 0:
        subprocess.run(["pkill", "-9", "-f", "scripts/train.py"], check=False)
        if time.time() > deadline:
            print("[modal] leftover training processes still alive; continuing", flush=True)
            return
        time.sleep(2)


def _reload_runs(timeout_s: int = 300) -> None:
    """runs_vol.reload(), retried while files are still open (processes of a killed attempt exiting).

    timeout_s: give up after this long. Returns: None; re-raises the last error on timeout.
    """
    deadline = time.time() + timeout_s
    while True:
        try:
            runs_vol.reload()
            return
        except (RuntimeError, OSError) as e:  # OSError: Modal's open-file report racing a process that just exited
            if time.time() > deadline:
                raise
            print(f"[modal] runs volume reload failed ({type(e).__name__}: {e}); retrying", flush=True)
            time.sleep(10)


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
    cmd: list[str],
    overrides: list[str],
    test_kill_after_first_state: bool = False,
    stage_data: bool = False,
    extra_env: dict | None = None,
    main_node: bool = True,
    commit_every_s: int = 600,
    peers: list[str] | None = None,
) -> dict:
    """Run `cmd + overrides` in the repo, outputs/ on the runs Volume, resuming its own run dir on retry.

    cmd: argv prefix to execute from REMOTE_REPO (train.sh / train_debug.sh invocation).
    overrides: Hydra CLI tokens (path defaults already merged).
    test_kill_after_first_state: test hook; kill training once right after the first resume
        state is written (marker file prevents a second kill) to exercise retry + auto-resume.
    stage_data: copy the /data/ datasets to container-local disk before training (see _stage_data).
    extra_env: env vars for train.sh (multi-node: NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT), or None.
    main_node: False on the other nodes of a multi-node cluster: they reach the same resume decision without
        touching the run dir, never write TRAINING_DONE and never raise for a retry (the main node decides).
    commit_every_s: periodic runs-Volume commit interval (multi-node: short, so other nodes' shards of a new
        resume state reach the Volume soon after the save).
    peers: other nodes' IPv4s (multi-node): a peer whose training died stops this node's (see _PeerWatch).
    Returns: {"ok": bool, "msg": str}. Raises (-> Modal retry) only if this attempt made
        progress, so a deterministic crash is not retried 10 times.
    """
    # A cancelled earlier input in this container can leave its training processes alive (own session): they would
    # hold the GPUs and keep files open, which makes runs_vol.reload() fail.
    _kill_leftovers()
    watch = _PeerWatch(peers) if peers else None
    # Relative `outputs/...` paths in train.yaml (run dirs, checkpoints, wandb) land on the Volume.
    out_dir = Path(f"/runs/{RUNS_SUBDIR}/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    link = Path(REMOTE_REPO) / "outputs"
    if not link.exists():
        link.symlink_to(out_dir)
    _reload_runs()  # see everything earlier attempts committed

    try:
        overrides, run_dir = _auto_resume(overrides, main_node=main_node)
    except FileExistsError as e:
        print(f"[modal] FAILED: {e}", flush=True)
        if watch:
            watch.close(finished_ok=False)
        return {"ok": False, "msg": str(e)}  # returned, not raised: no Modal retry
    env = {**os.environ, **(extra_env or {})}
    if run_dir is not None:
        # One wandb run per run dir: retries / resumes append to it instead of creating a new run.
        env["WANDB_RUN_ID"] = run_dir.name
    # Progress = a newer marker than at start (marker-only on both ends: other nodes' shards are not visible here).
    start_step = _latest_state_step(run_dir, require_all_shards=False) if run_dir else -1
    if stage_data:
        overrides = _stage_data(overrides)

    # Record which GPUs this container actually got (H100 vs B200) in the run log.
    subprocess.run(["nvidia-smi", "-L"], check=False)
    full = [*cmd, *overrides]
    print(f"[modal] {shlex.join(full)}", flush=True)
    stop = threading.Event()
    threading.Thread(target=_periodic_commit, args=(stop, commit_every_s), daemon=True).start()
    proc, rc = None, None
    try:
        proc = subprocess.Popen(full, cwd=REMOTE_REPO, env=env, start_new_session=True)
        marker = run_dir / ".test_killed_once" if run_dir and main_node else None
        next_probe = time.time() + PEER_PROBE_S
        while proc.poll() is None:
            time.sleep(5)
            if watch and time.time() >= next_probe:
                next_probe = time.time() + PEER_PROBE_S
                dead = watch.peer_died()
                if dead:
                    print(f"[modal] node {dead} stopped answering (its training died): stopping this node's", flush=True)
                    _kill_training(proc)
                    break
            if (test_kill_after_first_state and marker and not marker.exists()
                    and _latest_state_step(run_dir, require_all_shards=False) > start_step):
                marker.touch()
                # Freeze training (the other nodes block in their next collective), then let every node's periodic
                # commit carry its shards of this state before the kill (a real preemption can land right after a
                # save; then the previous state is used, see state_is_usable).
                subprocess.run(["pkill", "-STOP", "-f", "scripts/train.py"], check=False)
                time.sleep(2 * commit_every_s)
                _kill_training(proc)
                print("[modal] TEST: killed training after first resume state", flush=True)
        rc = proc.returncode
        if rc == 0 and run_dir is not None and main_node:
            # Training-finished signal for the sim-eval watcher (scripts/modal/simeval_watch.py: --done-marker).
            (run_dir / "TRAINING_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))
    finally:
        if proc is not None:
            _kill_training(proc)  # no orphaned rank may keep writing the run dir after we return / retry
        stop.set()
        runs_vol.commit()  # final commit, also on failure, so partial checkpoints/logs survive
        if watch:
            watch.close(finished_ok=rc == 0)  # a failure is seen by the peers within PEER_MISSES probes
    if rc == 0:
        return {"ok": True, "msg": "training finished"}
    if not main_node:
        # The main node decides on a retry (Modal restarts the whole cluster when any node's call fails).
        return {"ok": False, "msg": f"node training exited {rc}"}
    end_step = _latest_state_step(run_dir, require_all_shards=False) if run_dir else -1
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
# Nodes per run: 1 = the plain `train` function; > 1 = `train_multinode`, a Modal cluster of that many 8-GPU nodes.
NUM_NODES = int(os.environ.get("OPENWAM_NODES", "1"))
if modal.is_local() and (GPU_TYPE not in GPU_TYPES or GPU_COUNT not in (1, 8)):
    raise SystemExit(f"OPENWAM_GPU_TYPE must be one of {GPU_TYPES} and OPENWAM_GPU_COUNT 1 or 8, got {GPU_TYPE}:{GPU_COUNT}")
if modal.is_local() and NUM_NODES > 1 and GPU_COUNT != 8:
    raise SystemExit(f"OPENWAM_NODES={NUM_NODES} needs whole 8-GPU nodes (Modal clusters), got OPENWAM_GPU_COUNT={GPU_COUNT}")
MASTER_PORT = 29500  # torchrun rendezvous port on rank 0 of a cluster (Modal's private network allows any TCP port)


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


def train_multinode(
    overrides: list[str], stage_data: bool = True, verbose_nccl: bool = False, test_auto_resume: bool = False
) -> dict:
    """One node of a multi-node run: torchrun on this node's 8 GPUs, joined to the others through rank 0.

    Every node runs this with the same args (Modal broadcasts them); only rank 0's return value reaches the caller.
    Rank 0 owns the run dir (creation, config, weights, wandb), as the trainer's global rank 0 does on one node.
    Modal retries the whole cluster (timeout, preemption, a failed node); every node then reloads the same
    committed Volume and resumes from the newest state that has every rank's shard (see _auto_resume).

    overrides: Hydra CLI tokens appended after the path defaults.
    stage_data: copy datasets to each node's local disk first.
    verbose_nccl: NCCL_DEBUG=INFO in every rank (train.sh's OPENWAM_VERBOSE_NCCL), e.g. to see the transport chosen.
    test_auto_resume: kill once after the first resume state (main node), exercising the cluster retry + resume.
    Returns: {"ok": bool, "msg": str} from _run_training (rank 0's).
    """
    cluster = modal.Cluster.from_context()
    rank, ips = cluster.container_rank(), cluster.container_ips(family="ipv4")
    merged = _merge_overrides(overrides)
    stats = _stats_file(merged, staged=stage_data)
    if stats is not None and not stats.is_file():
        # Rank 0 would build it, but the other nodes only see the Volume as of their own reload: they would wait
        # forever. Build it with a single-node run (any length) of the same action_mode first.
        msg = f"multi-node needs the norm stats {stats} on the runs Volume already; run one single-node launch first"
        print(f"[modal] FAILED: {msg}", flush=True)
        return {"ok": False, "msg": msg}
    print(f"[modal] cluster node {rank}/{len(ips)}, master {ips[0]}:{MASTER_PORT}", flush=True)
    env = {"NNODES": str(len(ips)), "NODE_RANK": str(rank), "MASTER_ADDR": ips[0], "MASTER_PORT": str(MASTER_PORT),
           "OPENWAM_VERBOSE_NCCL": "1" if verbose_nccl else "0"}
    return _run_training(["bash", "scripts/train.sh"], merged, test_kill_after_first_state=test_auto_resume,
                         stage_data=stage_data, extra_env=env, main_node=rank == 0, commit_every_s=120,
                         peers=[ip for i, ip in enumerate(ips) if i != rank])


def _stats_file(overrides: list[str], staged: bool) -> Path | None:
    """The pooled norm stats file a run will read (``dataloader.normalization_stats_path``, action_mode filled in).

    overrides: Hydra CLI tokens (path defaults merged). staged: data staged to local disk (norm_stats_local/).
    Returns: the Path, or None when the path or the action_mode is not among the overrides.
    """
    path, mode = (_hydra_value(overrides, k) for k in ("dataloader.normalization_stats_path", "dataloader.action_mode"))
    if not (path and mode):
        return None
    path = path.replace("${.action_mode}", mode)
    return Path(path.replace("/norm_stats_modal/", "/norm_stats_local/") if staged else path)  # as _stage_data


# The cluster size is fixed at definition time like the GPU spec. Registered locally only for a multi-node launch
# (a single-node `modal run` never builds its image or asks for a cluster); always inside the container, which
# imports this file without OPENWAM_NODES and only needs the name to resolve (its size there is irrelevant).
if NUM_NODES > 1 or not modal.is_local():
    train_multinode = app.function(
        gpu=f"{GPU_TYPE}:8",
        cpu=CPU_COUNT,
        memory=MEMORY_GB * 1024,
        ephemeral_disk=512 * 1024,
        image=multinode_image,
        volumes=VOLUMES,
        secrets=COMMON["secrets"],
        # OPENWAM_TIMEOUT_S: a short per-attempt cap only to test the timeout -> retry -> resume path.
        timeout=int(os.environ.get("OPENWAM_TIMEOUT_S", COMMON["timeout"])),
        retries=COMMON["retries"],  # the whole cluster restarts and resumes (24 h cap, preemption, node failure)
    )(modal.clustered(size=max(NUM_NODES, 2), rdma=True)(train_multinode))


@app.local_entrypoint()
def main(overrides: str = "", debug: bool = False, test_auto_resume: bool = False) -> None:
    """CLI entry: `[OPENWAM_GPU_TYPE=B200] [OPENWAM_GPU_COUNT=1] modal run scripts/modal/train.py [--debug] [--overrides "..."]`.

    overrides: space-separated Hydra overrides (shell-quoted), e.g. "a.b=1 c=d".
    debug: run the tiny-model smoke test (use with OPENWAM_GPU_COUNT=1).
    test_auto_resume: test hook, kill once after the first resume state.
    Returns: None; exits non-zero if training failed without a retry.
    """
    stage = os.environ.get("OPENWAM_STAGE_DATA", "1") != "0"  # OPENWAM_STAGE_DATA=0 reads straight from the Volume
    print(f"[modal] requesting {NUM_NODES} x {GPU_TYPE}:{GPU_COUNT}, {CPU_COUNT} CPU, {MEMORY_GB} GB RAM per node, "
          f"stage data to local disk: {stage and not debug}")
    if NUM_NODES > 1:
        if debug:
            raise SystemExit("--debug is single-node only (OPENWAM_NODES=1)")
        # OPENWAM_VERBOSE_NCCL=1 in the launch env: NCCL INFO logs on every rank (the container does not see local env).
        verbose = os.environ.get("OPENWAM_VERBOSE_NCCL") == "1"
        res = train_multinode.remote(shlex.split(overrides), stage, verbose, test_auto_resume)
    else:
        res = train.remote(shlex.split(overrides), debug, test_auto_resume, stage)
    print(f"[modal] result: {res}")
    if not res["ok"]:
        raise SystemExit(1)
