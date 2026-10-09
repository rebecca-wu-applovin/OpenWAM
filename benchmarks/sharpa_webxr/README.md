# Closed-loop WebXR sim eval for GWP-0.5 Sharpa checkpoints

This harness runs the GWP-0.5 Sharpa EEF policy (`run_giga_sharpa_train_job.py --action-repr eef`) closed loop in the
WebXR-Teleop MuJoCo sim. It uses the training scenes from the live WebXR store and seeded resets, so every checkpoint
sees the same initial states. Each rollout gets automatic stage signals, 16 keyframes for a VLM judge, and a video.

## Recipe

The recipe follows the GWP-0.5 closed loop in wam.cpp (`eval/sim/run_robotwin_client.py` with `ActionChunkExecutor`).

- **Control loop:** a synchronous observe → predict → execute loop. The policy predicts a 32-step chunk at 20 Hz, all
  32 steps are executed, and then the policy re-observes. There is no ensembling.
- **Sampling:** 10 flow-matching steps, shift 5, no CFG, current frame only.
- **Policy inputs:** `ego_view` and `chest_view` in a T-layout at 320x384, identical to training (`_compose_views`).
  The state is the measured wrist pose plus the hand joints. Norm stats and T5 prompts come from the training prep dir.
- **Action decoding:** actions are decoded with `CODECS['giga']`, anchored at the measured state. The wrist poses are
  then converted to arm joints with `scripts/sharpa_eef/eef_kinematics.ik`; hand joints pass through unchanged.
- **Physics:** TacoSim with the teleop app defaults: 2400 Hz physics, 60 Hz tick, and each 20 Hz target held for 3 ticks.
- **Cameras:** dataset `camera_pose.<view>` extrinsics with fovy 60 (ego) and 90 (chest), at 640x480.
- **Reset:** `teleop_app.prepare_scene` with a seeded RNG: arm jitter 0.08, object xy ±5 cm, yaw ±180°, 1 s settle.
- **Episode length:** `ceil(1.5 × p90)` of the scene's demo lengths (`tasks.json`), capped at 1200 steps.

Stage signals, computed in `postprocess.py`:

- **reach:** a hand comes within 3 cm of an object.
- **grasp:** at least 0.5 s of hand contact while the object is lifted at least 2 cm or carried at least 3 cm.
- **move:** the object is displaced at least 5 cm, or a joint changes by at least 20° or 3 cm.
- **fallen:** an object fell off the table.
- **Rule verifier** (`benchmarks/sharpa/verifiers.py`): runs for the laptop and box-lid scenes.

Task completion and placement are scored by `judge.py` (Claude, from the keyframes).

Validated before use:

- Rendering matches the recorded dataset frames: PSNR 37.7 dB on the ego view and 42.1 dB on the chest view.
- Replaying recorded setpoints reproduces the recorded outcome to within 1.2 mm.
- Policy inputs are bit-identical to the training dataset (`check_policy_parity.py`).

## Setup

```bash
# headless rendering (no GPU GL needed)
apt-get install -y libosmesa6 libegl1 libglvnd0
git submodule update --init third_party/WebXR-Teleop third_party/giga-world-policy

# sim venv (python 3.12), separate from the training .venv
uv venv .venv_webxr --python 3.12
uv pip install --python .venv_webxr/bin/python -r third_party/WebXR-Teleop/requirements.txt \
    torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv_webxr/bin/python pyzmq imageio imageio-ffmpeg av anthropic

# scenes from the live store (needs gsutil access) -> data/webxr_scenes/<scene_id>/
.venv_webxr/bin/python benchmarks/sharpa_webxr/scenes.py $(.venv_webxr/bin/python -c \
    "import json;print(' '.join(json.load(open('benchmarks/sharpa_webxr/tasks.json'))))")
```

The policy server runs in the training `.venv`. It needs the Wan2.2 VAE at `assets/video_backbone_ckpt/Wan2.2-VAE-Diffusers/vae`.

## Automatic eval of a training run

```bash
benchmarks/sharpa_webxr/watch_run.sh <run> <gpu_ema> <gpu_bf16> [seeds_intermediate=0-4] [seeds_final=5-9] \
  > .sharpa_sim_eval/logs/watch_<run>.log 2>&1 &
```

`<run>` is the folder under `.gwp_runs/`. The watcher works like this:

- It copies each `checkpoint_stepN/transformer_bf16.pt` before training's rotation deletes it.
- It evaluates the EMA (`ema_stepN/`) and the raw weights of every checkpoint, oldest first. Each is served on its own
  GPU, using about 15 GB next to training.
- It renders videos from one shared low-priority queue, seed 0 first.
- After training exits, it adds the final seeds for the last step and evaluates the final `best_val`.

The watcher is resumable: rerunning the same command skips finished work. It prints `WATCH_ALL_DONE` when done.

| env | default | meaning |
|---|---|---|
| `VIDEO` | `1` | `0` skips `video.mp4` rendering. Signals and keyframes are always written. |
| `VIDEO_SEEDS` | `all` | Comma list of seeds to render, e.g. `0`. |
| `VIDEO_P` | `32` | Parallel renderers. Each holds about 2.3 GB RAM and renders on the CPU with OSMesa at nice 19. |
| `PORT_BASE` | `11600` | EMA server port; raw uses `PORT_BASE+1`. Change it to run two watchers at once. |
| `PREP_DIR` | `prep_eef_full_v1` | Training prep dir with norm stats and prompts. Must match the run's `--prep-dir`. |

Assumptions:

- The run uses the EEF action representation (62-D), 2 views in a T-layout, and horizon 32. Check a new run with
  `check_policy_parity.py` first: expect 24/24 identical inputs and a wrist error in the tens of mm or less.
- Training is detected by its `--out-dir .gwp_runs/<run>` argument.
- Rendering is slower than evaluation (about 3 videos/min on a loaded host), so videos lag behind the numbers.

## Manual pieces

```bash
# policy server (training venv)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u benchmarks/sharpa_webxr/policy_server.py --ckpt <transformer_*.pt> --port 11500
# one checkpoint: 58 scenes x seeds 0-9, 10 workers
RUN=<run> benchmarks/sharpa_webxr/run_suite.sh <tag> 11500 0 9 10
# one scene
.venv_webxr/bin/python benchmarks/sharpa_webxr/rollout.py --port 11500 --scene <scene_id> --seeds 0-2 --out <dir>
.venv_webxr/bin/python benchmarks/sharpa_webxr/postprocess.py <dir>/<scene_id>/seed_0
# checkpoint parity on val windows
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/sharpa_webxr/check_policy_parity.py --ckpt <ckpt> --n 24 --out g3.json
# VLM judge over keyframes (needs ANTHROPIC_API_KEY)
.venv_webxr/bin/python benchmarks/sharpa_webxr/judge.py .sharpa_sim_eval/<run>/<tag>/*/seed_*
```

## Results

- Table: `.venv/bin/python benchmarks/sharpa_webxr/aggregate.py <run>`. It prints stage rates, the all-three rate with a
  95% Wilson CI, and the gpt_fleet_depth vs simteleop-only split.
- Notebook: `notebooks/_build_gwp_sim_eval_nb.py`, which builds `notebooks/gwp_sim_eval.ipynb`. It covers stage rates,
  rate vs training step, outcome breakdown, a per-scene heatmap, IK/fallen/episode length, keyframes and embedded videos.
- Video page: `build_review_page.py <run> --tags <tag...> --out <dir>`. It writes one section per scene with
  checkpoints side by side.
- Per rollout: `.sharpa_sim_eval/<run>/<tag>/<scene>/seed_k/`, containing `rollout.json`, `trajectory.npz`,
  `signals.json`, `keyframes/` and `video.mp4`.
