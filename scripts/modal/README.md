# Training on Modal

Launcher: `scripts/modal/train.py`. It runs the unchanged `scripts/train.sh` inside a Modal
container in workspace **axon-foundational-research**, env **main**.

## One-time setup

```bash
modal token new --profile axon      # choose axon-foundational-research in the browser
modal profile activate axon
```

Everything else is already on Modal:

| Mount | Volume | Used for |
|---|---|---|
| `/data` | `openwam-sharpa-data` | `simteleop_1004_fleet/`, `gpt_fleet_depth/` (same episodes/frames/features as local `simteleop_1004` / `gpt_rollout`, checked against `meta/info.json` on 2026-10-12) |
| `/assets` | `openwam-assets` | `video_backbone_ckpt/Wan2.2-TI2V-5B` |
| `/runs` | `openwam-runs` | writes go **only** under `/runs/yjw/`: `norm_stats_modal/`, `outputs/` |
| secret | `wandb-yjw` | own **public wandb.ai** `WANDB_API_KEY` → runs land in `wandb.ai/jiangwei-yu-applovin/1008_wam_training`, NOT the company server `cbf-wandb.alcfd.com` (it resolves to the internal IP 10.132.0.109 and Modal can't reach it). Local runs via `run_dual.sh` also log to wandb.ai (it sets `WANDB_BASE_URL=https://api.wandb.ai`; key from `~/.netrc`, same entity), so local and Modal arms share one project. The teammate's `wandb-secret` gets "permission denied" on `1008_wam_training`. |

**Code shipping.** Every launch first uploads the checkout's code files (`openwam/ benchmarks/ scripts/ configs/`
+ root `*.sh *.toml`; `.py .sh .yaml .json .md ...` only, no tests / docs / outputs, ~1 MB) to `<run dir>/code/` on
`openwam-runs`. The image holds no source: the container copies that `code/` dir to `/opt/openwam` and trains from
it, so `code/` is exactly the code the run used (uncommitted edits included). A resume
(`training.resume_ckpt_path=...`) runs the run's own `code/` again, not the local checkout: to train with new code,
start a new run. Runs without a fixed dir (`--debug`) ship to `outputs/_code/<UTC time>/`.

Inside the container `/opt/openwam/outputs` is a symlink to `/runs/yjw/outputs`. Relative
`outputs/...` paths in `configs/train.yaml` (run dirs, checkpoints, wandb) end up on the Volume.
The local-only absolute paths are remapped by `PATH_OVERRIDES` in `train.py`. If you pass an
override with the same key, yours wins.

## Run

```bash
# Normal use: run_modal.sh at the repo root (ACTION_MODE / GPU_TYPE / WANDB_* env vars, see its header)
ACTION_MODE=unify_abs GPU_TYPE=B200 bash run_modal.sh
GPU_TYPE=B200 bash run_modal.sh --debug    # tiny-model smoke test, 1 GPU

# Direct call. The GPU spec comes from env vars that train.py reads when it defines the function
# (Modal fixes gpu= then), so the Modal UI shows the real GPU type. One function `train` serves
# both 1 and 8 GPUs.
OPENWAM_GPU_TYPE=B200 OPENWAM_GPU_COUNT=1 modal run scripts/modal/train.py --debug
OPENWAM_GPU_TYPE=B200 OPENWAM_GPU_COUNT=8 modal run --detach scripts/modal/train.py --overrides "..."
```

Runs launched with `run_modal.sh` land in `openwam-runs:/yjw/outputs/<WANDB_GROUP>/<index>_<WANDB_RUN_NAME>/`
(e.g. `82_fingertips_rel_b64`). `<index>` is a global counter in `/root/.core/index`, bumped once per
launch by `base.sh` (shared with `run_dual.sh`). The same name is the wandb run name and run ID, and,
with hyphens, the Modal run name (`82-fingertips-rel-b64`):

```
<group>/<index>_<run_name>/
  checkpoint_step_*.safetensors   accel_state_step_*/ (resume state, periodic saves only)
  vis/step_<N>_<i>.mp4            config.yaml  normalization_stats.npy  tokenizer/
  hydra/<launch time>/            .hydra/ (composed config + overrides) and train.log, one per launch
```

This layout comes from `training.output_path=outputs/<group>` plus `training.run_dir_name=<index>_<run_name>`.
`run_dir_name` defaults to null (a launch timestamp, which is the old behaviour). A run dir that
already holds a `config.yaml` is refused twice: once by `run_modal.sh` before launch, and once
by the trainer. Resume with `bash run_modal.sh training.resume_ckpt_path=outputs/<group>/<index>_<run_name>`. This takes no new index, reuses that dir, and its
Hydra logs get a new `hydra/<time>/` entry.
A resume runs the run's own `config.yaml`. Its launch must compose the same config: the same `run_modal.sh` env
(`BACKBONE`, `ACTION_MODE`, ...) and overrides as the first launch, or the arm script it came from. Only
`resume_ckpt_path` / `output_path` / `run_dir_name` may differ. Any other difference raises before training starts
(`openwam/train/run_mode.py::load_resume_cfg`).
`--debug` runs still go to `outputs/debug_checkpoints/` and Hydra's default `outputs/<date>/<time>/`.

Logs: `modal app logs <app id>` (find it in `modal app list` by the Modal run name), or the web UI.
Pull a run with `modal volume get openwam-runs /yjw/outputs/<group>/<index>_<run_name> ./outputs/<group>/`.

## Gotchas

- **Norm stats are rebuilt on Modal, not copied.** Each `.npy` records the source dirs it was
  built from, and the dataloader refuses a file whose sources don't match. Local files record
  `/root/projects/...`, so they are rejected. The first run of each `action_mode` scans the data and
  writes `/runs/yjw/norm_stats_modal/yam_sharpa_<mode>.npy`. For `fingertips_rel` this file matched
  the local `../norm_stats` file exactly (all arrays max|diff| = 0.0).

- **Auto-resume (24h limit, preemption, crashes).** Functions use `retries=modal.Retries(max_retries=10)`.
  Each retry gets a fresh 24h. A retry runs the *same call with the same arguments*, so its
  `training.run_dir_name` (`<index>_<run_name>`, fixed once by `run_modal.sh`) points at the dir the
  failed attempt created. No name matching across launches is done; a new `run_modal.sh` launch is always a
  new run. On each attempt, `_auto_resume` in `train.py` does the following:
  - the dir has a complete `accel_state_step_*` → adds `training.resume_ckpt_path=<that dir>`;
  - the dir has a `config.yaml` but no weights and no state (died before its first save) → renames it
    `<name>.failed_<ts>` and starts fresh;
  - the dir has weights but no state (a finished run) → refuses, and does not retry.
  An attempt that crashes **without** saving new resume state returns failure instead of raising.
  Modal then does not retry, so a deterministic bug does not burn 10 x 8 GPUs. At most 11 attempts run,
  which is about 11 days.
  `save_steps` bounds how much progress a timeout can lose. With about 4.5 s/step, `save_steps=2000` is
  about 2.5 h.
- **wandb continues the same run.** `WANDB_RUN_ID=<run dir name>` plus `resume="allow"` makes retries and
  manual resumes append to one wandb run. Steps between the last save and the kill were already logged,
  so wandb drops their re-run copies (step must increase). A SIGKILL also loses the last few seconds that
  wandb had not uploaded yet.
- **Manual resume** (e.g. after all retries are used up): `bash run_modal.sh training.resume_ckpt_path=outputs/<group>/<index>_<run_name>`.
- **Datasets are staged to local disk** (default for non-debug runs). Before training, `_stage_data` copies
  every `*.dataset_dir=/data/...` source to `/local_data/...` (32 parallel copies, `ephemeral_disk=512 GiB`, the Modal minimum)
  and points the overrides at the copy. The log line `[modal] staged ... GB in ... min` reports the
  throughput. Why: the first 8xH100 test reached ~13.5 samples/s vs ~47 locally, even with half the
  batch and gradient checkpointing off. That is suspected to be input-bound (Volume reads plus video
  decode on 64 CPUs) but was not measured. Staged runs use `norm_stats_local/` because stats record
  their source paths; it is rebuilt once per action mode. Turn staging off with
  `OPENWAM_STAGE_DATA=0 bash run_modal.sh`. Every retry gets a fresh container, so it copies again.
- **Weights are warmed with the data.** Before torchrun, every file under the `/assets/...` paths of the overrides is
  read once in parallel (log line `[modal] warmed ...`), so the 8 ranks load the backbone from the page cache, not
  cold from the Volume (~10 min -> ~1 min for Wan2.2-TI2V-5B, measured on `yjw/1008_our_data` 477cd51). Paths stay
  `/assets/...`; off together with staging (`OPENWAM_STAGE_DATA=0`).
- **Volume commits.** `train.py` commits `openwam-runs` every 10 min and once on exit.
  On preemption you can lose at most about 10 min of writes.
- **Environment.** `requirements-frozen.txt` is a freeze of the local `.venv`, so Modal and local runs use the same versions. If the local
  venv changes, regenerate it with the command in the header of that file. The next run then
  rebuilds the image.
- **Code is uploaded on every run** to `<run dir>/code/` (see Code shipping). Uncommitted local edits run too;
  `code/` records them, but commit before a real run so the wandb run can also be traced back to a git SHA.

## Verified

Detailed logs and the scripts behind them are kept locally (gitignored) in `outputs/investigations/`.
- Debug + real 8xH100 config, B200 debug, `run_dir_name` clash check, auto-resume (kill → retry resumes,
  one wandb run), crash-before-save not retried: all pass (2026-10-12/13).
- Modal reaches only public wandb.ai (company server is internal-only).
- 8xH100 `fingertips_abs`, batch 8/GPU: about 1.8 s/step on Modal and 1.4 s/step locally (2026-10-13). Data wait ≈ 0; see
  `timing/*` in wandb or `<run>/timing.jsonl`.
