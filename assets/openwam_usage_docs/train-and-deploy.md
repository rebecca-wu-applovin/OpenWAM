# Training and deployment

OpenWAM composes each run from Hydra YAML. The top-level configuration is [configs/train.yaml](../../configs/train.yaml); it selects the model and dataloader, while each model group selects its component backbones. The trainer saves the resolved configuration with each checkpoint.

## 1. Choose and assemble a model

### Architecture

The model group in [configs/train.yaml](../../configs/train.yaml) is `dual_system`; edit the `architecture` block in [configs/model/dual_system.yaml](../../configs/model/dual_system.yaml).

~~~yaml
# configs/train.yaml
defaults:
  # Model and data groups
  - model: dual_system       # the only model group
  - dataloader: yam_sharpa   # the only dataloader group
~~~

All architecture options are under `architecture:`. Hydra exposes the composed values as `model.architecture.*` for CLI overrides.

~~~yaml
# configs/model/dual_system.yaml
architecture:
  # Architecture identity
  framework: dual_system
  variant: joint_self_attn               # joint_self_attn | joint_cross_attn

  # Inputs (the action / proprio widths are common.action_dim / common.state_dim in configs/train.yaml)
  use_proprioception: true               # require a proprio field in each sample

  # Attention visibility
  attention_mask_mode: mutual            # only mutual is implemented (the other three values are rejected)
  clean_attention: bidirectional         # among the clean history latents: bidirectional | causal
  history_encoding: per_frame            # past history frames: per_frame | clip

  # Bridge and memory controls
  bridge_interval: 1                     # bridge layers = range(0, video geometry.num_layers, interval); 1 = every layer

  # Variant-specific options
  detach_bridge: false                   # joint_cross_attn only
~~~

Values several components share live in `common:` of `configs/train.yaml` (`action_dim`, `state_dim`: 48 for
`fingertips_*`, 80 for `unify_*`; the dataset checks them against `dataloader.action_mode`). Each component gets
`OmegaConf.merge(cfg.common, cfg.<component>)`. `training.mot_checkpoint_mixed_attn` checkpoints the mixed
attention to reduce GPU memory.

Implementations and registries are in [openwam/model/architectures/](../../openwam/model/architectures/). The model file composes the video and action groups through its `defaults` list.

### Video backbone

Choose a Hydra group from [configs/model/video_backbone/](../../configs/model/video_backbone/); implementations are in [openwam/model/video_backbone/](../../openwam/model/video_backbone/).

~~~text
# Choose one video group
model/video_backbone=wan22_ti2v_5b
model/video_backbone=cosmos3_edge
~~~

Edit the selected group file, for example [wan22_ti2v_5b.yaml](../../configs/model/video_backbone/wan22_ti2v_5b.yaml), or [cosmos3_edge.yaml](../../configs/model/video_backbone/cosmos3_edge.yaml):

~~~yaml
# configs/model/video_backbone/<selected_group>.yaml
# @package _global_          (the file writes model.video_backbone, its assets and its own training switches)
model:
  video_backbone:                       # what the model is (compared on finetune)
    name: wan22_ti2v_5b                 # registry key; use the selected backbone name
    shift_video: 5.0                    # flow-matching scheduler shift
    # Cosmos3 additionally defines fps, prompt_duration_template, and freeze_und.
assets:
  video_backbone:                       # where the upstream files are (not compared; finetune / deploy never read it)
    model_path: /path/to/video_backbone_checkpoint  # weights or bundle root
training:
  freeze: [video_backbone.text_encoder, video_backbone.vae]  # every path must exist on the model
  text_cache: true                      # cosmos3: false (true raises)
~~~

The selected backbone supplies video geometry to the architecture and action backbone. Keep `assets.video_backbone.model_path` consistent with the assets in [Assets Preparation](../../README.md#assets-preparation).

### Action backbone

DualSystem constructs [separate_action_dit.yaml](../../configs/model/action_backbone/separate_action_dit.yaml). Implementations are in [openwam/model/action_backbone/](../../openwam/model/action_backbone/), and there is no separate action-backbone registry.

~~~yaml
# configs/model/action_backbone/separate_action_dit.yaml
# ActionDiT dimensions
dim: 1024                        # ActionDiT hidden width
ffn_dim: 4096                    # ActionDiT feed-forward width
shift_action: 5.0                # action scheduler shift
freq_dim: 256                    # sinusoidal timestep embedding width
max_action_len: 1024             # learned action position table size
eps: 1.0e-06                     # norm epsilon
num_heads: ${..video_backbone.geometry.num_heads}   # attention geometry and the bridge / context widths
head_dim: ${..video_backbone.geometry.head_dim}     #   come from the video backbone yaml's geometry block
video_dim: ${..video_backbone.geometry.dim}
text_dim: ${..video_backbone.geometry.text_dim}
~~~

The only group is `model/action_backbone=separate_action_dit`; keep its output aligned with `common.action_dim` and the dataloader.

### Dataloader selection and loading

The built-in registry name is `yam_sharpa`; its fields are in [configs/dataloader/yam_sharpa.yaml](../../configs/dataloader/yam_sharpa.yaml).
Readers are under openwam/dataloader/ and configs under configs/dataloader/. Select one with Hydra:

~~~bash
# Select the dataloader and dataset
bash scripts/train.sh \
  dataloader=yam_sharpa \
  dataloader.action_mode=fingertips_rel
~~~

Each registered class implements from_config(config, split) and returns window samples. The trainer's DataLoader stacks each list of sample dictionaries into one `Batch` dataclass with `openwam.dataloader.collate.collate`; `compute_loss(batch)` takes that `Batch`. num_frames is the raw state/action window length; the action horizon is num_frames - 1. Video is the history rows t + history_offsets followed by the future rows t + future_offsets; the standard first-frame setting (history_offsets [0], future_offsets [4, 8, ..., 32]) yields 9 video frames. Camera layout, transforms, normalization, action maps, and masks belong to the dataloader.

## 2. Start training

### Fresh and fine-tuning runs

With a pretrained video backbone, a fresh run loads the selected components and trains the unfrozen modules from step zero.

~~~bash
# Model and dataloader
bash scripts/train.sh model=dual_system dataloader=yam_sharpa \
  dataloader.sources.<source_name>.dataset_dir=<dataset_dir_path> \
  training.output_path=<output_dir_path> \
  training.batch_size=1 \
  training.max_steps=20
~~~

These training fields come from [configs/train.yaml](../../configs/train.yaml); CLI overrides take precedence over the file.
Use training.finetune_ckpt_path to warm-start weights into a new run. Use training.resume_ckpt_path only to continue a run that saved full Accelerate state:

~~~yaml
training:
  # Start a new run from existing weights
  finetune_ckpt_path: /path/to/base_checkpoint

  # Continue a run with its saved optimizer and scheduler state
  resume_ckpt_path: null
  save_full_states_for_resume: true
~~~

The two paths are mutually exclusive. Fine-tuning starts at step 0 in a new output directory; resume restores optimizer, scheduler, RNG, and model state in the original run.

A finetune requires the launch's `model` + `common` blocks to equal the checkpoint's: any difference raises one error listing every differing key (nothing is merged). The upstream weight paths live in `assets` and are not compared; `training` and `dataloader` may differ.

A resume runs the run's own saved `config.yaml`: any other difference in the launch's config raises. Only the resume-location keys (`training.resume_ckpt_path` / `output_path` / `run_dir_name`) come from the launch. `assets` is compared too (a resume must not change any setting).

### Memory, precision, and distributed launch

The distributed and memory fields below are defined in [configs/train.yaml](../../configs/train.yaml). `scripts/train.sh` invokes `torchrun` and detects visible GPUs.

~~~yaml
training:
  # Precision and distributed optimizer
  mixed_precision: bf16                 # bf16, fp16, or no
  zero_stage: 2                         # ZeRO stage; stage 2 also shards gradients
  gradient_accumulation_steps: 1        # optimizer update every N micro-batches

  # Activation memory
  use_gradient_checkpointing: true      # recompute activations during backward
  use_gradient_checkpointing_offload: false  # offload checkpointed activations to CPU

  # Parameter and optimizer placement
  initialize_model_on_cpu: false        # initialize on CPU before accelerator placement
  offload_optimizer_device: none        # none or cpu
~~~

Set `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, and `MASTER_ADDR` for multi-node launch.

## 3. Deploy a checkpoint

A training output contains weights, `config.yaml`, normalization data and tokenizer/component assets. Deploy reads
the run's own `config.yaml` (re-checked with `resolve_config`, as training does) plus one `deploy` block from
[configs/deploy.yaml](../../configs/deploy.yaml); only `deploy.*` keys can be overridden, the model / data config is
the one the checkpoint was trained with. Everything is checked once at startup (`check_deploy`) and frozen.

~~~python
from openwam.deploy import InferenceEngine, build_deploy_cfg

cfg = build_deploy_cfg(["deploy.ckpt.dir=<run_dir>", "deploy.ckpt.name=checkpoint_step_N.safetensors",
                        "deploy.policy.steps_per_plan=20"])
engine = InferenceEngine.from_config(cfg)
session = engine.new_session()   # one per episode: its own observation history and seed
out = session.plan(obs)          # {"actions": (K, D), "hand_targets": {"wrist": (K, 2, 4, 4), "tips" | "fingers"}}
~~~

`obs` uses the training data's LeRobot feature names (`observation.images.<camera>`, `observation.state`,
`camera_pose.ego_view`, `joint_angles` for `unify_*`, `task`). For sim-policy-eval, serve a checkpoint file with
[benchmarks/simeval/server/serve.py](../../benchmarks/simeval/server/serve.py) (see its README).

### The deploy block

~~~yaml
ckpt: {dir: null, name: null}         # required: run dir + checkpoint_step_N.safetensors (no "latest" guess)
server: {host: 0.0.0.0, port: 8848, device: cuda:0}
policy:
  steps_per_plan: null                # required K: actions per plan; divides dataloader.fps and every history offset
  seed: 42                            # sampling seed of every plan of an episode, unless the client names one
sampler:
  denoise_steps: 10
  denoise_mode: sync                  # sync | async (denoising trajectory)
  lead_modality: video                # async only: action | video
  variance_shift_alpha: 1.0           # async only: >= 1
  linear_offset: 0.0                  # async only: 0 <= value < 1
  decode_video: false                 # also decode each plan's future video (debugging)
~~~

With `num_frames: 33`, each plan generates 32 actions (`common.layout.action_horizon`) and returns the first K.
Source code: `openwam/deploy/` (`config.py`, `engine.py`, `model_loader.py`, `history_buffer.py`).
