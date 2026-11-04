# Integrating a new benchmark

Benchmark integration has two adapters:

1. A dataloader emits OpenWAM sample dictionaries for training.
2. A client speaks the policy server WebSocket protocol and converts returned actions to the benchmark controller.

The benchmark owns action ordering, units, signs, ranges, and gripper meaning. Keep those definitions in the dataloader and client adapter.

## 1. Build a dataloader

### Choose a base and register it

For a custom source, subclass `BaseDataset` in [openwam/dataloader/bases/dataset.py](../../openwam/dataloader/bases/dataset.py). For LeRobot v3 data, subclass `LeRobotV3Reader` in [openwam/dataloader/bases/lerobot_v3_reader.py](../../openwam/dataloader/bases/lerobot_v3_reader.py), which provides parquet windows, video decoding, prompts, statistics, masks, and multiview composition.

Override only the source hooks that differ, such as `_getitem_impl(idx, rng)` (required: builds the sample; `rng` is the per-sample generator, the only random source a reader may use), `_action_20d`, `_resolve_cameras`, `_resolve_prompt`, and `_load_stats`. A `LeRobotV3Reader` is a source: the registered mixture (e.g. `MultiYamSharpaDataset`) calls its `sample(idx, rng)` with a `(seed, epoch, idx)` generator, and the reader reads every data shard into memory at construction. Register the class in [openwam/dataloader/registry.py](../../openwam/dataloader/registry.py):

~~~python
from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.registry import register_dataset

@register_dataset('my_benchmark')
class MyBenchmarkDataset(BaseDataset):
    @classmethod
    def from_config(cls, config, split='train'):
        return cls(dataset_dir=config.dataset_dir, split=split)

    def __len__(self): ...
    def __getitem__(self, index): ...
~~~

Import the module before calling `build_dataset(config)`. Every registered class must implement `from_config`; built-in readers are imported by the registry.

### Return the canonical sample

The trainer's `collate_fn` is [openwam/dataloader/collate.py](../../openwam/dataloader/collate.py) `collate`: it stacks the samples into one `Batch` (`video`, `prompt`, `action`, `action_is_pad = ~action_mask`, `proprio`, `proprio_mask`, `source`), which is the only input of `compute_loss`. A sample carries data only (no flag for a model branch). Return the fields below, with H = `len(history_offsets)` and F = `len(future_offsets)`:

~~~text
# Video and prompt
video             list[PIL.Image], length = H + F (history frames, the current frame t last, then the future)
prompt            str

# Action and state
# action width D must match architecture.action_dim
# state width D_state must match architecture.state_dim
action            float32 Tensor, shape (num_frames - 1, D)
action_mask       bool Tensor, shape (num_frames - 1, D)
proprio           float32 Tensor, shape (H, D_state), one row per history frame
proprio_mask      bool Tensor, shape (H, D_state)

# Added by the mixture dataset: the name of the source the sample came from
source            str
~~~

Every video frame is real (there is no video padding mask); a source window too short for the layout is not sampled. For video-only data, emit zero actions with an all-false `action_mask`; zero values alone do not disable loss. A missing head frame should raise; an auxiliary-camera failure may be black-filled. Keep camera composition and frame stride identical between training and deployment.

### Define an action space

An action may contain joints, EEF pose, base motion, or any combination. The final action width must be consistent across readers that are mixed together; `MixtureDataset` enforces this in strict mode.

For schema-free scatter/gather, use [openwam/dataloader/utils/unify_action.py](../../openwam/dataloader/utils/unify_action.py):

~~~python
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

# Build the destination map
dst = parse_unify_spec(config.unify_action_map, unify_dim=UNIFY_DIM)
unified, dim_mask = map_to_unify(raw_action, dst, unify_dim=UNIFY_DIM)
raw_again = unmap_from_unify(unified, dst)
~~~

A mapping must cover raw dimensions exactly, use every destination once, and stay within `[0, UNIFY_DIM)`. The built-in unified width is `UNIFY_DIM=80`; a non-unified reader may expose another model width. The helper assigns no slot semantics, so document physical meanings in the reader.

### Configure frames, transforms, splits, and statistics

Add a YAML file under [configs/dataloader/](../../configs/dataloader/):

~~~yaml
# configs/dataloader/my_benchmark.yaml
# Dataset and sampling
type: my_benchmark
dataset_dir: /path/to/dataset
num_frames: 33
history_offsets: [0]                  # rows of the history frames (last = the current frame t)
future_offsets: [4, 8, 12, 16, 20, 24, 28, 32]
history_sampling: fixed               # fixed | random (train split: H-1 random rows of the past history_window rows)
history_window: 100                   # random only; must reach the oldest history offset
window_stride: 1

# Image layout
height: 384
width: 320
multiview: false

# Action preprocessing
normalize_mode: min-max              # min-max, z-score, quantile, or null
~~~

Use [openwam/dataloader/utils/normalization.py](../../openwam/dataloader/utils/normalization.py) for normalization, [openwam/dataloader/utils/rotation.py](../../openwam/dataloader/utils/rotation.py) for the 6D rotation conversion and [openwam/dataloader/transforms/video.py](../../openwam/dataloader/transforms/video.py) for color jitter (train split, inside the dataset). Compute statistics from the training split and store them under `meta/`. A deployable reader writes `meta/normalization_stats.npy`; deployment gathers unified slots back to raw dimensions before unnormalizing. Keep the same statistics artifact with the checkpoint.

Inspect one sample through the registry:

~~~python
from omegaconf import OmegaConf
from openwam.dataloader.registry import build_dataset

config = OmegaConf.load('configs/dataloader/my_benchmark.yaml')
dataset = build_dataset(config)
sample = dataset[0]
print(len(dataset), sample['action'].shape, sample['video'])
~~~

## 2. Adapt the benchmark client

The wire protocol is implemented by `PolicyServer` in [openwam/deploy/server.py](../../openwam/deploy/server.py); clients (e.g. sim-policy-eval, launched through [benchmarks/simeval/serve.py](../../benchmarks/simeval/serve.py)) mirror it. The server decodes and composes images, forwards the prompt, normalizes state, and denormalizes actions.

### Messages and lifecycle

Keep one persistent WebSocket connection. Send one JSON observation per control step:

~~~json
{
  "type": "obs",
  "images": {
    "head_camera": "<base64_png>",
    "left_wrist_camera": null,
    "right_wrist_camera": null
  },
  "prompt": "<model_prompt>",
  "state": [0.0, 0.0]
}
~~~

`head_camera` is required. Wrist fields may be omitted or null; multiview servers black-fill missing slots and single-view servers ignore them. Encode RGB as base64 PNG. Do not resize unless the training reader pre-resized a tile.

The response is a flat action array in physical units:

~~~json
{
  "type": "action",
  "action": [0.1, -0.2],
  "step": 4,
  "latency_ms": 12.3
}
~~~

Reset at the start of every episode and after a hard control failure:

~~~json
{"type": "reset"}
~~~

~~~json
{"type": "reset_ack"}
~~~

Reset clears buffered chunks and in-flight asynchronous inference; it does not reload weights. `ping` returns a `pong` response. Malformed messages return an error object.

Synchronous and asynchronous execution use the same wire messages. The server may serve buffered action chunks without rerunning denoising on every step.

### Convert actions at the boundary

Read the returned list in the exact order documented by the reader. Apply only benchmark-specific conversion, such as rot6d-to-quaternion or EEF-to-OSC delta; the server response is already in physical units. Preserve the training action conventions.

Add an adapter under `benchmarks/<name>/`, document its payload and lifecycle, and keep action conversion next to the client boundary so training and deployment conventions remain visible together.
