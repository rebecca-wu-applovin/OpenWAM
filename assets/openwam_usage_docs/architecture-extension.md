# Extending OpenWAM architectures and backbones

OpenWAM keeps orchestration in an architecture and component behavior in backbones. The dependency direction is one way:

~~~text
architecture  ──calls──>  video / action backbones
~~~

A backbone must not import or call an architecture. An architecture selects backbones through registries, composes their public contracts, and owns the forward pass.

Source locations:

- Architectures and shared lifecycle: [openwam/model/architectures/](../../openwam/model/architectures/)
- Video backbones: [openwam/model/video_backbone/](../../openwam/model/video_backbone/)
- Action backbones: [openwam/model/action_backbone/](../../openwam/model/action_backbone/)
- Hydra model groups: [configs/model/](../../configs/model/)

The current registry keys are:

~~~text
# Video backbone
wan22_ti2v_5b
cosmos3_edge

# Architecture
dual_system_cross_attn
dual_system_self_attn
~~~

Architecture keys are resolved from `architecture.framework` and `architecture.variant`. A new registration should declare its current support status and fail clearly when a required capability is unavailable.

## Add a video DiT backbone

Subclass `VideoBackbone` in [openwam/model/video_backbone/base.py](../../openwam/model/video_backbone/base.py), implement the execution lifecycle, and register the class in [openwam/model/video_backbone/__init__.py](../../openwam/model/video_backbone/__init__.py).

~~~python
from openwam.model.video_backbone import VideoBackbone, BlockLoopState, register_video_backbone

@register_video_backbone('my_video')
class MyVideoBackbone(VideoBackbone):
    # Construction and geometry
    @classmethod
    def from_pretrained(cls, source, **kwargs): ...
    @property
    def dim(self) -> int: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def scheduler(self): ...

    # Video and text IO (the base class runs the shared preprocessing flow on top of these)
    def encode_pixels(self, pixels): ...        # (N, 3, T, H, W) in [-1, 1] -> (N, z, 1 + (T-1)/tc, h, w)
    def decode_latents(self, latents): ...      # (N, z, T_lat, h, w) -> (N, 3, T, H, W) in [-1, 1]
    def encode_text(self, prompts, *, grid, num_frames) -> TextCondition: ...

    # Block loop
    # layout = cfg.common.layout; proprio (B, H, D) is always passed, only a proprio_input=backbone_stream family reads it
    def prepare(self, *, latents, timestep, text, layout, proprio,
                use_gradient_checkpointing, use_gradient_checkpointing_offload) -> BlockLoopState: ...
    def run_block(self, block_id, state, attn_mask): ...  # video-only layer with this layer's token mask
    def finalize(self, state): ...
~~~

The base class implements `to_pixels`, `encode_frames` / `encode_clip`, `preprocess_condition` (deploy),
`preprocess_train` (training) and `decode_future` once on top of `encode_pixels` / `decode_latents` / `encode_text`;
all of them take the latent layout (`cfg.common.layout`, derived by `openwam.config.resolve_config`).

The execution order is `prepare → run_block → finalize`. `prepare` returns a `BlockLoopState` containing hidden tokens, timestep data, RoPE, context, and grid metadata. `finalize` returns video noise with shape `(B, C, T, H, W)`. Preserve gradient-checkpointing data in the state.

Expose structural metadata used by token geometry and masks:

~~~python
# openwam/model/video_backbone/<my_video>.py
self._dit_patch_size = (1, 2, 2)  # temporal, height, width
self._temporal_compression = 4
self._causal_temporal = True
self._shift_video = 5.0            # scheduler shift, when supported
~~~

Implement only the capability hooks that the backbone supports:

~~~text
# MoT
pre_attn_at_layer / post_attn_at_layer

# Masks and deployment
build_joint_attention_masks   # only when the token order is not frame-major (views, extra tokens, prefix keys)
build_text_cache / save_deploy_assets
~~~

Add a matching Hydra group under [configs/model/video_backbone/](../../configs/model/video_backbone/):

~~~yaml
# configs/model/video_backbone/my_video.yaml
name: my_video                         # registry key
model_path: /path/to/video_backbone_checkpoint  # weights or bundle root
~~~

Use `build_video_backbone('my_video', cfg)` to instantiate it. Keep architecture code on the public backbone contract and do not reach into implementation details such as `backbone.dit`.

## Add an action backbone

The base contracts are in [openwam/model/action_backbone/base.py](../../openwam/model/action_backbone/base.py):

~~~text
# DualSystem
ActionDiTBackbone: num_layers, num_heads, head_dim, bridge forward,
                  prepare_state, pre_attn_at_layer, post_attn_at_layer,
                  extract_prediction
~~~

There is no separate action-backbone registry. Each architecture constructs its action implementation directly. Keep the constructor compatible with the matching group under [configs/model/action_backbone/](../../configs/model/action_backbone/).

~~~yaml
# configs/model/action_backbone/separate_action_dit.yaml
dim: 1024                         # ActionDiT hidden width
ffn_dim: 4096                     # feed-forward width
shift_action: 5.0                 # action scheduler shift
# num_layers, video_dim, num_heads, and attn_head_dim resolve from video geometry
~~~

`ActionDiT` expects one `bridge_layers` entry per action block and an even positive `attn_head_dim`. MoT requires video and action layer/head geometry to match. Keep actions and predictions shaped `(B, T_action, action_dim)` and preserve scheduler and `shift_action` behavior through the base interface.

## Add an architecture

Subclass `BaseWAMArchitecture` in [openwam/model/architectures/base.py](../../openwam/model/architectures/base.py) and register a canonical name in the relevant [openwam/model/architectures/](../../openwam/model/architectures/) package:

~~~python
from openwam.model.architectures import BaseWAMArchitecture, register_architecture

@register_architecture(
    'my_architecture',
    framework='my_framework',
    variant='default',
)
class MyArchitecture(BaseWAMArchitecture):
    def __init__(self, cfg, *, layout, mode, ckpt_dir, train_options):
        super().__init__(cfg, layout=layout, mode=mode, ckpt_dir=ckpt_dir, train_options=train_options)
        self.action_backbone = ...

    def forward(self, noisy_actions, action_timestep, *, latents, timestep, text, proprio, proprio_mask,
                use_gradient_checkpointing, use_gradient_checkpointing_offload):
        video_text, action_text = self.build_contexts(text, proprio, proprio_mask)
        ...  # -> (video_noise, action_noise_or_none)
~~~

The base class owns the training step (`compute_loss(batch)` on the collated `Batch`) and inference
(`sample(hist_latents, text, proprio, schedule, seed=...)`); the forward method returns
`(video_noise_pred, action_noise_pred_or_none)`. `build_contexts` applies `proprio_input`: ActionDiT always reads
text + proprio tokens, the video backbone the same context (`context_token`) or the text alone (`backbone_stream`). Use public backbone methods, set architecture-owned trainable modules explicitly, and fail clearly when a required capability is absent. Import the implementation from the family `__init__.py` so the decorator runs.

Add a model YAML with `architecture.framework` and `architecture.variant`. [openwam/model/architectures/registry.py](../../openwam/model/architectures/registry.py) maps that pair to the canonical name. Keep mask rules in the architecture. A backbone should expose geometry and mask builders as methods and must not inspect architecture configuration or import an architecture module.

For runtime settings, see [Training and deployment](train-and-deploy.md). For dataset and client adapters, see [Benchmark integration](benchmark-integration.md).
