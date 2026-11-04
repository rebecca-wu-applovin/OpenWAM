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
# Video backbone (configs/model/video_backbone/<key>.yaml; one directory per family under openwam/model/video_backbone/)
wan22_ti2v_5b  longlive2_robot_s        # wan/      (wan22_ti2v_tiny is a debug yaml of wan22_ti2v_5b)
cosmos3_edge   cosmos3_nano             # cosmos3/
ge_base_fast                            # ge/
flux3_action_base                       # flux/

# Architecture ((framework, variant) as the model yaml spells it)
dual_system / joint_cross_attn
dual_system / joint_self_attn
~~~

Architecture classes are registered under and resolved from the pair `architecture.framework`, `architecture.variant`. A new registration should fail clearly when a required capability is unavailable.

## Add a video DiT backbone

Adding a backbone = **implement the 13 abstract members of `VideoBackbone` + write one yaml**. Everything else
(pixel conversion, history / future encoding, text of the latent clip, decoding, frame-major attention masks) is
implemented once in [openwam/model/video_backbone/base.py](../../openwam/model/video_backbone/base.py) on top of them.

1. Create `openwam/model/video_backbone/<family>/backbone.py` (plus a short `README.md`: upstream source / commit,
   the yamls, anything vendored) and subclass `VideoBackbone`. Keep the method order of the other families:
   construction → geometry → video and text IO → block loop → deploy assets.

~~~python
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone

class MyVideoBackbone(VideoBackbone):
    # Construction: 2 abstract classmethods + __init__. cfg = cfg.model.video_backbone, assets = cfg.assets.video_backbone
    # (upstream paths), both read-only; the entry point picks the constructor (LoadMode).
    def __init__(self, ...):                     # set _causal_temporal, _temporal_compression and
        self._scheduler = FlowScheduler(shift=shift_video, train_timestep=train_timestep)  # base `scheduler` returns it
    @classmethod
    def from_pretrained(cls, cfg, assets): ...   # upstream weights at the assets paths, checked against the yaml geometry
    @classmethod
    def from_checkpoint_dir(cls, cfg, ckpt_dir): ...  # meta-device shell of one of our checkpoints; the caller loads weights

    # Geometry (5 properties)
    dim, num_layers, num_heads, head_dim, text_dim  # ints

    # Video and text IO (3)
    def encode_pixels(self, pixels): ...         # (N, 3, T, H, W) in [-1, 1] -> (N, z, 1 + (T-1)/tc, h, w) normalized
    def decode_latents(self, latents): ...       # (N, z, T_lat, h, w) -> (N, 3, T, H, W) in [-1, 1]
    def encode_text(self, prompts, *, grid, num_frames) -> TextCondition: ...  # context (B, L, text_dim) + mask (B, L)

    # Block loop (3). layout = cfg.common.layout; proprio (B, H, D) is always passed
    def prepare(self, *, latents, timestep, text, layout, proprio,
                use_gradient_checkpointing, use_gradient_checkpointing_offload) -> BlockLoopState: ...
    def run_block(self, block_id, state, attn_mask) -> BlockLoopState: ...  # video-only layer with this layer's mask
    def finalize(self, state): ...               # -> (B, z, T_lat, h, w) predicted velocity
~~~

2. Register it at the bottom of [openwam/model/video_backbone/__init__.py](../../openwam/model/video_backbone/__init__.py):
   `register_video_backbone("my_video")(MyVideoBackbone)`.

3. Write `configs/model/video_backbone/my_video.yaml` (`# @package _global_`): under `model.video_backbone`,
   `name: my_video`, the module geometry (checked against the weights at build), `temporal_compression`,
   `shift_video`, `train_timestep` (the training noise law: `{law: grid, weighting: bell | uniform}`,
   `{law: logit_logistic, width, shift}` or `{law: ge_native}`), and what the backbone requires (`required_clean_attention`: null or a value,
   `required_proprio_input`; `resolve_config` checks them once); under `assets.video_backbone`, the upstream weight
   paths (not compared on finetune); under `training`, `freeze` (paths that exist on the model) and `text_cache`.

Optional hooks have working defaults; override only what the family supports:

~~~text
pre_attn_at_layer / post_attn_at_layer   # joint_self_attn (MoT); default raises
build_joint_attention_masks              # only when the token order is not frame-major (views, extra tokens, prefix keys)
build_text_cache                         # training.text_cache: true; base caches text_lookup (needs text_lookup / text_encoder /
                                         # _text_cache_batch_size); override to raise if the text cannot be cached
save_deploy_assets                       # non-weight files (tokenizer, configs) copied into the checkpoint; default none
set_dtype_device
~~~

Never override the shared flow (`to_pixels`, `encode_frames`, `encode_clip`, `preprocess_condition`,
`preprocess_train`, `decode_future`): `tests/test_video_backbone_base.py::test_families_do_not_override_base_flow`
checks every registered family. Build it through `build_video_backbone(cfg.model.video_backbone, mode, assets, ckpt_dir)`
(the architecture does this); architecture code stays on the public contract and never reaches into `backbone.dit`.

## Add an action backbone

The action stream lives in [openwam/model/action_backbone/separate_action_dit.py](../../openwam/model/action_backbone/separate_action_dit.py): one class per coupling on a shared base.

~~~text
ActionDiT (shared base): scheduler, num_layers, num_heads, head_dim, set_dtype_device, save_deploy_assets,
                         prepare_state(noisy_actions, timestep, context=, context_mask=) -> ActionDiTState
CrossAttnActionDiT (joint_cross_attn): forward(actions, bridges: list (one per block), timestep, context=, ...)
MoTActionDiT (joint_self_attn):        pre_attn_at_layer, post_attn_at_layer, extract_prediction (MoT driver)
~~~

There is no action-backbone registry and no variant argument: each architecture names its ActionDiT class (passed to
`BaseWAMArchitecture.__init__` as `action_dit_cls`), and that class's `from_cfg(ab, action_dim=, num_layers=)` reads
the matching group under [configs/model/action_backbone/](../../configs/model/action_backbone/).

~~~yaml
# configs/model/action_backbone/separate_action_dit.yaml
dim: 1024                         # ActionDiT hidden width
ffn_dim: 4096                     # feed-forward width
shift_action: 5.0                 # action scheduler shift
freq_dim: 256                     # timestep embedding width
max_action_len: 1024              # longest action chunk (action RoPE table length)
eps: 1.0e-06                      # norm epsilon
num_heads: ${..video_backbone.geometry.num_heads}   # geometry: the video backbone yaml's geometry block
head_dim: ${..video_backbone.geometry.head_dim}
video_dim: ${..video_backbone.geometry.dim}
text_dim: ${..video_backbone.geometry.text_dim}
~~~

Every video backbone yaml declares `model.video_backbone.geometry` (`dim`, `num_heads`, `head_dim`, `num_layers`, `text_dim`); the architecture checks it against the built module and raises on any difference. `CrossAttnActionDiT` has one block per bridge layer, `range(0, geometry.num_layers, model.architecture.bridge_interval)`; `MoTActionDiT` has one block per video layer (`bridge_interval` must be 1, checked by `resolve_config`). Both need an even positive `head_dim`; MoT requires video and action head geometry to match. Keep actions and predictions shaped `(B, T_action, action_dim)` and preserve scheduler and `shift_action` behavior through the base interface.

## Add an architecture

Subclass `BaseWAMArchitecture` in [openwam/model/architectures/base.py](../../openwam/model/architectures/base.py) and register its `(framework, variant)` pair in the relevant [openwam/model/architectures/](../../openwam/model/architectures/) package. The base builds the video backbone, the proprio context encoder and the action backbone (in that order); the subclass names the ActionDiT class and its block count:

~~~python
from openwam.model.action_backbone import MoTActionDiT
from openwam.model.architectures import BaseWAMArchitecture, register_architecture

@register_architecture(framework='my_framework', variant='default')
class MyArchitecture(BaseWAMArchitecture):
    # cfg is merge(cfg.common, cfg.model) (layout at cfg.layout); assets = cfg.assets.video_backbone
    def __init__(self, cfg, *, mode, assets, ckpt_dir, train_options):
        super().__init__(
            cfg, mode=mode, assets=assets, ckpt_dir=ckpt_dir, train_options=train_options,
            action_dit_cls=MoTActionDiT, num_action_layers=int(cfg.video_backbone.geometry.num_layers),
        )

    def forward(self, noisy_actions, action_timestep, *, latents, timestep, text, proprio, proprio_mask,
                use_gradient_checkpointing, use_gradient_checkpointing_offload):
        # build_contexts + video_backbone.prepare, shared by every coupling
        vstate, action_text = self._prepare_video(latents, timestep, text, proprio, proprio_mask,
                                                  use_gradient_checkpointing, use_gradient_checkpointing_offload)
        ...  # -> (video_noise, action_noise)
~~~

The base class owns the training step (`compute_loss(batch)` on the collated `Batch`) and inference
(`sample(hist_latents, text, proprio, schedule, seed=...)`); the forward method returns
`(video_noise_pred, action_noise_pred)` (both streams always run). `build_contexts` applies `proprio_input`: ActionDiT always reads
text + proprio tokens, the video backbone the same context (`context_token`) or the text alone (`backbone_stream`). Use public backbone methods, set architecture-owned trainable modules explicitly, and fail clearly when a required capability is absent. Import the implementation from the family `__init__.py` so the decorator runs.

Add a model YAML with `architecture.framework` and `architecture.variant`. [openwam/model/architectures/registry.py](../../openwam/model/architectures/registry.py) maps that pair to the class. Keep mask rules in the architecture. A backbone should expose geometry and mask builders as methods and must not inspect architecture configuration or import an architecture module.

For runtime settings, see [Training and deployment](train-and-deploy.md). For dataset and client adapters, see [Benchmark integration](benchmark-integration.md).
