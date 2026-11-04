"""Tests for WAM Architecture registry and implementations."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import open_dict

from openwam.model.architectures.base import GEOMETRY_KEYS
from openwam.model.video_backbone import LoadMode
from tests._arch import (
    TEST_ASSETS,
    TEST_TRAIN_OPTIONS,
    arch_params,
    backbone_double,
    build_test_arch,
    init_test_base,
    stub_backbone,
)
from tests._layout import make_layout
from tests._training_cfg import cpu_accelerator


def _action_context(ab, batch_size: int, seq_len: int = 4):
    return torch.randn(batch_size, seq_len, ab.text_dim), torch.ones(batch_size, seq_len, dtype=torch.bool)


def _make_dual_system_self_attn_mot_fixture():
    from tests.test_openwam_trainer import _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "num_layers": 2,
    }
    # 2 history + 2 future latents = the 4 one-token frames of _dual_system_self_attn_mot_states.
    backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch = build_test_arch("joint_self_attn", cfg, backbone, layout=make_layout(h=2))
    driver = arch._mot_driver
    arch.eval()
    return arch, driver


def _make_dual_system_cross_attn_fixture():

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 48,
        "num_layers": 2,
        "text_dim": 16,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    arch.eval()
    return arch


def _dual_system_self_attn_mot_states(arch, seed: int):
    from openwam.model.video_backbone.base import BlockLoopState

    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(1, 3, 7, generator=g)
    context = torch.randn(1, 4, arch.action_backbone.text_dim, generator=g)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    astate = arch.action_backbone.prepare_state(
        actions,
        torch.tensor([0.5]),
        context=context,
        context_mask=context_mask,
    )
    vstate = BlockLoopState(
        hidden_states=torch.randn(1, 4, 32, generator=g),
        time_mod=torch.zeros(1, 6, 32),
        rope_freqs=torch.zeros(4, 1, 1),
        context=torch.randn(1, 4, 32, generator=g),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        grid_frames=4,
        grid_height=1,
        grid_width=1,
        extras={},
    )
    return vstate, astate


def _dual_system_cross_attn_inputs(arch, seed: int):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(1, 3, 7, generator=g)
    timestep = torch.tensor([0.5])
    bridges = [torch.randn(1, 4, 48, generator=g) for _ in arch.bridge_layers]
    context = torch.randn(1, 4, arch.action_backbone.text_dim, generator=g)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    return actions, bridges, timestep, context, context_mask


def test_architecture_module_layout_imports():
    from openwam.model.architectures.dual_system import (
        DualSystemCrossAttnArchitecture,
        DualSystemSelfAttnArchitecture,
    )

    assert DualSystemCrossAttnArchitecture is not None
    assert DualSystemSelfAttnArchitecture is not None


def test_architecture_state_types_import():
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState

    assert ActionDiTState is not None


def test_freeze_modules_supports_dotted_path():
    from openwam.model.action_backbone.separate_action_dit import MoTActionDiT
    from openwam.model.architectures.base import BaseWAMArchitecture

    backbone = stub_backbone({})

    class TestArch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(
                arch_params({"dim": 32, "ffn_dim": 64, "action_dim": 1}, backbone),
                mode=LoadMode.PRETRAINED,
                assets=TEST_ASSETS,
                ckpt_dir=None,
                train_options=TEST_TRAIN_OPTIONS,
                action_dit_cls=MoTActionDiT,
                num_action_layers=backbone.num_layers,
            )
            self.aux_backbone = torch.nn.Module()
            self.aux_backbone.inner = torch.nn.Linear(2, 2)

        def forward(self, noisy_actions, action_timestep, **_kw):
            return None, noisy_actions

    with backbone_double(backbone):
        arch = TestArch()
    frozen = arch.freeze_modules(["aux_backbone.inner"])

    assert frozen == ["aux_backbone.inner"]
    assert not any(p.requires_grad for p in arch.aux_backbone.inner.parameters())


def test_openwam_trainer_uses_strategy_freeze_list(monkeypatch):
    """Sub-module freeze is owned by the backbone yaml's training.freeze list."""

    from omegaconf import OmegaConf

    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.ckpt_model_loader import RunMode

    class _Arch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.device = torch.device("cpu")
            self.aux_backbone = torch.nn.Module()
            self.aux_backbone.inner = torch.nn.Linear(2, 2)
            self._freeze_calls = []

        def set_dtype_device(self, dtype, device):  # noqa: ARG002
            return None

        @property
        def backbones(self):
            return {"aux_backbone": self.aux_backbone}

        def freeze_modules(self, names):
            self._freeze_calls.append(list(names))
            for name in names:
                self.get_submodule(name).requires_grad_(False)
            return list(names)

    holder = {}

    def _fake_build(model_cfg, *, mode, assets, ckpt_dir, train_options):  # noqa: ARG001
        holder["arch"] = _Arch()
        return holder["arch"]

    monkeypatch.setattr("openwam.model.build_architecture", _fake_build)

    cfg = OmegaConf.create(
        {
            "project": {"seed": 0},
            "training": {
                "initialize_model_on_cpu": False,
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
                "lambda_video": 1.0,
                "lambda_action": 1.0,
                "mot_checkpoint_mixed_attn": False,
                "freeze": ["aux_backbone.inner"],
                "text_cache": False,
                "mixed_precision": "bf16",
            },
            "model": {"architecture": {"framework": "dual_system", "variant": "joint_self_attn"}},
            "common": {"action_dim": 7, "state_dim": 7, "layout": make_layout()},
            "assets": {"video_backbone": {"model_path": "/upstream"}},
        }
    )

    trainer = OpenWAMTrainer(cfg, cpu_accelerator(), None, RunMode.FRESH, None, rank=0)
    arch = holder["arch"]

    assert trainer.architecture is arch
    assert arch._freeze_calls == [["aux_backbone.inner"]]
    assert not any(p.requires_grad for p in arch.aux_backbone.inner.parameters())


def test_build_architecture_dual_system():

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 128,
        "ffn_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 256,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    assert arch.action_dim == 7
    assert arch.bridge_layers == (0, 1)
    assert arch.action_backbone is not None


def test_build_architecture_unknown_variant_raises():
    """The (framework, variant) pair picks the class; an unregistered pair is a KeyError naming the choices."""
    from openwam.model import build_architecture as _build

    backbone = stub_backbone({})
    cfg = arch_params({}, backbone)
    with open_dict(cfg):
        cfg.architecture.variant = "vanilla"
    with backbone_double(backbone), pytest.raises(KeyError, match="Unsupported architecture"):
        _build(cfg, mode=LoadMode.PRETRAINED, assets=TEST_ASSETS, ckpt_dir=None, train_options=TEST_TRAIN_OPTIONS)


def test_dual_system_prepare_and_extract():
    """Smoke test: cross_attn ActionDiT.forward(action, bridges, timestep, context)."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 128,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    arch.eval()

    B, T_action = 1, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    bridges = [torch.randn(B, 20, 128) for _ in arch.bridge_layers]
    context, context_mask = _action_context(arch.action_backbone, B)

    with torch.no_grad():
        action_pred = arch.action_backbone(
            noisy_actions,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
    assert action_pred.shape == (B, T_action, 7)


def test_dual_system_bridge_interval_resolves():
    """bridge_interval strides over the video layer count (geometry.num_layers)."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_interval": 2,
        "num_layers": 30,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    assert arch.bridge_layers == tuple(range(0, 30, 2))
    assert len(arch.bridge_layers) == 15

    cfg["bridge_interval"] = 1
    arch_full = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    assert arch_full.bridge_layers == tuple(range(30))


def test_bridge_layers_interval():
    """select_bridge_layers strides over the video layers; resolve_config rejects an interval below 1 by key."""
    from omegaconf import OmegaConf

    from openwam.config.resolve import _check_bridge_interval
    from openwam.model.architectures.dual_system.joint_cross_attn import select_bridge_layers

    assert select_bridge_layers(30, 2) == tuple(range(0, 30, 2))
    assert select_bridge_layers(30, 1) == tuple(range(30))
    with pytest.raises(ValueError, match=r"model\.architecture\.bridge_interval must be >= 1"):
        _check_bridge_interval(OmegaConf.create({"variant": "joint_cross_attn", "bridge_interval": 0}))


@pytest.mark.parametrize("key", GEOMETRY_KEYS)
def test_geometry_mismatch_raises(key):
    """A yaml geometry block that disagrees with the built backbone in ``key`` fails at build, naming only it."""
    from openwam.model import build_architecture
    from tests._arch import TEST_ASSETS, TEST_TRAIN_OPTIONS, arch_params, backbone_double

    backbone = stub_backbone({})
    params = arch_params({"variant": "joint_cross_attn"}, backbone)
    params.video_backbone.geometry[key] = getattr(backbone, key) * 2  # a stale yaml value
    with backbone_double(backbone), pytest.raises(ValueError, match=f"'{key}'") as err:
        build_architecture(
            params, mode=LoadMode.PRETRAINED, assets=TEST_ASSETS, ckpt_dir=None, train_options=TEST_TRAIN_OPTIONS
        )
    # Only the differing key is listed (keys appear quoted in the diff dict, so 'dim' does not match 'head_dim').
    assert all(f"'{other}'" not in str(err.value) for other in GEOMETRY_KEYS if other != key)


def test_action_self_attention_rope_breaks_permutation_equivariance():
    """Without positional info, self-attention is permutation-equivariant.
    RoPE injects absolute position into Q/K, so permuting the input tokens
    must NOT merely permute the output (the model distinguishes positions).
    Also: real RoPE freqs must change the output relative to a no-rotation table (all ones).
    """
    from openwam.model.action_backbone.components import precompute_freqs_cis_1d
    from openwam.model.action_backbone.separate_action_dit import ActionSelfAttention

    dim, num_heads, seq = 32, 4, 4
    head_dim = dim // num_heads
    attn = ActionSelfAttention(hidden_dim=dim, num_heads=num_heads, attn_head_dim=head_dim, eps=1e-6).eval()

    x = torch.randn(1, seq, dim)
    freqs = precompute_freqs_cis_1d(head_dim, max_len=seq)
    # Unit complex numbers at angle 0: RoPE then rotates nothing (position-free attention).
    angle0 = torch.zeros(seq, head_dim // 2, dtype=torch.float64)
    no_rotation = torch.polar(torch.ones_like(angle0), angle0)
    perm = torch.tensor([3, 1, 2, 0])  # non-identity permutation
    x_perm = x[:, perm, :]

    with torch.no_grad():
        out_plain = attn(x, freqs=no_rotation)
        out_plain_perm = attn(x_perm, freqs=no_rotation)
        out_rope = attn(x, freqs=freqs)
        out_rope_perm = attn(x_perm, freqs=freqs)

    # Sanity: the no-rotation table is permutation-equivariant.
    assert torch.allclose(out_plain_perm, out_plain[:, perm, :], atol=1e-5)
    # RoPE must break that symmetry (content same, positions shuffled → non-permute-equivalent output).
    assert not torch.allclose(out_rope_perm, out_rope[:, perm, :], atol=1e-5), (
        "RoPE had no effect: permuted output equals permuted-input output"
    )
    # And RoPE output must differ from no-freqs output on the same input.
    assert not torch.allclose(out_rope, out_plain, atol=1e-5)


def test_action_dit_joint_cross_attn_uses_rope():
    """ActionDiT joint_cross_attn runs end-to-end with RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.separate_action_dit import CrossAttnActionDiT

    dit = CrossAttnActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        attn_head_dim=16,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    assert not hasattr(dit, "pos_encoding")
    assert hasattr(dit, "freqs") and dit.freqs.shape == (1024, 64 // 4 // 2)

    actions = torch.randn(2, 5, 7)
    bridges = [torch.randn(2, 10, 128) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])
    context, context_mask = _action_context(dit, 2)
    out = dit(
        actions, bridges, timestep, context=context, context_mask=context_mask,
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    )
    assert out.shape == (2, 5, 7)


def test_action_dit_rope_freqs_preserve_complex_dtype_on_dtype_to():
    """ActionDiT RoPE cache must not be cast to real by dtype-only Module.to()."""
    from openwam.model.action_backbone.separate_action_dit import CrossAttnActionDiT

    dit = CrossAttnActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        attn_head_dim=16,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    original_dtype = dit.freqs.dtype

    dit.to(dtype=torch.bfloat16)

    assert dit.freqs.dtype == original_dtype
    assert dit.freqs.is_complex()
    assert "freqs" not in dit.state_dict()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ActionDiT RoPE device migration check")
def test_action_dit_rope_freqs_follow_cuda_to_without_dtype_cast():
    """ActionDiT.to(cuda, bf16) should move RoPE freqs to CUDA but keep them complex."""
    from openwam.model.action_backbone.separate_action_dit import MoTActionDiT

    dit = MoTActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        text_dim=16,
        attn_head_dim=16,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    ).to(device="cuda", dtype=torch.bfloat16)

    assert dit.freqs.device.type == "cuda"
    assert dit.freqs.is_complex()
    assert dit.freqs.dtype.is_complex
    assert "freqs" not in dit.state_dict()

    context = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
    context_mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    astate = dit.prepare_state(
        torch.randn(2, 5, 7, device="cuda", dtype=torch.bfloat16),
        torch.tensor([0.5, 0.8], device="cuda", dtype=torch.bfloat16),
        context=context,
        context_mask=context_mask,
    )
    assert astate.action_freqs.device.type == "cuda"
    assert astate.action_freqs.is_complex()


def test_action_dit_joint_self_attn_uses_only_rope():
    """joint_self_attn relies solely on RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.separate_action_dit import MoTActionDiT

    dit = MoTActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        attn_head_dim=16,
        text_dim=4096,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    )
    assert not hasattr(dit, "pos_encoding")

    context, context_mask = _action_context(dit, 2)
    astate = dit.prepare_state(
        torch.randn(2, 5, 7), torch.tensor([0.5, 0.8]), context=context, context_mask=context_mask
    )
    # RoPE freqs on the action stream are pre-computed and stashed for
    # consumption by pre_attn_at_layer; their length is the action sequence the attention sees.
    assert astate.action_freqs.shape[0] == astate.x_action.shape[1]
    assert astate.action_freqs.device == astate.x_action.device


def test_dual_system_joint_self_attn_pre_post_attn_round_trip():
    """joint_self_attn pre/post_attn_at_layer round trip applies RoPE on Q/K."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "num_layers": 1,
    }
    arch = build_test_arch("joint_self_attn", cfg, stub_backbone(cfg), layout=make_layout())
    arch.eval()

    noisy_actions = torch.randn(2, 5, 7)
    timestep = torch.tensor([0.5, 0.8])
    context, context_mask = _action_context(arch.action_backbone, 2)
    astate = arch.action_backbone.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

    q, k, v, post = arch.action_backbone.pre_attn_at_layer(0, astate)
    # Q/K go through RoPE (different from V even when the same input feeds q/k/v):
    assert not torch.allclose(q, v)
    assert not torch.allclose(k, v)
    # Q/K/V are in (B, S, H*D) layout, ready for the driver to concat with
    # the video stream.
    B, S = noisy_actions.shape[0], noisy_actions.shape[1]
    assert q.shape == (B, S, arch.action_backbone.num_heads * arch.action_backbone.head_dim)

    # Round-trip through post_attn so we know the slot is wired.
    astate2 = arch.action_backbone.post_attn_at_layer(0, astate, torch.randn_like(q), post)
    assert astate2 is astate
    pred = arch.action_backbone.extract_prediction(astate2)
    assert pred.shape == (2, 5, 7)


def test_dual_system_joint_cross_attn_no_per_layer_state():
    """joint_cross_attn skips prepare_state and runs ActionDiT.forward directly."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "num_layers": 2,  # every layer of the 2-layer stub backbone
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())

    # cross_attn calls ab.forward(actions, bridges, timestep) directly —
    # no per-layer state machinery on the action backbone.
    bridges = [torch.randn(1, 9, 128) for _ in arch.bridge_layers]
    context, context_mask = _action_context(arch.action_backbone, 1)
    with torch.no_grad():
        out = arch.action_backbone(
            torch.randn(1, 5, 7),
            bridges,
            torch.tensor([0.5]),
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
    assert out.shape == (1, 5, 7)


def test_dual_system_joint_cross_attn_passes_appended_proprio_context_to_action():
    """cross_attn architecture should pass raw text+proprio context to ActionDiT."""
    from openwam.model.video_backbone.base import TextCondition
    from tests.test_openwam_trainer import _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "num_layers": 1,
        "use_proprioception": True,
        "state_dim": 7,
        "text_dim": 16,
    }
    backbone = _MockVideoBackbone(dim=32, num_layers=1, num_heads=4, text_dim=16)
    arch = build_test_arch("joint_cross_attn", cfg, backbone)
    arch.eval()

    seen = {}
    orig_forward = arch.action_backbone.forward

    def _capture(*args, **kwargs):
        seen["context"] = kwargs.get("context")
        seen["context_mask"] = kwargs.get("context_mask")
        return orig_forward(*args, **kwargs)

    arch.action_backbone.forward = _capture

    B = 1
    with torch.no_grad():
        video_pred, action_pred = arch(
            torch.randn(B, 5, 7),
            torch.tensor([0.5]),
            proprio=torch.randn(B, 1, 7),
            proprio_mask=torch.ones(B, 1, 7, dtype=torch.bool),
            latents=torch.randn(B, 16, 3, 2, 2),
            timestep=torch.tensor([0.5]),
            text=TextCondition(context=torch.randn(B, 3, 16), context_mask=torch.tensor([[True, True, False]])),
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )

    assert video_pred.shape[0] == B
    assert action_pred.shape == (B, 5, 7)
    assert seen["context"].shape == (B, 4, 16)
    assert seen["context_mask"].shape == (B, 4)
    assert seen["context_mask"].tolist() == [[True, True, False, True]]


def test_dual_system_detached_joint_cross_attn_blocks_grad_to_video():
    """detach_bridge=True: action loss must not produce grad on the bridge tensors."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "num_layers": 1,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    # detach is owned by the architecture; the action backbone has no such flag.
    assert arch._detach_bridge is True
    assert "detach_bridge" not in dir(arch.action_backbone)


def test_action_dit_cross_attn_bridges_one_per_block():
    """The bridge list is block-ordered, one per block: a short list or a wrong width raises, a right one runs."""

    arch = _make_dual_system_cross_attn_fixture()
    ab = arch.action_backbone
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 7)
    kw = dict(context=context, context_mask=context_mask, use_gradient_checkpointing=False,
              use_gradient_checkpointing_offload=False)

    with torch.no_grad():
        assert ab(actions, bridges, timestep, **kw).shape == actions.shape
        with pytest.raises(ValueError, match="2 action blocks"):
            ab(actions, bridges[:1], timestep, **kw)
        with pytest.raises(ValueError, match="video_dim=48"):
            ab(actions, [b[..., :8] for b in bridges], timestep, **kw)


def test_dual_system_joint_self_attn_creates_dit_state():
    """joint_self_attn: prepare_state builds the ActionDiTState the driver threads."""

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "num_layers": 2,
    }
    arch = build_test_arch("joint_self_attn", cfg, stub_backbone(cfg), layout=make_layout())

    context, context_mask = _action_context(arch.action_backbone, 1)
    state = arch.action_backbone.prepare_state(
        torch.randn(1, 5, 7), torch.tensor([0.5]), context=context, context_mask=context_mask
    )
    assert state.x_action.shape == (1, 5, 32)
    assert state.action_freqs.shape == (5, arch.action_backbone.head_dim // 2)


def test_build_architecture_injects_framework_variant_and_detach_option():

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "num_layers": 2,
    }
    arch = build_test_arch("joint_cross_attn", cfg, stub_backbone(cfg), layout=make_layout())
    from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture

    # (framework, variant) picked the class; detach_bridge reached the architecture.
    assert type(arch) is DualSystemCrossAttnArchitecture
    assert arch._detach_bridge is True


def test_dual_system_self_attn_state_is_action_dit_state():
    """joint_self_attn: prepare_state returns an ActionDiTState (no wrapper)."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState

    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "num_layers": 2,
    }
    arch = build_test_arch("joint_self_attn", cfg, stub_backbone(cfg), layout=make_layout())
    context, context_mask = _action_context(arch.action_backbone, 1)
    state = arch.action_backbone.prepare_state(
        torch.randn(1, 5, 7), torch.tensor([0.5]), context=context, context_mask=context_mask
    )
    assert isinstance(state, ActionDiTState)


def test_register_custom_architecture():
    """Verify that custom architectures can be registered under a (framework, variant) pair."""
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.architectures.registry import ARCHITECTURE_REGISTRY, register_architecture

    @register_architecture(framework="test", variant="custom")
    class TestArch(BaseWAMArchitecture):
        def forward(self, noisy_actions, action_timestep, **_kw):
            return None, noisy_actions

    assert ARCHITECTURE_REGISTRY[("test", "custom")] is TestArch
    # A pair registers once.
    with pytest.raises(ValueError, match="already registered"):
        register_architecture(framework="test", variant="custom")(TestArch)

    # Cleanup
    del ARCHITECTURE_REGISTRY[("test", "custom")]


def test_freeze_modules_disables_grad_and_wraps_forward_in_no_grad():
    """``freeze_modules`` is the single API for freezing: it both sets
    ``requires_grad=False`` AND wraps the named module's forward in
    ``torch.no_grad`` so the subtree never builds a backward graph.

    This is the contract every backbone and trainer relies on — there is no
    backbone-side freeze detection in OpenWAM.
    """
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            init_test_base(self)
            self.frozen_mod = nn.Linear(3, 4)
            self.trainable_mod = nn.Linear(3, 4)

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    frozen = arch.freeze_modules(["frozen_mod"])
    assert frozen == ["frozen_mod"]

    # 1) Params no longer require grad
    assert not any(p.requires_grad for p in arch.frozen_mod.parameters())
    # Trainable module untouched
    assert all(p.requires_grad for p in arch.trainable_mod.parameters())

    # 2) Forward wraps in no_grad — output does not require grad, even with grad-tracking inputs.
    x = torch.randn(2, 3, requires_grad=True)
    assert arch.frozen_mod(x).requires_grad is False
    # Trainable module still builds graph
    assert arch.trainable_mod(x).requires_grad is True

    # 3) Idempotent — second call doesn't double-wrap
    arch.freeze_modules(["frozen_mod"])
    out = arch.frozen_mod(x)
    assert out.requires_grad is False
    assert arch.frozen_mod._openwam_no_grad_wrapped is True


def test_freeze_unknown_path_raises():
    """A training.freeze path absent on the model is a config error, not a silent skip."""
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            init_test_base(self)
            self.real = nn.Linear(2, 2)

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    assert arch.freeze_modules(["real"]) == ["real"]
    with pytest.raises(ValueError, match=r"training\.freeze: 'video_backbone\.no_such'"):
        arch.freeze_modules(["video_backbone.no_such"])


def test_freeze_modules_wraps_all_descendants_in_no_grad():
    """``freeze_modules`` must wrap forward on every submodule in the frozen subtree,
    so calls that bypass the root (e.g. a frozen HF model whose inner ``.model``
    is called to skip the LM head) also see ``no_grad``. Without recursive wrap, the bypass
    silently leaves the frozen subtree grad-tracking and the activation memory
    isn't reclaimed.
    """
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 4)

        def forward(self, x):
            return self.lin(x)

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Inner()
            self.head = nn.Linear(4, 5)

        def forward(self, x):
            return self.head(self.inner(x))

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            init_test_base(self)
            self.outer = _Outer()

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    arch.freeze_modules(["outer"])
    x = torch.randn(2, 3, requires_grad=True)
    # Top-level call: wrapped
    assert arch.outer(x).requires_grad is False
    # Nested submodule called directly (bypasses outer.forward): also wrapped
    assert arch.outer.inner(x).requires_grad is False
    # Leaf submodule called directly: also wrapped
    assert arch.outer.inner.lin(x).requires_grad is False


def test_freeze_parent_blocks_trainable_child_grad():
    """Document limitation: freezing a parent freezes ALL descendants.

    _wrap_forward_in_no_grad is subtree-level: a trainable child registered
    under a frozen parent will NOT receive gradients. This is by design —
    partial-freeze (e.g. LoRA on a frozen base) requires freezing specific
    leaves, not the parent. See base.py:75-81.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            init_test_base(self)
            self.trunk = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
            # Register a trainable adapter UNDER the trunk
            self.trunk.adapter = torch.nn.Linear(8, 4)

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _Arch()
    assert any(p.requires_grad for p in arch.trunk.adapter.parameters()), "adapter should start trainable"

    arch.freeze_modules(["trunk"])

    x = torch.randn(1, 8, requires_grad=True)
    out = arch.trunk.adapter(x)
    # Adapter under frozen parent does NOT track grad — documented limitation
    assert not out.requires_grad, (
        "Expected trainable child under frozen parent to NOT track grad. "
        "This is by design; use leaf-level freeze for partial-freeze setups."
    )


def _video_scheduler():
    """Returns: the real video FlowScheduler (shift 5, grid / bell) the denoise loop steps the video latents with."""
    from openwam.model.common.flow_scheduler import FlowScheduler

    return FlowScheduler(shift=5.0, train_timestep={"law": "grid", "weighting": "bell"})


def test_sample_frozen_lag_stream_stays_in_forward_context():
    """A sigma plateau (linear_offset delay) freezes a stream's *update* only: its
    tokens/timestep must stay in every forward, because v→a-visible attention
    masks (mutual) read them."""

    from openwam.model.architectures.base import BaseWAMArchitecture

    forward_inputs = []
    action_steps = []

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def step(model_output, sigma, sigma_next, sample):
            action_steps.append((sigma, sigma_next))
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _video_scheduler()

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = 3

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            # First-frame layout: 1 clean + 2 noisy latents, a 3-step action chunk.
            init_test_base(self, layout=make_layout(num_frames=4))
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()

        def forward(self, noisy_actions, action_timestep, **kwargs):
            forward_inputs.append((noisy_actions is not None, float(action_timestep.item())))
            return torch.zeros_like(kwargs["latents"]), torch.zeros_like(noisy_actions)

    # linear_offset=0.5 shape: sigma_a plateaus at 1.0 for the first
    # two transitions, then catches up.
    schedule = [(1000.0, 1000.0), (750.0, 1000.0), (500.0, 1000.0), (250.0, 500.0), (0.0, 0.0)]
    _, actions = _Arch().sample(torch.zeros(1, 1, 1, 1, 1), _text(), torch.zeros(1, 1, 3), schedule, seed=0)

    # Every step is a joint forward with action tokens present; the frozen
    # head rides t_a=1000 (the training grid's sigma=1 endpoint).
    assert forward_inputs == [(True, 1000.0), (True, 1000.0), (True, 1000.0), (True, 500.0)]
    # The plateau gates only the update: the action stream steps once it moves.
    assert action_steps == [(1.0, 0.5), (0.5, 0.0)]
    assert actions.shape == (1, 3, 3)


_PIN_ACTION_DIM = 8
_PIN_ACTIVE_IDX = [0, 2, 5]  # non-contiguous, mimics a unify_action scatter map
_PIN_INACTIVE_IDX = [1, 3, 4, 6, 7]


def _make_inactive_dim_probe_arch():
    """Fake arch whose forward emits huge garbage flow on the non-active action
    channels — an adversarial stand-in for the unsupervised padding-dim flow of
    a unify_action checkpoint. Records every action forward input."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def step(model_output, sigma, sigma_next, sample):
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _Scheduler()

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = _PIN_ACTION_DIM

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            # First-frame layout: 1 clean + 2 noisy one-pixel latents, a 4-step action chunk.
            init_test_base(self, layout=make_layout(num_frames=5))
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()
            self.seen_action_inputs = []

        def forward(self, noisy_actions, action_timestep, **kwargs):  # noqa: ARG002
            self.seen_action_inputs.append(noisy_actions.detach().clone())
            pred = torch.empty_like(noisy_actions)
            pred[..., _PIN_ACTIVE_IDX] = 1.0
            pred[..., _PIN_INACTIVE_IDX] = 1000.0
            return torch.ones_like(kwargs["latents"]), pred

    return _Arch()


# Shapes sample() draws for the pin tests (from the probe arch's layout): 1 clean + 2 noisy one-pixel latents,
# a 4-step chunk.
_PIN_LATENT_SHAPE = (1, 1, 3, 1, 1)
_PIN_ACTION_SHAPE = (1, 4, _PIN_ACTION_DIM)


def _text():
    """A one-token TextCondition (the probe forwards ignore it). Returns: TextCondition (1, 1, 1)."""
    from openwam.model.video_backbone.base import TextCondition

    return TextCondition(context=torch.zeros(1, 1, 1), context_mask=torch.ones(1, 1, dtype=torch.bool))


def _probe_actions(arch, schedule, **kw):
    """Run ``arch.sample`` on the pin-test shapes (seed 7). Returns: (4, A) numpy actions (normalized space)."""
    _, actions = arch.sample(torch.zeros(1, 1, 1, 1, 1), _text(), torch.zeros(1, 1, 3), schedule, seed=7, **kw)
    return actions[0].numpy()


def test_sample_pins_inactive_action_dims_on_noise_path():
    """Unsupervised (non-scattered) unified-action dims must ride the analytic
    sigma * eps0 noise path through the denoising loop instead of integrating
    unconstrained flow back into the next forward."""
    import numpy as np

    schedule = [(1000.0, 1000.0), (750.0, 750.0), (500.0, 500.0), (250.0, 250.0), (0.0, 0.0)]
    sigmas = [t_a / 1000.0 for _, t_a in schedule]
    # sample() draws the video noise, then the action noise, from one seeded generator.
    gen = torch.Generator(device="cpu").manual_seed(7)
    torch.randn(_PIN_LATENT_SHAPE, generator=gen)
    eps0 = torch.randn(_PIN_ACTION_SHAPE, generator=gen)

    baseline = _probe_actions(_make_inactive_dim_probe_arch(), schedule)
    # Unpinned, the garbage flow drags inactive dims far off the noise path.
    assert np.abs(baseline[:, _PIN_INACTIVE_IDX]).max() > 100

    mask = torch.zeros(_PIN_ACTION_DIM, dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    pinned_arch = _make_inactive_dim_probe_arch()
    pinned = _probe_actions(pinned_arch, schedule, active_action_mask=mask)

    # Every forward saw inactive dims exactly on sigma_k * eps0 ...
    for k, seen in enumerate(pinned_arch.seen_action_inputs):
        torch.testing.assert_close(
            seen[..., _PIN_INACTIVE_IDX], sigmas[k] * eps0[..., _PIN_INACTIVE_IDX], rtol=0, atol=1e-7
        )
    # ... the final state lands exactly on 0 (sigma_end == 0) ...
    assert np.abs(pinned[:, _PIN_INACTIVE_IDX]).max() == 0.0
    # ... and active dims are untouched by the pin (input-independent fake pred).
    np.testing.assert_array_equal(pinned[:, _PIN_ACTIVE_IDX], baseline[:, _PIN_ACTIVE_IDX])


def test_sample_without_mask_pins_nothing():
    """No active_action_mask = every dim active: nothing is inferred from other attributes of the architecture
    (the pin is the caller's explicit choice, decided from action_mode by the deploy engine)."""
    import numpy as np

    schedule = [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)]
    baseline = _probe_actions(_make_inactive_dim_probe_arch(), schedule)
    arch = _make_inactive_dim_probe_arch()
    # What the removed inference read: a unify scatter map hanging off the architecture.
    arch.normalizer = SimpleNamespace(_dst_index=np.asarray(_PIN_ACTIVE_IDX), _unify_dim=_PIN_ACTION_DIM)
    assert arch._resolve_inactive_action_dims(None, torch.device("cpu")) is None
    np.testing.assert_array_equal(_probe_actions(arch, schedule), baseline)


def test_rollout_is_the_condition_encode_then_sample():
    """rollout = video_backbone.preprocess_condition on the history + prompts, then sample with the given mask."""
    import numpy as np

    schedule = [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)]
    mask = torch.zeros(_PIN_ACTION_DIM, dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    want = _probe_actions(_make_inactive_dim_probe_arch(), schedule, active_action_mask=mask)

    arch = _make_inactive_dim_probe_arch()
    calls = []

    def _preprocess_condition(history, prompts, layout):
        calls.append((history, prompts, layout))
        return torch.zeros(1, 1, 1, 1, 1), _text()

    arch.video_backbone.preprocess_condition = _preprocess_condition
    _, actions = arch.rollout([["frame"]], ["p"], torch.zeros(1, 1, 3), schedule, seed=7, active_action_mask=mask)
    assert calls == [([["frame"]], ["p"], arch.layout)]
    np.testing.assert_array_equal(actions[0].numpy(), want)
