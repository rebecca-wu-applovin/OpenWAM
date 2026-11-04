"""Verify that all openwam modules can be imported successfully."""


def test_import_openwam():
    import openwam  # noqa: F401

    assert hasattr(openwam, "__version__")


def test_import_data_base():
    from openwam.dataloader.bases import BaseDataset  # noqa: F401

    assert hasattr(BaseDataset, "__getitem__")
    assert hasattr(BaseDataset, "__len__")


def test_import_data_transforms():
    from openwam.dataloader.collate import Batch, collate  # noqa: F401
    from openwam.dataloader.transforms import Normalizer, VideoColorJitter  # noqa: F401
    from openwam.dataloader.utils.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix  # noqa: F401


def test_import_data_normalization_stats():
    import openwam.dataloader.utils.stats_computation.yam_sharpa_stats_computation  # noqa: F401


def test_import_data_init():
    from openwam.dataloader import (  # noqa: F401
        BaseDataset,
        MultiYamSharpaDataset,
        YamSharpaDataset,
    )


def test_import_inference_base():
    from openwam.deploy.engine import BaseInferenceEngine  # noqa: F401

    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    from openwam.deploy.denoise_schedule import (  # noqa: F401
        Schedule,
        make_schedule,
        schedule_sync,
    )


def test_import_inference_sparse_engine():
    from openwam.deploy.sparse_engine import SparseHistoryInferenceEngine  # noqa: F401


def test_import_inference_init():
    from openwam.deploy import (  # noqa: F401
        BaseInferenceEngine,
        Schedule,
        make_schedule,
    )


def test_import_training_base():
    from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: F401

    assert hasattr(OpenWAMTrainer, "compute_loss")
    assert hasattr(OpenWAMTrainer, "train")


def test_import_training_optimizer_groups():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters  # noqa: F401


def test_import_training_init():
    from openwam.train import OpenWAMTrainer  # noqa: F401


def test_import_action_dit():
    from openwam.model.action_backbone.separate_action_dit import (  # noqa: F401
        ActionDiT,
        ActionDiTState,
    )


def test_import_action_scheduler():
    from openwam.model.action_backbone.scheduler import ActionScheduler

    s = ActionScheduler()
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_video_backbone():
    from openwam.model.video_backbone import WanBackbone  # noqa: F401
    from openwam.model.video_backbone.common.weights import load_strict, read_weights  # noqa: F401


def test_import_architecture_registry():
    from openwam.model import ARCHITECTURE_REGISTRY
    from tests._arch import build_test_arch, stub_backbone

    assert "dual_system_cross_attn" in ARCHITECTURE_REGISTRY
    assert "dual_system_self_attn" in ARCHITECTURE_REGISTRY

    configs = {
        "dual_system_cross_attn": {
            "framework": "dual_system",
            "clean_attention": "bidirectional",
            "attention_mask_mode": "mutual",
            "variant": "joint_cross_attn",
            "detach_bridge": True,
            "bridge_layers": (0, 1),
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 128,
        },
        "dual_system_self_attn": {
            "framework": "dual_system",
            "clean_attention": "bidirectional",
            "attention_mask_mode": "mutual",
            "variant": "joint_self_attn",
            "bridge_layers": (0, 1),
            "action_dim": 7,
            # joint_self_attn requires action dim == video_dim (the MoT driver
            # runs a single mixed attention with no inter-modality projection).
            "dim": 128,
            "ffn_dim": 256,
            "num_heads": 4,
            "video_dim": 128,
        },
    }

    for arch_name in configs:
        arch = build_test_arch(arch_name, configs[arch_name], stub_backbone(configs[arch_name]))
        assert arch is not None


def test_wan_dit_block_public_attrs_for_mot():
    """Regression guard for Wan block sub-modules used by MoT pre/post hooks.

    The video pre/post attention split uses these PUBLIC sub-modules:
    ``block.{self_attn, cross_attn, norm1, norm2, norm3, modulation, ffn}``
    and ``self_attn.{q, k, v, o, norm_q, norm_k}``.

    On top of presence, three shape / semantic invariants are checked:
    ``modulation`` shape ``(1, 6, dim)``; ``norm1`` / ``norm2`` with
    ``elementwise_affine=False``; and ``cross_attn.forward`` keeping its
    binary ``(x, y)`` positional signature. A vendored ``wan/`` upgrade
    that changes any of these will trip CI immediately.
    """
    import inspect

    import torch.nn as nn

    from openwam.model.video_backbone.wan.dit import DiTBlock

    block = DiTBlock(dim=32, num_heads=4, ffn_dim=64, eps=1e-6)
    for name in ("self_attn", "cross_attn", "norm1", "norm2", "norm3", "ffn", "modulation"):
        assert hasattr(block, name), f"DiTBlock missing public attr: {name}"
    for name in ("q", "k", "v", "o", "norm_q", "norm_k"):
        assert hasattr(block.self_attn, name), f"SelfAttention missing public attr: {name}"
    # 6-param AdaLN: the pre/post split assumes the modulation second dim is 6
    # (shift / scale / gate, twice — once for self-attn, once for FFN).
    assert tuple(block.modulation.shape) == (1, 6, 32), (
        f"DiTBlock.modulation shape changed: {tuple(block.modulation.shape)} != (1, 6, 32)"
    )
    # AdaLN is applied externally around norm1 / norm2, so they
    # must keep elementwise_affine=False (no built-in affine bias).
    assert isinstance(block.norm1, nn.LayerNorm) and not block.norm1.elementwise_affine
    assert isinstance(block.norm2, nn.LayerNorm) and not block.norm2.elementwise_affine
    # cross_attn.forward must keep the binary (x, y) signature — the Wan
    # post-attention hook invokes ``block.cross_attn(block.norm3(x), context)``. The Motus
    # fork of wan uses a ternary (x, ctx, context_lens) signature, so if this
    # repo's wan/ ever aligns with that shape CI fires here.
    sig = inspect.signature(block.cross_attn.forward)
    n_positional = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    )
    assert n_positional == 2, (
        f"DiTBlock.cross_attn.forward required positional arg count changed: "
        f"{n_positional} != 2 (Wan post-attn invokes (x, y) — a ternary signature breaks it)"
    )
