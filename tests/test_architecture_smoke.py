"""Smoke tests for WAM architecture composition.

Each architecture exposes a different action-backbone contract:
  - joint_cross_attn → ActionDiT.forward(action_tokens, bridges, timestep)
  - joint_self_attn  → DualSystemMoTDriver coordinates pre/post_attn_at_layer

These tests exercise each path with fake tensors; no GPU required.
"""

import torch

from tests._arch import build_test_arch, stub_backbone


def _make_dual_system(variant="joint_cross_attn", detach_bridge=True, dim=64, video_dim=64):
    cfg = {
        "framework": "dual_system",
        "clean_attention": "bidirectional",
        "attention_mask_mode": "mutual",
        "variant": variant,
        "detach_bridge": detach_bridge,
        "action_dim": 7,
        "dim": dim,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": video_dim,
        "bridge_layers": (0, 2),
    }
    name = "dual_system_cross_attn" if variant == "joint_cross_attn" else "dual_system_self_attn"
    return build_test_arch(name, cfg, stub_backbone(cfg))


def test_dual_system_joint_cross_attn_flow():
    """DualSystem joint_cross_attn: ActionDiT.forward can consume context."""
    arch = _make_dual_system("joint_cross_attn", detach_bridge=True, dim=64, video_dim=128)
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    bridges = {bid: torch.randn(B, T_video, 128) for bid in ab.bridge_layers}
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)
    pred = ab(noisy_actions, bridges, timestep, context=context, context_mask=context_mask)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_joint_self_attn_flow():
    """DualSystem joint_self_attn: MoT pre/post_attn_at_layer round-trip."""
    # joint_self_attn requires action dim == video_dim (the MoT driver shares
    # per-head space across modalities).
    arch = _make_dual_system("joint_self_attn", detach_bridge=False, dim=64, video_dim=64)
    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)
    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

    # Drive the action half-step at each layer with a fake mixed-attention
    # output. The driver's job (plumbing video Q/K/V, mixed attention, splitting)
    # is covered separately in test_mot_driver.
    for layer_id in range(ab.num_layers):
        q, k, v, post = ab.pre_attn_at_layer(layer_id, astate)
        assert q.shape == k.shape == v.shape == (B, T_action, ab.num_heads * ab.head_dim)
        attn_out = torch.randn_like(q)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_out, post)

    pred = ab.extract_prediction(astate)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_self_attn_has_mot_driver():
    """DualSystemSelfAttn builds its DualSystemMoTDriver at construction (the default kernel; the driver itself
    is exercised in tests/test_mot_driver.py)."""
    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver

    arch = _make_dual_system("joint_self_attn", detach_bridge=False, dim=64, video_dim=64)
    assert type(arch.mot_driver) is DualSystemMoTDriver
    # Re-callable: a rebuild pairs the same two backbones.
    driver = arch.build_mot_driver()
    assert type(driver) is DualSystemMoTDriver and driver.vb is arch.video_backbone and driver.ab is arch.action_backbone


def test_all_architectures_registered():
    """Supported top-level architecture families should be in the registry."""
    from openwam.model import ARCHITECTURE_REGISTRY

    assert "dual_system_cross_attn" in ARCHITECTURE_REGISTRY
    assert "dual_system_self_attn" in ARCHITECTURE_REGISTRY
