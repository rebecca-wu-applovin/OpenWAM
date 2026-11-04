"""Smoke tests for action backbone components.

Verifies that all action model classes can be imported, instantiated with
small parameters, and produce correct output shapes. No GPU required.
"""

import torch


def test_components_import():
    """Shared components should be importable."""
    from openwam.model.action_backbone.components import (
        RMSNorm,
        TimestepEmbedding,
        TimestepModulation,
        sinusoidal_embedding_1d,
    )

    assert all(
        c is not None
        for c in [
            RMSNorm,
            TimestepEmbedding,
            TimestepModulation,
            sinusoidal_embedding_1d,
        ]
    )


def test_timestep_embedding_shape():
    """TimestepEmbedding should produce (B, dim) from (B,) timestep."""
    from openwam.model.action_backbone.components import TimestepEmbedding

    te = TimestepEmbedding(freq_dim=32, dim=64)
    t = torch.tensor([0.5, 0.8])
    out = te(t)
    assert out.shape == (2, 64)


def test_timestep_modulation_shape():
    """TimestepModulation should produce (B, n_params, dim)."""
    from openwam.model.action_backbone.components import TimestepModulation

    mod = TimestepModulation(dim=64, n_params=9)
    t_embed = torch.randn(2, 64)
    out = mod(t_embed)
    assert out.shape == (2, 9, 64)


def test_action_dit_small_instantiate():
    """ActionDiT should instantiate with small parameters."""
    import torch.nn as nn

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
    assert dit.action_dim == 7
    assert dit.num_layers == 2
    assert isinstance(dit.action_encoder, nn.Linear)
    assert dit.action_encoder.in_features == 7
    assert dit.action_encoder.out_features == 64
    assert isinstance(dit.text_embedding, nn.Sequential)
    assert dit.text_embedding[0].in_features == 4096
    assert dit.text_embedding[-1].out_features == 64
    assert isinstance(dit.action_decoder, nn.Linear)
    assert dit.action_decoder.in_features == 64
    assert dit.action_decoder.out_features == 7
    assert not torch.all(dit.action_decoder.weight == 0)
    assert not hasattr(dit, "action_embedding")
    assert not hasattr(dit, "action_output_head")


def test_action_dit_forward_shape():
    """ActionDiT forward should produce (B, T, action_dim)."""
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
    actions = torch.randn(2, 5, 7)
    bridges = [torch.randn(2, 10, 128) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])
    context, context_mask = torch.randn(2, 3, 4096), torch.ones(2, 3, dtype=torch.bool)

    out = dit(
        actions, bridges, timestep, context=context, context_mask=context_mask,
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    )
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_cross_attn_context_shape_and_effect():
    """joint_cross_attn should accept action-owned raw text/proprio context."""
    from openwam.model.action_backbone.separate_action_dit import CrossAttnActionDiT

    torch.manual_seed(0)
    dit = CrossAttnActionDiT(
        shift_action=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=2,
        video_dim=48,
        text_dim=16,
        attn_head_dim=8,
        freq_dim=256,
        max_action_len=1024,
        eps=1e-6,
    ).eval()
    actions = torch.randn(2, 5, 7)
    bridges = [torch.randn(2, 10, 48) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])
    context_a = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(11))
    context_b = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(22))
    context_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        out_a = dit(
            actions, bridges, timestep, context=context_a, context_mask=context_mask,
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        )
        out_b = dit(
            actions, bridges, timestep, context=context_b, context_mask=context_mask,
            use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
        )

    assert out_a.shape == (2, 5, 7)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_action_backbone_does_not_import_the_video_or_architecture_package():
    """The action package imports only openwam.model.common / model-level helpers, never video_backbone or the
    architectures (layering: architectures -> action_backbone -> common)."""
    import ast
    from pathlib import Path

    import openwam.model.action_backbone as action_pkg

    bad = []
    for path in sorted(Path(action_pkg.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            names += [a.name for a in node.names] if isinstance(node, ast.Import) else []
            bad += [
                f"{path.name}: {n}"
                for n in names
                if n.startswith(("openwam.model.video_backbone", "openwam.model.architectures"))
            ]
    assert not bad, bad
