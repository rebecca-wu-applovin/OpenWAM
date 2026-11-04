"""Test that the freeze strategy correctly freezes/unfreezes model components."""

import torch.nn as nn


class MockPipeline:
    """Minimal mock of WanVideoPipeline with named submodules."""

    def __init__(self):
        self.dit = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)
        self.text_encoder = nn.Linear(10, 10)
        self.aux_frozen = nn.Linear(10, 10)
        self.aux_trainable = nn.Linear(10, 10)


def _apply_freeze(pipe, freeze_list):
    """Replicate the freeze logic from OpenWAMTrainer."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)


def test_default_freeze_strategy():
    """Default freeze: text_encoder, vae, aux_frozen frozen; dit, aux_trainable trainable."""
    pipe = MockPipeline()
    freeze_list = ["text_encoder", "vae", "aux_frozen"]
    _apply_freeze(pipe, freeze_list)

    # Frozen
    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pipe.aux_frozen.parameters())

    # Trainable
    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in pipe.aux_trainable.parameters())


def test_freeze_everything_except_dit():
    """FastWAM-style: freeze all except dit."""
    pipe = MockPipeline()
    freeze_list = ["text_encoder", "vae", "aux_frozen", "aux_trainable"]
    _apply_freeze(pipe, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pipe.aux_frozen.parameters())
    assert not any(p.requires_grad for p in pipe.aux_trainable.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())


def test_empty_freeze_list():
    """Empty freeze list: everything remains trainable."""
    pipe = MockPipeline()
    _apply_freeze(pipe, [])

    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in pipe.vae.parameters())
    assert all(p.requires_grad for p in pipe.text_encoder.parameters())
    assert all(p.requires_grad for p in pipe.aux_frozen.parameters())
    assert all(p.requires_grad for p in pipe.aux_trainable.parameters())


def test_freeze_nonexistent_module():
    """Freezing a module that doesn't exist should not error."""
    pipe = MockPipeline()
    pipe.audio_encoder = None  # explicitly None
    freeze_list = ["text_encoder", "vae", "audio_encoder", "nonexistent_module"]
    _apply_freeze(pipe, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    # dit and others remain trainable
    assert all(p.requires_grad for p in pipe.dit.parameters())
