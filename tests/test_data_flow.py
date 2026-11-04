"""The flat training data flow: dataset sample -> ``collate`` -> ``Batch`` -> ``compute_loss(batch)``, and the
cfg-decided proprio routing (``model.architecture.proprio_input``)."""

import inspect
import os

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.dataloader.collate import Batch, collate
from openwam.model.video_backbone.base import TextCondition
from tests._arch import build_test_arch
from tests._layout import make_layout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sample(h: int, f: int, d: int, k: int, source: str) -> dict:
    """One dataset-shaped sample: H + F frames, a K-step action chunk, H proprio rows of width D."""
    return {
        "video": [Image.new("RGB", (8, 8)) for _ in range(h + f)],
        "prompt": "pick cup",
        "action": torch.randn(k, d),
        "action_mask": torch.ones(k, d, dtype=torch.bool),
        "proprio": torch.randn(h, d),
        "proprio_mask": torch.ones(h, d, dtype=torch.bool),
        "source": source,
    }


def test_collate_shapes():
    """2 samples with H=5, F=8 (13 frames, K=32 action steps) stack into one Batch; clips stay lists."""
    samples = [_sample(5, 8, 48, 32, "a"), _sample(5, 8, 48, 32, "b")]
    batch = collate(samples)
    assert isinstance(batch, Batch)
    assert batch.action.shape == (2, 32, 48) and batch.action_is_pad.shape == (2, 32, 48)
    assert batch.proprio.shape == (2, 5, 48) and batch.proprio_mask.shape == (2, 5, 48)
    assert isinstance(batch.video, list) and [len(v) for v in batch.video] == [13, 13]
    assert batch.prompt == ["pick cup", "pick cup"] and batch.source == ["a", "b"]
    # action_is_pad is the complement of the per-sample action_mask.
    assert not batch.action_is_pad.any()
    torch.testing.assert_close(batch.action[1], samples[1]["action"])
    # One call moves every tensor; float fields take the dtype, masks stay bool.
    moved = batch.to(torch.device("cpu"), torch.bfloat16)
    assert moved.action.dtype == torch.bfloat16 and moved.proprio.dtype == torch.bfloat16
    assert moved.action_is_pad.dtype == torch.bool and moved.proprio_mask.dtype == torch.bool


def test_sample_has_only_data_keys(tmp_path):
    """A mixture-dataset sample carries data only: no model-branch markers, ``source`` names the sub-dataset."""
    from openwam.dataloader.registry import build_dataset
    from tests.dataloader.test_yam_sharpa import _cfg, _mock_decoder, _write_source

    _write_source(tmp_path / "a", [40, 40])
    ds = build_dataset(
        _cfg({"src_a": (tmp_path / "a", 1.0)}, history_offsets=[-2, -1, 0], num_frames=5, future_offsets=[2, 4]), "train"
    )
    with _mock_decoder():
        s = ds[0]
    assert set(s) == {"video", "prompt", "action", "action_mask", "proprio", "proprio_mask", "source"}
    assert s["source"] == "src_a"
    # One proprio row (and its per-dim mask) per history frame.
    assert tuple(s["proprio"].shape) == (3, 48) and tuple(s["proprio_mask"].shape) == (3, 48)
    assert len(s["video"]) == 3 + 2


def test_compute_loss_signature():
    """``compute_loss`` takes exactly one Batch (no flag / dict bag)."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    params = list(inspect.signature(BaseWAMArchitecture.compute_loss).parameters)
    assert params == ["self", "batch"]


def _compose(overrides):
    """overrides: list[str] Hydra overrides of configs/train.yaml. Returns: the composed DictConfig."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        return compose("train", overrides=["dataloader=yam_sharpa", *overrides])


def test_flux_requires_backbone_stream():
    """flux reads proprio from its own stream: its yaml pins proprio_input=backbone_stream."""
    from openwam.config import resolve_config

    flux = ["model/video_backbone=flux3_action_base", "common.state_dim=48"]
    with pytest.raises(ValueError, match="proprio_input"):
        resolve_config(_compose([*flux, "model.architecture.proprio_input=context_token"]))
    resolve_config(_compose([*flux, "model.architecture.proprio_input=backbone_stream"]))
    # A backbone without a proprio stream cannot take backbone_stream (the proprio would reach no video token).
    with pytest.raises(ValueError, match="proprio_input"):
        resolve_config(_compose(["model.architecture.proprio_input=backbone_stream"]))


@pytest.fixture
def fake_arch():
    """Builds a joint_self_attn architecture (H = 5 history frames, 6-D proprio) for a given proprio_input."""
    from tests.test_openwam_trainer import _MockVideoBackbone

    def _build(proprio_input: str):
        cfg = {
            "variant": "joint_self_attn",
            "proprio_input": proprio_input,
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "num_dit_layers": 2,
            "text_dim": 16,
            "use_proprioception": True,
            "state_dim": 6,
        }
        backbone = _MockVideoBackbone(dim=64, num_layers=2, num_heads=4, text_dim=16)
        return build_test_arch("joint_self_attn", cfg, backbone, layout=make_layout(h=5))

    return _build


@pytest.mark.parametrize("proprio_input", ["backbone_stream", "context_token"])
def test_proprio_contexts(fake_arch, proprio_input):
    """ActionDiT always reads text + H proprio tokens; the video backbone only with context_token."""
    arch = fake_arch(proprio_input)
    b, n_text, h = 2, 4, 5
    text = TextCondition(context=torch.randn(b, n_text, 16), context_mask=torch.ones(b, n_text, dtype=torch.bool))
    proprio = torch.randn(b, h, 6)
    video_text, action_text = arch.build_contexts(text, proprio, torch.ones(b, h, 6, dtype=torch.bool))
    assert action_text.context.shape == (b, n_text + h, 16) and action_text.context_mask.shape == (b, n_text + h)
    torch.testing.assert_close(action_text.context[:, :n_text], text.context)
    if proprio_input == "backbone_stream":
        assert video_text is text
    else:
        assert video_text.context.shape == (b, n_text + h, 16)
        torch.testing.assert_close(video_text.context, action_text.context)
    # A sample whose proprio rows are all masked gets zero tokens and False mask entries.
    mask = torch.ones(b, h, 6, dtype=torch.bool)
    mask[1] = False
    _, masked = arch.build_contexts(text, proprio, mask)
    assert not masked.context_mask[1, n_text:].any() and masked.context_mask[0].all()
    assert torch.count_nonzero(masked.context[1, n_text:]) == 0
    np.testing.assert_array_equal(masked.context_mask[:, :n_text].numpy(), text.context_mask.numpy())
