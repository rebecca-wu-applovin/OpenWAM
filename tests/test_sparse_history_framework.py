"""Framework pieces of the sparse-history layout (07_sparse_history_backbones).

- joint attention mask with H clean history latents ("clean never reads noisy")
- N proprio tokens with a learned slot embedding, and its optimizer visibility
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.utils.mask_modes import build_frame_mask
from openwam.model.video_backbone.base import expand_frame_mask
from tests._arch import proprio_context_arch

H, F, TPF, SA = 3, 2, 2, 4  # 3 history + 2 future latents x 2 tokens, 4 action tokens
SV = (H + F) * TPF
NH = H * TPF  # history token count


def _joint(clean_attention):
    """(SV+SA, SV+SA) token mask of the frame mask (mutual), expanded frame-major."""
    layout = OmegaConf.create({"num_latents": H + F, "num_hist_latents": H})
    return expand_frame_mask(build_frame_mask(layout, clean_attention, "mutual"), [TPF] * (H + F), SA, "cpu")


@pytest.mark.parametrize("clean", ["bidirectional", "causal"])
def test_history_rows_never_read_noisy(clean):
    m = _joint(clean)
    # History rows: no future video, no action.
    assert not m[:NH, NH:].any()
    # Future rows: all video + all action.
    assert m[NH:SV, :].all()
    # Action rows: everything.
    assert m[SV:, :].all()


def test_history_inner_pattern():
    bi = _joint("bidirectional")
    assert bi[:NH, :NH].all()  # bidirectional inside the history
    causal = _joint("causal")
    for i in range(H):
        rows = slice(i * TPF, (i + 1) * TPF)
        # history frame i sees frames 0..i only (Long-WAM)
        assert causal[rows, : (i + 1) * TPF].all()
        assert not causal[rows, (i + 1) * TPF : NH].any()


def _append(arch, proprio, b=2, ctx=8, text_len=4):
    """Run ``build_contexts`` on a padded text (lengths 2 / 4). Returns: ActionDiT's TextCondition."""
    from openwam.model.video_backbone.base import TextCondition

    mask = torch.arange(text_len)[None] < torch.tensor([2, 4])[:, None]
    text = TextCondition(context=torch.zeros(b, text_len, ctx), context_mask=mask)
    return arch.build_contexts(text, proprio, torch.ones(proprio.shape, dtype=torch.bool))[1]


def test_n_proprio_tokens_with_slot_embedding():
    torch.manual_seed(0)
    arch = proprio_context_arch(state_dim=3, text_dim=8, h=5)
    nn.init.normal_(arch.proprio_slot_emb.weight)
    proprio = torch.randn(1, 3).expand(2, 5, 3).clone()  # same state in all 5 slots
    out = _append(arch, proprio)
    assert out.context.shape == (2, 4 + 5, 8)
    assert out.context_mask.shape == (2, 9)
    assert out.context_mask[:, 4:].all()
    tok = out.context[:, 4:]
    # Identical states differ only by the slot embedding → order is recoverable.
    expected = arch.proprio_encoder(proprio) + arch.proprio_slot_emb.weight
    torch.testing.assert_close(tok, expected)
    assert not torch.allclose(tok[:, 0], tok[:, 1])


def test_single_proprio_token_unchanged():
    torch.manual_seed(0)
    arch = proprio_context_arch(state_dim=3, text_dim=8, h=1)
    assert arch.proprio_slot_emb is None
    proprio = torch.randn(2, 1, 3)
    out = _append(arch, proprio)
    assert out.context.shape == (2, 5, 8)
    torch.testing.assert_close(out.context[:, 4:], arch.proprio_encoder(proprio))


def test_wrong_token_count_raises():
    with pytest.raises(ValueError):
        _append(proprio_context_arch(state_dim=3, text_dim=8, h=5), torch.randn(2, 4, 3))


def test_slot_embedding_reaches_optimizer():
    # build_trainable_parameters only walks top-level child modules.
    class _Arch(nn.Module):
        get_trainable_modules = BaseWAMArchitecture.get_trainable_modules

    arch = _Arch()
    arch.proprio_encoder = nn.Linear(3, 8)
    arch.proprio_slot_emb = nn.Embedding(5, 8)
    assert "proprio_slot_emb" in arch.get_trainable_modules()
