"""Registry + construction for the cosmos3 backbone (Edge / Nano share one class).

Bad-config cases raise from ``_check_cfg`` before any heavy import. The checkpoint path builds meta
shells from the yaml ``dit`` / ``vae`` blocks, so it runs on CPU without weights (tokenizer stubbed, a
tiny ``dit`` / ``vae`` where storage is allocated).
"""

import os

import pytest
from omegaconf import OmegaConf

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    Cosmos3EdgeVideoBackbone,
    register_video_backbone,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Tiny Cosmos3OmniTransformer / AutoencoderKLWan kwargs (same as test_cosmos3_block_loop_parity's MINI).
MINI_DIT = dict(
    attention_bias=False,
    head_dim=6,
    hidden_size=12,
    intermediate_size=24,
    latent_channel=2,
    latent_patch_size=1,
    num_attention_heads=2,
    num_hidden_layers=2,
    num_key_value_heads=1,
    patch_latent_dim=2,
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    hidden_act="relu2",
    rms_norm_eps=1e-5,
    rope_axes_dim=[1, 1, 1],
    rope_theta=1e8,
    vocab_size=32,
)


def _edge_cfg(**changes):
    """changes: top-level key -> value replacing the composed cosmos3_edge yaml's. Returns: read-only DictConfig."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose("train", overrides=["model/video_backbone=cosmos3_edge"])
    vb = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    vb.update(changes)
    out = OmegaConf.create(vb)
    OmegaConf.set_readonly(out, True)
    return out


@pytest.fixture
def stub_tokenizer(monkeypatch, tmp_path):
    """A ckpt dir with an empty text_tokenizer/ and the tokenizer load stubbed. Returns: the ckpt dir."""
    from transformers import PreTrainedTokenizerFast

    (tmp_path / "text_tokenizer").mkdir()
    monkeypatch.setattr(
        PreTrainedTokenizerFast, "from_pretrained", classmethod(lambda cls, *a, **k: object()), raising=True
    )
    return str(tmp_path)


def test_cosmos3_edge_registered():
    assert _VIDEO_BACKBONE_REGISTRY["cosmos3_edge"] is Cosmos3EdgeVideoBackbone


def test_cosmos3_nano_registered():
    # Nano shares the class; the variant is the yaml's dit / vae blocks.
    assert _VIDEO_BACKBONE_REGISTRY["cosmos3_nano"] is Cosmos3EdgeVideoBackbone


def test_double_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)


@pytest.mark.parametrize(
    "change, error, match",
    [
        ({"max_text_tokens": 2}, ValueError, "max_text_tokens"),
        ({"fps": 0}, ValueError, "fps"),
        ({"freeze_und": False}, NotImplementedError, "freeze_und"),
    ],
)
def test_bad_cfg_rejected_in_both_modes(change, error, match, tmp_path):
    """The yaml knobs are validated before any heavy import, whichever constructor runs."""
    cfg = _edge_cfg(**change)
    with pytest.raises(error, match=match):
        Cosmos3EdgeVideoBackbone.from_pretrained(cfg)
    with pytest.raises(error, match=match):
        Cosmos3EdgeVideoBackbone.from_checkpoint_dir(cfg, str(tmp_path))


def test_nonexistent_model_path_fails_fast(tmp_path):
    # With diffusers installed this is a FileNotFoundError on the missing transformer/ subfolder;
    # without diffusers it is the install-hint ImportError. Both are acceptable fail-fast outcomes on CPU.
    with pytest.raises((FileNotFoundError, ImportError)):
        Cosmos3EdgeVideoBackbone.from_pretrained(_edge_cfg(model_path=str(tmp_path / "nonexistent")))


def test_checkpoint_dir_needs_tokenizer(tmp_path):
    pytest.importorskip("diffusers")
    with pytest.raises(FileNotFoundError, match="text_tokenizer"):
        Cosmos3EdgeVideoBackbone.from_checkpoint_dir(_edge_cfg(), str(tmp_path))


def test_freeze_applies_to_the_checkpoint_shell(stub_tokenizer):
    """Finetune / resume build from a ckpt dir too, so the und tower must be frozen there as well.

    Regression: the freeze once sat behind ``not deploy`` (deploy = a ckpt_dir build), so any
    finetune / resume run handed the whole und tower to the optimizer — 2.19x the trainable surface,
    for parameters that run under ``no_grad`` and can never receive a gradient.
    """
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    vb = Cosmos3EdgeVideoBackbone.from_checkpoint_dir(_edge_cfg(), stub_tokenizer)
    net = vb.dit
    assert all(p.is_meta for p in net.parameters()), "checkpoint shells hold no weights"
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable, "everything frozen — the gen pathway must stay trainable"
    # und tower: shared embedding, final norm, per-layer und attn/MLP/norms.
    leaked = sorted(
        n
        for n in trainable
        if n.startswith(("embed_tokens", "norm.", "action_proj_", "audio_proj_"))
        or any(f".self_attn.{c}." in n for c in ("to_q", "to_k", "to_v", "to_out", "k_norm_und_for_gen"))
        or any(f".{c}." in n for c in ("input_layernorm", "post_attention_layernorm", "mlp"))
    )
    assert not leaked, f"und/native-head params left trainable on the ckpt_dir path: {leaked[:6]}"
    # The gen half is untouched by the freeze; lm_head never enters the checkpoint.
    assert any(".self_attn.add_q_proj." in n for n in trainable)
    assert net.lm_head is None
    # Geometry comes from the yaml dit block.
    assert (vb.dim, vb.num_layers, vb.num_heads, vb.head_dim, vb.text_dim) == (2048, 28, 16, 128, 2048)


def test_resume_shell_gets_storage(stub_tokenizer):
    """Resume fills the checkpoint shell only after ``prepare``, so it gets zeroed storage first.

    ``set_dtype_device`` (before ``load_state``) used to raise "Cannot copy out of meta tensor" on a
    meta shell. The rotary table (non-persistent, absent from every checkpoint) must survive as the
    recomputed fp32 values, not be zeroed.
    """
    pytest.importorskip("diffusers")
    import torch

    from openwam.model.video_backbone.common.weights import materialize_meta

    tiny_vae = dict(OmegaConf.to_container(_edge_cfg().vae), base_dim=8, decoder_base_dim=8)
    vb = Cosmos3EdgeVideoBackbone.from_checkpoint_dir(_edge_cfg(dit=MINI_DIT, vae=tiny_vae), stub_tokenizer)
    assert any(p.is_meta for p in vb.dit.parameters()), "deploy / finetune get the cheap meta shell"
    materialize_meta(vb, "cpu")
    assert not any(t.is_meta for t in [*vb.parameters(), *vb.buffers()])
    # Zeroed, not uninitialized: DeepSpeed may read the values while flattening, before load_state.
    assert all(torch.all(p == 0) for p in vb.dit.parameters())
    inv_freq = vb.dit.rotary_emb.inv_freq
    assert inv_freq.dtype == torch.float32 and torch.any(inv_freq != 0)
    vb.set_dtype_device(torch.bfloat16, torch.device("cpu"))
