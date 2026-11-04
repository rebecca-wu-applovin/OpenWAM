"""cosmos3 / GE / flux construction: ``from_pretrained(cfg, assets)`` / ``from_checkpoint_dir(cfg, ckpt_dir)`` + ``LoadMode``.

The same two constructors Wan got (``tests/test_wan_construction.py``):
  - ``from_pretrained(cfg, assets)``: upstream weights at the ``assets.video_backbone`` paths (each family's own
    loaders for its frozen third-party parts);
  - ``from_checkpoint_dir(cfg, ckpt_dir)``: a meta shell built from the yaml geometry plus the non-weight
    assets ``save_deploy_assets`` copied into the checkpoint dir (tokenizer / processor / ``config.json``).
``build_video_backbone(cfg, mode, assets, ckpt_dir)`` picks one of the two from the explicit ``LoadMode``.

``gpu``-marked tests read the real weights (skipped when they are not on this machine).
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAMILIES = ("cosmos3_edge", "ge_base_fast", "flux3_action_base")
# One DiT geometry key per family whose change no upstream checkpoint can satisfy.
_GEOMETRY_KEY = {
    "cosmos3_edge": ("dit.num_hidden_layers", 27),
    "ge_base_fast": ("dit.num_layers", 27),
    "flux3_action_base": ("dit.depth_single_blocks", 27),
}


def _compose_vb(name: str, *overrides: str):
    """name: video_backbone group option. overrides: extra Hydra overrides.
    Returns: the composed, read-only ``cfg.model.video_backbone`` (DictConfig)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose("train", overrides=[f"model/video_backbone={name}", *overrides])
    vb = cfg.model.video_backbone
    OmegaConf.set_readonly(vb, True)
    return vb


def _with(cfg, **changes):
    """cfg: read-only video_backbone cfg. changes: dotted-key -> value. Returns: a read-only copy with them applied."""
    out = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in changes.items()]))
    OmegaConf.set_readonly(out, True)
    return out


def _cls(name: str):
    """name: registry key. Returns: the registered VideoBackbone class."""
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY

    return _VIDEO_BACKBONE_REGISTRY[name]


def _assets(name: str):
    """name: family. Returns: its read-only ``cfg.assets.video_backbone`` (upstream paths)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        assets = compose("train", overrides=[f"model/video_backbone={name}"]).assets.video_backbone
    OmegaConf.set_readonly(assets, True)
    return assets


def _real_cfg(name: str):
    """name: family. Returns: (model.video_backbone, assets.video_backbone); skips when the upstream weights are
    not on this machine."""
    cfg, assets = _compose_vb(name), _assets(name)
    if not os.path.isdir(assets.model_path):
        pytest.skip(f"{name} weights not found at {assets.model_path}")
    return cfg, assets


# ----------------------------------------------------------------------------- LoadMode dispatch (CPU)


@pytest.mark.parametrize("name", FAMILIES)
def test_build_mode_checkpoint_skips_pretrained(name, monkeypatch, tmp_path):
    """CHECKPOINT never touches the upstream weights; it needs the ckpt dir."""
    from openwam.model.video_backbone import LoadMode, build_video_backbone

    cls = _cls(name)
    calls = []
    monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, cfg, a: calls.append(("pretrained", a)) or "P"))
    monkeypatch.setattr(cls, "from_checkpoint_dir", classmethod(lambda c, cfg, d: calls.append(("ckpt", d)) or "C"))
    cfg = _compose_vb(name)

    assets = _assets(name)

    assert build_video_backbone(cfg, LoadMode.CHECKPOINT, None, str(tmp_path)) == "C"
    assert calls == [("ckpt", str(tmp_path))]
    with pytest.raises(ValueError, match="ckpt_dir"):
        build_video_backbone(cfg, LoadMode.CHECKPOINT, None, None)
    # PRETRAINED hands the assets slice (not the model slice) to from_pretrained.
    assert build_video_backbone(cfg, LoadMode.PRETRAINED, assets, None) == "P"
    assert calls[-1] == ("pretrained", assets)


def test_text_cache_switch_in_backbone_yaml():
    """The text-cache switch lives in the backbone yaml's training block: wan / ge / flux cache, cosmos3 cannot."""
    from hydra import compose, initialize_config_dir

    expected = {
        "wan22_ti2v_5b": True,
        "longlive2_robot_s": True,
        "ge_base_fast": True,
        "flux3_action_base": True,
        "cosmos3_edge": False,
        "cosmos3_nano": False,
    }
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        for name, value in expected.items():
            assert compose("train", overrides=[f"model/video_backbone={name}"]).training.text_cache is value, name


def test_cosmos3_text_cache_error_names_training_key():
    """cosmos3's text encode needs the latent grid, so a prompt cache is impossible; the error names the key."""
    from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone

    vb = Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)
    with pytest.raises(ValueError, match=r"training\.text_cache=false"):
        vb.build_text_cache(["pick the cup"])


def test_materialize_meta_keeps_real_tensors():
    """Resume shells: meta tensors get zeroed storage, tensors that already have values keep them."""
    from openwam.model.video_backbone.common.weights import materialize_meta

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
            self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)

    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        toy = _Toy()
    assert toy.lin.weight.is_meta and not toy.inv_freq.is_meta
    materialize_meta(toy, "cpu")
    assert not any(t.is_meta for t in [*toy.parameters(), *toy.buffers()])
    assert torch.equal(toy.lin.weight, torch.zeros(2, 3))
    # A real non-persistent buffer (no checkpoint restores it) is untouched.
    assert torch.equal(toy.inv_freq, torch.arange(4.0))


def test_load_strict_keeps_tied_weights():
    """assign=True assigns every key separately; tied weights (T5 / Qwen embeddings) must stay one tensor."""
    from accelerate import init_empty_weights

    from openwam.model.video_backbone.common.weights import load_strict

    class _Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)

        def tie(self):
            self.head.weight = self.emb.weight
            return self

    w = torch.randn(4, 3)
    with init_empty_weights(include_buffers=False):
        shell = _Tied()
    shell.tie()  # as HF tie_weights() after the meta build
    load_strict(shell, {"emb.weight": w, "head.weight": w.clone()}, assign=True)
    assert shell.head.weight is shell.emb.weight and torch.equal(shell.emb.weight, w)
    with init_empty_weights(include_buffers=False):
        shell = _Tied()
    shell.tie()
    with pytest.raises(ValueError, match="tied"):
        load_strict(shell, {"emb.weight": w, "head.weight": w + 1}, assign=True)


# ----------------------------------------------------------------------------- yaml geometry contract (CPU)


def _flux_ckpt_assets(tmp_path) -> str:
    """Copy the non-weight Qwen3-VL files of the flux bundle into a fake ckpt dir (what save_deploy_assets writes).
    Returns: the ckpt dir; skips when the bundle is absent."""
    from openwam.model.video_backbone.flux.backbone import FluxActionVideoBackbone

    assets = _assets("flux3_action_base")
    if not os.path.isdir(assets.model_path):
        pytest.skip(f"flux bundle not found at {assets.model_path}")
    vb = FluxActionVideoBackbone.__new__(FluxActionVideoBackbone)
    object.__setattr__(vb, "_text_encoder_dir", os.path.join(assets.model_path, "text_encoder"))
    vb.save_deploy_assets(str(tmp_path))
    return str(tmp_path)


def test_flux_checkpoint_shell_from_yaml(tmp_path):
    """flux builds its meta shell from the yaml geometry + the ckpt's Qwen3-VL config (no weights read)."""
    from openwam.model.video_backbone.flux.backbone import FluxActionVideoBackbone

    ckpt = _flux_ckpt_assets(tmp_path)
    assert not any(f.endswith(".safetensors") for f in os.listdir(os.path.join(ckpt, "text_encoder")))
    vb = FluxActionVideoBackbone.from_checkpoint_dir(_compose_vb("flux3_action_base"), ckpt)
    assert all(p.is_meta for p in vb.parameters())
    assert (vb.dim, vb.num_layers, vb.num_heads, vb.text_dim) == (3072, 33, 24, 20480)
    assert vb.state_dim == 48  # the P stream width (yaml action_cond_dim)


def test_flux_action_cond_dim_must_equal_state_dim():
    """The P stream input width is declared in the backbone yaml; resolve_config rejects a different common.state_dim."""
    from hydra import compose, initialize_config_dir

    from openwam.config import resolve_config

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose(
            "train",
            overrides=[
                "model/video_backbone=flux3_action_base",
                "model=dual_system",
                "model.architecture.proprio_input=backbone_stream",
                "common.state_dim=80",
                "common.action_dim=80",
            ],
        )
    with pytest.raises(ValueError, match="action_cond_dim"):
        resolve_config(cfg)


# ----------------------------------------------------------------------------- real weights


def _assert_same_backbone(a, b) -> None:
    """a: from_pretrained backbone. b: from_checkpoint_dir shell after a strict load of a's state.
    Asserts equal state, equal non-persistent buffers, equal frozen flags and equal geometry."""
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    bad = [k for k in sa if sa[k].dtype != sb[k].dtype or not torch.equal(sa[k], sb[k])]
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:5]}"
    # Non-persistent buffers (rotary tables) are not in the checkpoint: the shell must compute the same ones.
    ba, bb = dict(a.named_buffers()), dict(b.named_buffers())
    assert ba.keys() == bb.keys()
    bad = [k for k in ba if ba[k].dtype != bb[k].dtype or not torch.equal(ba[k], bb[k])]
    assert not bad, f"{len(bad)} buffers differ, e.g. {bad[:5]}"
    assert {n: p.requires_grad for n, p in a.named_parameters()} == {
        n: p.requires_grad for n, p in b.named_parameters()
    }
    for prop in ("dim", "num_layers", "num_heads", "head_dim", "text_dim", "temporal_compression"):
        assert getattr(a, prop) == getattr(b, prop), prop
    assert a.scheduler.shift == b.scheduler.shift


@pytest.mark.gpu
@pytest.mark.parametrize("name", FAMILIES)
def test_checkpoint_matches_pretrained(name, tmp_path):
    """from_checkpoint_dir(cfg, ckpt) + a strict load of the pretrained state == from_pretrained(cfg, assets)."""
    from openwam.model.video_backbone.common.weights import load_strict

    cfg, assets = _real_cfg(name)
    cls = _cls(name)
    a = cls.from_pretrained(cfg, assets)
    a.save_deploy_assets(str(tmp_path))
    b = cls.from_checkpoint_dir(cfg, str(tmp_path))
    assert all(p.is_meta for p in b.parameters()), "from_checkpoint_dir must not read any weight"
    # Shapes only: dtypes follow the checkpoint (assign=True), e.g. cosmos3 keeps its time embedder in fp32.
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}
    load_strict(b, a.state_dict(), assign=True)
    _assert_same_backbone(a, b)
    # The shell's assets came from the ckpt dir, so a re-save from it is self-contained too.
    out = tmp_path / "resaved"
    b.save_deploy_assets(str(out))
    assert sorted(os.listdir(out)) == sorted(n for n in os.listdir(tmp_path) if n != "resaved")


@pytest.mark.gpu
@pytest.mark.parametrize("name", FAMILIES)
def test_geometry_mismatch_raises(name, tmp_path):
    """A yaml geometry the weights do not have fails at build, in both modes."""
    from openwam.model.video_backbone.common.weights import load_strict

    cfg, assets = _real_cfg(name)
    key, wrong = _GEOMETRY_KEY[name]
    bad = _with(cfg, **{key: wrong})
    cls = _cls(name)
    with pytest.raises(ValueError):
        cls.from_pretrained(bad, assets)
    a = cls.from_pretrained(cfg, assets)
    a.save_deploy_assets(str(tmp_path))
    shell = cls.from_checkpoint_dir(bad, str(tmp_path))
    with pytest.raises(ValueError):
        load_strict(shell, a.state_dict(), assign=True)
    del a, shell
    shutil.rmtree(tmp_path)
