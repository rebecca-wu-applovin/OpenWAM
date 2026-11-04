"""Wan construction: yaml geometry -> meta skeleton -> strict weight load.

Covers ``common/weights.py`` (``load_strict`` / ``read_weights``) and the two
``WanBackbone`` constructors:
  - ``from_pretrained(cfg, assets)``: upstream weights from ``assets.model_path`` (``init: pretrained``)
    or a random init (``init: random``, the tiny debug yaml);
  - ``from_checkpoint_dir(cfg, ckpt_dir)``: a meta shell whose tokenizer comes from the ckpt dir.

CPU tests use the tiny yaml (random init) and write tiny weights in the upstream file
formats to exercise the pretrained path. ``gpu``-marked tests read the real 5B weights.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from openwam.model.video_backbone.common.weights import load_strict, read_weights

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _compose(name: str, *overrides: str):
    """name: video_backbone group option. overrides: extra Hydra overrides.
    Returns: (``cfg.model.video_backbone``, ``cfg.assets.video_backbone``), both read-only DictConfigs."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose("train", overrides=[f"model/video_backbone={name}", *overrides])
    vb, assets = cfg.model.video_backbone, cfg.assets.video_backbone
    OmegaConf.set_readonly(vb, True)
    OmegaConf.set_readonly(assets, True)
    return vb, assets


def _with(cfg, **changes):
    """cfg: read-only video_backbone cfg. changes: dotted-key -> value. Returns: a read-only copy with them applied."""
    out = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in changes.items()]))
    OmegaConf.set_readonly(out, True)
    return out


def _require_tokenizer(assets) -> None:
    """Skip when the umT5 tokenizer files ``assets.tokenizer_path`` points at are not on this machine."""
    if not os.path.isdir(assets.tokenizer_path):
        pytest.skip(f"umT5 tokenizer not found at {assets.tokenizer_path}")


@pytest.fixture(scope="module")
def tiny():
    """Returns: (model.video_backbone, assets.video_backbone) of the tiny debug yaml (random init)."""
    vb, assets = _compose("wan22_ti2v_tiny")
    _require_tokenizer(assets)
    return vb, assets


@pytest.fixture(scope="module")
def tiny_cfg(tiny):
    return tiny[0]


@pytest.fixture(scope="module")
def tiny_assets(tiny):
    return tiny[1]


@pytest.fixture(scope="module")
def longlive():
    """Returns: (model.video_backbone, assets.video_backbone) of the LongLive yaml."""
    vb, assets = _compose("longlive2_robot_s")
    _require_tokenizer(assets)
    return vb, assets


@pytest.fixture(scope="module")
def longlive_cfg(longlive):
    return longlive[0]


def _shapes(module: nn.Module) -> dict:
    """module: any nn.Module. Returns: {state_dict key: shape}."""
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def _write_upstream_dir(bb, out_dir: str) -> None:
    """Write ``bb``'s three modules in the upstream Wan2.2 file layout (the ``weights`` names of the yaml).

    bb: a built WanBackbone. out_dir: directory to create.
    DiT -> two safetensors shards, VAE -> ``.pth`` without the ``model.`` prefix, umT5 -> ``.pth``.
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir)
    dit = {k: v.contiguous() for k, v in bb.dit.state_dict().items()}
    keys = sorted(dit)
    half = len(keys) // 2
    # Two shards, like the real diffusion_pytorch_model-0000k-of-00003 split.
    save_file(
        {k: dit[k] for k in keys[:half]}, os.path.join(out_dir, "diffusion_pytorch_model-00001-of-00002.safetensors")
    )
    save_file(
        {k: dit[k] for k in keys[half:]}, os.path.join(out_dir, "diffusion_pytorch_model-00002-of-00002.safetensors")
    )
    # Upstream VAE keys are the inner VideoVAE38_ keys (no ``model.``).
    vae = {k[len("model.") :]: v for k, v in bb.vae.state_dict().items()}
    torch.save(vae, os.path.join(out_dir, "Wan2.2_VAE.pth"))
    torch.save(bb.text_encoder.state_dict(), os.path.join(out_dir, "models_t5_umt5-xxl-enc-bf16.pth"))


# ----------------------------------------------------------------------------- common/weights.py


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.register_buffer("buf", torch.zeros(2))


def _meta_toy() -> _Toy:
    """Returns: a _Toy whose params and buffers are on meta (as an empty shell)."""
    with torch.device("meta"):
        return _Toy()


def _toy_state() -> dict:
    """Returns: a complete state dict for _Toy."""
    return {"lin.weight": torch.randn(2, 3), "lin.bias": torch.randn(2), "buf": torch.ones(2)}


def test_load_strict_fills_meta_shell():
    toy = _meta_toy()
    state = _toy_state()
    load_strict(toy, state, assign=True)
    assert all(p.device.type == "cpu" for p in toy.parameters())
    assert torch.equal(toy.lin.weight, state["lin.weight"]) and torch.equal(toy.buf, state["buf"])


def test_load_strict_rejects_leftover_meta():
    # One key short: missing-key error, and nothing may stay meta silently.
    state = _toy_state()
    del state["lin.bias"]
    with pytest.raises(ValueError, match="missing"):
        load_strict(_meta_toy(), state, assign=True)
    # A meta parameter that is not in the module's state_dict keys at all (non-persistent) is left on meta.
    toy = _meta_toy()
    toy.register_buffer("scratch", torch.empty(1, device="meta"), persistent=False)
    with pytest.raises(ValueError, match="meta"):
        load_strict(toy, _toy_state(), assign=True)
    # Without assign, copying into a meta shell is a no-op, so the params stay meta -> raise.
    with pytest.raises(ValueError, match="meta"):
        load_strict(_meta_toy(), _toy_state(), assign=False)


def test_load_strict_rejects_unexpected_and_shape():
    state = {**_toy_state(), "extra": torch.zeros(1)}
    with pytest.raises(ValueError, match="unexpected"):
        load_strict(_Toy(), state, assign=False)
    state = {**_toy_state(), "lin.weight": torch.zeros(5, 3)}
    with pytest.raises(ValueError, match="shape"):
        load_strict(_Toy(), state, assign=False)


def test_read_weights_shards_and_pth_wrappers(tmp_path):
    from safetensors.torch import save_file

    a, b = {"x": torch.randn(2)}, {"y": torch.randn(3)}
    save_file(a, str(tmp_path / "s-00001-of-00002.safetensors"))
    save_file(b, str(tmp_path / "s-00002-of-00002.safetensors"))
    got = read_weights([str(tmp_path / "s-00001-of-00002.safetensors"), str(tmp_path / "s-00002-of-00002.safetensors")])
    assert set(got) == {"x", "y"} and torch.equal(got["y"], b["y"])
    # .pth: exactly one wrapper layer (state_dict / module / model_state) is unwrapped, nothing else.
    for wrapper in ("state_dict", "module", "model_state"):
        torch.save({wrapper: a}, str(tmp_path / f"{wrapper}.pth"))
        assert set(read_weights([str(tmp_path / f"{wrapper}.pth")])) == {"x"}
    torch.save(b, str(tmp_path / "plain.pth"))
    assert set(read_weights([str(tmp_path / "plain.pth")])) == {"y"}
    # The same key in two files is ambiguous.
    with pytest.raises(ValueError, match="x"):
        read_weights([str(tmp_path / "s-00001-of-00002.safetensors"), str(tmp_path / "model_state.pth")])


# ----------------------------------------------------------------------------- WanBackbone


def test_checkpoint_matches_pretrained_wan(tiny_cfg, tiny_assets, tmp_path):
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    a = WanBackbone.from_pretrained(tiny_cfg, tiny_assets)
    a.save_deploy_assets(str(tmp_path))
    assert os.path.isfile(os.path.join(tmp_path, "tokenizer", "google", "umt5-xxl", "tokenizer_config.json"))
    b = WanBackbone.from_checkpoint_dir(tiny_cfg, str(tmp_path))
    assert _shapes(a) == _shapes(b)
    # Same dtypes too (bf16 everywhere), so a strict ckpt load needs no cast.
    assert {k: v.dtype for k, v in a.state_dict().items()} == {k: v.dtype for k, v in b.state_dict().items()}
    # from_checkpoint_dir is an empty shell: every weight is on meta until the caller fills it.
    assert all(p.device.type == "meta" for p in b.parameters())
    load_strict(b, a.state_dict(), assign=True)
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())


def test_from_checkpoint_dir_needs_tokenizer(tiny_cfg, tmp_path):
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    with pytest.raises(FileNotFoundError, match="tokenizer"):
        WanBackbone.from_checkpoint_dir(tiny_cfg, str(tmp_path))


def test_pretrained_loads_upstream_layout(tiny_cfg, tiny_assets, tmp_path):
    """init: pretrained reads the upstream files (DiT shards, VAE .pth + ``model.`` prefix, umT5 .pth) bitwise."""
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    ref = WanBackbone.from_pretrained(tiny_cfg, tiny_assets)
    _write_upstream_dir(ref, str(tmp_path / "up"))
    assets = _with(
        tiny_assets, init="pretrained", model_path=str(tmp_path / "up"), tokenizer_path=tiny_assets.tokenizer_path
    )
    bb = WanBackbone.from_pretrained(tiny_cfg, assets)
    assert _shapes(bb) == _shapes(ref)
    assert all(torch.equal(ref.state_dict()[k], v) for k, v in bb.state_dict().items())
    assert all(p.dtype == torch.bfloat16 for p in bb.parameters())


def test_geometry_mismatch_raises(tiny_cfg, tiny_assets, tmp_path):
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    _write_upstream_dir(WanBackbone.from_pretrained(tiny_cfg, tiny_assets), str(tmp_path / "up"))
    assets = _with(
        tiny_assets, init="pretrained", model_path=str(tmp_path / "up"), tokenizer_path=tiny_assets.tokenizer_path
    )
    with pytest.raises(ValueError, match="num_layers"):
        WanBackbone.from_pretrained(_with(tiny_cfg, **{"dit.num_layers": 5}), assets)


def test_dit_weights_override_strips_prefix(tiny_cfg, tiny_assets, tmp_path):
    """dit_weights (LongLive model.pt) replaces the model_path DiT; its ``model.`` prefix is stripped."""
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    base = WanBackbone.from_pretrained(tiny_cfg, tiny_assets)
    _write_upstream_dir(base, str(tmp_path / "up"))
    other = WanBackbone.from_pretrained(tiny_cfg, tiny_assets)  # a second random DiT, different values
    torch.save({f"model.{k}": v for k, v in other.dit.state_dict().items()}, str(tmp_path / "model.pt"))
    assets = _with(
        tiny_assets,
        init="pretrained",
        model_path=str(tmp_path / "up"),
        tokenizer_path=tiny_assets.tokenizer_path,
        dit_weights=str(tmp_path / "model.pt"),
        dit_weights_prefix="model.",
    )
    bb = WanBackbone.from_pretrained(tiny_cfg, assets)
    assert all(torch.equal(other.dit.state_dict()[k], v) for k, v in bb.dit.state_dict().items())
    assert all(torch.equal(base.vae.state_dict()[k], v) for k, v in bb.vae.state_dict().items())
    # A file whose keys lack the declared prefix is rejected, not loaded half-renamed.
    torch.save(other.dit.state_dict(), str(tmp_path / "noprefix.pt"))
    with pytest.raises(ValueError, match="prefix"):
        WanBackbone.from_pretrained(tiny_cfg, _with(assets, dit_weights=str(tmp_path / "noprefix.pt")))


def test_longlive_text_len_both_modes(longlive, tmp_path):
    """LongLive's 128-token prompt budget comes from the cfg on the checkpoint path too (was 512 there)."""
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    longlive_cfg, longlive_assets = longlive
    shutil.copytree(longlive_assets.tokenizer_path, os.path.join(tmp_path, "tokenizer", "google", "umt5-xxl"))
    bb = WanBackbone.from_checkpoint_dir(longlive_cfg, str(tmp_path))
    assert bb.text_max_length == 128
    # The 5B geometry from the yaml, built as a meta shell (no weights read).
    assert (bb.dim, bb.num_layers, bb.num_heads, bb.text_dim) == (3072, 30, 24, 4096)


def test_wan_registry_names_map_to_wan_backbone():
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    assert _VIDEO_BACKBONE_REGISTRY["wan22_ti2v_5b"] is WanBackbone
    assert _VIDEO_BACKBONE_REGISTRY["longlive2_robot_s"] is WanBackbone


def test_required_clean_attention_checked_by_resolve_config(longlive_cfg):
    """The yaml's required_clean_attention (LongLive: causal) is enforced by resolve_config, before any build."""
    from omegaconf import OmegaConf

    from openwam.config import resolve_config
    from tests._training_cfg import CHECKED_TRAINING

    def _cfg(clean):
        return OmegaConf.create({
            "project": {"seed": 0},
            "training": CHECKED_TRAINING,
            "common": {"action_dim": 48, "state_dim": 48},
            "dataloader": {"history_offsets": [-20, 0], "future_offsets": [4, 8, 12, 16], "num_frames": 33,
                           "history_sampling": "fixed", "history_window": 100,
                           "caption_dropout_p": 0.0},
            "model": {
                "architecture": {"history_encoding": "per_frame", "clean_attention": clean,
                                 "attention_mask_mode": "mutual", "proprio_input": "context_token"},
                "video_backbone": OmegaConf.to_container(longlive_cfg, resolve=True),
            },
        })

    with pytest.raises(ValueError, match="causal"):
        resolve_config(_cfg("bidirectional"))
    assert resolve_config(_cfg("causal")).common.layout.num_latents == 3


# ----------------------------------------------------------------------------- real weights


@pytest.mark.gpu
def test_wan22_5b_from_pretrained_real_weights():
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    cfg, assets = _compose("wan22_ti2v_5b")
    if not os.path.isdir(assets.model_path):
        pytest.skip(f"Wan2.2-TI2V-5B not found at {assets.model_path}")
    bb = WanBackbone.from_pretrained(cfg, assets)
    assert (bb.dim, bb.num_layers, bb.num_heads, bb.head_dim, bb.text_dim) == (3072, 30, 24, 128, 4096)
    assert bb.text_max_length == 512 and bb.scheduler.shift == 5.0
    assert not any(p.device.type == "meta" for p in bb.parameters())
    assert all(p.dtype == torch.bfloat16 for p in bb.parameters())


@pytest.mark.gpu
def test_longlive_from_pretrained_real_weights():
    """from_pretrained(longlive): DiT == model.pt (prefix stripped, bf16), text budget 128."""
    from openwam.model.video_backbone.wan.backbone import WanBackbone

    cfg, assets = _compose("longlive2_robot_s")
    if not os.path.isfile(assets.dit_weights):
        pytest.skip(f"LongLive model.pt not found at {assets.dit_weights}")
    bb = WanBackbone.from_pretrained(cfg, assets)
    assert bb.text_max_length == 128
    ref = torch.load(assets.dit_weights, map_location="cpu", mmap=True, weights_only=True)
    for k, v in bb.dit.state_dict().items():
        assert torch.equal(v, ref[f"model.{k}"].to(torch.bfloat16)), k


# ---------------------------------------------------------------------------
# Tiny CPU WanBackbone: time modulation + prepare (merged from the former test_video_backbone_smoke.py)
# ---------------------------------------------------------------------------


def _build_tiny_wan_backbone():
    """A minimal real WanBackbone wrapping a CPU WanModel (no VAE / text encoder).

    Returns:
        (backbone, model): WanBackbone and its WanModel (dim 64, 4 latent channels, patch (1, 2, 2)).
    """
    from openwam.model.video_backbone.wan.backbone import WanBackbone
    from openwam.model.video_backbone.wan.dit import WanModel

    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
    )
    backbone = WanBackbone(
        dit=model,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        tokenizer_dir="",
        shift_video=5.0,
        train_timestep={"law": "grid", "weighting": "bell"},
        text_max_length=512,
    )
    return backbone, model


def _time_embedding(model, t: float):
    """model: WanModel. t: scalar timestep. Returns: (dim,) ``time_embedding(t)`` computed on its own."""
    from openwam.model.video_backbone.wan.dit import sinusoidal_embedding_1d

    return model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, torch.tensor([t])))[0]


def test_time_modulation_single_path():
    """One layout: num_hist_latents = 2, F = 4 -> the first 2 frames' tokens carry time_embedding(0), the rest
    time_embedding(t); time_mod is time_projection of those rows, (B, L, 6, dim)."""
    from openwam.model.video_backbone.wan.dit import build_time_modulation

    _, model = _build_tiny_wan_backbone()
    batch, frames, tokens_per_frame, t = 2, 4, 3, 500.0
    with torch.no_grad():
        time_embed, time_mod = build_time_modulation(
            model, torch.full((batch,), t), num_hist_latents=2, num_latents=frames, tokens_per_frame=tokens_per_frame
        )
        e0, et = _time_embedding(model, 0.0), _time_embedding(model, t)
        m0 = model.time_projection(e0).unflatten(0, (6, model.dim))
        mt = model.time_projection(et).unflatten(0, (6, model.dim))
    seq = frames * tokens_per_frame
    assert time_embed.shape == (batch, seq, model.dim) and time_mod.shape == (batch, seq, 6, model.dim)
    clean = slice(0, 2 * tokens_per_frame)
    noisy = slice(2 * tokens_per_frame, seq)
    # Every token of a frame shares one row; clean frames at t = 0, the others at t.
    torch.testing.assert_close(time_embed[:, clean], e0.expand(batch, 2 * tokens_per_frame, -1))
    torch.testing.assert_close(time_embed[:, noisy], et.expand(batch, 2 * tokens_per_frame, -1))
    torch.testing.assert_close(time_mod[:, clean], m0.expand(batch, 2 * tokens_per_frame, -1, -1))
    torch.testing.assert_close(time_mod[:, noisy], mt.expand(batch, 2 * tokens_per_frame, -1, -1))
    assert not torch.allclose(e0, et)


def test_wan_prepare_takes_text_condition_and_layout():
    """``WanBackbone.prepare`` reads context / context_mask from the TextCondition and gives the first
    ``layout.num_hist_latents`` latent frames t = 0 (``build_time_modulation`` on the layout)."""
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.base import TextCondition
    from openwam.model.video_backbone.wan.dit import build_time_modulation

    backbone, model = _build_tiny_wan_backbone()
    latents = torch.randn(1, 4, 3, 4, 4)
    timestep = torch.tensor([500.0])
    mask = torch.tensor([[True, True, False]])
    text = TextCondition(context=torch.randn(1, 3, 32), context_mask=mask)

    layout = OmegaConf.create({"num_hist_latents": 2, "num_latents": 3})
    state = backbone.prepare(
        latents=latents, timestep=timestep, text=text, layout=layout, proprio=torch.zeros(1, 2, 3),
        use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    )
    assert torch.equal(state.context_mask, mask)
    # Frames 0-1 (4 tokens each) carry the t = 0 modulation, frame 2 the sampled one.
    time_embed, expected = build_time_modulation(model, timestep, num_hist_latents=2, num_latents=3, tokens_per_frame=4)
    assert torch.equal(state.time_mod, expected) and torch.equal(state.extras["time_embed"], time_embed)
    assert torch.equal(state.time_mod[:, 0], state.time_mod[:, 7])
    assert not torch.allclose(state.time_mod[:, 0], state.time_mod[:, 8])


def test_unknown_backbone_name_raises():
    from openwam.model.video_backbone import LoadMode, build_video_backbone

    with pytest.raises(KeyError, match="Unknown video backbone"):
        build_video_backbone(OmegaConf.create({"name": "nonexistent_backbone"}), LoadMode.PRETRAINED, {}, None)


def test_build_video_backbone_mode_and_args_must_agree(tiny_cfg, tiny_assets, tmp_path):
    """PRETRAINED reads only assets, CHECKPOINT only ckpt_dir: any other combination raises before a build."""
    from openwam.model.video_backbone import LoadMode, build_video_backbone

    with pytest.raises(ValueError, match="PRETRAINED"):
        build_video_backbone(tiny_cfg, LoadMode.PRETRAINED, None, None)
    with pytest.raises(ValueError, match="PRETRAINED"):
        build_video_backbone(tiny_cfg, LoadMode.PRETRAINED, tiny_assets, str(tmp_path))
    with pytest.raises(ValueError, match="CHECKPOINT"):
        build_video_backbone(tiny_cfg, LoadMode.CHECKPOINT, tiny_assets, str(tmp_path))
    with pytest.raises(ValueError, match="CHECKPOINT"):
        build_video_backbone(tiny_cfg, LoadMode.CHECKPOINT, None, None)
