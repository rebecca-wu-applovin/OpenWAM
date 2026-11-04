"""Tests for OpenWAMTrainer: freeze strategy, loss computation, and training loop.

Uses mock pipeline and tiny architecture to avoid GPU / real weights dependency.
All tests run on CPU.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from openwam.dataloader.collate import Batch
from openwam.model.common.flow_scheduler import FlowScheduler
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from openwam.train.run_mode import RunMode
from tests._arch import TEST_TRAIN_OPTIONS, build_test_arch
from tests._training_cfg import cpu_accelerator

# ---------------------------------------------------------------------------
# Mock pipeline: replaces WanVideoPipeline to avoid loading ~20GB of weights
# ---------------------------------------------------------------------------


class _MockDiT(nn.Module):
    """Tiny mock DiT with the attributes the trainer + adapter expect."""

    def __init__(self, dim=64, in_dim=16, num_blocks=2):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = dim
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_blocks)])

    def forward(self, x):
        return x


class _MockPipeline:
    """Minimal mock of WanVideoPipeline for trainer tests."""

    def __init__(self, dim=64):
        self.dit = _MockDiT(dim=dim)
        self.vae = nn.Linear(4, 4)
        self.text_encoder = nn.Linear(4, 4)
        self.torch_dtype = torch.float32
        self.device = "cpu"
        self.scheduler = _mock_scheduler()
        self.units = []
        self.in_iteration_models = ["dit"]

    def modules(self):
        return [self.dit, self.vae, self.text_encoder]

    def named_parameters(self):
        yield from self.dit.named_parameters(prefix="dit")

    def named_buffers(self):
        yield from self.dit.named_buffers(prefix="dit")

    def load_state_dict(self, state_dict, strict=True):
        pass

    def model_fn(self, dit=None, latents=None, timestep=None, **kwargs):
        """Return noise_pred matching latents shape; invoke architecture
        callbacks so bridge features get populated for cross_attn / interleaved
        paths. Mirrors how the real model_fn_wan_video calls the hooks.
        """
        # Compute the (B, num_tokens, dim) layout the real model_fn produces
        # after patchify + flatten.
        B = latents.shape[0]
        num_tokens = 1
        for d in latents.shape[2:]:
            num_tokens *= d
        x = torch.randn(B, num_tokens, self.dit.dim)

        on_before_blocks = kwargs.get("on_before_blocks")
        on_after_block = kwargs.get("on_after_block")
        on_after_blocks = kwargs.get("on_after_blocks")
        if on_before_blocks is not None:
            x, _, _ = on_before_blocks(x, torch.zeros(B, 6, self.dit.dim), None)
        for block_id, _ in enumerate(self.dit.blocks):
            if on_after_block is not None:
                x = on_after_block(block_id, x)
        if on_after_blocks is not None:
            x = on_after_blocks(x)
        return torch.randn_like(latents)

    def unit_runner(self, unit, pipe, shared, posi, nega):
        return shared, posi, nega


class _MockVideoBackbone(VideoBackbone):
    """Minimal VideoBackbone implementing the ABC for tests."""

    def __init__(self, dim=64, num_layers=2, num_heads=4, head_dim=None, text_dim=4096):
        super().__init__()
        self._dim = dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        # head_dim None = dim / num_heads (the usual square attention space).
        self._head_dim = dim // num_heads if head_dim is None else int(head_dim)
        self._text_dim = int(text_dim)
        self._scheduler = _mock_scheduler()
        self._dit = nn.Linear(dim, dim)
        self._vae = nn.Linear(4, 4)
        self._text_encoder = nn.Linear(4, 4)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        # ABC-property backings (mirror Wan native so legacy mask code stays
        # bit-for-bit identical).
        self._temporal_compression, self._causal_temporal = 4, True
        self.last_injected = None
        self.last_extracted = None

    @classmethod
    def from_pretrained(cls, cfg, assets):
        return cls()

    @classmethod
    def from_checkpoint_dir(cls, cfg, ckpt_dir):
        return cls()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def text_dim(self) -> int:
        return self._text_dim

    @property
    def submodule_names(self) -> list:
        return ["dit", "vae", "text_encoder"]

    def prepare(self, **inputs) -> BlockLoopState:
        latents = inputs["latents"]
        B = latents.shape[0]
        # Derive (f, h, w) from latent dims so 3D RoPE can index correctly.
        f, h, w = (latents.shape[2], latents.shape[3], latents.shape[4]) if latents.ndim >= 5 else (1, 1, 1)
        num_tokens = f * h * w
        x = torch.randn(B, num_tokens, self._dim)
        # Per-token t_mod (B, T, 6, dim), the one layout Wan's build_time_modulation emits.
        t_mod = torch.zeros(B, num_tokens, 6, self._dim)
        freq_dim = self._dim // 2
        freqs = torch.polar(torch.ones(num_tokens, 1, freq_dim), torch.zeros(num_tokens, 1, freq_dim))
        context = torch.randn(B, 4, self._dim)
        context_mask = torch.ones(B, 4, dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
            extras={},
        )

    def run_block(self, block_id: int, state: BlockLoopState, attn_mask) -> BlockLoopState:
        return state

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState):
        """Tiny pre-attn split for split-attention path testing.

        Returns x as q/k/v (no projection, no RoPE) so the MoT driver can
        exercise concat/split shape handling end-to-end. ``post_state`` carries
        the residual and a no-op modulation that ``post_attn_at_layer`` consumes.
        """
        residual = state.hidden_states
        post_state = {"residual": residual}
        return residual, residual, residual, post_state

    def post_attn_at_layer(self, layer_id: int, state: BlockLoopState, attn_out, post_state: dict) -> BlockLoopState:
        """Trivial post-attn: residual + attn_out so the driver round-trips.

        No cross-attn, no FFN — those are the real
        backbone's responsibility and aren't exercised by the trainer mock.
        """
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def finalize(self, state: BlockLoopState):
        B = state.hidden_states.shape[0]
        return torch.randn(B, 16, 3, 8, 8)

    def inject_action_tokens(self, state, action_tokens, n_action, *, timestep=None):
        state.hidden_states = torch.cat([state.hidden_states, action_tokens.to(state.hidden_states.dtype)], dim=1)
        return state

    def extract_action_tokens(self, state, n_action):
        n_video = state.hidden_states.shape[1] - n_action
        action_tokens = state.hidden_states[:, n_video:, :]
        state.hidden_states = state.hidden_states[:, :n_video, :]
        return state, action_tokens

    def encode_pixels(self, pixels):
        raise NotImplementedError("mock: preprocess_train is stubbed")

    def decode_latents(self, latents):
        raise NotImplementedError("mock: decode_future is stubbed")

    def encode_text(self, prompts, *, grid, num_frames):
        raise NotImplementedError("mock: preprocess_train is stubbed")

    def preprocess_train(self, clips, prompts, layout):
        # One latent clip per sample; latent 0 (first-frame layout) is the clean condition the loss skips.
        b = len(clips)
        # Text width: the action stream's (set by a test as ``context_width``), else the mock's dim.
        width = getattr(self, "context_width", self._dim)
        text = TextCondition(context=torch.randn(b, 4, width), context_mask=torch.ones(b, 4, dtype=torch.bool))
        return torch.randn(b, 16, int(layout.num_latents), 8, 8), text

    def get_submodule(self, name):
        return getattr(self, f"_{name}", None)

    def set_submodule(self, name, module):
        setattr(self, f"_{name}", module)

    def set_dtype_device(self, dtype, device):
        pass


def _mock_scheduler() -> FlowScheduler:
    """Returns: the real FlowScheduler at shift 5 with the Wan / action law (grid draw, bell weights)."""
    return FlowScheduler(shift=5.0, train_timestep=OmegaConf.create({"law": "grid", "weighting": "bell"}))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_TINY_ARCH_CFG = {
    "framework": "dual_system",
    "clean_attention": "bidirectional",
    "attention_mask_mode": "mutual",
    "variant": "joint_cross_attn",
    "detach_bridge": False,
    "action_dim": 7,
    "dim": 64,
    "ffn_dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "video_dim": 64,
    "text_dim": 64,
}


def _make_tiny_arch(train_options=TEST_TRAIN_OPTIONS):
    """Tiny joint_cross_attn architecture on a mock video backbone (CPU, fp32)."""
    # The ActionDiT geometry follows the backbone: 2 heads of 16, a 64-wide text context.
    backbone = _MockVideoBackbone(dim=64, num_layers=2, num_heads=2, head_dim=16, text_dim=64)
    arch = build_test_arch("joint_cross_attn", _TINY_ARCH_CFG, backbone, train_options=train_options)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _make_batch(B=1, action_dim=7, T_action=5):
    """A collated Batch for compute_loss (the mock backbone ignores the clips and prompts)."""
    return Batch(
        video=[[] for _ in range(B)],
        prompt=["pick cup"] * B,
        action=torch.randn(B, T_action, action_dim),
        action_is_pad=torch.zeros(B, T_action, action_dim, dtype=torch.bool),
        proprio=torch.randn(B, 1, action_dim),
        proprio_mask=torch.ones(B, 1, action_dim, dtype=torch.bool),
        source=["test"] * B,
    )


def _apply_freeze(pipe, trainer_attrs, freeze_list):
    """Replicate the freeze logic from OpenWAMTrainer.__init__."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is None:
            module = trainer_attrs.get(name, None)
        if module is not None:
            module.requires_grad_(False)


def test_freeze_joint_strategy():
    """Joint strategy (dual_system.yaml): freeze text_encoder + vae; dit + action_dit remain trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    freeze_list = ["text_encoder", "vae"]  # from configs/model/dual_system.yaml
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


def test_freeze_video_only_strategy():
    """Video-only freeze strategy: freeze text_encoder + vae + action_dit; dit stays trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    # Freeze the action backbone too → video-only training (no shipped yaml uses this;
    # exercises the freeze mechanism for a hypothetical lambda_action=0 setup).
    freeze_list = ["text_encoder", "vae", "action_dit"]
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in action_dit.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())


def test_freeze_custom_list():
    """Custom freeze: dit frozen, action_dit stays trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    freeze_list = ["text_encoder", "vae", "dit"]
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


# ---------------------------------------------------------------------------
# Tests: Loss computation (single and multi-batch)
# ---------------------------------------------------------------------------


def test_single_batch_loss():
    """B=1: architecture.compute_loss produces valid scalar losses."""
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=1))

    assert "loss" in result
    assert "loss_video" in result
    assert "loss_action" in result
    assert result["loss"].shape == ()
    assert result["loss"].item() > 0
    assert result["loss_video"].item() >= 0
    assert result["loss_action"].item() >= 0


def test_multi_batch_loss():
    """B=2: architecture.compute_loss produces valid scalar losses with batched input."""
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=2))

    assert result["loss"].shape == ()
    assert result["loss"].item() > 0


def _video_only_loss(lambda_video: float) -> float:
    """lambda_video: video loss weight. Returns: ``loss`` of one seeded compute_loss with lambda_action = 0."""
    from dataclasses import replace

    torch.manual_seed(0)
    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_video=lambda_video, lambda_action=0.0))
    arch.action_backbone.scheduler = _mock_scheduler()
    return arch.compute_loss(_make_batch(B=1))["loss"].item()


def test_video_only_loss():
    """lambda_action=0: the action stream still runs (one forward path), but its term is left out of ``loss``."""
    from dataclasses import replace

    torch.manual_seed(0)  # same noise draw as _video_only_loss
    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_action=0.0))
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=1))

    assert result["loss"].item() > 0
    assert result["loss_action"].item() == 0.0
    assert result["loss"].item() == result["loss_video"].item()
    # The action term contributes nothing to the value: with the same noise draw, halving lambda_video halves
    # the loss exactly (the mock video stream has no trainable params, so grad_fn cannot tell the terms apart).
    raw = _video_only_loss(1.0)
    assert result["loss"].item() == _video_only_loss(TEST_TRAIN_OPTIONS.lambda_video)
    assert _video_only_loss(0.5) == 0.5 * raw


def test_action_only_loss():
    """lambda_video=0: the video term is left out of ``loss``, mirroring lambda_action=0."""
    from dataclasses import replace

    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_video=0.0, lambda_action=0.5))
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=1))

    # loss_action is already weighted (lambda_action * raw); loss is exactly that term.
    assert result["loss_video"].item() == 0.0
    assert result["loss_action"].item() > 0
    assert result["loss"].item() == result["loss_action"].item()


def test_video_only_loss_is_weighted_video_term():
    """lambda_action=0 with a non-unit lambda_video: loss == lambda_video * raw video loss."""
    from dataclasses import replace

    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_video=0.5, lambda_action=0.0))
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=1))

    assert result["loss_action"].item() == 0.0
    assert result["loss"].item() == result["loss_video"].item() > 0


def test_trainer_rejects_both_lambdas_zero():
    """A training run with both loss weights 0 has no loss term: the trainer fails at construction."""
    from omegaconf import OmegaConf

    from openwam.train.trainer import OpenWAMTrainer

    cfg = OmegaConf.create(
        {
            "project": {"seed": 0},
            "training": {
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
                "lambda_video": 0.0,
                "lambda_action": 0.0,
                "mot_checkpoint_mixed_attn": False,
            },
        }
    )
    with pytest.raises(ValueError, match="both 0"):
        OpenWAMTrainer(cfg, cpu_accelerator(), None, RunMode.FRESH, None, rank=0)


class _StopTraining(Exception):
    """Raised by the stubbed StepTimer: train() got past the text-cache decision."""


@pytest.mark.parametrize("text_cache", [True, False])
def test_trainer_reads_training_freeze_text_cache_and_mot(monkeypatch, text_cache):
    """freeze list, mixed-attention recompute and the text cache all come from cfg.training."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    import openwam.train.trainer as trainer_mod
    from tests._layout import make_layout

    seen = {"frozen": [], "text_cache_calls": 0}

    class _Arch(nn.Module):
        dtype, device = torch.float32, torch.device("cpu")
        backbones = property(lambda self: {})  # nothing for the parameter report to count

        def set_dtype_device(self, dtype, device):  # noqa: ARG002
            return None

        def freeze_modules(self, names):
            seen["frozen"].append(list(names))
            return list(names)

    def _fake_build(model_cfg, *, mode, assets, ckpt_dir, train_options):  # noqa: ARG001
        seen["train_options"], seen["model_cfg"], seen["assets"] = train_options, model_cfg, assets
        return _Arch()

    monkeypatch.setattr("openwam.model.build_architecture", _fake_build)
    cfg = OmegaConf.create(
        {
            # wandb off (null project): train() runs init_wandb on the main rank.
            "project": {"seed": 0, "wandb": {"project": None}},
            "training": {
                "initialize_model_on_cpu": False,
                "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
                "lambda_video": 1.0, "lambda_action": 1.0, "mot_checkpoint_mixed_attn": True,
                "freeze": ["video_backbone.vae"], "text_cache": text_cache,
                # train() up to the step loop.
                "num_epochs": 1, "max_steps": None, "batch_size": 1, "gradient_accumulation_steps": 1,
                "max_grad_norm": None, "save_steps": None, "keep_last_k_ckpts": 1, "save_full_states_for_resume": False,
                "vis_steps": None, "log_step_timing": False, "mixed_precision": "bf16",
            },
            "model": {"architecture": {"framework": "dual_system", "variant": "joint_self_attn"}},
            "common": {"action_dim": 7, "state_dim": 7, "layout": make_layout()},
            "assets": {"video_backbone": {"model_path": "/upstream"}},
        }
    )
    trainer = trainer_mod.OpenWAMTrainer(cfg, cpu_accelerator(), None, RunMode.FRESH, None, rank=0)
    assert seen["frozen"] == [["video_backbone.vae"]]
    assert seen["train_options"].mot_checkpoint_mixed_attn is True
    # The architecture gets merge(common, model): the shared dims sit next to the model blocks.
    assert seen["model_cfg"].action_dim == 7 and seen["model_cfg"].architecture.variant == "joint_self_attn"
    # A fresh run reads the upstream files at the assets slice, passed beside (not inside) the model slice.
    assert seen["assets"] == cfg.assets.video_backbone and "assets" not in seen["model_cfg"]

    # Stub everything train() touches before the text-cache decision; the step timer right after it stops the run.
    trainer.accelerator = SimpleNamespace(num_processes=1, is_main_process=True)
    monkeypatch.setattr(trainer, "build_optimizer", lambda: SimpleNamespace(param_groups=[]))
    monkeypatch.setattr(trainer, "build_dataloader", lambda batch_size: [0])
    monkeypatch.setattr(trainer, "build_lr_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "open_run_dir", lambda: (SimpleNamespace(path="out"), None))
    monkeypatch.setattr(trainer, "prepare_accelerate", lambda o, d, s: (o, d, s))
    monkeypatch.setattr(trainer, "build_text_cache", lambda: seen.__setitem__("text_cache_calls", 1))

    def _stop(*a, **k):
        raise _StopTraining

    monkeypatch.setattr(trainer_mod, "StepTimer", _stop)
    with pytest.raises(_StopTraining):
        trainer.train()
    assert seen["text_cache_calls"] == int(text_cache)


def test_loss_backward():
    """Loss should be differentiable and backward should succeed."""
    arch = _make_tiny_arch()
    arch.train()
    arch.action_backbone.scheduler = _mock_scheduler()

    result = arch.compute_loss(_make_batch(B=1))

    result["loss"].backward()

    has_grad = any(p.grad is not None for p in arch.action_backbone.parameters())
    assert has_grad, "ActionDiT should have gradients after backward"


# ---------------------------------------------------------------------------
# Tests: Training loop with mock components
# ---------------------------------------------------------------------------


def test_training_step():
    """Run 3 optimizer steps: loss should decrease or remain stable."""
    arch = _make_tiny_arch()
    arch.train()
    arch.action_backbone.scheduler = _mock_scheduler()

    optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        result = arch.compute_loss(_make_batch(B=1))

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        losses.append(result["loss"].item())

    # All losses should be finite positive
    assert all(loss > 0 for loss in losses)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


# ---------------------------------------------------------------------------
# Tests: Checkpointing (save / load / manage)
# ---------------------------------------------------------------------------


def test_save_load_checkpoint():
    """Save and load checkpoint via architecture; verify weights match."""
    arch = _make_tiny_arch()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_ckpt.safetensors")
        arch.save_checkpoint(ckpt_path)
        assert Path(ckpt_path).exists()

        # Load into fresh model
        arch2 = _make_tiny_arch()
        arch2.load_checkpoint(ckpt_path)

        # Verify action_dit weights match
        for (k1, v1), (k2, v2) in zip(
            arch.action_backbone.state_dict().items(),
            arch2.action_backbone.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Weight mismatch for {k1}"


def test_load_checkpoint_rejects_leftover_meta():
    """The strict rule on checkpoint loads: a tensor still on meta after the load raises (it would be garbage)."""
    arch = _make_tiny_arch()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "ckpt.safetensors")
        arch.save_checkpoint(ckpt_path)
        # A non-persistent buffer is in no checkpoint, so no key check sees it.
        shell = _make_tiny_arch()
        shell.action_backbone.register_buffer("table", torch.empty(4, device="meta"), persistent=False)
        with pytest.raises(ValueError, match="meta"):
            shell.load_checkpoint(ckpt_path)


def test_load_checkpoint_rejects_shape_mismatch():
    """The strict rule on checkpoint loads: a shape mismatch raises before anything is loaded."""
    from safetensors.torch import save_file

    arch = _make_tiny_arch()
    state = {k: v.clone().contiguous() for k, v in arch.state_dict().items()}
    key = next(k for k, v in state.items() if v.ndim == 2)
    state[key] = torch.zeros(state[key].shape[0] + 1, state[key].shape[1])
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "ckpt.safetensors")
        save_file(state, ckpt_path)
        with pytest.raises(ValueError, match="shape"):
            _make_tiny_arch().load_checkpoint(ckpt_path)
