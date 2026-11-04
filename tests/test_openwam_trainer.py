"""Tests for OpenWAMTrainer: freeze strategy, loss computation, and training loop.

Uses mock pipeline and tiny architecture to avoid GPU / real weights dependency.
All tests run on CPU.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from openwam.dataloader.collate import Batch
from openwam.model.flow_grid import sample_grid
from openwam.model.video_backbone.base import BlockLoopState, TextCondition, VideoBackbone
from tests._arch import TEST_TRAIN_OPTIONS, build_test_arch

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
        self.scheduler = _MockScheduler()
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

    def __init__(self, dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self._dim = dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._scheduler = _MockScheduler()
        self._dit = nn.Linear(dim, dim)
        self._vae = nn.Linear(4, 4)
        self._text_encoder = nn.Linear(4, 4)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        # ABC-property backings (mirror Wan native so legacy mask code stays
        # bit-for-bit identical).
        self._temporal_compression, self._causal_temporal = 4, True
        self._shift_video = 5.0
        self.last_injected = None
        self.last_extracted = None

    @classmethod
    def from_pretrained(cls, cfg):
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
        return self._dim // self._num_heads

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def text_dim(self) -> int:
        return 4096

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


class _MockScheduler:
    """Minimal scheduler mock for loss tests."""

    num_train_timesteps = 1000

    def __init__(self, n=1000):
        self.timesteps = torch.linspace(1, 0, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)

    def set_timesteps(self, n, **kwargs):
        self.timesteps = torch.linspace(1, 0, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def sample_train(self, batch_size, *, device, dtype):
        """Uniform grid draw, like the real grid schedulers. Returns (timesteps, sigmas, weights), each (B,)."""
        return sample_grid(
            self.timesteps, self.sigmas, self.linear_timesteps_weights, batch_size, device=device, dtype=dtype
        )

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


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
    "bridge_layers": (0, 1),
}


def _make_tiny_arch(train_options=TEST_TRAIN_OPTIONS):
    """Tiny joint_cross_attn architecture on a mock video backbone (CPU, fp32)."""
    arch = build_test_arch(
        "dual_system_cross_attn", _TINY_ARCH_CFG, _MockVideoBackbone(dim=64, num_layers=2), train_options=train_options
    )
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


def test_resolve_architecture_config_merges_action_backbone_fields():
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "dual_system",
            "clean_attention": "bidirectional",
            "attention_mask_mode": "mutual",
            "variant": "joint_cross_attn",
            "detach_bridge": False,
            "action_dim": 7,
        },
        action_backbone={"dim": 64, "ffn_dim": 128, "num_heads": 2},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=64, num_dit_layers=12)

    assert resolved.registry_name == "dual_system_cross_attn"
    assert resolved.params["dim"] == 64
    assert resolved.params["ffn_dim"] == 128
    assert resolved.params["num_heads"] == 2
    assert resolved.params["video_dim"] == 64


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
    arch.action_backbone.scheduler = _MockScheduler()

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
    arch.action_backbone.scheduler = _MockScheduler()

    result = arch.compute_loss(_make_batch(B=2))

    assert result["loss"].shape == ()
    assert result["loss"].item() > 0


def _video_only_loss(lambda_video: float) -> float:
    """lambda_video: video loss weight. Returns: ``loss`` of one seeded compute_loss with lambda_action = 0."""
    from dataclasses import replace

    torch.manual_seed(0)
    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_video=lambda_video, lambda_action=0.0))
    arch.action_backbone.scheduler = _MockScheduler()
    return arch.compute_loss(_make_batch(B=1))["loss"].item()


def test_video_only_loss():
    """lambda_action=0: the action stream still runs (one forward path), but its term is left out of ``loss``."""
    from dataclasses import replace

    torch.manual_seed(0)  # same noise draw as _video_only_loss
    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_action=0.0))
    arch.action_backbone.scheduler = _MockScheduler()

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
    arch.action_backbone.scheduler = _MockScheduler()

    result = arch.compute_loss(_make_batch(B=1))

    # loss_action is already weighted (lambda_action * raw); loss is exactly that term.
    assert result["loss_video"].item() == 0.0
    assert result["loss_action"].item() > 0
    assert result["loss"].item() == result["loss_action"].item()


def test_video_only_loss_is_weighted_video_term():
    """lambda_action=0 with a non-unit lambda_video: loss == lambda_video * raw video loss."""
    from dataclasses import replace

    arch = _make_tiny_arch(replace(TEST_TRAIN_OPTIONS, lambda_video=0.5, lambda_action=0.0))
    arch.action_backbone.scheduler = _MockScheduler()

    result = arch.compute_loss(_make_batch(B=1))

    assert result["loss_action"].item() == 0.0
    assert result["loss"].item() == result["loss_video"].item() > 0


def test_trainer_rejects_both_lambdas_zero():
    """A training run with both loss weights 0 has no loss term: the trainer fails at construction."""
    from omegaconf import OmegaConf

    from openwam.train.openwam_trainer import OpenWAMTrainer

    cfg = OmegaConf.create(
        {
            "project": {"seed": None},
            "training": {
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
                "lambda_video": 0.0,
                "lambda_action": 0.0,
            },
        }
    )
    with pytest.raises(ValueError, match="both 0"):
        OpenWAMTrainer(cfg, accelerator=None, dataset=None)


def test_loss_backward():
    """Loss should be differentiable and backward should succeed."""
    arch = _make_tiny_arch()
    arch.train()
    arch.action_backbone.scheduler = _MockScheduler()

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
    arch.action_backbone.scheduler = _MockScheduler()

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


def test_manage_checkpoints():
    """manage_checkpoints should keep only the latest K files."""
    from openwam.train.utils.checkpointing import manage_checkpoints

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 fake checkpoints
        for i in range(1, 6):
            path = Path(tmpdir) / f"checkpoint_step_{i * 100}.safetensors"
            path.write_text("fake")

        manage_checkpoints(tmpdir, keep_last_k=2)

        remaining = sorted(Path(tmpdir).glob("checkpoint_step_*"))
        assert len(remaining) == 2
        names = [r.name for r in remaining]
        assert "checkpoint_step_400.safetensors" in names
        assert "checkpoint_step_500.safetensors" in names


def test_architecture_variant_strings_for_sampler_guard():
    """The guard keys on canonical.variant == 'joint_self_attn'; pin the variant
    string for every architecture so the guard cannot silently drift."""
    import openwam.model.architectures  # noqa: F401  (run every @register_architecture)
    from openwam.model.architectures.registry import normalize_architecture_spec

    assert normalize_architecture_spec("dual_system_self_attn").variant == "joint_self_attn"
    assert normalize_architecture_spec("dual_system_cross_attn").variant != "joint_self_attn"
