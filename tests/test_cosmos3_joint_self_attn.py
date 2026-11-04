"""joint_self_attn (MoT) support for cosmos3_edge — CPU parity tests.

1. Split parity: ``post(pre → driver-style SDPA) == run_block`` per layer on the
   tiny random Edge-flavoured config. The split path uses the driver's exact
   flat single-``num_heads`` layout (KV heads pre-expanded 8→16-style), so this
   is the canary for GQA-expand / rope / residual mistakes.
2. Driver integration: with a frame mask whose latents are all clean (video cannot
   see action) and bidirectional among them, the video stream that comes out of
   ``DualSystemMoTDriver.run_joint_loop`` must equal the pure ``run_block``
   loop — exercising the rectangular prefix-KV mask end to end.
"""

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("diffusers")

from omegaconf import OmegaConf  # noqa: E402

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver  # noqa: E402
from openwam.model.architectures.utils.mask_modes import build_frame_mask  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit as cosmos3_dit  # noqa: E402
from openwam.model.video_backbone.cosmos3 import text as cosmos3_text  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import (  # noqa: E402
    Cosmos3OmniTransformer,
)
from tests._cosmos3_shell import edge_scalar_kwargs, unused_io_kwargs  # noqa: E402

MINI = dict(
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

# Nano-flavoured twin (SiLU + RMSNorm, und QK-norm, no k_norm_und_for_gen).
NANO_MINI = dict(MINI, qk_norm_for_text=True, use_und_k_norm_for_gen=False, hidden_act="silu", rms_norm_eps=1e-6)



def _visible(state):
    """state: BlockLoopState with hidden_states (B, S, dim). Returns: (S, S) all-True gen mask (fully visible)."""
    s = state.hidden_states.shape[1]
    return torch.ones(s, s, dtype=torch.bool)

def _fm(t: int, n_clean: int, clean_attention: str = "bidirectional"):
    """Frame mask of t latents, the first n_clean clean (mutual). Returns: (t+1, t+1) bool."""
    layout = OmegaConf.create({"num_latents": t, "num_hist_latents": n_clean})
    return build_frame_mask(layout, clean_attention, "mutual")


def _build_state(net, ids, lat, ncp=0, *, padded_und: bool = True):
    """``padded_und=False`` reproduces the unpadded production shape, where the
    backbone passes ``und_mask=None`` all the way into SDPA."""
    text_pos = cosmos3_text.text_mrope_positions(ids.numel(), float_positions=True)
    grid = cosmos3_text.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = cosmos3_text.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_und, sin_und = cosmos3_dit.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool) if padded_und else None
    ids_b = ids.unsqueeze(0).expand(lat.shape[0], -1)
    context, und_kv, _ = cosmos3_dit.run_und_tower(net, ids_b, und_mask, cos_und, sin_und)
    state = cosmos3_dit.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=context,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=ncp,
    )
    return state


@pytest.mark.parametrize("cfg", [MINI, NANO_MINI], ids=["edge", "nano"])
@pytest.mark.parametrize("padded_und", [True, False], ids=["und_mask_tensor", "und_mask_none"])
def test_split_parity_matches_run_block(padded_und, cfg):
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**cfg).eval()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, cfg["latent_channel"], 3, 2, 2)

    with torch.no_grad():
        ref = _build_state(net, ids, lat, padded_und=padded_und)
        split = _build_state(net, ids, lat, padded_und=padded_und)
        assert split.prefix_kv_len == 4
        assert (split.prefix_kv_mask is not None) is padded_und

        for i in range(len(net.layers)):
            ref = cosmos3_dit.run_block(net, i, ref, gen_mask=_visible(ref))

            q, k, v, post = cosmos3_dit.state_pre_attn(net, i, split)
            assert k.shape[1] == q.shape[1] + split.prefix_kv_len
            # Driver-style mixed attention: single num_heads, rectangular mask.
            n = MINI["num_attention_heads"]
            qh = q.view(q.shape[0], q.shape[1], n, -1).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], n, -1).transpose(1, 2)
            vh = v.view(v.shape[0], v.shape[1], n, -1).transpose(1, 2)
            mask = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool)
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
            out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)
            split = cosmos3_dit.state_post_attn(split, out, post)

            diff = (split.hidden_states - ref.hidden_states).abs().max().item()
            assert diff < 1e-5, f"layer {i}: split vs run_block diff {diff}"


class _TinyActionBackbone(nn.Module):
    """Minimal MoT-compatible action stream for driver integration tests."""

    def __init__(self, dim, num_layers, num_heads, head_dim):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        width = num_heads * head_dim
        self.q = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.k = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.v = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.o = nn.ModuleList(nn.Linear(width, dim) for _ in range(num_layers))

    def pre_attn_at_layer(self, layer_id, astate):
        x = astate.x_action
        return self.q[layer_id](x), self.k[layer_id](x), self.v[layer_id](x), {"layer": layer_id}

    def post_attn_at_layer(self, layer_id, astate, attn_out, post_state):
        astate.x_action = astate.x_action + self.o[post_state["layer"]](attn_out)
        return astate


@pytest.mark.parametrize("padded_und", [True, False], ids=["und_mask_tensor", "und_mask_none"])
def test_driver_video_blind_to_action_matches_run_block_loop(padded_und):
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)

    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        **unused_io_kwargs(),
        **edge_scalar_kwargs(),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        clip_fps=24.0,
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    # All 3 latents clean: the video rows read neither the action nor anything outside the (bidirectional) video.
    driver = DualSystemMoTDriver(vb, ab, frame_mask=_fm(3, 3), mot_checkpoint_mixed_attn=True)

    with torch.no_grad():
        ref = _build_state(net, ids, lat, padded_und=padded_und)
        for i in range(len(net.layers)):
            ref = cosmos3_dit.run_block(net, i, ref, gen_mask=_visible(ref))

        joint = _build_state(net, ids, lat, padded_und=padded_und)
        astate = types.SimpleNamespace(x_action=torch.randn(1, 5, MINI["hidden_size"]))
        joint_v, joint_a = driver.run_joint_loop(joint, astate)

    diff = (joint_v.hidden_states - ref.hidden_states).abs().max().item()
    assert diff < 1e-5, f"action-blind video stream deviates from run_block: {diff}"
    assert torch.isfinite(joint_a.x_action).all()


def _padded_state(net, poison: bool, pad: int = 2):
    """B=2 state whose second prompt is right-padded; optionally poison the
    padded und K/V so a masking mistake shows up as a numeric change."""
    torch.manual_seed(7)
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    und_mask = torch.tensor([[True] * 5, [True] * (5 - pad) + [False] * pad])
    lat = torch.randn(2, MINI["latent_channel"], 2, 2, 2)
    text_pos = cosmos3_text.text_mrope_positions(5, float_positions=True)
    grid = cosmos3_text.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = cosmos3_text.build_joint_positions(
        5, grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = cosmos3_dit.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    ctx, und_kv, _ = cosmos3_dit.run_und_tower(net, ids, und_mask, cos_u, sin_u)
    if poison:
        und_kv = [(k.clone(), v.clone()) for k, v in und_kv]
        for k, v in und_kv:  # sample 1's padded slots only
            k[1, 5 - pad : 5] = 1e4
            v[1, 5 - pad : 5] = 1e4
    return cosmos3_dit.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((2,), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=0,
    )


@pytest.mark.parametrize("n_clean", [1, 2], ids=["noisy_latent", "all_clean"])
def test_joint_loop_keeps_padded_und_keys_out(n_clean):
    """Padded und keys must not reach the video or action stream through MoT.

    ``run_joint_loop`` widens its cross-modal mask by ``prefix_kv_len`` and gates
    the prefix columns with ``prefix_kv_mask``. Poisoning the padded und K/V is
    the end-to-end check that the gate survives the widening whether or not the
    video rows see the action — the mask helper's own unit tests cannot see a wiring mistake here.
    """
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        **unused_io_kwargs(),
        **edge_scalar_kwargs(),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        clip_fps=24.0,
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    mode = f"n_clean={n_clean}"
    driver = DualSystemMoTDriver(vb, ab, frame_mask=_fm(2, n_clean), mot_checkpoint_mixed_attn=False)

    outs = []
    for poison in (False, True):
        state = _padded_state(net, poison)
        astate = types.SimpleNamespace(
            x_action=torch.arange(2 * 3 * MINI["hidden_size"]).float().view(2, 3, -1) * 1e-2
        )
        with torch.no_grad():
            v, a = driver.run_joint_loop(state, astate)
        outs.append((v.hidden_states, a.x_action))

    d_video = (outs[0][0] - outs[1][0]).abs().max().item()
    d_action = (outs[0][1] - outs[1][1]).abs().max().item()
    assert d_video < 1e-6, f"{mode}: padded und keys leaked into the video stream ({d_video:.3e})"
    assert d_action < 1e-6, f"{mode}: padded und keys leaked into the action stream ({d_action:.3e})"


def test_padded_und_poison_is_detectable():
    """Positive control for the test above: without the gate the poison must move
    the output, otherwise the leak assertions would pass vacuously."""
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    outs = []
    for poison in (False, True):
        state = _padded_state(net, poison)
        state.prefix_kv_mask = None  # drop the gate
        state.extras = {**state.extras, "und_mask": None}
        with torch.no_grad():
            for i in range(len(net.layers)):
                state = cosmos3_dit.run_block(net, i, state, gen_mask=_visible(state))
            outs.append(cosmos3_dit.finalize_block_loop(net, state))
    assert (outs[0] - outs[1]).abs().max().item() > 1e-3, "poison is inert; the leak tests prove nothing"


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("padded_und", [False, True])
@pytest.mark.parametrize("clean_attention", ["bidirectional", "causal"])
def test_action_blind_video_matches_direct_blocks_with_matching_mask(batch, padded_und, clean_attention):
    """One-way coupling (video never reads the action) without dropping the video's own mask: the MoT step on
    the joint mask equals the video-only ``run_block`` on the same frame mask's video block."""
    torch.manual_seed(17)
    net = Cosmos3OmniTransformer(**MINI).eval()
    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        **unused_io_kwargs(),
        **edge_scalar_kwargs(),
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
        clip_fps=24.0,
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    # All latents clean: the video rows never read the action; the clean rule is the one under test.
    frame_mask = _fm(3, 3, clean_attention)
    driver = DualSystemMoTDriver(vb, ab, frame_mask=frame_mask, mot_checkpoint_mixed_attn=True)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(batch, MINI["latent_channel"], 3, 2, 2)
    astate = types.SimpleNamespace(x_action=torch.randn(batch, 5, MINI["hidden_size"]))

    with torch.no_grad():
        direct = _build_state(net, ids, lat, ncp=1, padded_und=padded_und)
        joint = _build_state(net, ids, lat, ncp=1, padded_und=padded_und)
        for state in (direct, joint):
            if padded_und:
                padding_mask = state.extras["und_mask"].clone()
                padding_mask[-1, -1] = False
                state.extras["und_mask"] = padding_mask
                state.prefix_kv_mask = padding_mask
                for k, v in state.extras["und_kv"]:
                    k[-1, -1] = 1e4
                    v[-1, -1] = 1e4
        video_masks = vb.build_joint_attention_masks(frame_mask, direct, 0)
        joint_masks = vb.build_joint_attention_masks(frame_mask, joint, 5)
        assert joint_masks[0].dim() == (4 if padded_und else 2)
        for i in range(vb.num_layers):
            direct = vb.run_block(i, direct, video_masks[i])
            joint, astate = driver.step(i, joint, astate, attn_mask=joint_masks[i])
            torch.testing.assert_close(direct.hidden_states, joint.hidden_states, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vb.finalize(direct), vb.finalize(joint), rtol=1e-5, atol=1e-5)
        assert torch.isfinite(astate.x_action).all()
