# Vendored into OpenWAM from black-forest-labs/flux-action @ 830f1ce (src/flux_action/models/transformer.py),
# Apache-2.0. Changes: the upstream forwards (``JointSingleSeq`` / ``ModeBlock`` / ``SingleStreamBlock``) and the
# attention-backend plumbing they used (attn_mode, flash / cuDNN / packed-window attention, PAG) are removed; OpenWAM
# drives the blocks from ``flux/backbone.py`` with its own attention.
# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JointSingleSeq DiT. Module tree matches the production checkpoint layout so safetensors keys load without renaming."""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn


@dataclass
class JointSingleSeqParams:
    in_channels: dict = field(
        default_factory=lambda: {
            "video": 96,
            "video_cond": 96,
        }
    )
    sequence: dict = field(
        default_factory=lambda: {
            "x_video": "video",
            "x_video_cond": "video_cond",
        }
    )
    vec_in_dim: int | None = 768
    context_in_dim: int = 20480
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    depth_late_blocks: int = 0
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 10000
    mlp_ratio: float = 3.0
    qkv_bias: bool = False
    gate_type: str | None = None


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0, "RoPE dimension must be even"
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(out_dtype if out_dtype is not None else t.dtype)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        out = self.lin(F.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if vec.ndim == 2:
            vec = vec[:, None, :]
        s = self.adaLN_modulation[0](vec)  # type: ignore
        shift_w, scale_w = self.adaLN_modulation[1].weight.chunk(2)

        # x = (1 + scale) * self.norm_final(x) + shift
        # using in place ops and avoid materializing full (seq, 2*hidden) projection to reduce memory
        x = self.norm_final(x)
        x.mul_(F.linear(s, scale_w).add_(1))
        x.add_(F.linear(s, shift_w))
        return self.linear(x)


class ModeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        gate_type: str | None = None,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1
        self.gate_type = gate_type

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        if gate_type is not None:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")

class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        gate_type: str | None = None,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2 if use_gated_mlp else 1
        self.gate_type = gate_type

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        if gate_type is not None:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation() if use_gated_mlp else nn.GELU(approximate="tanh")

class JointSingleSeq(nn.Module):
    def __init__(self, params: JointSingleSeqParams = JointSingleSeqParams()):
        super().__init__()
        self.params = params

        self.sequence = params.sequence
        for _, v in params.sequence.items():
            assert v in params.in_channels, (params.sequence, params.in_channels)

        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.depth = params.depth
        self.depth_single_blocks = params.depth_single_blocks
        self.depth_late_blocks = params.depth_late_blocks
        self.mlp_ratio = params.mlp_ratio
        self.qkv_bias = params.qkv_bias
        self.gate_type = params.gate_type

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.emb_in = nn.ModuleDict(
            {m: nn.Linear(c, params.hidden_size, bias=False) for m, c in params.in_channels.items()}
        )
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size, bias=False)

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size, disable_bias=True)
        self.vector_in: nn.Module = (
            MLPEmbedder(params.vec_in_dim, params.hidden_size, disable_bias=True)
            if params.vec_in_dim is not None
            else nn.Identity()
        )

        content_modalities = sorted(params.in_channels.keys())
        self.early_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modalities
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                **{
                    m: Modulation(params.hidden_size, double=False, disable_bias=True)
                    for m in content_modalities
                },
                "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
            }
        )
        if params.depth_late_blocks > 0:
            self.late_stream_modulations = nn.ModuleDict(
                {
                    **{
                        m: Modulation(params.hidden_size, double=False, disable_bias=True)
                        for m in content_modalities
                    },
                    "txt": Modulation(params.hidden_size, double=False, disable_bias=True),
                }
            )

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        ModeBlock(
                            params.hidden_size,
                            params.num_heads,
                            mlp_ratio=params.mlp_ratio,
                            qkv_bias=params.qkv_bias,
                            gate_type=params.gate_type,
                        )
                        for _ in range(params.depth)
                    ]
                )
                for m in content_modalities
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                ModeBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    gate_type=params.gate_type,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    gate_type=params.gate_type,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if params.depth_late_blocks > 0:
            self.late_content_mode_blocks = nn.ModuleDict(
                {
                    m: nn.ModuleList(
                        [
                            ModeBlock(
                                params.hidden_size,
                                params.num_heads,
                                mlp_ratio=params.mlp_ratio,
                                qkv_bias=params.qkv_bias,
                                gate_type=params.gate_type,
                            )
                            for _ in range(params.depth_late_blocks)
                        ]
                    )
                    for m in content_modalities
                }
            )
            self.late_txt_mode_blocks = nn.ModuleList(
                [
                    ModeBlock(
                        params.hidden_size,
                        params.num_heads,
                        mlp_ratio=params.mlp_ratio,
                        qkv_bias=params.qkv_bias,
                        gate_type=params.gate_type,
                    )
                    for _ in range(params.depth_late_blocks)
                ]
            )

        self.final_layer = nn.ModuleDict(
            {m: LastLayer(params.hidden_size, c) for m, c in params.in_channels.items()}
        )

    def embed_timesteps(self, ts: Tensor, out_dtype: torch.dtype) -> Tensor:
        """Sinusoid + ``time_in`` in the precision of ``time_in``'s own weights, outside autocast.

        The reference trainer keeps this MLP in fp32 under bf16 mixed precision (its own FSDP group)
        and casts the result where the modulations consume it; :func:`shard_dit` does the same, and a
        bf16 inference model computes it in bf16 as before.
        """
        weight_dtype = self.time_in.in_layer.weight.dtype
        with torch.autocast(device_type=ts.device.type, enabled=False):
            vec = self.time_in(timestep_embedding(ts, 256, out_dtype=weight_dtype))
        return vec.to(out_dtype)
