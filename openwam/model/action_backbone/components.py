"""Building blocks of the ActionDiT action stream (timestep embedding / modulation, RMSNorm, 1D RoPE)."""

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding of a timestep (cos half, then sin half), computed in float64.

    Args:
        dim: int embedding width (even).
        position: (B,) timesteps.

    Returns:
        (B, dim) embedding in ``position``'s dtype.
    """
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_1d(head_dim: int, max_len: int = 1024, theta: float = 10000.0) -> torch.Tensor:  # noqa: B008
    """Precompute complex rotary frequencies for 1D RoPE.

    Returns a complex64 tensor of shape (max_len, head_dim // 2).

    Note:
        This mirrors FastWAM's `precompute_freqs_cis` in `wan_video_dit.py`.
        The default `max_len=1024` matches FastWAM to support action sequences
        up to 1024 steps (e.g., ~3.5s at 300Hz or ~7s at 150Hz).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).double() / head_dim))
    freqs = torch.outer(torch.arange(max_len).double(), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D rotary position embedding to Q or K.

    Mirrors FastWAM/Wan RoPE numerics by doing the complex multiply through a
    float64 complex view, then restoring the caller's dtype.

    Args:
        x:     (B, H, S, D) head-split tensor.
        freqs: (S, D // 2) complex frequencies from ``precompute_freqs_cis_1d``.

    Returns:
        Rotated tensor with the same dtype and shape as ``x``.
    """
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(x_c.device).view(1, 1, x_c.shape[-2], x_c.shape[-1])
    return torch.view_as_real(x_c * freqs).flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization over the last dim, with a learned per-channel scale.

    Args: dim: int normalized width. eps: float added to the mean square.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (..., dim). Returns: (..., dim) normalized in float32, cast back to x's dtype, then scaled."""
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection.

    Input: (B,) timestep scalar
    Output: (B, dim) timestep embedding
    """

    def __init__(self, freq_dim: int, dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Args: timestep: (B,) timesteps. Returns: (B, dim) embedding."""
        t = sinusoidal_embedding_1d(self.freq_dim, timestep)
        return self.mlp(t)


class TimestepModulation(nn.Module):
    """Projects timestep embedding to per-block modulation parameters.

    Input: (B, dim) from TimestepEmbedding
    Output: (B, n_params, dim)
    """

    def __init__(self, dim: int, n_params: int):
        super().__init__()
        self.n_params = n_params
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * n_params),
        )

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        """Args: t_embed: (B, dim) timestep embedding. Returns: (B, n_params, dim) modulation parameters."""
        return self.proj(t_embed).unflatten(-1, (self.n_params, -1))
