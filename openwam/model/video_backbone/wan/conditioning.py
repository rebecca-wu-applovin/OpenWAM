"""First-frame conditioning construction for the Wan TI2V backbone, deploy side.

Free functions — every backbone-owned input (``vae`` / ``dtype`` / ``device``) is
passed explicitly so this module stays Wan-internal (no backbone/ABC import).
Encode/decode goes through :mod:`wan.encode`; pixel helpers through
:mod:`wan.preprocess`.
"""

from __future__ import annotations

from torch import Tensor

from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.preprocess import generate_noise


def build_deploy_noise(*, height, width, num_frames, seed, rand_device, vae, dtype, device) -> Tensor:
    """Initial Gaussian latent noise for deploy (replaces NoiseInitializer)."""
    z_dim = vae.model.z_dim
    upsample = vae.upsampling_factor
    length = (num_frames - 1) // 4 + 1
    shape = (1, z_dim, length, height // upsample, width // upsample)
    return generate_noise(shape, seed=seed, rand_device=rand_device, dtype=dtype, device=device)


def finalize_ti2v_first_frame_latents(inputs_shared: dict, first_frame_image, *, vae, dtype, device):
    """Emit ``first_frame_latents`` for TI2V deploy: its ``seperated_timestep``
    DiT needs both ``fuse_vae_embedding_in_latents=True`` AND ``first_frame_latents``
    so frame-0 tokens get t=0 and base.generate can clean-replace ``latents[:, :, 0:1]``.

    Args:
        inputs_shared: dict, the deploy input dict; mutated in place.
        first_frame_image: PIL image, list of PIL images (the clean reference frames), or None.
        vae: the Wan VAE used to encode the reference frames.
        dtype / device: dtype and device of the emitted latents.

    Returns: None. With a reference, sets ``first_frame_latents`` (1, z, T_ref, h, w),
        ``fuse_vae_embedding_in_latents=True`` and ``num_clean_prefix_frames=0``; without one,
        removes ``first_frame_latents`` and resets both flags.
    """
    if first_frame_image is None:
        inputs_shared.pop("first_frame_latents", None)
        inputs_shared["fuse_vae_embedding_in_latents"] = False
        inputs_shared["num_clean_prefix_frames"] = 0
        return
    inputs_shared["fuse_vae_embedding_in_latents"] = True
    inputs_shared["num_clean_prefix_frames"] = 0
    ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
    ref_tensor = wan_encode.preprocess_video(ref_frames, dtype=dtype, device=device)
    ref_image_latents = wan_encode.encode_video(ref_tensor.to(device), vae=vae).to(
        dtype=dtype, device=device
    )
    inputs_shared["first_frame_latents"] = ref_image_latents
