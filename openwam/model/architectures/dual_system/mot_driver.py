"""DualSystem MoT (Mixture-of-Transformers) driver: per-layer mixed attention
across a video backbone and an action backbone.

Each backbone runs the prefix of one DiT block (``pre_attn_at_layer``) to yield
Q/K/V; the driver concatenates the two modalities along the sequence dimension,
runs a single mixed self-attention with a joint mask, splits the result, and
feeds each slice through that modality's block suffix (``post_attn_at_layer``).
Inspired by FastWAM-Joint's ``MoT.forward`` + ``_build_mot_attention_mask``
(``references/FastWAM/src/fastwam/models/wan22/{mot.py,fastwam_joint.py}``).
The driver owns no parameters — a plain Python class, absent from ``state_dict``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

if TYPE_CHECKING:
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState, MoTActionDiT
    from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone


class DualSystemMoTDriver:
    """Drives joint self-attention across a video backbone and an action backbone.

    Validates structural compatibility at construction time:

    - ``vb.num_layers == ab.num_layers`` (one joint attention per layer)
    - ``vb.num_heads == ab.num_heads`` and ``vb.head_dim == ab.head_dim``
      (so concatenated Q/K/V can run through a single attention; FastWAM's
      "two experts share the per-head attention space" pattern)

    Hidden dim (``vb.dim`` vs ``ab.dim``) does **not** need to match — each
    backbone owns its own Q/K/V projections that map their residual streams
    into the shared ``num_heads * head_dim`` attention space.

    The per-layer token masks come from the frame-level ``frame_mask`` the
    architecture built once (``utils.mask_modes.build_frame_mask``), expanded by
    ``vb.build_joint_attention_masks`` once per :meth:`run_joint_loop`.
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "MoTActionDiT",
        *,
        frame_mask: Tensor,
        mot_checkpoint_mixed_attn: bool,
    ) -> None:
        """
        Args:
            vb / ab: the video and action backbones.
            frame_mask: (T + 1, T + 1) bool frame-level mask of ``[latents, action block]`` (``build_frame_mask``).
            mot_checkpoint_mixed_attn: bool, grad-checkpoint the mixed attention in training.
        """
        if vb.num_layers != ab.num_layers:
            raise ValueError(
                f"DualSystemMoTDriver: video num_layers ({vb.num_layers}) must equal "
                f"action num_layers ({ab.num_layers}) for joint self-attention."
            )
        if vb.num_heads != ab.num_heads:
            raise ValueError(
                f"DualSystemMoTDriver: video num_heads ({vb.num_heads}) must equal "
                f"action num_heads ({ab.num_heads}). Per-head attention layout must match so "
                f"the concatenated Q/K/V can run through a single attention."
            )
        if vb.head_dim != ab.head_dim:
            raise ValueError(
                f"DualSystemMoTDriver: video head_dim ({vb.head_dim}) must equal action head_dim ({ab.head_dim})."
            )

        self.vb = vb
        self.ab = ab
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.frame_mask = frame_mask

    # ------------------------------------------------------------------
    # Mixed attention
    # ------------------------------------------------------------------

    def _mixed_attention(
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
    ) -> Tensor:
        """Single mixed self-attention over the concatenated [v, a] sequence.

        Inputs are ``(B, S, H*D)`` (the layout produced by ``pre_attn_at_layer``
        in both backbones). Output is the same layout. SDPA is used so an
        optional bool ``attn_mask`` (True = keep) can be honored.
        """
        n = self.num_heads
        q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(out, "b n s d -> b s (n d)", n=n)

    # ------------------------------------------------------------------
    # Per-layer step
    # ------------------------------------------------------------------

    def step(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionDiTState",
        attn_mask: Optional[Tensor] = None,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionDiTState"]:
        """Run one mixed-attention layer.

        Pulls Q/K/V from each backbone, concatenates along sequence,
        runs mixed attention (optionally checkpointed), splits the
        result, and feeds each slice through its post-attention suffix.

        ``attn_mask`` is this layer's joint ``[Sv+Sa, Sv+Sa]`` mask built once at
        :meth:`run_joint_loop`.

        When ``use_gradient_checkpointing`` is enabled (and the action
        backbone is in training mode), the entire pre→mixed→post triplet
        runs under :func:`torch.utils.checkpoint.checkpoint`, mirroring
        what the other architectures get from ``vb.run_block``, with a plain
        mixed attention inside (the outer wrapper already recomputes it).
        Otherwise ``mot_checkpoint_mixed_attn`` (training only) checkpoints
        just the mixed attention.
        """
        # The one place the checkpointing choice is made: which wrapper, and which mixed attention runs inside.
        if use_gradient_checkpointing and self.ab.training:
            return self._step_checkpointed(
                layer_id, vstate, astate, attn_mask=attn_mask, offload=use_gradient_checkpointing_offload
            )
        if self.mot_checkpoint_mixed_attn and self.ab.training:
            return self._step_impl(layer_id, vstate, astate, attn_mask, self._checkpointed_mixed_attention)
        return self._step_impl(layer_id, vstate, astate, attn_mask, self._mixed_attention)

    def _checkpointed_mixed_attention(
        self, q_cat: Tensor, k_cat: Tensor, v_cat: Tensor, attn_mask: Optional[Tensor]
    ) -> Tensor:
        """:meth:`_mixed_attention` under ``torch.utils.checkpoint`` (same inputs / output; activations recomputed)."""
        return torch.utils.checkpoint.checkpoint(
            self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
        )

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionDiTState",
        attn_mask: Optional[Tensor],
        mixed_attention: Callable[[Tensor, Tensor, Tensor, Optional[Tensor]], Tensor],
    ) -> Tuple["BlockLoopState", "ActionDiTState"]:
        """Unwrapped per-layer body. See :meth:`step` for the public entry point.

        ``mixed_attention`` is :meth:`_mixed_attention` or :meth:`_checkpointed_mixed_attention`, chosen by
        :meth:`step` (same math either way).

        INVARIANT (consumed by :meth:`_step_checkpointed`): this method must
        only reassign the two layer-varying tensor fields ``vstate.hidden_states`` and
        ``astate.x_action``. It MUST NOT mutate any layer-invariant
        field of ``vstate`` / ``astate`` in place — concretely, do not
        write into ``extras``, or mutate ``context`` /
        ``freqs`` / ``t_mod``. ``_step_checkpointed`` runs this body inside
        ``torch.utils.checkpoint`` against shallow copies of the state
        objects; only ``x`` / ``x_action`` are isolated, everything else is
        shared by reference. In-place writes there would be re-applied on
        the backward recompute and silently corrupt the outer state without
        any test detecting it.
        """
        vb = self.vb
        ab = self.ab

        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)

        if q_v.dtype != q_a.dtype:
            raise RuntimeError(
                f"DualSystemMoTDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"DualSystemMoTDriver: device mismatch at layer {layer_id} (video={q_v.device}, action={q_a.device})."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
        # passes it in for every layer. _step_impl does NOT rebuild the mask
        # itself; any direct caller must pass the same pre-built mask.

        mixed = mixed_attention(q_cat, k_cat, v_cat, attn_mask)

        attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        return vstate, astate

    def _step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionDiTState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ) -> Tuple["BlockLoopState", "ActionDiTState"]:
        """Run :meth:`_step_impl` under ``torch.utils.checkpoint``.

        ``_step_impl`` mutates ``vstate.hidden_states`` and ``astate.x_action``
        as it walks the block, which is fine on forward but lethal on
        backward: ``torch.utils.checkpoint`` re-runs the closure during
        recompute, and that second mutation would clobber the post-forward
        value held by the outer state objects, leaving them pointing at a
        recomputed activation from an earlier layer once backward unwinds.

        We avoid that by giving the closure shallow copies of the state
        containers (``copy.copy`` on both dataclasses — same field
        references, fresh wrappers). Layer-invariant fields like
        ``context`` / ``freqs`` / ``t_mod`` / ``extras``
        are still shared by reference (cheap), but the two tensor fields
        the body writes (``x`` / ``x_action``) live on the local copies so
        recompute never touches the outer references the caller and the
        autograd graph hold. After the checkpoint returns we propagate the
        new tensor values onto the outer ``vstate`` / ``astate`` for the
        next layer's iteration in ``run_joint_loop``.
        """

        def _run(vx: Tensor, ax: Tensor) -> Tuple[Tensor, Tensor]:
            # Fresh wrappers around the shared layer-invariant fields; only the two hidden states are local.
            local_vstate = copy.copy(vstate)
            local_astate = copy.copy(astate)
            local_vstate.hidden_states = vx
            local_astate.x_action = ax
            self._step_impl(layer_id, local_vstate, local_astate, attn_mask, self._mixed_attention)
            return local_vstate.hidden_states, local_astate.x_action

        vx0 = vstate.hidden_states
        ax0 = astate.x_action

        if offload:
            with torch.autograd.graph.save_on_cpu():
                new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)
        else:
            new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)

        vstate.hidden_states = new_vx
        astate.x_action = new_ax
        return vstate, astate

    def run_joint_loop(
        self,
        vstate: "BlockLoopState",
        astate: "ActionDiTState",
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionDiTState"]:
        """Run the full per-layer loop over both modalities.

        Expands the frame mask to per-layer token masks once (token counts are
        layer-invariant during one forward) and reuses them across layers. The
        two ``use_gradient_*`` flags are forwarded to :meth:`step` so each layer
        may opt into step-level activation checkpointing.
        """
        # Action tokens in the joint sequence.
        s_action = astate.x_action.shape[1]
        # The backbone owns its token order (views, text / proprio tokens, prefix keys); the rules are the frame mask's.
        layer_masks = self.vb.build_joint_attention_masks(self.frame_mask, vstate, s_action)
        if len(layer_masks) != self.num_layers:
            raise ValueError(
                f"build_joint_attention_masks returned {len(layer_masks)} masks for {self.num_layers} layers"
            )

        for layer_id in range(self.num_layers):
            vstate, astate = self.step(
                layer_id,
                vstate,
                astate,
                attn_mask=layer_masks[layer_id],
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vstate, astate


__all__ = ["DualSystemMoTDriver"]
