"""Deploy engine for sparse-history checkpoints (07 layout: H history frames + H proprio tokens -> action chunk).

The request carries the H history observations picked by :class:`ObsHistoryBuffer` (``conditions["history"]``).
The engine turns them into the training representation with the same functions the dataset uses
(``encode_proprio_in_anchor_cam``, the checkpoint's normalizer), conditions the model through
``architecture.prepare_sparse_condition`` (the backbone's training condition encode) and denoises the future
latents + action chunk with ``architecture.sample``. It depends on the model and the data-representation
functions only; nothing from the training loop or the training visualization.

Only yam_sharpa checkpoints have this layout. Each history observation needs ``image`` (the server-composed
canvas), ``state`` (48,) ``observation.state``, ``camera_pose`` (16,) ``camera_pose.ego_view`` and, for
``unify_*`` action modes, ``joint_angles`` (56,).
"""

import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.deploy.engine import BaseInferenceEngine, build_schedule

logger = logging.getLogger(__name__)


def _rows(history: list, key: str, *, required: bool = True):
    """history: H observation dicts; key: field name. Returns: (H, ...) float64 array, or None if absent and optional."""
    vals = [o.get(key) for o in history]
    if any(v is None for v in vals):
        if required:
            raise ValueError(f"sparse-history checkpoint: every observation needs obs[{key!r}]")
        return None
    return np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vals])


class SparseHistoryInferenceEngine(BaseInferenceEngine):
    """Joint video-action inference from an H-frame observation history.

    Args:
        cfg: merged deploy config (training config + deploy overrides; reads ``inference`` and ``dataloader``).
        architecture: WAM architecture loaded from the checkpoint (normalizer / action decoder attached).
        history_offsets: H ints, the checkpoint's ``dataloader.history_offsets`` (rows at 20 Hz).
    """

    require_architecture = True

    def __init__(self, cfg, architecture, *, history_offsets):
        super().__init__(cfg, architecture=architecture)
        dl = cfg.dataloader
        if OmegaConf.select(cfg, "dataloader.type") != "yam_sharpa":
            raise ValueError(f"sparse-history deploy supports yam_sharpa checkpoints only, got {dl.get('type')}")
        self.history_offsets = [int(o) for o in history_offsets]
        self.action_mode = str(dl.action_mode)
        self.num_future_frames = len(dl.future_offsets)
        # Action chunk = rows t .. t + num_frames - 2 (the dataset keeps num_frames - 1 targets).
        self.action_horizon = int(dl.num_frames) - 1
        self._decode_video = bool(OmegaConf.select(cfg, "optimization.decode_video", default=False))
        logger.info(
            "[sparse] history_offsets=%s, future frames=%d, action chunk=%d, action_mode=%s",
            self.history_offsets, self.num_future_frames, self.action_horizon, self.action_mode,
        )

    def _proprio(self, history: list):
        """History observations -> (raw (H, D) float32 in the frame-t camera, raw frame-t proprio (D,)).

        Same representation as ``YamSharpaDataset._proprio_history`` (wrists re-expressed in the frame-t ego
        camera, the action's frame) and as the request proprio the relative-action decoder composes with.
        """
        from openwam.dataloader.utils.yam_sharpa_repr import (
            encode_proprio,
            encode_proprio_in_anchor_cam,
            is_unify,
        )

        state = _rows(history, "state")
        cam = _rows(history, "camera_pose")
        joints = _rows(history, "joint_angles", required=is_unify(self.action_mode))
        raw = encode_proprio_in_anchor_cam(self.action_mode, state, joints, cam, cam[-1])
        raw_t = encode_proprio(self.action_mode, state[-1], None if joints is None else joints[-1])[0]
        return raw, raw_t

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        """Denoise the action chunk (and optionally the future video) for the current step.

        Args:
            conditions: dict with ``history`` (H preprocessed observation dicts, oldest first, from
                :class:`ObsHistoryBuffer`), ``prompt`` (str), optional ``seed`` (int, default 42) and the
                per-request schedule overrides of :func:`build_schedule`.

        Returns: dict ``actions`` (K, D) float32 in physical units (relative modes decoded to absolute targets),
            ``video`` list of F decoded future PIL canvases, or None unless ``optimization.decode_video``.
        """
        arch = self.architecture
        history = conditions["history"]
        if len(history) != len(self.history_offsets):
            raise ValueError(f"expected {len(self.history_offsets)} history observations, got {len(history)}")
        schedule, _, _ = build_schedule(self.cfg.inference, arch, conditions)

        raw, raw_t = self._proprio(history)
        proprio = arch.normalize_deploy_proprio(raw).unsqueeze(0)  # (1, H, D) normalized
        cond = arch.prepare_sparse_condition(
            [[o["image"] for o in history]],
            [conditions.get("prompt", "") or ""],
            proprio,
            num_future_frames=self.num_future_frames,
        )
        b, z, h, hl, wl = cond["first_frame_latents"].shape
        t_lat = h + arch.video_backbone.num_future_latents(self.num_future_frames)
        latents, actions = arch.sample(
            cond,
            schedule,
            latent_shape=(b, z, t_lat, hl, wl),
            action_shape=(1, self.action_horizon, arch.action_dim),
            seed=int(conditions.get("seed", 42)),
        )

        actions = actions[0].float().cpu().numpy()
        normalizer = getattr(arch, "normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)
        # Relative action modes: compose the chunk with the frame-t proprio (set by model_loader).
        decoder = getattr(arch, "action_decoder", None)
        if decoder is not None:
            actions = decoder(actions, raw_t)
        video = arch.video_backbone.decode_video(latents[:, :, h:]) if self._decode_video else None
        return {"actions": actions, "video": video}
