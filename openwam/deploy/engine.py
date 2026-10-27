"""InferenceEngine: one sparse-history yam_sharpa checkpoint behind a plain observation -> action-plan interface.

The engine owns the model and all of its I/O; callers (a server, an in-process script) never see tensors, the
normalizer or the action representation. One :class:`Session` is one episode: it keeps that episode's observation
history (:class:`ObsHistoryBuffer`), so sessions never share state.

Per plan (``Session.plan``), the same functions the training dataset uses:
Observations use the LeRobot dataset's feature names (what the model was trained on, and what a simulator's
LeRobot-style preprocessor emits), one flat dict per step:
    ``observation.images.<camera>`` uint8 (H, W, 3) for every camera of ``dataloader.camera_layout``,
    ``observation.state`` (48,), ``camera_pose.ego_view`` (16,) row-major base_from_camera, ``joint_angles`` (56,)
    (read for ``unify_*`` only), ``task`` str (the prompt, verbatim).

  1. camera images -> one vstack canvas (``resize_camera_tile`` LANCZOS tiles + ``assemble_vstack_layout``);
  2. record it in the history; take the H history observations (``history_rows``: clamped to the episode start);
     ``history_layout: panel`` tiles the 4 past canvases into one 2x2 panel (``compose_history_panel``);
  3. proprio of every history row in the frame-t camera (``encode_proprio_in_anchor_cam``), normalized;
  4. ``prepare_sparse_condition`` + seeded ``sample`` -> the normalized action chunk;
  5. unnormalize, decode ``*_rel`` chunks to absolute targets with the frame-t proprio, keep the first K steps;
  6. split into wrist poses + fingertips / finger joints (``split_hand_targets``).
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from openwam.deploy.history_buffer import ObsHistoryBuffer

logger = logging.getLogger(__name__)

# Action α-shift of backbones that leave ``shift_action`` unset: the pre-refactor deploy's value.
DEFAULT_ACTION_SHIFT = 5.0


@dataclass(frozen=True)
class EngineSpec:
    """What a checkpoint serves; plain values only, so ``dataclasses.asdict`` can cross a process boundary."""

    ckpt: str                    # <run dir>/<checkpoint file>
    action_mode: str             # dataloader.action_mode
    action_dim: int              # D of the returned physical actions (48 fingertips_* / 62 unify_*)
    hand_targets: str            # the plan's hand key: "tips" (fingertips_*) or "fingers" (unify_*)
    chunk: int                   # actions the model generates per plan (dataloader.num_frames - 1)
    steps_per_plan: int          # K, actions returned per plan = steps executed between two observations
    action_fps: int              # dataloader.fps, rate of the actions (and of the robot's control steps)
    history_offsets: tuple       # dataloader.history_offsets, in action steps
    history_layout: str          # dataloader.history_layout (frames | panel)
    camera_layout: tuple         # dataloader.camera_layout: the image keys of an observation, top to bottom


class InferenceEngine:
    """The loaded model + its I/O. Build with :meth:`from_config`; plan through :meth:`new_session`.

    Args:
        architecture: the loaded WAM architecture (normalizer attached), eval mode.
        normalizer: the checkpoint's action / proprio normalizer (``model_loader.build_normalizer``).
        schedule: (t_video, t_action) denoise schedule (``make_schedule``), fixed for the engine's life.
        spec: :class:`EngineSpec`.
        height, width: canvas size (dataloader.height / width).
        num_future_frames: F, future video frames the sampled latents cover (len(dataloader.future_offsets)).
        seed: default episode seed (deploy.policy.seed).
        decode_video: also decode the future video of every plan (debugging).
    """

    def __init__(self, architecture, normalizer, schedule, spec: EngineSpec, *, height: int, width: int,
                 num_future_frames: int, seed: int, decode_video: bool):
        from openwam.dataloader.utils.yam_sharpa_repr import is_relative, is_unify

        self.architecture, self.normalizer, self.schedule, self.spec = architecture, normalizer, schedule, spec
        self.height, self.width, self.num_future_frames = int(height), int(width), int(num_future_frames)
        self.default_seed, self.decode_video = int(seed), bool(decode_video)
        # Representation switches, resolved once from action_mode.
        self.unify, self.relative = is_unify(spec.action_mode), is_relative(spec.action_mode)
        self.panel = spec.history_layout == "panel"

    @classmethod
    def from_config(cls, cfg) -> "InferenceEngine":
        """Load the checkpoint named by ``cfg.deploy.ckpt``.

        Args:
            cfg: frozen cfg from :func:`openwam.deploy.config.build_deploy_cfg` (checkpoint cfg + ``deploy``).
        Returns:
            InferenceEngine on ``cfg.deploy.server.device``.
        """
        from openwam.deploy.model_loader import DTYPES, build_normalizer, load_architecture
        from openwam.model.denoise_schedule import make_schedule

        dl, dep = cfg.dataloader, cfg.deploy
        arch = load_architecture(cfg.model, dep.ckpt.dir, dep.ckpt.name,
                                 dtype=DTYPES[str(cfg.training.mixed_precision)], device=dep.server.device)
        normalizer = build_normalizer(dl, dep.ckpt.dir)
        arch.attach_normalizer(normalizer)  # normalize_deploy_proprio reads it
        # Both streams' α-shifts come from the checkpoint's backbones; the sampler config sets the trajectory.
        shift = arch.action_backbone.shift_action
        sm = dep.sampler
        schedule = make_schedule(
            sm.denoise_mode, video_scheduler=arch.video_scheduler, action_scheduler=arch.action_scheduler,
            num_steps=sm.denoise_steps, shift=DEFAULT_ACTION_SHIFT if shift is None else shift,
            shift_video=arch.video_backbone.shift_video, lead=sm.lead_modality, alpha=sm.variance_shift_alpha,
            offset=sm.linear_offset)
        spec = EngineSpec(
            ckpt=f"{dep.ckpt.dir}/{dep.ckpt.name}", action_mode=str(dl.action_mode),
            action_dim=_raw_dim(dl.action_mode), hand_targets=_hand_targets(dl.action_mode),
            chunk=int(dl.num_frames) - 1,
            steps_per_plan=int(dep.policy.steps_per_plan), action_fps=int(dl.fps),
            history_offsets=tuple(int(o) for o in dl.history_offsets), history_layout=str(dl.history_layout),
            camera_layout=tuple(str(c) for c in dl.camera_layout))
        logger.info("[engine] %s", spec)
        return cls(arch, normalizer, schedule, spec, height=dl.height, width=dl.width,
                   num_future_frames=len(dl.future_offsets), seed=dep.policy.seed,
                   decode_video=sm.decode_video)

    def new_session(self, seed=None) -> "Session":
        """A new episode. seed: int sampling seed of every plan of the episode; None = ``deploy.policy.seed``."""
        return Session(self, self.default_seed if seed is None else int(seed))

    # -- one plan, called by Session ------------------------------------------------------------------------------

    def observation(self, obs: dict) -> dict:
        """Client observation -> the record the history keeps.

        Args:
            obs: LeRobot-named features: ``observation.images.<camera>`` uint8 (H, W, 3) for every
                ``camera_layout`` key, ``observation.state`` (48,), ``camera_pose.ego_view`` (16,), ``joint_angles``
                (56,) (``unify_*`` only), ``task`` str.
        Returns:
            dict ``canvas`` PIL (width x height), ``state`` (48,) float64, ``camera_pose`` (16,) float64,
            ``joint_angles`` (56,) float64 (``unify_*``).
        """
        from PIL import Image

        from openwam.dataloader.transforms.multiview import assemble_vstack_layout, resize_camera_tile

        cams = self.spec.camera_layout
        missing = [c for c in cams if c not in obs]
        if missing:
            raise ValueError(f"observation lacks {missing}; the checkpoint's cameras are {list(cams)}")
        # Each camera straight to its tile with LANCZOS, then the training composer: byte-equal to training.
        tile_h = self.height // len(cams)
        tiles = {c: resize_camera_tile(Image.fromarray(np.asarray(obs[c], dtype=np.uint8)), tile_h,
                                       self.width) for c in cams}
        rec = {
            "canvas": assemble_vstack_layout(tiles, list(cams), out_h=self.height, out_w=self.width),
            # observation.state is float32 data; the pre-refactor server rounded it to float32 as well.
            "state": np.asarray(obs["observation.state"], dtype=np.float32).astype(np.float64).reshape(-1),
            "camera_pose": np.asarray(obs["camera_pose.ego_view"], dtype=np.float64).reshape(-1),
        }
        if self.unify:
            rec["joint_angles"] = np.asarray(obs["joint_angles"], dtype=np.float64).reshape(-1)
        return rec

    @torch.no_grad()
    def plan(self, window: list, prompt: str, seed: int) -> dict:
        """Plan from the H history records of one step.

        Args:
            window: H records from :meth:`observation`, oldest first (``ObsHistoryBuffer.window()``).
            prompt: str, the task prompt (fed to the model verbatim).
            seed: int, seeds the video and action noise.
        Returns:
            dict ``actions`` (K, D) physical absolute targets, ``wrist`` (K, 2, 4, 4) and ``tips`` (K, 2, 5, 3) or
            ``fingers`` (K, 2, 22) (``split_hand_targets``); plus ``video`` (F future PIL canvases) with
            ``deploy.sampler.decode_video``.
        """
        from openwam.dataloader.transforms.multiview import compose_history_panel
        from openwam.dataloader.utils.yam_sharpa_repr import (
            decode_action,
            encode_proprio,
            encode_proprio_in_anchor_cam,
            split_hand_targets,
        )

        arch, mode = self.architecture, self.spec.action_mode
        state = np.stack([r["state"] for r in window])
        cam = np.stack([r["camera_pose"] for r in window])
        joints = np.stack([r["joint_angles"] for r in window]) if self.unify else None
        # Proprio of every history row in the frame-t camera (the action's frame), as the dataset builds it.
        raw = encode_proprio_in_anchor_cam(mode, state, joints, cam, cam[-1])
        proprio = arch.normalize_deploy_proprio(raw).unsqueeze(0)  # (1, H, D')
        frames = [r["canvas"] for r in window]
        if self.panel:
            frames = [compose_history_panel(frames[:4])] + frames[4:]  # (2x2 panel of the past, frame t)
        cond = arch.prepare_sparse_condition([frames], [prompt], proprio, num_future_frames=self.num_future_frames)
        b, z, h, hl, wl = cond["first_frame_latents"].shape
        t_lat = h + arch.video_backbone.num_future_latents(self.num_future_frames)
        latents, actions = arch.sample(cond, self.schedule, latent_shape=(b, z, t_lat, hl, wl),
                                       action_shape=(1, self.spec.chunk, arch.action_dim), seed=seed)

        actions = self.normalizer.unnormalize(actions[0].float().cpu().numpy())
        if self.relative:
            # Deltas w.r.t. frame t -> absolute targets, composed with the frame-t raw proprio.
            raw_t = encode_proprio(mode, state[-1], None if joints is None else joints[-1])[0]
            actions = decode_action(mode, actions, raw_t)
        actions = actions[: self.spec.steps_per_plan]
        out = {"actions": actions, **split_hand_targets(mode, actions)}
        if self.decode_video:
            out["video"] = arch.video_backbone.decode_video(latents[:, :, h:])
        return out


class Session:
    """One episode of one client: its observation history and seed. Never shared between clients.

    Args:
        engine: the :class:`InferenceEngine`.
        seed: int, sampling seed of every plan of this episode.
    """

    def __init__(self, engine: InferenceEngine, seed: int):
        self.engine, self.seed = engine, int(seed)
        s = engine.spec
        self.history = ObsHistoryBuffer(s.history_offsets, action_fps=s.action_fps, steps_per_obs=s.steps_per_plan)

    def plan(self, obs: dict) -> dict:
        """Record this step's observation and plan the K actions to execute before the next one.

        Args:
            obs: one observation (:meth:`InferenceEngine.observation`).
        Returns:
            as :meth:`InferenceEngine.plan`.
        """
        self.history.append(self.engine.observation(obs))
        return self.engine.plan(self.history.window(), str(obs["task"]), self.seed)


def _hand_targets(action_mode) -> str:
    """action_mode: str. Returns: the key ``split_hand_targets`` puts the hand under ("fingers" / "tips")."""
    from openwam.dataloader.utils.yam_sharpa_repr import is_unify

    return "fingers" if is_unify(str(action_mode)) else "tips"


def _raw_dim(action_mode) -> int:
    """action_mode: str. Returns: D, width of the physical action (``yam_sharpa_repr.raw_dim``)."""
    from openwam.dataloader.utils.yam_sharpa_repr import raw_dim

    return raw_dim(str(action_mode))
