"""Per-model action codecs: Canonical (wrist pose + hand joints) <-> each checkpoint's action vector.

A codec fixes three things for one model:
  rot_format  how a rotation is written (canonical.ROT_DIMS)
  reference   what a wrist action is relative to (below)
  layout      where each field sits in the model's action vector, its width, and the mask

Hands are always absolute joint angles (every checkpoint keeps its gripper absolute).

References (verified against upstream; file:line):
  absolute          T_k
  prev_step         dT_k = T_{k-1}^-1 T_k, over [anchor, a_0, ..., a_{T-1}]
                    Cosmos3 'backward_framewise' (pose_utils.py:479, _apply... :455 current @ delta_T).
                    LDA-1B calculate_delta_eef (lda/utils/rotation_convert.py:67-98) aligns to frame 0 and
                    then takes T_{0->i}^-1 T_{0->i+1}, which is algebraically T_i^-1 T_{i+1}. LDA's anchor is
                    the first ACTION of the chunk (datasets.py:957-1000 diffs consecutive action poses),
                    so its codec uses anchor='first_action' and consumes T+1 action rows for T outputs.
  current_state     dT_k = T_s^-1 T_k, s = measured state at the current row. Our pose analogue of Giga's
                    joint convention a[t+k] - s[t] (wa_transforms_lerobot_pretrain.py:316-322); also used
                    for Motus (user decision 2026-10-07). Not an upstream pose convention: no upstream
                    code to compare against.
  clip_start        dp = p_k - p_0 (world-frame difference), dR = R_0^-1 R_k, anchor = first pose of the
                    clip. LingBot-VA get_relative_pose (lerobot_latent_dataset.py:112-125, used for
                    robotwin_tshape at :356-358).
  per_step_state    dp = p_a[t] - p_s[t], dR = R_a[t] R_s[t]^T (world-frame, left-multiplied), state at the
                    SAME step t. X-WAM _build_delta_action_tensor (data/robot_dataset.py:661-691).
                    Decoding needs the per-step proprio (X-WAM predicts it).
  step_delta_euler  d_k = [p_k - p_{k-1}, wrap(e_k - e_{k-1})] with e = euler_xyz(R), over
                    [prev_action, a_0, ...]. Genie-Envisioner get_actions_eef (data/utils/get_actions.py:
                    41-46) and the 'delta' branch of lerobot_like_dataset.py:478-488 (prev = action at t-1).

Frame dependence. Codecs operate in whatever frame the Canonical is in; the canonical frame is the
ego camera (canonical.py). prev_step and current_state are body-frame relative transforms
(T_ref^-1 T), identical in any fixed frame. clip_start, per_step_state and step_delta_euler take
position differences (and, for X-WAM, left-multiplied rotations; for Genie, Euler differences of the
absolute orientation) along the reference frame's axes, so LingBot-VA, X-WAM and Genie-Envisioner
actions are expressed along ego-camera axes (x right, y down, z forward). Encode refuses mixed frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import canonical as C

S = C.K.S  # sharpa78_schema
REFERENCES = ("absolute", "prev_step", "current_state", "clip_start", "per_step_state", "step_delta_euler")
ANCHORS = (None, "state", "first_action", "prev_action", "clip_start", "per_step_state")


@dataclass
class Codec:
    name: str
    rot_format: str
    reference: str
    anchor: str | None
    width: int
    # field -> first index; fields: "pos0","rot0","pos1","rot1" (0 = left, 1 = right); hands below
    pose_slots: dict
    # hand placement: "contiguous" (hand0 then hand1 at hand_start) or "lingbot78" (functional slots + signs)
    hand_mode: str = "contiguous"
    hand_starts: tuple = (None, None)
    state_rot_format: str | None = None
    notes: str = ""
    _static_mask: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert self.reference in REFERENCES and self.anchor in ANCHORS
        self.state_rot_format = self.state_rot_format or self.rot_format
        m = np.zeros(self.width, dtype=bool)
        rd = C.ROT_DIMS[self.rot_format]
        for k in (0, 1):
            m[self.pose_slots[f"pos{k}"]:self.pose_slots[f"pos{k}"] + 3] = True
            m[self.pose_slots[f"rot{k}"]:self.pose_slots[f"rot{k}"] + rd] = True
        for _, _, slot, _ in self._hand_index():
            m[slot] = True
        self._static_mask = m

    # ---- layout ------------------------------------------------------------------------------
    def _hand_index(self):
        """[(side_k, joint_i, slot, sign)] for all 44 hand channels."""
        out = []
        if self.hand_mode == "contiguous":
            for k in (0, 1):
                out += [(k, i, self.hand_starts[k] + i, 1.0) for i in range(22)]
        elif self.hand_mode == "lingbot78":
            sg = S.hand_signs()
            for k, (side, base) in enumerate((("left", S.L_HAND_BASE), ("right", S.R_HAND_BASE))):
                for i, j in enumerate(S.SHARPA_HAND_JOINTS):
                    out.append((k, i, base + S.HAND_SLOT_NAMES.index(S.JOINT_TO_SLOT[j]), float(sg[side][j])))
        else:
            raise ValueError(self.hand_mode)
        return out

    @property
    def extra_action_rows(self) -> int:
        """Action rows beyond the horizon the encoder consumes (LDA-style anchors use one)."""
        return 1 if self.anchor == "first_action" else 0

    @property
    def state_width(self) -> int:
        return 2 * (3 + C.ROT_DIMS[self.state_rot_format]) + 44

    # ---- relative pose math -----------------------------------------------------------------
    def _rel(self, a: C.Canonical, anchor: C.Canonical | None, states: C.Canonical | None):
        """Returns (dp [T,2,3], rot vectors [T,2,rd]) for action rows a (already trimmed to targets)."""
        p, Rm = a.pos, a.rot
        if self.reference == "absolute":
            return p, C.matrix_to(self.rot_format, Rm)
        if self.reference == "prev_step":
            pp = np.concatenate([anchor.pos, p[:-1]])
            Rp = np.concatenate([anchor.rot, Rm[:-1]])
            RpT = np.swapaxes(Rp, -1, -2)
            dp = np.einsum("tkij,tkj->tki", RpT, p - pp)
            return dp, C.matrix_to(self.rot_format, RpT @ Rm)
        if self.reference == "current_state":
            RsT = np.swapaxes(anchor.rot, -1, -2)
            dp = np.einsum("tkij,tkj->tki", np.broadcast_to(RsT, Rm.shape), p - anchor.pos)
            return dp, C.matrix_to(self.rot_format, RsT @ Rm)
        if self.reference == "clip_start":
            R0T = np.swapaxes(anchor.rot, -1, -2)
            Rr = R0T @ Rm
            if self.rot_format.startswith("quat"):
                # Sign-continuous along time, per arm (q and -q are the same rotation; scipy picks a hemisphere
                # per row, which made the left-wrist target flip sign 88 times on the v2 subset). Starts at
                # w >= 0; the first row relative to itself is the identity.
                return p - anchor.pos, np.stack([C.matrix_to(self.rot_format, Rr[:, k], continuous=True)
                                                 for k in range(Rr.shape[1])], axis=1)
            return p - anchor.pos, C.matrix_to(self.rot_format, Rr)
        if self.reference == "per_step_state":
            dR = Rm @ np.swapaxes(states.rot, -1, -2)
            return p - states.pos, C.matrix_to(self.rot_format, dR)
        if self.reference == "step_delta_euler":
            e = C.matrix_to("euler_xyz", Rm)
            ep = np.concatenate([C.matrix_to("euler_xyz", anchor.rot), e[:-1]])
            pp = np.concatenate([anchor.pos, p[:-1]])
            return p - pp, C.wrap_angle(e - ep)
        raise ValueError(self.reference)

    def _abs(self, dp, rv, anchor: C.Canonical | None, states: C.Canonical | None):
        """Inverse of _rel: returns absolute (pos [T,2,3], rot [T,2,3,3])."""
        if self.reference == "absolute":
            return dp, C.to_matrix(self.rot_format, rv)
        if self.reference in ("prev_step", "step_delta_euler"):
            T = dp.shape[0]
            pos = np.zeros_like(dp)
            rot = np.zeros(dp.shape[:2] + (3, 3))
            p_prev, R_prev = anchor.pos[0], anchor.rot[0]
            if self.reference == "prev_step":
                dR = C.to_matrix(self.rot_format, rv)
                for t in range(T):
                    p_prev = p_prev + np.einsum("kij,kj->ki", R_prev, dp[t])
                    R_prev = R_prev @ dR[t]
                    pos[t], rot[t] = p_prev, R_prev
            else:
                e_prev = C.matrix_to("euler_xyz", R_prev)
                for t in range(T):
                    p_prev = p_prev + dp[t]
                    e_prev = e_prev + rv[t]
                    pos[t], rot[t] = p_prev, C.to_matrix("euler_xyz", e_prev)
            return pos, rot
        dR = C.to_matrix(self.rot_format, rv)
        if self.reference == "current_state":
            pos = anchor.pos + np.einsum("tkij,tkj->tki", np.broadcast_to(anchor.rot, dR.shape), dp)
            return pos, anchor.rot @ dR
        if self.reference == "clip_start":
            return anchor.pos + dp, anchor.rot @ dR
        if self.reference == "per_step_state":
            return states.pos + dp, dR @ states.rot
        raise ValueError(self.reference)

    def _anchor(self, actions, state, prev_action, clip_start):
        return {None: None, "state": state, "first_action": actions[0] if actions is not None else None,
                "prev_action": prev_action, "clip_start": clip_start, "per_step_state": None}[self.anchor]

    # ---- public API --------------------------------------------------------------------------
    def encode(self, actions: C.Canonical, *, state: C.Canonical | None = None,
               prev_action: C.Canonical | None = None, clip_start: C.Canonical | None = None,
               states: C.Canonical | None = None):
        """Canonical action rows -> (x [T, width] float32, mask [T, width] bool).

        actions: T rows, or T+1 rows when extra_action_rows == 1 (row 0 is the anchor).
        state: measured state at the current row (1 row); prev_action: action at the row before
        the chunk; clip_start: first pose of the clip; states: measured states for the T rows.
        """
        anchor = self._anchor(actions, state, prev_action, clip_start)
        tgt = actions[self.extra_action_rows:]
        for ref in (anchor, states):
            if ref is not None and ref.frame != tgt.frame:
                raise ValueError(f"{self.name}: actions in frame {tgt.frame!r} but reference in {ref.frame!r}")
        dp, rv = self._rel(tgt, anchor, states)
        T = len(tgt)
        x = np.zeros((T, self.width))
        rd = rv.shape[-1]
        for k in (0, 1):
            x[:, self.pose_slots[f"pos{k}"]:self.pose_slots[f"pos{k}"] + 3] = dp[:, k]
            x[:, self.pose_slots[f"rot{k}"]:self.pose_slots[f"rot{k}"] + rd] = rv[:, k]
        for k, i, slot, sign in self._hand_index():
            x[:, slot] = sign * tgt.hand[:, k, i]
        row_ok = tgt.valid.copy()
        if anchor is not None:
            row_ok &= bool(np.all(anchor.valid))
        if states is not None and self.reference == "per_step_state":
            row_ok &= states.valid
        mask = np.broadcast_to(self._static_mask, x.shape) & row_ok[:, None]
        x[~mask] = 0.0
        return x.astype(np.float32), mask.copy()

    def decode(self, x: np.ndarray, *, state: C.Canonical | None = None,
               prev_action: C.Canonical | None = None, clip_start: C.Canonical | None = None,
               states: C.Canonical | None = None, first_action: C.Canonical | None = None) -> C.Canonical:
        """Model action vectors [T, width] -> absolute Canonical action rows (T rows).
        For anchor='first_action' pass `first_action` (deploy: the last commanded pose)."""
        x = np.asarray(x, dtype=np.float64)
        anchor = first_action if self.anchor == "first_action" else self._anchor(None, state, prev_action, clip_start)
        rd = C.ROT_DIMS[self.rot_format]
        dp = np.stack([x[:, self.pose_slots[f"pos{k}"]:self.pose_slots[f"pos{k}"] + 3] for k in (0, 1)], 1)
        rv = np.stack([x[:, self.pose_slots[f"rot{k}"]:self.pose_slots[f"rot{k}"] + rd] for k in (0, 1)], 1)
        pos, rot = self._abs(dp, rv, anchor, states)
        hand = np.zeros((len(x), 2, 22))
        for k, i, slot, sign in self._hand_index():
            hand[:, k, i] = x[:, slot] / sign
        ref = anchor if anchor is not None else states
        frame = ref.frame if ref is not None else "ego_camera"
        return C.Canonical(pos, rot, hand, np.ones(len(x), dtype=bool), frame=frame)

    def state_encode(self, state: C.Canonical) -> np.ndarray:
        """Absolute state vector: [pos0, rot0, pos1, rot1, hand0(22), hand1(22)] in state_rot_format."""
        rv = C.matrix_to(self.state_rot_format, state.rot)
        parts = []
        for k in (0, 1):
            parts += [state.pos[:, k], rv[:, k]]
        parts += [state.hand[:, 0], state.hand[:, 1]]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def channel_names(self) -> list[str]:
        names = [""] * self.width
        comp = {"rot6d": ["r11", "r21", "r31", "r12", "r22", "r32"], "quat_xyzw": ["qx", "qy", "qz", "qw"],
                "quat_wxyz": ["qw", "qx", "qy", "qz"], "euler_xyz": ["ex", "ey", "ez"],
                "axis_angle": ["ax", "ay", "az"]}[self.rot_format]
        for k, side in ((0, "left"), (1, "right")):
            for j, c in enumerate("xyz"):
                names[self.pose_slots[f"pos{k}"] + j] = f"{side}_wrist_{c}"
            for j, c in enumerate(comp):
                names[self.pose_slots[f"rot{k}"] + j] = f"{side}_wrist_{c}"
        for k, i, slot, _ in self._hand_index():
            names[slot] = f"{('left', 'right')[k]}_{C.HAND_JOINTS[i]}"
        return [n or "pad" for n in names]


def _arms(rd: int, start: int = 0):
    """Per arm [pos 3, rot rd], left then right."""
    return {"pos0": start, "rot0": start + 3, "pos1": start + 3 + rd, "rot1": start + 6 + rd}


_C62 = dict(rot_format="rot6d", width=62, pose_slots=_arms(6), hand_starts=(18, 40))

CODECS = {
    # Cosmos3: rot6d, framewise relative from the measured state; 62 used, padded to 64.
    "cosmos3": Codec("cosmos3", "rot6d", "prev_step", "state", 64, _arms(6), hand_starts=(18, 40),
                     notes="DROID midtrain-style backward_framewise; hands absolute; pad 62->64"),
    # LDA-1B: euler rpy, delta between consecutive actions (anchor = first action of the chunk).
    # Slots follow LDA's eval script eval_relative_eef.py:546-549: L pos 0:3, L rot 3:6, R pos 6:9, R rot 9:12,
    # hand blocks 12:75 and 75:138 (22 of 63 used each). LDA's TRAINING dataloader writes keys in data-config
    # order instead (pretraining EEF/MANO: L pos 0:3, L rot 3:6, L hand 6:69, R pos 69:72, R rot 72:75,
    # R hand 75:138; GR1 joints 0:29), so this is not a pretrained slot meaning. Sharpa is embodiment row 32 with
    # freshly initialized action encoder/decoder rows, so no pretrained slot semantics are reused either way.
    # State: euler angles as sin/cos (continuous across +-pi; the right-wrist yaw sits near +-pi), as LDA feeds
    # angle state (StateActionSinCosTransform): per arm pos 3 + sin 3 + cos 3, 62 per row (changed 2026-10-09
    # from raw euler, 56).
    "lda1b": Codec("lda1b", "euler_xyz", "prev_step", "first_action", 138,
                   {"pos0": 0, "rot0": 3, "pos1": 6, "rot1": 9}, hand_starts=(12, 75),
                   state_rot_format="euler_xyz_sincos",
                   notes="LDA delta EEF; T+1 action rows -> T outputs"),
    # Giga: rot6d relative to the current measured state.
    "giga": Codec(name="giga", reference="current_state", anchor="state", **_C62),
    # Motus: rot6d relative to the current measured state (user decision 2026-10-07).
    "motus": Codec(name="motus", reference="current_state", anchor="state", **_C62),
    # LingBot-VA: quat xyzw relative to the clip start, in its 30-D EEF slots 0-13; arm joint and
    # gripper slots masked; hands in the 24-slot functional blocks of the 78-D layout (with signs).
    "lingbot": Codec("lingbot", "quat_xyzw", "clip_start", "clip_start", 78,
                     {"pos0": 0, "rot0": 3, "pos1": 7, "rot1": 10}, hand_mode="lingbot78",
                     notes="LingBot get_relative_pose; slots 14-29 masked"),
    # Genie-Envisioner: its checkpoints never predicted wrist poses (GE-Base is video-only, GE-Act was
    # trained on AgiBot joints), so per the 2026-10-07 rule it uses the default: rot6d relative to the
    # current measured state (s0), like Giga and Motus. Euler step deltas also hit gimbal lock here.
    "genie": Codec(name="genie", reference="current_state", anchor="state",
                   notes="no native wrist convention in the checkpoint -> s0 + rot6d; hands absolute", **_C62),
    # X-WAM: axis-angle delta from the measured state at the same step; proprio pos + quat wxyz.
    "xwam": Codec("xwam", "axis_angle", "per_step_state", "per_step_state", 56, _arms(3), hand_starts=(12, 34),
                  state_rot_format="quat_wxyz", notes="X-WAM per-step delta; decode needs predicted proprio"),
}


def get_codec(name: str) -> Codec:
    return CODECS[name]


def codec_inputs(sample: dict, codec: Codec) -> dict:
    """Slice one SharpaEEFDataset sample into `codec.encode` keyword arguments.

    The reader returns H+1 action/state rows (c .. c+H); codecs that anchor on the first action
    consume all H+1, the rest use rows c .. c+H-1.
    """
    H = len(sample["actions"]) - 1
    n = H + codec.extra_action_rows
    return {
        "actions": sample["actions"][:n],
        "state": sample["state"],
        "prev_action": sample["prev_action"],
        "clip_start": sample["clip_start"],
        "states": sample["states"][:H],
    }


def decode_kwargs(sample: dict, codec: Codec) -> dict:
    """Matching `codec.decode` keyword arguments (first_action = action row c for LDA-style codecs)."""
    kw = codec_inputs(sample, codec)
    kw.pop("actions")
    kw["first_action"] = sample["actions"][0]
    return kw
