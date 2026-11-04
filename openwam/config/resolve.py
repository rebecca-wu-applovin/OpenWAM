"""``resolve_config``: every value derived from the run config, and every cross-block check, in one place.

Entry points (training, resume, deploy) call :func:`resolve_config` once right after reading the config. It
writes the latent layout into ``cfg.common.layout`` and makes the whole config read-only (interpolations stay
unresolved, so a saved config.yaml keeps the yaml text); every layer below reads the layout from there (passed
down explicitly), never from samples, tensor shapes or key presence.

Latent layout of one sample (H history frames, the last one is the current frame t, then F future frames):

* past frames (the first P = H - 1 history frames, clean): ``history_encoding`` ``per_frame`` encodes each frame
  alone (P latents); ``clip`` encodes them as one causal clip (1 + (P-1)/tc latents, P = 1 + tc*k required);
* current frame t (clean): encoded alone, 1 latent;
* future (noisy target): the clip [t, F future frames] encoded as one, its latent 0 (= t) dropped: F/tc latents.

It also runs the cross-block checks (attention keys, proprio routing, random-history window) and validates the
dataloader's sampling keys (``history_sampling`` / ``history_window`` / ``caption_dropout_p``) in this one place: the
dataset reads them without re-checking.
"""

from __future__ import annotations

from typing import Sequence

from omegaconf import DictConfig, OmegaConf, open_dict

HISTORY_ENCODINGS = ("per_frame", "clip")
CLEAN_ATTENTIONS = ("bidirectional", "causal")
# How the video backbone gets the proprio (ActionDiT always reads the proprio tokens of its context): the same
# appended context tokens, or the backbone's own stream (flux).
PROPRIO_INPUTS = ("context_token", "backbone_stream")
# v<->a visibility vocabulary; only ``mutual`` is implemented and verified.
VERIFIED_VA_MODES = ("mutual",)
UNVERIFIED_VA_MODES = ("action_sees_video", "video_sees_action", "isolated")
# How the train split picks the history rows: always ``history_offsets``, or per sample H-1 random rows of the
# past ``history_window`` rows + frame t. Val and deploy always use ``history_offsets``.
HISTORY_SAMPLINGS = ("fixed", "random")
# Hand-written architecture keys the layout replaced; a launch that still sets them is stale.
REMOVED_ARCH_KEYS = ("num_history_frames", "num_proprio_tokens", "video_attention_mask_mode")
# Per-backbone caption-dropout keys -> their old meaning: caption dropout moved to the dataset
# (dataloader.caption_dropout_p, drawn from the per-sample RNG), so a config that still sets them is stale.
REMOVED_BACKBONE_KEYS = {
    "caption_dropout_p": "GE caption-dropout probability",
    "text_dropout_p": "flux caption-dropout probability",
    "text_encoder_dropout": "cosmos3 caption-dropout probability",
    "text_encoder_dropout_seed": "cosmos3 caption-dropout RNG seed",
}
# GE's generic-named ``seed`` only seeded its caption dropout; checked for GE alone (another backbone may own a seed).
REMOVED_GE_KEYS = {"seed": "GE caption-dropout RNG seed"}


def latent_layout(
    history_offsets: Sequence[int],
    future_offsets: Sequence[int],
    num_frames: int,
    history_encoding: str,
    temporal_compression: int,
) -> dict:
    """The latent layout of one sample.

    Args:
        history_offsets: H row offsets of the history frames (the last one is the current frame t), H >= 1.
        future_offsets: F row offsets of the future frames, F = tc * k.
        num_frames: int, the dataloader's action window (the action chunk is num_frames - 1 steps).
        history_encoding: ``per_frame`` | ``clip``, how the P = H - 1 past frames are encoded.
        temporal_compression: int tc, the video VAE's temporal compression.

    Returns:
        dict with num_hist_frames H, num_future_frames F, history_encoding, num_past_latents, num_hist_latents
        (past + current, all clean), num_future_latents F/tc (all noisy), num_latents, action_horizon
        num_frames - 1, num_proprio_tokens H. Raises ValueError for an unknown encoding or frame counts the
        encoding / VAE cannot represent.
    """
    h, f, tc = len(history_offsets), len(future_offsets), int(temporal_compression)
    if h < 1:
        raise ValueError(f"layout: history_offsets needs at least the current frame, got {list(history_offsets)}")
    if f < 1 or f % tc:
        raise ValueError(f"layout: the future must be tc * k = {tc}k frames, got F = {f}")
    if history_encoding not in HISTORY_ENCODINGS:
        raise ValueError(f"layout: history_encoding must be one of {HISTORY_ENCODINGS}, got {history_encoding!r}")
    past = h - 1
    if history_encoding == "per_frame":
        num_past = past
    elif past == 0:
        num_past = 0  # clip with no past frames: the segment is empty
    elif (past - 1) % tc:
        raise ValueError(f"layout: history_encoding=clip needs P = H - 1 = 1 + {tc}k past frames, got P = {past}")
    else:
        num_past = 1 + (past - 1) // tc
    return {
        "num_hist_frames": h,
        "num_future_frames": f,
        "history_encoding": str(history_encoding),
        "num_past_latents": num_past,
        "num_hist_latents": num_past + 1,
        "num_future_latents": f // tc,
        "num_latents": num_past + 1 + f // tc,
        "action_horizon": int(num_frames) - 1,
        "num_proprio_tokens": h,
    }


def _check_attention(model_cfg: DictConfig) -> None:
    """Cross-checks of the attention keys: clean_attention value, the backbone's required one, the v<->a mode.

    Args:
        model_cfg: ``cfg.model`` (``architecture`` and ``video_backbone`` blocks).

    Returns:
        None. ValueError for an unknown value or a backbone requirement the architecture does not meet;
        NotImplementedError for a known but unverified v<->a mode.
    """
    arch = model_cfg.architecture
    clean = arch.clean_attention
    if clean not in CLEAN_ATTENTIONS:
        raise ValueError(f"model.architecture.clean_attention must be one of {CLEAN_ATTENTIONS}, got {clean!r}")
    # Every backbone yaml states it: null = no requirement, a value pins it (LongLive: causal); the backbone
    # code never reads this key.
    required = model_cfg.video_backbone.required_clean_attention
    if required is not None and required != clean:
        raise ValueError(
            f"video_backbone {model_cfg.video_backbone.name!r} requires "
            f"model.architecture.clean_attention={required!r}, got {clean!r}"
        )
    va = arch.attention_mask_mode
    if va in UNVERIFIED_VA_MODES:
        raise NotImplementedError(
            f"attention_mask_mode={va!r} is not verified; implement it in build_frame_mask together with a "
            "reference test before enabling it (only 'mutual' is supported)"
        )
    if va not in VERIFIED_VA_MODES:
        raise ValueError(f"unknown attention_mask_mode {va!r}; choose from {VERIFIED_VA_MODES + UNVERIFIED_VA_MODES}")


def _check_proprio_input(model_cfg: DictConfig, state_dim: int) -> None:
    """Cross-checks of ``model.architecture.proprio_input`` against the video backbone.

    Every backbone yaml states ``required_proprio_input``: ``backbone_stream`` for a backbone with its own
    proprio stream (flux), ``context_token`` for every other (``backbone_stream`` would leave its video side
    without proprio).

    Args:
        model_cfg: ``cfg.model`` (``architecture`` and ``video_backbone`` blocks).
        state_dim: int proprio width, ``common.state_dim``.

    Returns:
        None. ValueError for an unknown value, a value the backbone does not take, or (backbone_stream) a
        proprio stream width ``video_backbone.action_cond_dim`` that differs from ``state_dim``.
    """
    video_backbone = model_cfg.video_backbone
    proprio_input = model_cfg.architecture.proprio_input
    if proprio_input not in PROPRIO_INPUTS:
        raise ValueError(f"model.architecture.proprio_input must be one of {PROPRIO_INPUTS}, got {proprio_input!r}")
    supported = video_backbone.required_proprio_input
    if proprio_input != supported:
        raise ValueError(
            f"video_backbone {video_backbone.name!r} takes model.architecture.proprio_input={supported!r}, "
            f"got {proprio_input!r}"
        )
    if proprio_input == "backbone_stream" and int(video_backbone.action_cond_dim) != state_dim:
        # The backbone's proprio stream has a fixed input width.
        raise ValueError(
            f"video_backbone.action_cond_dim={video_backbone.action_cond_dim} must equal "
            f"common.state_dim={state_dim}"
        )


def _check_history_sampling(dl: DictConfig) -> None:
    """Cross-checks of ``dataloader.history_sampling`` / ``history_window`` against ``history_offsets``.

    Random history keeps H = len(history_offsets) (the layout is unchanged); the window must reach the oldest
    fixed offset and hold H - 1 distinct past rows, so the fixed (val / deploy) layout is one training can draw.

    Args:
        dl: ``cfg.dataloader`` (``history_offsets``, ``history_sampling``, ``history_window``).

    Returns:
        None. ValueError for an unknown mode, or (random) a window that does not cover the offsets.
    """
    mode, window, offsets = dl.history_sampling, int(dl.history_window), list(dl.history_offsets)
    if mode not in HISTORY_SAMPLINGS:
        raise ValueError(f"dataloader.history_sampling must be one of {HISTORY_SAMPLINGS}, got {mode!r}")
    # Fixed sampling reads the offsets alone; the window only bounds the random draw.
    need = max(len(offsets) - 1, -int(offsets[0]))
    if mode == "random" and window < need:
        raise ValueError(
            f"dataloader.history_window={window} must cover history_offsets {offsets} (>= {need} rows) "
            "so the fixed val / deploy layout is one the random training draw can produce"
        )


def _check_seed(project: DictConfig) -> None:
    """Check ``project.seed`` is an int (the training entry seeds from it before any dataset exists).

    Args:
        project: ``cfg.project``.

    Returns:
        None. ValueError when the seed is null, a bool (yaml ``true``) or any other non-int.
    """
    seed = project.seed
    # bool is an int subclass: reject it explicitly so a yaml `true` is not read as seed 1.
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            "project.seed must be an int (null is not supported): the dataset's per-sample RNG, the per-step "
            "seeds and the vis sample all derive from it"
        )


def _check_caption_dropout(dl: DictConfig) -> None:
    """Check ``dataloader.caption_dropout_p`` (the dataset reads it without re-checking).

    Args:
        dl: ``cfg.dataloader``.

    Returns:
        None. ValueError when the probability is outside [0, 1] (0.0 = no caption dropout).
    """
    p = dl.caption_dropout_p
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"dataloader.caption_dropout_p must be in [0, 1], got {p!r}")


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Derive the latent layout into ``cfg.common.layout``, run the cross-block checks, freeze the whole config.

    Args:
        cfg: the run config (``project``, ``common``, ``dataloader``, ``model.architecture``,
            ``model.video_backbone``). A checkpoint's config already carries ``common.layout``: it is then only
            checked against a fresh derivation.

    Returns:
        ``cfg`` itself, with ``cfg.common.layout`` set and the whole config read-only (``OmegaConf.set_readonly``,
        interpolations unresolved; struct off so component slices can be merged). Raises ValueError when a stale
        hand-written architecture key or a removed backbone caption-dropout key is present, an existing layout
        disagrees with the derived one, ``project.seed`` is not an int, or a check above fails.
    """
    _check_seed(cfg.project)
    arch = cfg.model.architecture
    stale = [k for k in REMOVED_ARCH_KEYS if k in arch]
    if stale:
        raise ValueError(
            f"model.architecture.{stale} were replaced by the derived cfg.common.layout / clean_attention; "
            "remove them (the history length comes from dataloader.history_offsets alone)"
        )
    video_backbone = cfg.model.video_backbone
    removed = {**REMOVED_BACKBONE_KEYS, **(REMOVED_GE_KEYS if video_backbone.name == "ge_base_fast" else {})}
    stale = {k: meaning for k, meaning in removed.items() if k in video_backbone}
    if stale:
        raise ValueError(
            f"model.video_backbone keys {stale} were removed: caption dropout is dataloader.caption_dropout_p "
            "(the dataset drops prompts with the per-sample RNG); remove them and set that key instead"
        )
    _check_attention(cfg.model)
    _check_proprio_input(cfg.model, int(cfg.common.state_dim))
    dl = cfg.dataloader
    want = latent_layout(
        list(dl.history_offsets),
        list(dl.future_offsets),
        int(dl.num_frames),
        str(arch.history_encoding),
        int(cfg.model.video_backbone.temporal_compression),
    )
    # After the layout: history_offsets is known to hold at least frame t.
    _check_history_sampling(dl)
    _check_caption_dropout(dl)
    # A saved config.yaml (deploy / resume) carries the layout; a composed launch config does not.
    if "layout" in cfg.common:
        # Re-resolving a saved config: the stored layout must be exactly the derived one.
        have = OmegaConf.to_container(cfg.common.layout)
        if have != want:
            raise ValueError(f"cfg.common.layout {have} disagrees with the layout derived from the config {want}")
    else:
        with open_dict(cfg):
            cfg.common.layout = want
    # The run config is fixed from here on; no OmegaConf.resolve (interpolations stay as the yaml wrote them).
    # Read-only already forbids new keys; dropping Hydra's struct flag lets OmegaConf.merge(cfg.common,
    # cfg.<component>) build each component's slice (merge adds the component's keys to a copy of common).
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_readonly(cfg, True)
    return cfg


__all__ = ["HISTORY_SAMPLINGS", "latent_layout", "resolve_config"]
