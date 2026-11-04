"""``resolve_config``: every value derived from the run config, and every cross-block check, in one place.

Entry points (training, resume, deploy) call :func:`resolve_config` once right after reading the config. It
writes the latent layout into ``cfg.common.layout`` and makes that block read-only; every layer below reads the
layout from there (passed down explicitly), never from samples, tensor shapes or key presence.

Latent layout of one sample (H history frames, the last one is the current frame t, then F future frames):

* past frames (the first P = H - 1 history frames, clean): ``history_encoding`` ``per_frame`` encodes each frame
  alone (P latents); ``clip`` encodes them as one causal clip (1 + (P-1)/tc latents, P = 1 + tc*k required);
* current frame t (clean): encoded alone, 1 latent;
* future (noisy target): the clip [t, F future frames] encoded as one, its latent 0 (= t) dropped: F/tc latents.

It also runs the cross-block checks (attention keys, proprio routing, random-history window). Later blocks of this
function (the config refactor) add the remaining derived values (action_dim / state_dim) and freeze the whole
config; for now only ``common.layout`` is frozen.
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
    clean = arch.get("clean_attention")
    if clean not in CLEAN_ATTENTIONS:
        raise ValueError(f"model.architecture.clean_attention must be one of {CLEAN_ATTENTIONS}, got {clean!r}")
    # A backbone yaml may pin it (LongLive: causal); the backbone code never reads this key.
    required = model_cfg.video_backbone.get("required_clean_attention")
    if required is not None and required != clean:
        raise ValueError(
            f"video_backbone {model_cfg.video_backbone.get('name')!r} requires "
            f"model.architecture.clean_attention={required!r}, got {clean!r}"
        )
    va = arch.get("attention_mask_mode")
    if va in UNVERIFIED_VA_MODES:
        raise NotImplementedError(
            f"attention_mask_mode={va!r} is not verified; implement it in build_frame_mask together with a "
            "reference test before enabling it (only 'mutual' is supported)"
        )
    if va not in VERIFIED_VA_MODES:
        raise ValueError(f"unknown attention_mask_mode {va!r}; choose from {VERIFIED_VA_MODES + UNVERIFIED_VA_MODES}")


def _check_proprio_input(model_cfg: DictConfig) -> None:
    """Cross-checks of ``model.architecture.proprio_input`` against the video backbone.

    A backbone yaml with its own proprio stream pins ``required_proprio_input: backbone_stream`` (flux); every
    other backbone has no such stream, so ``backbone_stream`` would leave the video side without proprio.

    Args:
        model_cfg: ``cfg.model`` (``architecture`` and ``video_backbone`` blocks).

    Returns:
        None. ValueError for an unknown value, a value the backbone does not take, or (backbone_stream) a
        proprio stream width ``video_backbone.action_cond_dim`` that differs from ``architecture.state_dim``.
    """
    arch, video_backbone = model_cfg.architecture, model_cfg.video_backbone
    proprio_input = arch.get("proprio_input")
    if proprio_input not in PROPRIO_INPUTS:
        raise ValueError(f"model.architecture.proprio_input must be one of {PROPRIO_INPUTS}, got {proprio_input!r}")
    supported = video_backbone.get("required_proprio_input", "context_token")
    if proprio_input != supported:
        raise ValueError(
            f"video_backbone {video_backbone.get('name')!r} takes model.architecture.proprio_input={supported!r}, "
            f"got {proprio_input!r}"
        )
    if proprio_input == "backbone_stream" and int(video_backbone.action_cond_dim) != int(arch.state_dim):
        # The backbone's proprio stream has a fixed input width.
        raise ValueError(
            f"video_backbone.action_cond_dim={video_backbone.action_cond_dim} must equal "
            f"model.architecture.state_dim={arch.state_dim}"
        )


def _check_history_sampling(dl: DictConfig) -> None:
    """Cross-checks of ``dataloader.history_sampling`` / ``history_window`` against ``history_offsets``.

    Random history keeps H = len(history_offsets) (the layout is unchanged); the window must reach the oldest
    fixed offset and hold H - 1 distinct past rows, so the fixed (val / deploy) layout is one training can draw.

    Args:
        dl: ``cfg.dataloader`` (``history_offsets``, ``history_sampling``, ``history_window``).

    Returns:
        None. ValueError for a missing key, an unknown mode, or (random) a window that does not cover the offsets.
    """
    for key in ("history_sampling", "history_window"):
        if dl.get(key) is None:
            raise ValueError(f"dataloader.{key} is required")
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


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Derive the latent layout into ``cfg.common.layout`` (read-only) and run the cross-block checks.

    Args:
        cfg: the run config (``dataloader``, ``model.architecture``, ``model.video_backbone``). A checkpoint's
            config already carries ``common.layout``: it is then only checked against a fresh derivation.

    Returns:
        ``cfg`` itself, with ``cfg.common.layout`` set and read-only. Raises ValueError when a stale hand-written
        key is present, an existing layout disagrees with the derived one, or a check above fails.
    """
    arch = cfg.model.architecture
    stale = [k for k in REMOVED_ARCH_KEYS if k in arch]
    if stale:
        raise ValueError(
            f"model.architecture.{stale} were replaced by the derived cfg.common.layout / clean_attention; "
            "remove them (the history length comes from dataloader.history_offsets alone)"
        )
    _check_attention(cfg.model)
    _check_proprio_input(cfg.model)
    dl = cfg.dataloader
    for key in ("history_offsets", "future_offsets"):
        if dl.get(key) is None:
            raise ValueError(f"dataloader.{key} is required (it defines the latent layout)")
    want = latent_layout(
        list(dl.history_offsets),
        list(dl.future_offsets),
        int(dl.num_frames),
        str(arch.history_encoding),
        int(cfg.model.video_backbone.temporal_compression),
    )
    # After the layout: history_offsets is known to hold at least frame t.
    _check_history_sampling(dl)
    have = OmegaConf.select(cfg, "common.layout", default=None)
    if have is not None:
        # Re-resolving a saved config (deploy / resume): the stored layout must be exactly the derived one.
        have = OmegaConf.to_container(have)
        if have != want:
            raise ValueError(f"cfg.common.layout {have} disagrees with the layout derived from the config {want}")
    else:
        with open_dict(cfg):
            if cfg.get("common") is None:
                cfg.common = {}
            cfg.common.layout = want
    OmegaConf.set_readonly(cfg.common.layout, True)
    return cfg


__all__ = ["HISTORY_SAMPLINGS", "latent_layout", "resolve_config"]
