"""The deploy config of one checkpoint: assembled once, checked once, then frozen.

``build_deploy_cfg`` = the checkpoint's own ``config.yaml`` (the training cfg it was trained with, unchanged) plus one
``deploy`` block (``configs/deploy.yaml`` + ``deploy.*`` dotlist overrides). Nothing else is merged or rewritten, so
the model / data config a checkpoint is served with is exactly the one it was trained with. ``check_deploy`` is the
only place deploy-time conditions are validated; every component downstream reads its own slice and trusts it.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEPLOY_YAML = Path(__file__).resolve().parents[2] / "configs" / "deploy.yaml"


def build_deploy_cfg(overrides, deploy_yaml=DEPLOY_YAML) -> DictConfig:
    """Assemble, check and freeze the deploy cfg of one checkpoint.

    Args:
        overrides: list of ``deploy.<key>=<value>`` dotlist strings; must name ``deploy.ckpt.dir`` (run dir with
            ``config.yaml`` + ``normalization_stats.npy``), ``deploy.ckpt.name`` (``checkpoint_step_N.safetensors``)
            and ``deploy.policy.steps_per_plan`` unless the yaml sets them.
        deploy_yaml: path of the deploy yaml (default ``configs/deploy.yaml``).
    Returns:
        read-only DictConfig: the checkpoint's training cfg with one extra ``deploy`` block.
    """
    bad = [o for o in overrides if not str(o).startswith("deploy.")]
    if bad:
        raise ValueError(f"deploy overrides must start with 'deploy.' (the checkpoint's model / data config is fixed): {bad}")
    # The deploy yaml is the content of the `deploy` block; strip the prefix off the dotlist to merge into it.
    deploy = OmegaConf.merge(OmegaConf.load(deploy_yaml), OmegaConf.from_dotlist([o[len("deploy."):] for o in overrides]))
    if deploy.ckpt.dir is None or deploy.ckpt.name is None:
        raise ValueError("set deploy.ckpt.dir (run dir) and deploy.ckpt.name (checkpoint_step_N.safetensors)")
    cfg = OmegaConf.merge(OmegaConf.load(Path(deploy.ckpt.dir) / "config.yaml"), {"deploy": deploy})
    check_deploy(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def check_deploy(cfg: DictConfig) -> None:
    """Every deploy-time condition, checked once at startup; raises ValueError on the first violation.

    Args:
        cfg: the assembled cfg (checkpoint training cfg + ``deploy`` block).
    """
    from openwam.model.denoise_schedule import normalize_denoise_config

    dl, dep = cfg.dataloader, cfg.deploy
    ckpt = Path(dep.ckpt.dir) / dep.ckpt.name
    if not ckpt.is_file():
        raise ValueError(f"no checkpoint file {ckpt}")
    # The engine serves sparse-history yam_sharpa checkpoints (vstack canvas, history buffer) and nothing else.
    if dl.type != "yam_sharpa":
        raise ValueError(f"deploy serves yam_sharpa checkpoints only, got dataloader.type={dl.type!r}")
    if not dl.multiview or dl.multiview_layout != "vstack":
        raise ValueError(f"deploy composes the vstack canvas only, got multiview={dl.multiview} "
                         f"multiview_layout={dl.multiview_layout!r}")
    if dl.normalize_mode in (None, "none", "null"):
        raise ValueError("deploy needs a normalized checkpoint (dataloader.normalize_mode)")
    # Keys older checkpoints were trained without: required, never defaulted (scripts/patch_ckpt_config.py adds them).
    for key in ("history_offsets", "history_layout", "fps"):
        if OmegaConf.select(dl, key) is None:
            raise ValueError(f"{Path(dep.ckpt.dir) / 'config.yaml'}: dataloader.{key} is missing; this deploy serves "
                             "sparse-history checkpoints and needs it written in the checkpoint config "
                             "(older runs: scripts/patch_ckpt_config.py)")
    # Data shape: deploy states it, the checkpoint decides it; they must agree.
    for key in ("num_frames", "height", "width"):
        if dep.data[key] != dl[key]:
            raise ValueError(f"deploy.data.{key}={dep.data[key]} but the checkpoint was trained with "
                             f"dataloader.{key}={dl[key]}")
    # K: one observation every K action steps, so it must fit the chunk, whole seconds and every history offset.
    k, fps, chunk = dep.policy.steps_per_plan, int(dl.fps), int(dl.num_frames) - 1
    if not isinstance(k, int) or not 1 <= k <= chunk:
        raise ValueError(f"deploy.policy.steps_per_plan must be an int in [1, {chunk}] (the action chunk), got {k!r}")
    if fps % k or any(int(o) % k for o in dl.history_offsets):
        raise ValueError(f"deploy.policy.steps_per_plan={k} must divide dataloader.fps={fps} and every history "
                         f"offset {list(dl.history_offsets)} (else a history frame falls between two observations)")
    normalize_denoise_config(dep.sampler)
