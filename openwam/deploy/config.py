"""The deploy config of one checkpoint: assembled once, checked once, then frozen.

``build_deploy_cfg`` = the checkpoint's own ``config.yaml`` (the training cfg it was trained with, re-checked by
``resolve_config`` exactly as the training entry does) plus one ``deploy`` block (``configs/deploy.yaml`` +
``deploy.*`` dotlist overrides). Nothing else is merged or rewritten, so the model / data config a checkpoint is
served with is exactly the one it was trained with. ``check_deploy`` is the only place deploy-time conditions are
validated; every component downstream reads its own slice and trusts it.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from openwam.dataloader.utils.normalization import normalization_on

DEPLOY_YAML = Path(__file__).resolve().parents[2] / "configs" / "deploy.yaml"


def build_deploy_cfg(overrides, deploy_yaml=DEPLOY_YAML) -> DictConfig:
    """Assemble, check and freeze the deploy cfg of one checkpoint.

    Args:
        overrides: list of ``deploy.<key>=<value>`` dotlist strings; must name ``deploy.ckpt.dir`` (run dir with
            ``config.yaml`` + ``normalization_stats.npy``), ``deploy.ckpt.name`` (``checkpoint_step_N.safetensors``)
            and ``deploy.policy.steps_per_plan`` unless the yaml sets them.
        deploy_yaml: path of the deploy yaml (default ``configs/deploy.yaml``).
    Returns:
        read-only DictConfig: the checkpoint's resolved training cfg (``common.layout`` set) with one extra
        ``deploy`` block.
    """
    from openwam.config import resolve_config

    bad = [o for o in overrides if not str(o).startswith("deploy.")]
    if bad:
        raise ValueError(f"deploy overrides must start with 'deploy.' (the checkpoint's model / data config is fixed): {bad}")
    # The deploy yaml is the content of the `deploy` block; strip the prefix off the dotlist to merge into it.
    base, dotlist = OmegaConf.load(deploy_yaml), [o[len("deploy."):] for o in overrides]
    # Only keys the yaml defines: a typo or a removed key is an error, never silently carried along.
    unset = object()
    unknown = [f"deploy.{o}" for o in dotlist if OmegaConf.select(base, o.split("=", 1)[0], default=unset) is unset]
    if unknown:
        raise ValueError(f"unknown deploy overrides {unknown}: {deploy_yaml} defines the deploy keys")
    deploy = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
    if deploy.ckpt.dir is None or deploy.ckpt.name is None:
        raise ValueError("set deploy.ckpt.dir (run dir) and deploy.ckpt.name (checkpoint_step_N.safetensors)")
    # The same resolve the training entry runs: derives / re-checks common.layout and the cross-block rules.
    ckpt_cfg = resolve_config(OmegaConf.load(Path(deploy.ckpt.dir) / "config.yaml"))
    cfg = OmegaConf.merge(ckpt_cfg, {"deploy": deploy})
    check_deploy(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def check_deploy(cfg: DictConfig) -> None:
    """Every deploy-time condition, checked once at startup; raises ValueError on the first violation.

    Args:
        cfg: the assembled cfg (resolved checkpoint training cfg + ``deploy`` block).
    """
    from openwam.deploy.history_buffer import ObsHistoryBuffer
    from openwam.model.denoise_schedule import normalize_denoise_config

    dl, dep = cfg.dataloader, cfg.deploy
    ckpt = Path(dep.ckpt.dir) / dep.ckpt.name
    if not ckpt.is_file():
        raise ValueError(f"no checkpoint file {ckpt}")
    # The engine serves sparse-history yam_sharpa checkpoints (vstack canvas, history buffer) and nothing else.
    if dl.type != "yam_sharpa":
        raise ValueError(f"deploy serves yam_sharpa checkpoints only, got dataloader.type={dl.type!r}")
    if not normalization_on(dl.normalize_mode):
        raise ValueError("deploy needs a normalized checkpoint (dataloader.normalize_mode)")
    # Older checkpoints were trained without fps: required, never defaulted.
    if "fps" not in dl or dl.fps is None:
        raise ValueError(f"{Path(dep.ckpt.dir) / 'config.yaml'}: dataloader.fps is missing; deploy serves actions at "
                         "the data rate and needs it written in the checkpoint config")
    # K: one observation every K action steps, so it must fit the chunk resolve_config derived ...
    k, chunk = dep.policy.steps_per_plan, int(cfg.common.layout.action_horizon)
    # A bool is an int in Python: steps_per_plan=true is a typo, not K = 1.
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= chunk:
        raise ValueError(f"deploy.policy.steps_per_plan must be an int in [1, {chunk}] (the action chunk), got {k!r}")
    # ... and divide fps and every history offset: the history buffer's own rule, checked by building one.
    ObsHistoryBuffer(dl.history_offsets, action_fps=int(dl.fps), steps_per_obs=k)
    normalize_denoise_config(dep.sampler)
