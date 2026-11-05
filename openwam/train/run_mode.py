"""Run mode (fresh / finetune / resume): decided and checked once, then the training architecture each mode builds.

``scripts/train.py`` calls :func:`decide_run_mode` on the resolved launch config and hands ``(run_mode, ckpt_path)``
to the trainer, which builds its architecture with :func:`build_training_architecture`:

  - FRESH: the upstream pretrained backbone weights (``assets.video_backbone``).
  - FINETUNE (a new run from a checkpoint's weights): the launch's ``model`` + ``common`` (minus the derived
    ``common.layout``) must equal the checkpoint's; ``training`` / ``dataloader`` may differ. Then a strict
    ``load_checkpoint`` of the named / latest safetensors.
  - RESUME (the same run continued): the whole launch config must equal the run's ``config.yaml`` except the
    ``RESUME_LAUNCH_KEYS``, so the launch config IS the run's config and trains as is. Storage is given to the meta
    shell (``materialize_meta``); accelerate's ``load_state`` fills it after ``prepare``.

Finetune / resume build the shell with the same :func:`openwam.model.build_architecture_from_ckpt_dir` the deploy
loader uses: module skeletons from the model config, non-weight assets (tokenizers / processors / third-party
``config.json``) from the checkpoint dir, so the upstream backbone dir need not exist on the training host.
"""

import enum
import logging
import os

from omegaconf import DictConfig, OmegaConf

from openwam.config.diff import check_same_cfg
from openwam.model.architectures.base import BaseWAMArchitecture, TrainOptions
from openwam.train.checkpointing import find_latest_weights

logger = logging.getLogger(__name__)

# Keys a resume launch legitimately sets differently from the run's config.yaml: where to resume from, and the
# fresh-run dir fields a resume launch drops (the run dir is resume_ckpt_path's).
RESUME_LAUNCH_KEYS = ("training.resume_ckpt_path", "training.output_path", "training.run_dir_name")


class RunMode(enum.Enum):
    """How a training run starts: from upstream weights, from a checkpoint's weights, or continuing a run."""

    FRESH = "fresh"
    FINETUNE = "finetune"
    RESUME = "resume"


def decide_run_mode(cfg: DictConfig) -> tuple[RunMode, str | None]:
    """The run mode from the two ckpt-path switches (null = off), with its source checked against this launch.

    Args:
        cfg: DictConfig, the launch's whole run config after ``resolve_config`` (carries ``common.layout``);
            ``training.finetune_ckpt_path`` / ``training.resume_ckpt_path`` pick the mode.

    Returns:
        (run_mode, ckpt_path): FRESH with None; FINETUNE with the source as given (a run dir, or one
        ``checkpoint_step_*.safetensors`` in it); RESUME with the run dir. Raises ValueError when both switches are
        set, a resume source is not a directory, or the source's ``config.yaml`` differs where it must not
        (finetune: ``model`` + ``common`` minus the layout; resume: everything but ``RESUME_LAUNCH_KEYS``).
    """
    finetune_path, resume_path = cfg.training.finetune_ckpt_path, cfg.training.resume_ckpt_path
    if finetune_path is not None and resume_path is not None:
        raise ValueError("finetune_ckpt_path and resume_ckpt_path are mutually exclusive; set at most one.")

    if finetune_path is not None:
        source = str(finetune_path)
        # A weights file names its run dir; a typo'd file fails here, not after minutes of skeleton building.
        if not os.path.exists(source):
            raise FileNotFoundError(f"training.finetune_ckpt_path={source!r} does not exist")
        path = os.path.join(source if os.path.isdir(source) else os.path.dirname(source), "config.yaml")
        saved = OmegaConf.load(path)
        # The model's identity, interpolations evaluated (what the values are, not how the yaml spells them).
        saved, live = ({k: OmegaConf.to_container(c[k], resolve=True) for k in ("model", "common") if k in c}
                       for c in (saved, cfg))
        check_same_cfg(
            saved, live, ignore=("common.layout",),
            context=f"[finetune] the checkpoint's weights only fit the model they were trained as; set these keys "
                    f"to {path}'s values. This launch's model / common config differs",
        )
        return RunMode.FINETUNE, source

    if resume_path is not None:
        run_dir = str(resume_path)
        if not os.path.isdir(run_dir):
            raise ValueError(
                f"training.resume_ckpt_path={run_dir!r} is not a directory: resume continues a run dir (its latest "
                "accel_state_step_*). To start a new run from a weights file use training.finetune_ckpt_path."
            )
        path = os.path.join(run_dir, "config.yaml")
        # Unresolved on both sides: compare what the yaml says, not what interpolations evaluate to on this host.
        check_same_cfg(
            OmegaConf.to_container(OmegaConf.load(path), resolve=False),
            OmegaConf.to_container(cfg, resolve=False),
            ignore=RESUME_LAUNCH_KEYS,
            context=f"[resume] a resume must run the run's own config (use training.finetune_ckpt_path to start a "
                    f"new run from these weights with a different config); this launch differs from {path}",
        )
        return RunMode.RESUME, run_dir

    return RunMode.FRESH, None


def build_training_architecture(
    run_mode: RunMode, ckpt_path: str | None, cfg: DictConfig, train_options: TrainOptions
) -> BaseWAMArchitecture:
    """The architecture a training run starts from, per run mode (the one place the mode picks the build).

    Args:
        run_mode: RunMode from :func:`decide_run_mode` (which already checked the source).
        ckpt_path: str, that mode's checkpoint source (None for FRESH).
        cfg: the run config, resolved and read-only; ``merge(cfg.common, cfg.model)`` builds the shell.
        train_options: TrainOptions the trainer decided (passed to the architecture constructor).

    Returns:
        BaseWAMArchitecture: FRESH = upstream pretrained weights; FINETUNE = the checkpoint's weights loaded;
        RESUME = zeroed storage for ``load_state``. Weights land before the trainer's freeze.
    """
    from openwam.model import build_architecture, build_architecture_from_ckpt_dir
    from openwam.model.video_backbone import LoadMode
    from openwam.model.video_backbone.common.weights import materialize_meta

    # The architecture's slice: the shared values (action / state dims, layout) under the model block.
    model_cfg = OmegaConf.merge(cfg.common, cfg.model)
    if run_mode is RunMode.FRESH:
        # The upstream file paths come separately (assets, not part of the model).
        return build_architecture(
            model_cfg, mode=LoadMode.PRETRAINED, assets=cfg.assets.video_backbone, ckpt_dir=None,
            train_options=train_options,
        )

    # FINETUNE may name one weights file; its dir is the checkpoint dir.
    is_file = run_mode is RunMode.FINETUNE and not os.path.isdir(ckpt_path)
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path)) if is_file else ckpt_path
    logger.info("[%s] building architecture from ckpt dir: %s", run_mode.value, ckpt_dir)
    architecture = build_architecture_from_ckpt_dir(ckpt_dir, model_cfg, train_options)

    if run_mode is RunMode.FINETUNE:
        weights = os.path.abspath(ckpt_path) if is_file else find_latest_weights(ckpt_dir)
        # Plain print so the warm-start shows on the launch terminal (non-main ranks have print disabled).
        print(f"[finetune] loading pretrained weights: {weights}", flush=True)
        architecture.load_checkpoint(weights)
        print("[finetune] pretrained weights loaded OK", flush=True)
    else:
        # RESUME: the safetensors are not read. load_state restores a strict superset (frozen params too) after
        # prepare, and the latest safetensors may be truncated by a mid-save kill while the accel state is intact.
        # set_dtype_device runs before load_state and cannot copy out of meta tensors, so give them storage.
        materialize_meta(architecture, "cpu")
    return architecture
