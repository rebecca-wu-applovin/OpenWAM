"""Write the dataloader keys older yam_sharpa checkpoints were trained without into their config.yaml.

Deploy (``openwam/deploy/config.py: check_deploy``) requires ``dataloader.fps`` and ``dataloader.history_layout`` in
the checkpoint config and never defaults them in code. Runs trained before those keys existed have the training
defaults implicitly: 20 fps (the LeRobot data) and one history frame per row (``frames``). This writes exactly
those values, only where a key is missing, and keeps the original next to it as ``config.yaml.bak``.

    .venv/bin/python scripts/patch_ckpt_config.py <run_dir> [<run_dir> ...] [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

# The values those runs trained with (yam_sharpa data rate; YamSharpaDataset's history_layout default then).
MISSING_DEFAULTS = {"fps": 20, "history_layout": "frames"}


def patch(run_dir: Path, dry_run: bool) -> list:
    """Add the missing keys to <run_dir>/config.yaml.

    Args:
        run_dir: training run dir holding config.yaml.
        dry_run: only report.
    Returns:
        list of "dataloader.<key>: <value>" lines added (empty if nothing was missing).
    """
    path = run_dir / "config.yaml"
    cfg = OmegaConf.load(path)
    if cfg.dataloader.type != "yam_sharpa":
        raise SystemExit(f"{path}: dataloader.type={cfg.dataloader.type!r}, only yam_sharpa runs are patched")
    added = [f"dataloader.{k}: {v}" for k, v in MISSING_DEFAULTS.items() if OmegaConf.select(cfg.dataloader, k) is None]
    if added and not dry_run:
        shutil.copy2(path, path.with_name("config.yaml.bak"))
        for k, v in MISSING_DEFAULTS.items():
            if OmegaConf.select(cfg.dataloader, k) is None:
                cfg.dataloader[k] = v
        OmegaConf.save(cfg, path)
    return added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for run_dir in args.run_dirs:
        added = patch(run_dir, args.dry_run)
        print(f"{run_dir}: " + (("would add " if args.dry_run else "added ") + ", ".join(added) if added else "complete"))


if __name__ == "__main__":
    sys.exit(main())
