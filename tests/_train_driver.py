"""Test-only driver: one single-process training run of ``configs/train_debug.yaml``, then dump the text cache.

Same path as ``scripts/train_debug.sh`` -> ``scripts/train.py`` (its ``_build_trainer``: run mode, config, dataset), run
in a subprocess by ``tests/test_text_cache_after_resume.py`` so each run gets a fresh DeepSpeed process group.

Usage (env RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT set, as train_debug.sh does):
  python tests/_train_driver.py <cache_out.pt> [--keep-resume-state] [--stop-after-step=N] <hydra overrides...>
"""

from __future__ import annotations

import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_train_script():
    """Returns: the ``scripts/train.py`` module (loaded by path; ``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location("openwam_train_script", os.path.join(REPO, "scripts", "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop(Exception):
    """Raised by the --stop-after-step save hook to end a run mid-way."""


def main() -> None:
    """Parse argv, run the trainer, write ``{prompt: cached text tensors}`` (CPU) to the cache path. Returns: None."""
    import logging

    import torch
    from hydra import compose, initialize_config_dir

    sys.path.insert(0, REPO)
    # INFO on stdout: the test reads the trainer's "[resume] resumed at global_step=N" line.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    out, args = sys.argv[1], sys.argv[2:]
    keep_state = "--keep-resume-state" in args
    stop = [int(a.split("=", 1)[1]) for a in args if a.startswith("--stop-after-step=")]
    overrides = [a for a in args if a != "--keep-resume-state" and not a.startswith("--stop-after-step=")]
    train = _load_train_script()
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose("train_debug", overrides=overrides)

    from openwam.train import checkpointing

    if keep_state:
        # A finished run drops its accel_state dirs; keep everything so a second run can resume from step N.
        checkpointing.prune_checkpoints = lambda *a, **k: None
    if stop:
        # Stop right after the resume state of step N is written (a timeout mid-run), via the run dir's save hook.
        save = checkpointing.save_full_state

        def save_then_stop(accelerator, output_path, global_step, *a, **k):
            """Same args as save_full_state; raises _Stop once step N is on disk. Returns: None."""
            save(accelerator, output_path, global_step, *a, **k)
            if global_step == stop[0]:
                raise _Stop

        checkpointing.save_full_state = save_then_stop
    # The entry point's own path: run mode decided once, resume config swapped in, resolve, seed, dataset, trainer.
    trainer = train._build_trainer(cfg)
    try:
        trainer.train()
    except _Stop:
        pass
    # The cache the training steps used: prompt -> tuple of tensors (Wan: context (1, L, D), seq_lens (1,)).
    cache = trainer.accelerator.unwrap_model(trainer.architecture).video_backbone.text_lookup.entries
    torch.save({p: tuple(t.detach().cpu() for t in v) for p, v in cache.items()}, out)


if __name__ == "__main__":
    main()
