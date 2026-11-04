"""OpenWAM training: the trainer (``trainer``) and the pieces it calls; import the submodules directly (no re-exports,
so a light importer such as the Modal supervisor does not pull the trainer stack).

  run_mode       fresh / finetune / resume: decided once from cfg.training, and the architecture each mode builds
  checkpointing  the run dir: config / stats assets, weights + resume-state files, pruning, resume position
  optim          trainable-parameter groups, the step budget, the warmup + cosine LR schedule
  step_log       per-step reporting: phase timer, metric reduction, progress bar / wandb / CSV sinks
  seeding        process and per-step RNG seeds
  visualization  training-time world-model rollouts
"""
