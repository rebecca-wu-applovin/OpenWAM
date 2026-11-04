"""Deploy: one sparse-history yam_sharpa checkpoint as an observation -> action-plan engine.

    cfg = build_deploy_cfg(["deploy.ckpt.dir=<run>", "deploy.ckpt.name=checkpoint_step_N.safetensors",
                            "deploy.policy.steps_per_plan=20"])
    engine = InferenceEngine.from_config(cfg)
    session = engine.new_session()          # one per episode
    out = session.plan(obs)                 # {"actions": (K, D), "hand_targets": {"wrist", "tips" | "fingers"}}

Transports (the sim_policy_v1 server in benchmarks/simeval, a robot server) live outside this package.
"""

from openwam.deploy.config import build_deploy_cfg, check_deploy
from openwam.deploy.engine import EngineSpec, InferenceEngine, Session

__all__ = ["EngineSpec", "InferenceEngine", "Session", "build_deploy_cfg", "check_deploy"]
