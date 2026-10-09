"""Policy adapters: everything model-specific about running a policy in the env.

An adapter declares what it needs from the env (``obs``, ``action_space``, ``eef_frame``) and decides how to
execute: ``act(obs)`` returns the list of actions to run open loop before the next observation. Return the whole
chunk to execute it fully, its first k steps to replan every k steps, or one action to replan every step
(e.g. temporal ensembling over overlapping chunks kept inside the adapter).

Add a model: write policies/<name>.py with a ``Policy`` subclass and register it in REGISTRY.
"""
from __future__ import annotations

import importlib

from env import ObsConfig


class Policy:
    """Base adapter. Subclasses set the class attributes and implement ``act``."""

    name = "base"
    action_space = "joint"   # "joint" or "eef" (see env.SharpaWebXREnv)
    eef_frame = "ego_view"   # frame of "eef" observations and actions
    obs = ObsConfig()        # what the env returns when the adapter asks for an observation

    def reset(self, task: str, episode_seed: int) -> None:
        """Called once per episode, before the first ``act``."""

    def act(self, obs: dict) -> list:
        """Actions to execute before the next observation (at least one)."""
        raise NotImplementedError

    def episode_stats(self) -> dict:
        """Extra fields for rollout.json (requests, latency, ...)."""
        return {}


REGISTRY = {"gwp05": "policies.gwp05:GWP05Policy"}


def load(name: str, **kwargs) -> Policy:
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
