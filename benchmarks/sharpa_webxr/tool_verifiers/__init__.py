"""Task-specific state verifiers for the tool-use tasks that used to end in a vision judge.

One module per task. Each module defines CHECKS = {check_name: fn(r, s) -> (mask, metrics, Conds)} and uses the
shared rules of verify.py through the module-level name V, which register() points at the running verify module
(verify.py imports this package at the end of its own import, so there is no circular import and no second copy of
verify when it runs as a script).
"""
from __future__ import annotations

from importlib import import_module

MODULES = ("common", "drum", "hammer", "brush", "stir", "scramble", "scoop", "camera_button")


def register(V) -> None:
    for name in MODULES:
        try:
            mod = import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as e:  # a task module not written yet
            if e.name == f"{__name__}.{name}":
                continue
            raise
        mod.V = V
        V.CHECKS.update(getattr(mod, "CHECKS", {}))
