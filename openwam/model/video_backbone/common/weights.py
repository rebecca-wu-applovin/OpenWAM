"""Weight-file reading and strict loading shared by the video backbone families.

Construction is: build the module skeleton (parameters on ``meta``) from the yaml geometry,
``read_weights`` the files, then ``load_strict(..., assign=True)``. Any key / shape drift between
the yaml-declared skeleton and the weights raises here instead of surfacing later as a bad forward.
A checkpoint shell that is filled only later (training resume) gets storage from ``materialize_meta``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Single wrapper keys some ``.pth`` / ``.pt`` files put around the actual state dict.
_PTH_WRAPPER_KEYS = ("state_dict", "module", "model_state")


def read_weights(paths: list[str]) -> dict[str, Tensor]:
    """Read and merge weight files.

    Args:
        paths: list of file paths. ``*.safetensors`` (one file or several shards) are read with
            safetensors; anything else with ``torch.load`` (``weights_only``, mmap). A torch file that
            is ``{wrapper: state_dict}`` with wrapper one of state_dict / module / model_state is
            unwrapped once; no other conversion (no renames, no dtype cast).

    Returns:
        dict[str, Tensor]: the union of every file's tensors, keys as stored (CPU, stored dtype).
        Raises ValueError when two files carry the same key.
    """
    from safetensors.torch import load_file

    merged: dict[str, Tensor] = {}
    for path in paths:
        if str(path).endswith(".safetensors"):
            state = load_file(str(path))
        else:
            state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
            # Exactly one wrapper layer is peeled off, e.g. {"model_state": {...}}.
            if len(state) == 1 and next(iter(state)) in _PTH_WRAPPER_KEYS:
                state = next(iter(state.values()))
        dup = sorted(set(state) & set(merged))
        if dup:
            raise ValueError(f"read_weights: {path} repeats {len(dup)} keys already read, e.g. {dup[:3]}")
        merged.update(state)
    return merged


def tied_groups(module: nn.Module) -> list[list[str]]:
    """Names of parameters that are one shared tensor (tied weights, e.g. T5 shared / embed_tokens).

    Args:
        module: nn.Module.

    Returns:
        list of groups, each a list of >= 2 dotted parameter names bound to the same Parameter object.
    """
    by_id: dict[int, list[str]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(param), []).append(name)
    return [names for names in by_id.values() if len(names) > 1]


def retie(module: nn.Module, groups: list[list[str]]) -> None:
    """Re-bind every name of each group to the first name's parameter, after an ``assign=True`` load.

    ``load_state_dict(assign=True)`` assigns each key's tensor separately, which unties tied weights (two
    copies, two optimizer / memory entries). The values must already be equal (a checkpoint of a tied model).

    Args:
        module: nn.Module just loaded with ``assign=True``.
        groups: ``tied_groups(module)`` taken before the load.

    Returns:
        None. Raises ValueError when the loaded copies of a tied group differ.
    """

    def _owner(name: str):
        # Plain attribute walk: some modules override get_submodule.
        *path, leaf = name.split(".")
        owner = module
        for part in path:
            owner = getattr(owner, part)
        return owner, leaf

    for names in groups:
        first_owner, first_leaf = _owner(names[0])
        keep = getattr(first_owner, first_leaf)
        for name in names[1:]:
            owner, leaf = _owner(name)
            if not torch.equal(getattr(owner, leaf), keep):
                raise ValueError(f"tied weights {names} were loaded with different values")
            setattr(owner, leaf, keep)


def load_strict(module: nn.Module, state: dict[str, Tensor], *, assign: bool) -> None:
    """Load ``state`` into ``module``, rejecting any mismatch.

    Args:
        module: the skeleton to fill (parameters may be on ``meta``).
        state: dict[str, Tensor], must match ``module.state_dict()`` key-for-key and shape-for-shape.
        assign: forwarded to ``load_state_dict``. True rebinds the module's tensors to ``state``'s
            (the only way to fill a meta skeleton); False copies in place.

    Returns:
        None. Raises ValueError on missing / unexpected keys, on a shape mismatch, or when any
        parameter or buffer is still on ``meta`` after the load. Tied weights stay tied.
    """
    expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"load_strict: {len(missing)} missing keys {missing[:5]}, {len(unexpected)} unexpected keys {unexpected[:5]}"
        )
    # Compare shapes up front so the message names every bad key, not just the first.
    bad = [(k, expected[k], tuple(state[k].shape)) for k in expected if tuple(state[k].shape) != expected[k]]
    if bad:
        raise ValueError(f"load_strict: {len(bad)} shape mismatches (key, module, weights), e.g. {bad[:5]}")
    groups = tied_groups(module)
    module.load_state_dict(state, strict=True, assign=assign)
    retie(module, groups)
    # Non-persistent buffers and in-place copies into meta are not covered by the key check.
    left = [n for n, t in [*module.named_parameters(), *module.named_buffers()] if t.device.type == "meta"]
    if left:
        raise ValueError(f"load_strict: {len(left)} tensors still on meta after the load, e.g. {left[:5]}")


@torch.no_grad()
def materialize_meta(module: nn.Module, device: str | torch.device) -> None:
    """Give every parameter / buffer still on ``meta`` zeroed storage, in place (the resume shell).

    Resume builds the same checkpoint shell as deploy / finetune, but its weights arrive only after
    ``accelerator.prepare`` (accelerate ``load_state``), while ``set_dtype_device`` runs before that. Tensors
    that already hold values (non-persistent buffers such as rotary tables, built for real on the shell) are
    left alone: no checkpoint restores them, so ``module.to_empty`` would turn them into garbage.

    Args:
        module: nn.Module, possibly a partial meta shell.
        device: where the new storage lives (the trainer uses "cpu").

    Returns:
        None. Zeros, not uninitialized memory: DeepSpeed may read the values while flattening param groups,
        before ``load_state`` overwrites them; NaN/Inf there would be nondeterministic.
    """
    # id(meta tensor) -> its replacement, so tied weights (one tensor under two names) stay tied.
    done: dict[int, Tensor] = {}
    for sub in module.modules():
        # Parameters keep their requires_grad; buffers keep their persistence (register_buffer is not re-run).
        for name, param in list(sub._parameters.items()):
            if param is not None and param.is_meta:
                if id(param) not in done:
                    done[id(param)] = nn.Parameter(
                        torch.zeros_like(param, device=device), requires_grad=param.requires_grad
                    )
                sub._parameters[name] = done[id(param)]
        for name, buf in list(sub._buffers.items()):
            if buf is not None and buf.is_meta:
                if id(buf) not in done:
                    done[id(buf)] = torch.zeros_like(buf, device=device)
                sub._buffers[name] = done[id(buf)]
