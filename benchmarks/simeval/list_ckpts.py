"""List a training run's complete checkpoints for sim-policy-eval's ``watch --list-ckpts``.

    .venv/bin/python benchmarks/simeval/list_ckpts.py <run_dir> --steps-per-plan K [--settle-s 120]

Prints one line per complete checkpoint, ``<step> <ckpt file> deploy.policy.steps_per_plan=K`` (the watcher passes
the trailing arguments to ``server/serve.py``). The trainer writes ``checkpoint_step_N.safetensors`` in place with no
completion marker, so a file counts once it has not changed for ``--settle-s`` seconds and its safetensors header
covers the whole file (a volume committed mid-write shows a truncated file with an old mtime).
Nothing is printed before the run's ``config.yaml`` exists.
"""

import argparse
import json
import re
import struct
import time
from pathlib import Path


def intact(path: Path) -> bool:
    """path: a .safetensors file. Returns: True when the file reaches the end of its last tensor's byte range."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if 8 + n > size:
                return False
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return False
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    return 8 + n + end <= size


def complete_checkpoints(run_dir: Path, settle_s: float) -> list:
    """run_dir: training run dir. settle_s: seconds a file must be unchanged.
    Returns: [(step, path)] of the complete checkpoints, by step."""
    if not (run_dir / "config.yaml").is_file():
        return []
    now, found = time.time(), []
    for path in run_dir.glob("checkpoint_step_*.safetensors"):
        m = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", path.name)
        try:
            settled = now - path.stat().st_mtime > settle_s
        except FileNotFoundError:  # pruned meanwhile
            continue
        if m and settled and intact(path):
            found.append((int(m.group(1)), path))
    return sorted(found)


def main(argv=None) -> None:
    """argv: command-line arguments (None = sys.argv). Prints ``<step> <ckpt> deploy.policy.steps_per_plan=K`` per
    complete checkpoint of the run, by step; nothing before the run's config.yaml exists."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--steps-per-plan", type=int, required=True, help="K, actions executed per plan")
    ap.add_argument("--settle-s", type=float, default=120.0)
    args = ap.parse_args(argv)
    for step, path in complete_checkpoints(args.run_dir, args.settle_s):
        print(f"{step} {path.resolve()} deploy.policy.steps_per_plan={args.steps_per_plan}")


if __name__ == "__main__":
    main()
