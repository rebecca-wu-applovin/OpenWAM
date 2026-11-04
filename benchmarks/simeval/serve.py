"""Start the OpenWAM policy server for one checkpoint FILE, as sim-policy-eval's checkpoint watcher launches servers.

    .venv/bin/python -u benchmarks/simeval/serve.py --ckpt <run_dir>/checkpoint_step_N.safetensors \
        --inference-horizon K --port P [any openwam.deploy.server flag or dotlist override]

The watcher passes ``--ckpt <file>``; OpenWAM's server takes the run dir (config.yaml, normalization stats) and the
file name separately, so this splits the path and hands every other argument to ``openwam.deploy.server.main``
unchanged: one server code path for robots, simeval_bench and sim-policy-eval.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from openwam.deploy.server import main  # noqa: E402


def server_argv(argv: list[str]) -> list[str]:
    """Translate the watcher's command line to the server's.

    Args:
        argv: ``--ckpt <path to checkpoint_step_N.safetensors>`` plus any server flags.
    Returns:
        the server's argv: ``--ckpt-dir <run dir> --ckpt-name <file name>`` plus the other flags, in order.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt", required=True, type=Path, help="checkpoint_step_N.safetensors inside its run dir")
    args, rest = parser.parse_known_args(argv)
    if not args.ckpt.is_file():
        raise SystemExit(f"--ckpt {args.ckpt}: no such checkpoint file")
    return ["--ckpt-dir", str(args.ckpt.parent), "--ckpt-name", args.ckpt.name, *rest]


if __name__ == "__main__":
    main(server_argv(sys.argv[1:]))
