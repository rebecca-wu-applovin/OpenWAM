"""Serve one checkpoint over sim-policy-eval's ``sim_policy_v1`` protocol: one episode per connection.

    .venv/bin/python -u benchmarks/simeval/server/serve.py --ckpt <run_dir>/checkpoint_step_N.safetensors \
        deploy.policy.steps_per_plan=K [deploy.<key>=<value> ...] --port P

The command line is the sim-policy-eval watcher's (``{python} -u {script} --ckpt {ckpt} {args} --port {port}``):
``--ckpt`` / ``--port`` become ``deploy.ckpt.dir`` / ``deploy.ckpt.name`` / ``deploy.server.port``, everything else
is a ``deploy.*`` dotlist. The model is loaded before the port opens, so an open port means ready.

``sim_policy_v1`` (sim-policy-eval ``policy_io/clients/sim_policy.py``): ``multiprocessing.connection``, pickled
tuples, replies ``("ok", value)`` / ``("error", str)`` (the connection stays usable):
    ("spec",)                                   -> the protocol spec (``sim_policy.protocol_spec``)
    ("reset", {"instruction": str, "seed": int | None}) -> None; starts this connection's episode
    ("plan", obs)                               -> {"wrist", "tips" | "fingers"} for the next K steps
Each connection is served by its own thread with its own :class:`Session`; one lock sends the plans to the GPU one at
a time. The observation / action translation is ``benchmarks/simeval/sim_policy.py``.
"""

import argparse
import logging
import sys
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "third_party")]

from benchmarks.simeval.sim_policy import AUTHKEY, engine_obs, hand_targets_reply, protocol_spec  # noqa: E402

logger = logging.getLogger("simeval.serve")


def serve_connection(conn, engine, gpu_lock: threading.Lock) -> None:
    """Answer one client's requests until it disconnects.

    Args:
        conn: an accepted ``multiprocessing.connection.Connection``.
        engine: the InferenceEngine (shared by all connections).
        gpu_lock: serializes plans across connections.
    """
    session, instruction = None, None  # this connection's episode, set by "reset"
    try:
        while True:
            verb, *payload = conn.recv()
            try:
                if verb == "spec":
                    reply = protocol_spec(engine.spec)
                elif verb == "reset":
                    session, instruction, reply = engine.new_session(payload[0]["seed"]), payload[0]["instruction"], None
                elif verb == "plan":
                    if session is None:
                        raise RuntimeError("'plan' before 'reset' on this connection")
                    obs = engine_obs(payload[0], instruction, engine.spec.camera_layout)
                    with gpu_lock:
                        plan = session.plan(obs)
                    reply = hand_targets_reply(plan, engine.spec.hand_targets)
                else:
                    raise ValueError(f"unknown request {verb!r} (spec | reset | plan)")
                conn.send(("ok", reply))
            except Exception as e:  # noqa: BLE001  (reported to the client; the connection stays up)
                logger.exception("request %r failed", verb)
                conn.send(("error", f"{type(e).__name__}: {e}"))
    except (EOFError, ConnectionError):
        pass  # client gone
    finally:
        conn.close()


def serve(engine, host: str, port: int) -> None:
    """Accept connections forever, one thread (and session) each.

    Args:
        engine: anything with ``spec`` (EngineSpec fields) and ``new_session(seed)`` -> object with ``plan(obs)``.
        host, port: where to listen.
    """
    gpu_lock = threading.Lock()
    with Listener((host, port), authkey=AUTHKEY) as listener:
        logger.info("serving %s on %s:%d", engine.spec.ckpt, host, port)
        while True:
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError, ConnectionError):
                continue  # a port probe (connect + close) or a stranger: keep listening
            threading.Thread(target=serve_connection, args=(conn, engine, gpu_lock), daemon=True).start()


def main(argv=None) -> None:
    """Parse the watcher's command line, load the engine, then serve forever."""
    ap = argparse.ArgumentParser(description="Serve one checkpoint over sim_policy_v1.")
    ap.add_argument("--ckpt", required=True, type=Path, help="<run_dir>/checkpoint_step_N.safetensors")
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("overrides", nargs="*", help="deploy.<key>=<value> dotlist (e.g. deploy.policy.steps_per_plan=20)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from openwam.deploy import InferenceEngine, build_deploy_cfg

    cfg = build_deploy_cfg([f"deploy.ckpt.dir={args.ckpt.parent}", f"deploy.ckpt.name={args.ckpt.name}",
                            f"deploy.server.port={args.port}", *args.overrides])
    engine = InferenceEngine.from_config(cfg)
    serve(engine, cfg.deploy.server.host, int(cfg.deploy.server.port))


if __name__ == "__main__":
    main()
