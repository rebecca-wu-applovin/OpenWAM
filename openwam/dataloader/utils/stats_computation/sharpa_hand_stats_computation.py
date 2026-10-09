"""Compute SharpaHand (YAM Ultra + Sharpa Wave) joint-angle normalization statistics.

Action and state are pooled SEPARATELY (unlike MukaFranka's shared EEF10 pool,
which is valid there because action[t] IS state[t+1] by construction) —
SharpaHand's action is a COMMANDED target and state is a MEASURED achieved
position; these are not the same quantity and can have materially different
distributions (e.g. PD-controller tracking error), so pooling them would
produce normalization stats that are subtly wrong for one or both streams.

The output payload nests both under the single ``DEPLOY_ACTION_MODE`` key
(matching the ``{action_mode: {...}}`` convention every other reader's deploy
artifact uses), with "action"/"state" as a second level:

    {"joint": {"action": {mean,std,min,max,q01,q99}, "state": {...},
               "action_rows", "state_rows", "angle_unit", "action_alignment",
               "stats_population", "split", "num_episodes"}}

Example::

    python -m openwam.dataloader.utils.stats_computation.sharpa_hand_stats_computation \
      --config configs/dataloader/pretrain_data/sharpa_hand.yaml \
      --output /path/to/sharpa_hand_lerobot_v3/meta/normalization_stats.npy
"""

from __future__ import annotations

import argparse
import os
import socket
import uuid
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.sharpa_hand import ACTION_DIM, ANGLE_UNIT, STATS_POPULATION, SharpaHandDataset
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_bucket_arrays(bucket: SharpaHandDataset):
    """Yield per-episode ``(action, state)`` arrays, raw joint angles."""
    for pos, (_, row) in enumerate(bucket._eps_df.iterrows()):  # noqa: SLF001
        table = bucket._load_data_table(  # noqa: SLF001
            int(row["data/chunk_index"]), int(row["data/file_index"])
        )
        offset = int(bucket._ep_data_row_offset[pos])  # noqa: SLF001
        win = table.slice(offset, int(row["length"])).to_pandas()
        yield bucket._raw_action(win), bucket._raw_state(win)  # noqa: SLF001


def _compute_global_stats(dataset: SharpaHandDataset, reservoir_cap: int):
    if not isinstance(dataset, SharpaHandDataset):
        raise TypeError(f"expected SharpaHandDataset, got {type(dataset).__name__}")
    if dataset._split != "train":  # noqa: SLF001
        raise ValueError(
            f"SharpaHand stats must be computed from the 'train' split only, got split={dataset._split!r}. "  # noqa: SLF001
            "Construct the dataset with split='train' (the stats CLI's default)."
        )

    action_accumulator = Accumulator(dim=ACTION_DIM, reservoir_cap=reservoir_cap)
    state_accumulator = Accumulator(dim=ACTION_DIM, reservoir_cap=reservoir_cap)
    action_rows = 0
    state_rows = 0
    for action, state in _iter_bucket_arrays(dataset):
        action = np.asarray(action, np.float32).reshape(-1, ACTION_DIM)
        state = np.asarray(state, np.float32).reshape(-1, ACTION_DIM)
        action_accumulator.update_batch(action)
        state_accumulator.update_batch(state)
        action_rows += action.shape[0]
        state_rows += state.shape[0]
    if action_rows == 0 or state_rows == 0:
        raise ValueError("cannot compute SharpaHand stats from an empty dataset")

    action_stats = action_accumulator.finalize()
    state_stats = state_accumulator.finalize()
    combined = {
        "action": action_stats,
        "state": state_stats,
        "action_rows": action_rows,
        "state_rows": state_rows,
        "angle_unit": ANGLE_UNIT,
        "action_alignment": "action[t] = commanded joint targets at row t (row-aligned, not shifted)",
        "stats_population": STATS_POPULATION,
        "split": dataset._split,  # noqa: SLF001 — asserted == "train" above
        "num_episodes": len(dataset._eps_df),  # noqa: SLF001
    }
    return dataset.action_mode, ACTION_DIM, combined, action_rows, state_rows


def build_and_save_sharpa_hand_stats(
    dataset: SharpaHandDataset,
    output: str | Path,
    reservoir_cap: int = 1_000_000,
):
    """Compute stats and atomically merge the joint block into ``output``."""
    output = Path(output)
    action_mode, dim, combined, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap)
    payload = {}
    if output.exists():
        try:
            previous = np.load(output, allow_pickle=True).item()
            if isinstance(previous, dict):
                payload.update(previous)
        except (ValueError, EOFError):
            pass
    payload[action_mode] = combined
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, payload)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return action_mode, dim, action_rows, state_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/pretrain_data/sharpa_hand.yaml")
    parser.add_argument("--output", required=True, help="Deploy-compatible .npy output")
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    requested_split = str(OmegaConf.select(cfg, "split", default="train"))
    if requested_split != "train":
        raise ValueError(
            f"--config declares split={requested_split!r}, but statistics must be computed from the "
            "'train' split only. Fix the config or pass a config whose split is 'train'."
        )
    dataset = SharpaHandDataset.from_config(cfg, split=requested_split)
    action_mode, dim, action_rows, state_rows = build_and_save_sharpa_hand_stats(dataset, output, args.reservoir_cap)
    print(
        f"wrote {output} mode={action_mode} pool=separate_action_state dim={dim} "
        f"action_rows={action_rows} state_rows={state_rows} "
        f"total_rows={action_rows + state_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
