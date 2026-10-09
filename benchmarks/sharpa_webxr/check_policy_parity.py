"""G3: eval policy path == training path on val windows.

(1) images/state/prompt from GWPSharpaPolicy.preprocess (raw frames) are bit-identical to the training dataset's
    __getitem__; (2) training flow-matching action loss of the loaded weights on val windows (compare to the
    run's val/action_loss); (3) sampled chunk (eval sampler, 10 and 20 steps) vs ground truth: normalized MSE and
    decoded absolute wrist position / rotation / hand-joint errors.
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "benchmarks/sharpa_webxr")
from policy_server import GWPSharpaPolicy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--n", type=int, default=24)
p.add_argument("--out", required=True)
a = p.parse_args()

pol = GWPSharpaPolicy(a.ckpt, num_steps=10)
T = pol.T
T.SPLIT_FILE = "assets/sharpa_full_meta/split_v1.json"
dirs = ["assets/sharpa_full_meta/gpt_fleet_depth", "assets/sharpa_full_meta/simteleop_1004_fleet"]
norm = {"observation.state": {"q01": pol.s_lo, "q99": pol.s_hi}, "action": {"q01": pol.a_lo, "q99": pol.a_hi}}
ds = T.SharpaGigaEEFDataset("val", None, norm, pol.prompt_embeds, False, dirs)
rng = np.random.default_rng(0)
pool = rng.choice(ds.pool_size(), size=a.n, replace=False)
ds.indices = pool

from giga_train import ModuleDict  # noqa: E402
loss_ns = T.build_loss_ns(ModuleDict({"transformer": pol.model}), pol.vae, pol.device)
res = {"ckpt": a.ckpt, "n": a.n, "identical_inputs": [], "flow_action_loss": [], "s10": [], "s20": []}
A = ds.A
for i, w in enumerate(pool):
    item = ds[i]
    s = ds.sample(int(w))
    # (1) preprocess parity
    imgs = {c.split(".")[-1]: np.asarray(s["video"][c][0]) for c in T.EEF_CAMERAS}
    state = ds.codec.state_encode(s["state"])[0]
    raw = pol.preprocess(imgs, state, s["prompt"])
    same = (torch.equal(raw["images"][0, 0], item["images"][0]) and torch.allclose(raw["state"][0], item["state"])
            and torch.equal(raw["prompt_embeds"][0], item["prompt_embeds"]))
    res["identical_inputs"].append(bool(same))
    # (2) training loss (train mode = training contract), fixed noise
    pol.model.train()
    with torch.no_grad(), torch.random.fork_rng(devices=[pol.device]):
        torch.manual_seed(1234 + i)
        batch = T.to_batch({k: v[None] for k, v in item.items()}, pol.device)
        res["flow_action_loss"].append(float(loss_ns.forward_step(batch)["action_loss"]))
    pol.model.eval()
    # (3) sampled chunk vs GT
    gt = (item["action"].numpy() + 1) / 2 * np.maximum(pol.a_hi - pol.a_lo, 1e-8) + pol.a_lo
    gt_abs = ds.codec.decode(gt, **A.decode_kwargs(s, ds.codec))
    for key, steps in (("s10", 10), ("s20", 20)):
        pred = pol.predict(imgs, state, s["prompt"], seed=i, num_steps=steps)
        pn = (pred - pol.a_lo) / np.maximum(pol.a_hi - pol.a_lo, 1e-8) * 2 - 1
        pa = ds.codec.decode(pred, **A.decode_kwargs(s, ds.codec))
        dR = np.einsum("tkji,tkjl->tkil", pa.rot, gt_abs.rot)
        ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
        res[key].append({"mse_norm": float(((pn - item["action"].numpy()) ** 2).mean()),
                         "pos_err_mm": float(np.linalg.norm(pa.pos - gt_abs.pos, axis=-1).mean() * 1e3),
                         "pos_err_mm_last": float(np.linalg.norm(pa.pos[-1] - gt_abs.pos[-1], axis=-1).mean() * 1e3),
                         "rot_err_deg": float(ang.mean()),
                         "hand_err_deg": float(np.degrees(np.abs(pa.hand - gt_abs.hand)).mean())})
    print(i, res["identical_inputs"][-1], round(res["flow_action_loss"][-1], 4), res["s10"][-1], flush=True)

summary = {"identical_inputs": int(sum(res["identical_inputs"])), "n": a.n,
           "flow_action_loss_mean": float(np.mean(res["flow_action_loss"]))}
for key in ("s10", "s20"):
    summary[key] = {m: float(np.mean([r[m] for r in res[key]])) for m in res[key][0]}
res["summary"] = summary
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
