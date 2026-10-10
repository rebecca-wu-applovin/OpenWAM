"""GWP-0.5 Sharpa policy server for closed-loop sim eval (runs in the OpenWAM .venv, one per GPU).

Model side reuses the training script verbatim (scripts/parallel_sharpa_forward_checks/run_giga_sharpa_train_job.py):
TRANSFORMER_CFG (62-D I/O, embodiment 2), SharpaGigaDataset._process_images/_compose_views (ego on top, chest
below, 320x384, val-mode center crop), quantile normalization and ``sample_actions`` (upstream action-only
loop, FlowMatchEuler shift 5, no CFG). Default 10 denoising steps = GWP-0.5 deployment (wam.cpp released
configs); training's decode check used 20.

Transport is upstream's ZeroMQ RobotInferenceServer (giga_models.sockets), endpoint "inference":
  in : {"images": {"ego_view": uint8 [480,640,3], "chest_view": ...}, "state": float [62] (codec state_encode),
        "task": str, "seed": int}
  out: {"action": float32 [32,62] denormalized giga-codec chunk, "timing": {...}}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)  # the training module inserts repo-relative third_party paths

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

TRAIN_SCRIPT = REPO / "scripts/parallel_sharpa_forward_checks/run_giga_sharpa_train_job.py"
PREP_DIR = REPO / "assets/giga_sharpa_train_job/prep_eef_full_v1"
VAE_DIR = REPO / "assets/video_backbone_ckpt/Wan2.2-VAE-Diffusers/vae"


def load_train_module():
    spec = importlib.util.spec_from_file_location("gwp_train", TRAIN_SCRIPT)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    # main() sets these for --action-repr eef --views 2 --view-concat t
    T.ACTION_DIM = T.EEF_DIM
    T.TRANSFORMER_CFG.update(in_action_channels=T.EEF_DIM, out_action_channels=T.EEF_DIM)
    T.VIEWS, T.VIEW_CONCAT = 2, "t"
    return T


class GWPSharpaPolicy:
    def __init__(self, ckpt: str, device: str = "cuda", num_steps: int = 10, prep_dir: Path = PREP_DIR):
        self.T = T = load_train_module()
        from diffusers.models import AutoencoderKLWan
        from world_action_model.models import CasualWorldActionTransformer_MoT

        self.device = torch.device(device)
        self.num_steps = int(num_steps)
        t0 = time.time()
        model = CasualWorldActionTransformer_MoT(**T.TRANSFORMER_CFG)
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(sd, strict=True)
        del sd
        self.model = model.to(self.device, dtype=torch.bfloat16).eval()
        self.vae = AutoencoderKLWan.from_pretrained(str(VAE_DIR)).to(self.device, dtype=torch.float32).eval()
        self.vae.requires_grad_(False)
        norm = json.load(open(prep_dir / "norm_stats.json"))["norm_stats"]
        s, a = norm["observation.state"], norm["action"]
        self.s_lo, self.s_hi = np.asarray(s["q01"], np.float32), np.asarray(s["q99"], np.float32)
        self.a_lo, self.a_hi = np.asarray(a["q01"], np.float32), np.asarray(a["q99"], np.float32)
        t5 = torch.load(prep_dir / "t5_task_embeddings.pt", map_location="cpu", weights_only=False)
        self.prompt_embeds = {k: v.float() for k, v in t5["by_text"].items()}
        # training's view composition, bound to a val-mode (center-crop) stand-in for the dataset object
        self._views = types.SimpleNamespace(train=False)
        self._views._process_images = types.MethodType(T.SharpaGigaDataset._process_images, self._views)
        self._views._compose_views = types.MethodType(T.SharpaGigaDataset._compose_views, self._views)
        self.ckpt = ckpt
        print(f"[policy] loaded {ckpt} in {time.time() - t0:.1f}s; steps={self.num_steps}", flush=True)

    def preprocess(self, images: dict, state: np.ndarray, task: str) -> dict:
        T = self.T
        cams = T.EEF_CAMERAS[: T.VIEWS]
        video = {c: [Image.fromarray(np.asarray(images[c.split(".")[-1]], np.uint8))] * len(T.IMAGE_FRAME_OFFSETS)
                 for c in cams}
        imgs = self._views._compose_views(video)  # [5,3,384,320] in [-1,1]; only frame 0 conditions the action
        st = T.SharpaGigaDataset._quantile_norm(np.asarray(state, np.float32)[None], self.s_lo, self.s_hi)
        if task not in self.prompt_embeds:
            raise KeyError(f"task {task!r} has no training prompt embedding")
        p = self.prompt_embeds[task][: T.MAX_PROMPT_LEN]
        p = F.pad(p, (0, 0, 0, T.MAX_PROMPT_LEN - p.shape[0]), value=0)
        return {"images": imgs[None], "state": torch.from_numpy(st.astype(np.float32))[None],
                "prompt_embeds": p[None]}

    @torch.no_grad()
    def predict(self, images: dict, state: np.ndarray, task: str, seed: int = 0, num_steps: int | None = None):
        raw = self.preprocess(images, state, task)
        a = self.T.sample_actions(self.model, self.vae, raw, self.device, num_steps=num_steps or self.num_steps,
                                  seed=int(seed))[0].cpu().numpy()
        return (a + 1.0) / 2.0 * np.maximum(self.a_hi - self.a_lo, 1e-8) + self.a_lo

    def inference(self, data: dict) -> dict:
        t0 = time.perf_counter()
        action = self.predict(data["images"], data["state"], data["task"], data.get("seed", 0), data.get("num_steps"))
        torch.cuda.synchronize(self.device)
        return {"action": action.astype(np.float32), "timing": {"server_ms": (time.perf_counter() - t0) * 1e3},
                "ckpt": self.ckpt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=11411)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--prep-dir", type=Path, default=PREP_DIR, help="training prep dir (norm_stats.json, t5_task_embeddings.pt)")
    a = p.parse_args()
    sys.path.insert(0, str(REPO / "third_party/giga-world-policy/third_party/giga-models"))
    from giga_models.sockets.server import RobotInferenceServer

    policy = GWPSharpaPolicy(a.ckpt, num_steps=a.num_steps, prep_dir=a.prep_dir)
    RobotInferenceServer.start_server(policy, a.port)


if __name__ == "__main__":
    main()
