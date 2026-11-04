# OpenWAM

<p align="center"><strong>An Open, Modular Exploration Towards Systematic World–Action Model Pretraining</strong></p>

<p align="center">
  <a href="https://openwam-official.github.io/"><img src="https://img.shields.io/badge/Project%20Page-OPENWAM--OFFICIAL.GITHUB.IO-blue?style=flat&amp;logo=github" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2609.07398"><img src="https://img.shields.io/badge/arXiv-2609.07398-red?style=flat&amp;logo=arxiv" alt="Paper on arXiv"></a>
  <a href="https://huggingface.co/OpenWAM"><img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-orange?style=flat&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

## What is OpenWAM

OpenWAM is an open research stack for systematically developing **World-Action Models (WAMs)**. It turns tightly coupled design choices into modular components and controlled experiments. It consists of:

- **OpenWAM-Infra:** A modular infrastructure for composing and comparing model, representation, training, inference, deployment, and evaluation choices.
- **OpenWAM-Study:** Controlled studies that derive practical principles for inheriting world knowledge, coupling world and action learning, and scaling across domains.
- **OpenWAM-α:** An open pretrained WAM that applies these principles at scale, trained on 518.5M frames (about 6,400 hours) of egocentric human and robot data.

<!--
## Repository Layout

```text
OpenWAM/
├── openwam/
│   ├── dataloader/    # Dataset adapters (yam_sharpa), transforms, processors, registry
│   ├── model/
│   │   ├── architectures/    # WAM family: dual_system
│   │   ├── action_backbone/  # ActionDiTBackbone ABC, separate ActionDiT, scheduler
│   │   └── video_backbone/   # VideoBackbone ABC + one directory per family (wan / cosmos3 / ge / flux)
│   ├── train/         # OpenWAMTrainer, flow-match loss, checkpointing, optimizer utils
│   └── deploy/        # Policy server, model loader, inference engine, executors, optimizations
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tooling, LAPA tooling
├── configs/           # Hydra configs for model, dataloader, training, deploy
├── benchmarks/
│   └── simeval/       # sim-policy-eval server launcher
└── assets/            # Base-model checkpoints (created by the download script; git-ignored)
```

## Support Status

### Architectures

| Architecture | Variant | Description |
|---|---|---|
| `dual_system` | `joint_self_attn` | Separate ActionDiT + video DiT, fused per layer via one mixed self-attention (MoT driver). |
| `dual_system` | `joint_cross_attn` | Video DiT runs to completion → bridge features → ActionDiT runs once with cross-attention to them. Sub-variants via `detach_bridge`: `false` lets action gradients flow back into the video DiT, `true` blocks them (ActionDiT trains on detached video features) |

All architectures are selected via `configs/model/<framework>.yaml` with `architecture.variant`. The video backbone is composed from the Hydra `video_backbone` group (default `wan22_ti2v_5b`).

### Benchmarks and Evaluation

| Benchmark | Status | Notes |
|---|---|---|
| sim-policy-eval | Supported | Server launcher in `benchmarks/simeval/serve.py` |
-->

## News
- **[2026/09/09]** 📄 OpenWAM [Paper](https://arxiv.org/pdf/2609.07398) is released on arXiv.
- **[2026/09/06]** 🤖 OpenWAM is integrated into [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).
- **[2026/09/06]** 🤗 We release all pretrained and finetuned models on [huggingface](https://huggingface.co/OpenWAM).
- **[2026/09/06]** 🔥 OpenWAM Codebase Release！

## Installation

Create an environment with **conda**:

```bash
# Requires Python >= 3.10
conda create -n openwam python=3.10
conda activate openwam
```

or with **venv**:

```bash
# Requires Python >= 3.10 (check with `python3 --version`)
python3 -m venv .venv
source .venv/bin/activate
```

We recommend using PyTorch 2.7.1 with CUDA 12.8 (others may also work):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install OpenWAM:

```bash
pip install -e .
```

> `deepspeed` ships as a source distribution and is compiled during `pip install`, so a C compiler (`gcc`) must be on the PATH.

## Assets Preparation

The downloaders are interactive by default; every menu step also has a flag (`--name`, `--source`, `--root`, `--yes`, see `--help`) so they can run unattended, and the default storage location is resolved relative to the repository regardless of the working directory. The downloader stores assets under
`assets/` and updates the matching YAML path.

### 1. Video Backbone

~~~bash
python scripts/download_assets/download_video_backbone.py
~~~

**Supported video backbones**

<table>
<tr>
<td>Wan2.2-TI2V-5B ✅</td>
<td>Cosmos3-Edge ✅</td>
</tr>
</table>

Weights are saved under `assets/video_backbone_ckpt/` and the selected
`configs/model/video_backbone/` file is updated with the downloaded path.

### 2. Training Data

The yam_sharpa LeRobot v3 data and its normalization stats are configured in
[`configs/dataloader/yam_sharpa.yaml`](configs/dataloader/yam_sharpa.yaml)
(one `sources.<name>.dataset_dir` per source, plus `normalization_stats_path`); there is no bundled downloader.

## Quick Start

Quick Start provides a minimal end-to-end example: prepare the assets, train
a policy, deploy its checkpoint, and run a first inference check.

The example uses the DualSystem JointSelfAttention architecture, the
Wan2.2-TI2V-5B video backbone, and the Mutual attention mask.

| Component | Selection | Configuration |
|---|---|---|
| Architecture | dual_system / joint_self_attn | [`configs/model/dual_system.yaml`](configs/model/dual_system.yaml) |
| Video backbone | wan22_ti2v_5b | [`configs/model/video_backbone/wan22_ti2v_5b.yaml`](configs/model/video_backbone/wan22_ti2v_5b.yaml) |
| Attention mask | mutual | model.architecture.attention_mask_mode |
| Dataset | yam_sharpa | [`configs/dataloader/yam_sharpa.yaml`](configs/dataloader/yam_sharpa.yaml) |

> **Resource recommendation:** We recommend 8 GPUs with 80 GB VRAM each for
> training. This configuration supports normal training for all architectures
> using Wan2.2-5B or smaller video backbones. More GPUs are better when
> available and can further improve training throughput.

### From Scratch Training

1. Point [`configs/dataloader/yam_sharpa.yaml`](configs/dataloader/yam_sharpa.yaml) at the
   yam_sharpa data (`sources.<name>.dataset_dir`, e.g. `dataloader.sources.simteleop.dataset_dir=...`) and its
   normalization stats (`normalization_stats_path`).

2. Download Wan2.2-TI2V-5B:

   ~~~bash
   python scripts/download_assets/download_video_backbone.py
   ~~~

   Select Wan2.2-TI2V-5B and the desired model source.

3. Start a debug run with the complete model selection. [`run_dual.sh`](run_dual.sh) (local 8 GPUs) and
   [`run_modal.sh`](run_modal.sh) (Modal) wrap the same overrides; extra Hydra overrides are appended:

   ~~~bash
   bash run_dual.sh training.debug=true
   ~~~

   training.debug=true runs 20 steps, saves at steps 10 and 20, and uses a
   constant learning rate. Check the run output, then drop it for normal
   training. Training defaults and CLI overrides are defined in
   [`configs/train.yaml`](configs/train.yaml). Outputs use training.output_path,
   whose default is `outputs/openwam_checkpoints`.

4. Deploy the debug checkpoint:

   ~~~bash
   bash scripts/deploy.sh <debug_ckpt_dir_path>
   ~~~

   Deployment enables compile by default, so the first inference may take
   longer while compilation warms up. For sim-policy-eval, launch the server
   per checkpoint file with [`benchmarks/simeval/serve.py`](benchmarks/simeval/serve.py).

## OpenWAM Usage Guidance

OpenWAM is configured through composable Hydra YAML files. Select an architecture,
backbone, dataloader, and runtime behavior by changing configuration values or
overriding them on the command line. The task guides below are the maintained
entry points for using and extending the repository:

| Guide | Use it when you want to… |
|---|---|
| [Training and deployment](assets/openwam_usage_docs/train-and-deploy.md) | choose a model/dataloader, prepare assets, train, fine-tune, resume, or deploy a policy |
| [Architecture extension](assets/openwam_usage_docs/architecture-extension.md) | add a video backbone (implement the base class's 14 abstract members + write one yaml), an action backbone, or a WAM architecture |
| [Benchmark integration](assets/openwam_usage_docs/benchmark-integration.md) | extend a dataloader and connect a benchmark client to the WebSocket protocol |

> 💡 **Agent tip:** Pick the guide that matches your task and feed it directly
> to your agent — less explaining, more building.

Installation and Assets Preparation above cover environment setup and model
downloads.

## License

OpenWAM is released under the [Apache License 2.0](LICENSE).

## Citation

If you use OpenWAM, please cite:

```bibtex
@article{wang2026openwam,
  title   = {OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining},
  author  = {Yuran Wang and Siqiao Huang and Mingleyang Li and Chenhao Zhang and Jiaqi Liang and Weiyang Jin and Yue Chen and Xuemin Chi and Donghao Zhou and Qize Yu and Yu-Kai Wang and Yuhan Rui and Shenzhe Yao and Zhen Yuan and Zhenhao Shen and Kefei Zhu and Zijie Zhu and Ning Gao and Xiaowei Chi and Guanqi He and Shanghang Zhang and Hao Dong and Lin Shao and Hang Zhao},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2609.07398}
}
```

