# ge

- **Source:** GE-Base-fast (Genie Envisioner) multi-view LTX transformer, vendored in `dit.py` from AgibotTech/Genie-Envisioner `d41358f` (`models/ltx_models/transformer_ltx_multiview.py`, Apache 2.0; action expert and upstream forwards removed). VAE (`AutoencoderKLLTXVideo`), T5 and tokenizer are LTX-Video's, loaded with diffusers / transformers.
- **Yamls:** `configs/model/video_backbone/ge_base_fast.yaml`.
- **`_vendor`:** none (the vendored DiT is `dit.py`). `masks.py` adds GE's per-view / cross-view pattern to the frame mask; GE's native training-noise law is the yaml `train_timestep: {law: ge_native}`, drawn by the shared `openwam/model/common/flow_scheduler.py`.
