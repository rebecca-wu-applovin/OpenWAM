# flux

- **Source:** the video side of FLUX 3 Action, vendored from black-forest-labs/flux-action `830f1ce` (Apache 2.0): `dit.py` (trunk, upstream forwards removed), `vae.py` (ViTNorm video VAE, needs NATTEN), `text_encoder.py` (Qwen3-VL-4B embedder).
- **Yamls:** `configs/model/video_backbone/flux3_action_base.yaml` (requires `proprio_input: backbone_stream`).
- **`_vendor`:** none (the vendored files sit in the family directory, each with its provenance header). `masks.py` maps the `[T | F | H | P | A]` sequence onto the frame mask; flux's native training timestep law is the yaml `train_timestep: {law: logit_logistic, ...}`, drawn by the shared `openwam/model/common/flow_scheduler.py`.
