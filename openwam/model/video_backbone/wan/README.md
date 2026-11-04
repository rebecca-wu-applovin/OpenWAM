# wan

- **Source:** Wan2.2 modules (`dit.py` WanModel, `vae.py` Wan2.2 VAE38, `text_encoder.py` umT5) derived from DiffSynth-Studio (Apache 2.0, see `license/`); no pinned upstream commit, taken over from OpenWAM's earlier vendored DiffSynth package.
- **Yamls:** `configs/model/video_backbone/wan22_ti2v_5b.yaml`, `longlive2_robot_s.yaml` (LongLive2.0-Robot-S DiT weights, 128-token prompts) and `wan22_ti2v_tiny.yaml` (debug, random init); all build the one `WanBackbone`.
- **`_vendor`:** none. `backbone.py` is the `VideoBackbone` implementation; `dit.py` holds the DiT network and `build_time_modulation`. The training noise law is the yaml `train_timestep: {law: grid, weighting: bell}`, drawn by the shared `openwam/model/common/flow_scheduler.py`.
