# cosmos3

- **Source:** NVIDIA Cosmos3 dual-stream transformer from huggingface/diffusers `6ad357395d936c4d27347463f938cdbb400a6e59`; the VAE (`AutoencoderKLWan`) and the tokenizer come from the installed diffusers / transformers.
- **Yamls:** `configs/model/video_backbone/cosmos3_edge.yaml` (4B) and `cosmos3_nano.yaml` (16B); one class, the variant is the yaml's `dit` / `vae` blocks.
- **`_vendor`:** `transformer_cosmos3.py` (`Cosmos3OmniTransformer`), byte-identical to upstream except absolute imports; provenance and edit policy in `_vendor/README.md`. It can go once a released diffusers ships Edge support (followups).
- `dit.py` drives the vendored layers block by block (und tower, gen blocks, MoT split); `text.py` packs prompts and mRoPE positions.
