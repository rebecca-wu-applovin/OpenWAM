wan_series = [
    {
        # Example: ModelConfig(model_id="krea/krea-realtime-video", origin_file_pattern="krea-realtime-video-14b.safetensors")
        "model_hash": "5ec04e02b42d2580483ad69f4e76346a",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
        },
        "state_dict_converter": "openwam.model.video_backbone.wan.shared.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "openwam.model.video_backbone.wan.text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "openwam.model.video_backbone.wan.vae.WanVideoVAE",
        "state_dict_converter": "openwam.model.video_backbone.wan.shared.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "eps": 1e-06,
        },
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "aafcfd9672c3a2456dc46e1cb6e52c70",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 16,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
        },
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "2267d489f0ceb9f21836532952852ee5",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 52,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
            # Upstream model attr for weight loading; the OpenWAM adapter
            # does not use reference_latents.
            "has_ref_conv": True,
        },
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "5b013604280dd715f8457c6ed6d6a626",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "eps": 1e-06,
        },
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
        "extra_kwargs": {
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "num_layers": 30,
            "eps": 1e-06,
            "seperated_timestep": True,
            "fuse_vae_embedding_in_latents": True,
        },
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "openwam.model.video_backbone.wan.vae.WanVideoVAE38",
        "state_dict_converter": "openwam.model.video_backbone.wan.shared.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
]

MODEL_CONFIGS = wan_series
