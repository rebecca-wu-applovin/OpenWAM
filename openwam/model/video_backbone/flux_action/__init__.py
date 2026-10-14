"""flux-3-action-base (FLUX 3 Action) video trunk for OpenWAM; see ``backbone.py``.

The VAE module (``video_vae.py``) imports NATTEN and the text encoder imports ``transformers``; both
are imported lazily inside ``FluxActionVideoBackbone.from_pretrained`` so this package stays importable
on CPU-only CI.
"""

from openwam.model.video_backbone.flux_action.backbone import FluxActionVideoBackbone

__all__ = ["FluxActionVideoBackbone"]
