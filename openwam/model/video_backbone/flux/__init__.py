"""flux-3-action-base (FLUX 3 Action) video trunk for OpenWAM; see ``backbone.py``.

The VAE module (``vae.py``) imports NATTEN and the text encoder imports ``transformers``; both
are imported lazily inside the ``FluxActionVideoBackbone`` constructors so this package stays importable
on CPU-only CI.
"""

from openwam.model.video_backbone.flux.backbone import FluxActionVideoBackbone

__all__ = ["FluxActionVideoBackbone"]
