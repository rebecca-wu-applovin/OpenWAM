"""API hygiene for the cosmos3_edge backbone (predict2.5 parity rules).

The backbone class may expose zero public members beyond the VideoBackbone ABC;
its decode hook must be wired (a real validation error, not NotImplementedError).
"""

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone, VideoBackbone


def _public_members(cls):
    return {n for n in dir(cls) if not n.startswith("_")}


def test_no_public_surface_beyond_base_contract():
    extra = _public_members(Cosmos3EdgeVideoBackbone) - _public_members(VideoBackbone)
    assert extra == set(), f"cosmos3_edge leaks public members beyond the ABC: {sorted(extra)}"


def test_is_video_backbone_subclass():
    assert issubclass(Cosmos3EdgeVideoBackbone, VideoBackbone)


def _bare_instance():
    return Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)


def test_decode_hook_is_wired_not_stub():
    vb = _bare_instance()
    try:
        vb.decode_latents(None)  # type: ignore[arg-type]
    except ValueError:
        pass  # the real gate: no VAE attached
    except NotImplementedError as exc:  # pragma: no cover
        raise AssertionError("decode_latents is an unwired stub") from exc
