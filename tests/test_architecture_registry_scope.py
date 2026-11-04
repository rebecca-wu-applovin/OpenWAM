"""The architecture registry holds exactly the two kept dual_system variants (2026-10-21 deprecation).

single_system / tri_system / dual_system idm were deleted; a cfg naming them must fail loudly at resolve time
instead of silently falling back to another class.
"""

from types import SimpleNamespace

import pytest

import openwam.model.architectures  # noqa: F401  (run every @register_architecture)
from openwam.model.architectures.registry import _FRAMEWORK_VARIANT_INDEX, resolve_architecture_config


def _model_cfg(framework, variant):
    """framework / variant: str or None. Returns: the minimal model cfg resolve_architecture_config reads."""
    return SimpleNamespace(architecture={"framework": framework, "variant": variant, "action_dim": 7},
                           action_backbone={})


def test_only_dual_system_variants_registered():
    assert set(_FRAMEWORK_VARIANT_INDEX) == {("dual_system", "joint_self_attn"), ("dual_system", "joint_cross_attn")}


@pytest.mark.parametrize("framework,variant", [("dual_system", "idm"), ("single_system", "vanilla"),
                                               ("single_system", "moe"), ("tri_system", "joint_self_attn")])
def test_removed_variant_raises(framework, variant):
    # A removed architecture must not resolve to any surviving class.
    with pytest.raises(KeyError, match="Unsupported"):
        resolve_architecture_config(_model_cfg(framework, variant))


def test_variant_none_is_ambiguous():
    # Two dual_system variants remain, so an unset variant cannot pick one silently.
    with pytest.raises(KeyError, match="Ambiguous"):
        resolve_architecture_config(_model_cfg("dual_system", None))
