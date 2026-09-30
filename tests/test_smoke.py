"""Smoke tests: the package imports and resolves CPU. CPU-only."""

from idiom.utils.device import is_cpu_only, resolve_device


def test_package_skeleton_imports():
    """Verify that the package and its public components import successfully."""
    import idiom.data
    import idiom.model
    import idiom.sae
    import idiom.train

    assert all(m is not None for m in (idiom.data, idiom.model, idiom.sae, idiom.train))


def test_explicit_cpu():
    """Verify that explicit CPU selection returns a CPU device."""
    assert resolve_device("cpu").type == "cpu"


def test_cpu_first_default():
    """Verify that device resolution selects CPU in the test environment."""
    assert is_cpu_only()
