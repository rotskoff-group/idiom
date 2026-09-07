"""Smoke tests: the package imports and resolves CPU. CPU-only."""

from idiom.utils.device import is_cpu_only, resolve_device


def test_package_skeleton_imports():
    import idiom.data
    import idiom.model
    import idiom.sae
    import idiom.train

    assert all(m is not None for m in (idiom.data, idiom.model, idiom.sae, idiom.train))


def test_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_cpu_first_default():
    assert is_cpu_only()
