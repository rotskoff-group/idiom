"""P0 smoke tests: the v2 package skeleton imports and resolves CPU. CPU-only."""

from idiom.utils.device import is_cpu_only, resolve_device


def test_package_skeleton_imports():
    import idiom.data  # noqa: F401
    import idiom.model  # noqa: F401
    import idiom.sae  # noqa: F401
    import idiom.train  # noqa: F401


def test_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_cpu_first_default():
    # conftest sets IDIOM_DEVICE=cpu, so the default must resolve to CPU here.
    assert is_cpu_only()
