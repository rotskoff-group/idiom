"""How a config spec names the thing it wants, and how that name becomes a callable.

Rewards and shaping rules are built the same way, from the same kind of spec: a **factory** named
either by an alias or by a "module:function" path, called with the spec's own arguments.

    entropy                                         # a bare name, no arguments
    {name: quadratic, target: 3.65, width: 0.2}     # a name plus that factory's arguments
    {name: "mypkg.scoring:make_scorer", cutoff: 0.3}

A reward factory returns a *reward*: one call per GRPO step, mapping the step's decoded IDRs to one
raw value each. Wrap a function that scores a single IDR with lift:

    def fraction_charged():
        return lift(lambda idr: sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0)

A shaping factory returns f(raw) -> float. Both are called once, while the config is validated, so
a factory that raises on a bad argument fails in seconds rather than on the first training step.

The aliases are short names for the factories the library ships; they map to "module:function"
paths and are resolved before anything is imported, so naming one imports only what it points at.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable

# A reward scores a whole step at once: one raw value per IDR, in the order they were given.
Reward = Callable[[list[str]], list[float]]

_BUILTIN = "idiom.train.grpo.reward.builtin"

REWARD_ALIASES: dict[str, str] = {
    "entropy": f"{_BUILTIN}:entropy",
    "length": f"{_BUILTIN}:length",
    "scorer": "idiom.train.grpo.reward.external:scorer",
    "sae_signature": "idiom.train.grpo.reward.sae_feature:sae_signature",
}

SHAPING_ALIASES: dict[str, str] = {
    "identity": "idiom.train.grpo.reward.shaping:identity",
    "quadratic": "idiom.train.grpo.reward.shaping:quadratic",
    "gaussian": "idiom.train.grpo.reward.shaping:gaussian",
}


def lift(score: Callable[[str], float]) -> Reward:
    """Turn a function scoring one IDR into a reward scoring a whole step.

    Args:
        score (Callable[[str], float]): Maps one decoded IDR to one raw value.

    Returns:
        Reward: Maps a list of IDRs to one raw value each, in order.
    """
    return lambda idrs: [float(score(idr)) for idr in idrs]


def import_module(spec: str):
    """Import a module by dotted name or by path to a .py file.

    Args:
        spec (str): A dotted module path, or a path ending in ".py".

    Returns:
        ModuleType: The imported module.
    """
    if spec.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("idiom_reward_module", spec)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def load_callable(path: str, where: str = "spec"):
    """Import and return the callable a "module:function" path names.

    Args:
        path (str): "package.module:function", or "path/to/file.py:function".
        where (str): The config path this came from, used in error messages.

    Returns:
        Callable: The named attribute.

    Raises:
        ValueError: If the path is malformed, the module cannot be imported, the attribute is
            missing, or it is not callable.
    """
    mod_name, _, attr = path.rpartition(":")
    if not mod_name or not attr:
        raise ValueError(f"{where}: {path!r} is not of the form 'module:function'")
    try:
        module = import_module(mod_name)
    except Exception as e:  # ImportError, FileNotFoundError, or anything the module raises
        raise ValueError(f"{where}: cannot import {mod_name!r} ({type(e).__name__}: {e})") from e
    try:
        fn = getattr(module, attr)
    except AttributeError:
        raise ValueError(f"{where}: {mod_name!r} has no attribute {attr!r}") from None
    if not callable(fn):
        raise ValueError(f"{where}: {attr!r} is not callable (got {type(fn).__name__})")
    return fn


def spec_name(spec, where: str) -> tuple[str, dict]:
    """Split a spec into the factory it names and the arguments it passes.

    Args:
        spec: A bare name, or a mapping with a "name" plus that factory's arguments.
        where (str): The config path this came from, used in error messages.

    Returns:
        tuple[str, dict]: The name, and the keyword arguments for its factory.

    Raises:
        ValueError: If the spec is neither a string nor a mapping, or a mapping without a name.
    """
    if isinstance(spec, str):
        return spec, {}
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a name or a mapping with a name, got "
                         f"{type(spec).__name__}")
    kwargs = dict(spec)
    name = kwargs.pop("name", None)
    if not name:
        raise ValueError(f"{where}: a mapping needs a name naming the factory to call, plus that "
                         f"factory's arguments")
    return name, kwargs


def build_from_spec(spec, aliases: dict[str, str], what: str, where: str):
    """Resolve a spec and call the factory it names, returning what that factory built.

    Args:
        spec: A bare name, or a mapping with a "name" plus that factory's arguments.
        aliases (dict[str, str]): Short names for the factories the library ships.
        what (str): "reward" or "shaping", used in error messages.
        where (str): The config path this came from, used in error messages.

    Returns:
        Callable: The reward or shaping rule the factory returned.

    Raises:
        ValueError: If the spec is malformed, the name resolves to nothing importable, the
            arguments do not fit the factory, or the factory does not return a callable.
    """
    name, kwargs = spec_name(spec, where)
    path = aliases.get(name, name)
    if ":" not in path:
        raise ValueError(f"{where}: unknown {what} {name!r}; the shipped names are "
                         f"{sorted(aliases)}, or give a 'module:function' path to your own")
    fn = load_callable(path, f"{where}: {what} {name!r}")
    try:
        built = fn(**kwargs)
    except TypeError as e:  # a missing argument, or one this factory does not take
        raise ValueError(f"{where}: bad arguments for {what} {name!r}: {e}") from e
    if not callable(built):
        raise ValueError(f"{where}: {what} {name!r} must be a factory returning a callable, but "
                         f"it returned {type(built).__name__}; see the reward section of "
                         f"cookbook/rewards/README.md")
    return built
