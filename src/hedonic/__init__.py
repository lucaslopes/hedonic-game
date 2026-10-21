import os as _os
import sys as _sys
import types as _types

if _os.environ.get("HEDONIC_CORE") == "rust":
    # The Rust port of the core (rust/, built with maturin) in place of the C
    # extension: the same functions, bit for bit. The compiled module itself,
    # not maturin's package around it, so _core.__file__ is the binary.
    from hedonic_core_rs import hedonic_core_rs as _core

    _sys.modules[__name__ + "._core"] = _core


__all__ = ["Game", "arrays", "metrics", "stability"]


class _Package(_types.ModuleType):
    """``hedonic.Game`` is the class even after the submodule of that name is
    imported (``import hedonic.Game``, or unpickling a Game), which would
    otherwise bind the module in its place."""

    def __setattr__(self, name, value):
        if name == "Game" and isinstance(value, _types.ModuleType):
            value = value.Game
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package


def __getattr__(name):
    # ``Game`` (and igraph with it) is imported on first use, so the array
    # path (``hedonic.arrays``) runs without igraph. ``hedonic.metrics`` is
    # importable as an attribute without importing it (and its optional SciPy
    # path) with the package.
    if name == "Game":
        from .Game import Game

        return Game
    if name in {"arrays", "metrics", "stability"}:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name == "__version__":  # the installed distribution's version (a tag build sets it, e.g. 2.0.0.dev7)
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("hedonic")
        except PackageNotFoundError:  # a bare source tree on sys.path
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
