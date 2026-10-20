import os as _os

if _os.environ.get("HEDONIC_CORE") == "rust":
    # The Rust port of the core (rust/, built with maturin) in place of the C
    # extension: the same functions, bit for bit. The compiled module itself,
    # not maturin's package around it, so _core.__file__ is the binary.
    import sys as _sys

    from hedonic_core_rs import hedonic_core_rs as _core

    _sys.modules[__name__ + "._core"] = _core

from .Game import Game


__all__ = ["Game", "metrics", "stability"]


def __getattr__(name):
    # ``hedonic.metrics`` is importable as an attribute without importing it
    # (and its optional SciPy path) with the package.
    if name in {"metrics", "stability"}:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name == "__version__":  # the installed distribution's version (a tag build sets it, e.g. 2.0.0.dev7)
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("hedonic")
        except PackageNotFoundError:  # a bare source tree on sys.path
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
