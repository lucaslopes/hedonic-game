from .Game import Game


__all__ = ["Game", "metrics", "stability"]


def __getattr__(name):
    # ``hedonic.metrics`` is importable as an attribute without importing it
    # (and its optional SciPy path) with the package.
    if name in {"metrics", "stability"}:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
