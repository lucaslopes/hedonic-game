from .Game import Game


__all__ = ["Game", "metrics"]


def __getattr__(name):
    # ``hedonic.metrics`` is importable as an attribute without importing it
    # (and its optional SciPy path) with the package.
    if name == "metrics":
        import importlib

        return importlib.import_module(".metrics", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
