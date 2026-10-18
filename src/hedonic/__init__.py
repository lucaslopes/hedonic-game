from .Game import Game


__all__ = ["Game", "metrics"]


def __getattr__(name):
    # ``hedonic.metrics`` is importable as an attribute without importing it
    # (and its optional SciPy path) with the package.
    if name == "metrics":
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name == "__version__":  # the installed distribution's version (a tag build sets it, e.g. 2.0.0.dev7)
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("hedonic")
        except PackageNotFoundError:  # a bare source tree on sys.path
            return "unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
