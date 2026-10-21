import os as _os


def _select_core():
    """HEDONIC_CORE=<name> runs another core in place of hedonic._core: the module hedonic._core_<name>, installed
    beside it, with the same functions (a build of the core in another language, to compare the two). The name of
    the core hedonic._core already is (its ``backend``, e.g. ``rust``) selects it; unset or empty, hedonic._core."""
    name = _os.environ.get("HEDONIC_CORE", "")
    if not name:
        return
    import importlib
    import sys

    if not name.isidentifier():
        raise ImportError(f"HEDONIC_CORE={name!r} is not a core's name (hedonic._core_<name>); unset HEDONIC_CORE to "
                          "use hedonic._core")
    module = f"{__name__}._core_{name}"
    try:
        core = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        default = importlib.import_module(f"{__name__}._core")
        if getattr(default, "backend", None) == name:
            return
        raise ImportError(f"HEDONIC_CORE={name!r}: no module {module} is installed beside hedonic._core (whose backend "
                          f"is {getattr(default, 'backend', 'unknown')!r}); build it, or unset HEDONIC_CORE") from None
    sys.modules[f"{__name__}._core"] = core
    globals()["_core"] = core


_select_core()

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
