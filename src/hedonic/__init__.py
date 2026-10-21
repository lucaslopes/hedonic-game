import os as _os
import sys as _sys
import types as _types


def _select_core():
    """HEDONIC_CORE=<name> runs another core in place of hedonic._core: the module hedonic._core_<name>, installed
    beside it, with the same functions (a build of the core in another language, to compare the two). The name of
    the core hedonic._core already is (its ``backend``, e.g. ``rust``) selects it; unset or empty, hedonic._core."""
    name = _os.environ.get("HEDONIC_CORE", "")
    if not name:
        return
    import importlib

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
    _sys.modules[f"{__name__}._core"] = core
    globals()["_core"] = core


_select_core()


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
