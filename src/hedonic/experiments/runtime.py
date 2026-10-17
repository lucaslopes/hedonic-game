"""What ran an experiment: the versions and identities every record carries.

A record names the detector that produced it (``ALGORITHM_IDENTITY``) and the
build it ran on (``runtime_versions``). The identity follows the installed
``hedonic`` release; the SHA-256 of the compiled core tells two builds of one
release apart, so a resume check can refuse records made by another build.
"""

from __future__ import annotations

import functools
import hashlib
import platform
import sys
from pathlib import Path

import igraph


@functools.cache
def hedonic_version() -> str:
    """The installed ``hedonic`` version (``"unknown"`` without metadata)."""
    try:
        from importlib.metadata import version

        return version("hedonic")
    except Exception:  # pragma: no cover - metadata absent in a bare checkout
        return "unknown"


@functools.cache
def core_sha256() -> str:
    """SHA-256 of the compiled C core (``hedonic._core``) that this process loaded."""
    from hedonic import _core

    digest = hashlib.sha256()
    with open(_core.__file__, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def algorithm_identity() -> str:
    """The detector's name in records: ``community_hedonic/hedonic-<version>``."""
    return f"community_hedonic/hedonic-{hedonic_version()}"


ALGORITHM_IDENTITY = algorithm_identity()


def runtime_versions() -> dict[str, str | None]:
    """Versions and paths of everything that ran a detector call.

    ``hedonic_core_sha256`` identifies the build; ``igraph_c`` is the C core of
    the stock python-igraph wheel, which only the comparison methods (full CPM
    Leiden, graph loading) use.
    """
    from hedonic import _core

    native = getattr(igraph, "_igraph", None)
    return {
        "hedonic": hedonic_version(),
        "hedonic_core": getattr(_core, "core_version", None),
        "hedonic_core_path": str(Path(_core.__file__).resolve()),
        "hedonic_core_sha256": core_sha256(),
        "python_igraph": getattr(igraph, "__version__", None),
        "igraph_c": getattr(native, "__igraph_version__", None),
        "python_igraph_path": getattr(igraph, "__file__", None),
        "igraph_extension_path": getattr(native, "__file__", None),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
