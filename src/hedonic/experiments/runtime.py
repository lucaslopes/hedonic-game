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
    """SHA-256 of the compiled core (``hedonic._core``) that this process loaded."""
    from hedonic import _core

    digest = hashlib.sha256()
    with open(_core.__file__, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Source digests of cores that compute everything a record holds exactly as an earlier core does: records name the
# earlier digest, so their work stays valid. An entry needs that evidence, not just an unchanged detector.
EQUIVALENT_CORES: dict[str, str] = {
    # 2.0.0rc10: only the stability audit's search changed. hig_audit_intervals now evaluates only the cells convexity
    # cannot settle and returns the exhaustive search's intervals bit for bit (native/README.md; `make -C native test`
    # compares them, and every audit of the overlap queue's ledger recomputed identically). Detector sources unchanged.
    "c86846640631141ecf334b1b34050110e8b93c1b1e9447596f31a49499f5dc3e":
        "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4",
}


@functools.cache
def core_built_source_sha256() -> str:
    """SHA-256 of the sources the loaded core was compiled from (``hedonic._core.source_sha256``, set by its build:
    ``native/rust/build.rs``).
    A core built without it falls back to the SHA-256 of its bytes."""
    from hedonic import _core

    return getattr(_core, "source_sha256", None) or core_sha256()


@functools.cache
def core_source_sha256() -> str:
    """The source digest records name: the loaded core's (``core_built_source_sha256``), or the earlier one it is
    equivalent to (``EQUIVALENT_CORES``).

    The core's rounding is fixed by its source, so two builds of one source (another platform, compiler, wheel or
    devN) compute the same results: records name this digest, not the compiled bytes."""
    built = core_built_source_sha256()
    return EQUIVALENT_CORES.get(built, built)


def release_series() -> str:
    """``major.minor`` of the installed version (``2.0`` for 2.0.0.dev9, 2.0.0rc1 and 2.0.3). A change of the Python
    layer that changes results bumps the minor version; dev, rc and patch releases share the series."""
    import re

    match = re.match(r"(\d+)\.(\d+)", hedonic_version())
    return f"{match.group(1)}.{match.group(2)}" if match else hedonic_version()


def build_identity() -> dict[str, str]:
    """What a record of a Hedonic detector call depends on: the release series and the core's source digest."""
    return {"hedonic": release_series(), "hedonic_core_source_sha256": core_source_sha256()}


def algorithm_identity() -> str:
    """The detector's name in records: ``community_hedonic/hedonic-<series>+core-<16 hex of the source digest>``."""
    return f"community_hedonic/hedonic-{release_series()}+core-{core_source_sha256()[:16]}"


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
        "hedonic_core_source_sha256": core_source_sha256(),
        "hedonic_core_built_source_sha256": core_built_source_sha256(),
        "python_igraph": getattr(igraph, "__version__", None),
        "igraph_c": getattr(native, "__igraph_version__", None),
        "python_igraph_path": getattr(igraph, "__file__", None),
        "igraph_extension_path": getattr(native, "__file__", None),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
