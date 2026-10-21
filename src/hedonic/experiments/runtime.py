"""What ran an experiment: the versions and identities every record carries.

A record names the detector that produced it (``algorithm_identity``) and the
build it ran on (``build_identity``): the results series (``RESULTS_SERIES``)
and the source digest of the core's *results class*. Cores whose three
behaviour fingerprints (``hedonic.experiments.fingerprint``) are equal compute
the same partitions, covers and audits bit for bit, so they share one identity
(``CORE_CLASSES``): the C core of 2.0.0rc11, a rebuilt wheel, a later C source
or a Rust port that keeps the fingerprints all name the digest the records
already carry. A core of unknown fingerprints is a new class, named by its own
source digest. The full version, the real source digest and the compiled
bytes' SHA-256 are provenance (``runtime_versions``), never identity.
"""

from __future__ import annotations

import functools
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

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


# The series records name for the Python layer (``build_identity()["hedonic"]``). It moves only when a change of the
# Python layer changes results, never with the package version: 2.0.0.devN, rcN, 2.0.x and a 2.1 that computes the
# same results all keep "2.0". Bump it (and say why in the changelog) when results change; every record is then redone.
RESULTS_SERIES = "2.0"

# The results classes of the core: the three fingerprints (``fingerprint.GRIDS`` order) of every core that computes
# them, the source digest the class's records name (``records``), and the core sources shown to be in the class
# (``members``: source digest -> the release that shipped it and the evidence). A core is in a class when its own three
# fingerprints are the key: it then takes ``records`` whatever its own digest, so a source change that keeps every
# result (a refactor, a faster search, another language) keeps every finished record. A member is listed so the table
# says which sources were checked; ``tests/experiments/test_core_identity.py`` checks the loaded core against it and
# rebuilds every member from its tag to recompute its fingerprints (HEDONIC_REBUILD_CORES=0 skips the rebuild).
CORE_CLASSES: dict[tuple[str, str, str], dict] = {
    ("partitions-v1:0668d1011dc78b6fa63e6c9a0c17901e4738d421a270e906858fb415593c9c12",
     "covers-v1:ddc34c763b38292fd19c03e9a89da401336fcddb098ea1c7983dbb964b93966a",
     "audit-v1:39d5ad0e58d4e6816d33c7209da31937226f19ebff88d0c004e9f1629c0d7e48"): {
        "records": "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4",
        "members": {
            "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4": {
                "tag": "v2.0.0rc9", "releases": "2.0.0rc5 to rc9",
                "evidence": "the class's first core: the overlap queue's records name it"},
            "c86846640631141ecf334b1b34050110e8b93c1b1e9447596f31a49499f5dc3e": {
                "tag": "v2.0.0rc11", "releases": "2.0.0rc10, rc11",
                "evidence": "only the stability audit's search changed (hig_audit_intervals evaluates the cells "
                            "convexity cannot settle; `make -C native test` compares it with the exhaustive search bit "
                            "for bit, and every audit of the overlap queue's ledger recomputed identically); the same "
                            "three fingerprints, measured on the rc9 and rc11 wheels"},
            "19bc0c978909e803503bc40fa25de1f83ecc27784cb7c04c46ad77093ed33325": {
                "tag": "v2.1.0.dev1", "releases": "2.1.0 (the Rust core)",
                "evidence": "a port of the rc11 C core to Rust (native/rust): the crate's tests hold test_core.c's and "
                            "test_audit.c's pins, test_differential.py compares every result, trace, verifier report "
                            "and audit with the C core bit for bit, and the overlap queue's dry run reads the same; "
                            "the same three fingerprints"},
            "c2014342699829a3046c5c512d5d566835584f18eaa3c4ccfbe230b38cc4d1e7": {
                "tag": "v2.1.0.dev2", "releases": "2.1.0 (buffer entry points for hedonic.arrays)",
                "evidence": "the binding gains detect_arrays and _audit_prepare_arrays (the same run as detect and "
                            "_audit_prepare over two endpoint buffers); the engines are unchanged: the crate's pins, "
                            "tools/core_pins.py, the pinned Python tests and tests/test_arrays.py (against Game, bit "
                            "for bit) pass unchanged; the same three fingerprints"},
        },
    },
}

# Source digests of cores that compute everything a record holds exactly as an earlier core does: records name the
# earlier digest. Since 2.0.0rc12 the fingerprints decide (``CORE_CLASSES``); this table is the same facts per source
# digest, kept for readers of earlier releases, and a test checks that each alias and its target are members of one
# class whose records name the target.
EQUIVALENT_CORES: dict[str, str] = {
    # 2.0.0rc10: only the stability audit's search changed (see CORE_CLASSES).
    "c86846640631141ecf334b1b34050110e8b93c1b1e9447596f31a49499f5dc3e":
        "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4",
    # 2.1.0: the Rust core (see CORE_CLASSES).
    "19bc0c978909e803503bc40fa25de1f83ecc27784cb7c04c46ad77093ed33325":
        "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4",
    # 2.1.0: the binding's buffer entry points for hedonic.arrays (see CORE_CLASSES).
    "c2014342699829a3046c5c512d5d566835584f18eaa3c4ccfbe230b38cc4d1e7":
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
def core_fingerprints() -> tuple[str, str, str]:
    """The loaded core's three behaviour fingerprints (computed once per process, cached on disk per compiled core;
    a few seconds the first time on a machine)."""
    from hedonic.experiments.fingerprint import fingerprints

    return fingerprints()


def core_class() -> dict | None:
    """The loaded core's results class (an entry of ``CORE_CLASSES``), or None when no class has its fingerprints."""
    return CORE_CLASSES.get(core_fingerprints())


@functools.cache
def core_source_sha256() -> str:
    """The source digest records name: the one of the loaded core's results class (``CORE_CLASSES``), else, for a
    core of fingerprints no class has, its own (``core_built_source_sha256``): a new class.

    The core's rounding is fixed by its source, so two builds of one source (another platform, compiler, wheel or
    devN) compute the same results, and so does another source with the same fingerprints: records name this digest,
    not the compiled bytes."""
    results = core_class()
    return results["records"] if results else core_built_source_sha256()


def release_series() -> str:
    """``major.minor`` of the installed version (``2.0`` for 2.0.0.dev9, 2.0.0rc1 and 2.0.3): provenance. Identities
    name ``RESULTS_SERIES``, which a release that keeps the results (2.1 with the same fingerprints) does not move."""
    import re

    match = re.match(r"(\d+)\.(\d+)", hedonic_version())
    return f"{match.group(1)}.{match.group(2)}" if match else hedonic_version()


def build_identity() -> dict[str, str]:
    """What a record of a Hedonic detector call depends on: the results series and the core's results class."""
    return {"hedonic": RESULTS_SERIES, "hedonic_core_source_sha256": core_source_sha256()}


def algorithm_identity() -> str:
    """The detector's name in records: ``community_hedonic/hedonic-<series>+core-<16 hex of the source digest>``."""
    return f"community_hedonic/hedonic-{RESULTS_SERIES}+core-{core_source_sha256()[:16]}"


def __getattr__(name: str):
    # ``ALGORITHM_IDENTITY`` is computed on first use, not at import: it needs the core's fingerprints.
    if name == "ALGORITHM_IDENTITY":
        return algorithm_identity()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def runtime_versions() -> dict[str, Any]:
    """Versions and paths of everything that ran a detector call.

    ``hedonic_core_sha256`` identifies the build, ``hedonic_core_built_source_sha256`` its sources and
    ``hedonic_core_fingerprints`` what it computes; ``hedonic_core_source_sha256`` is the identity records name.
    ``igraph_c`` is the C core of the stock python-igraph wheel, which only the comparison methods (full CPM
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
        "hedonic_core_fingerprints": list(core_fingerprints()),
        "python_igraph": getattr(igraph, "__version__", None),
        "igraph_c": getattr(native, "__igraph_version__", None),
        "python_igraph_path": getattr(igraph, "__file__", None),
        "igraph_extension_path": getattr(native, "__file__", None),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
