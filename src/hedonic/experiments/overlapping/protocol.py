"""Locked protocol identity and read-only evidence reconciliation.

This module never loads a raw SNAP archive or runs a detector.  It reconstructs
the persisted bounded analysis graph solely to re-score covers and independently
replay the equilibrium audit, then explains whether every requested condition
is admissible under the locked identity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import subprocess
import tomllib
from collections import Counter
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import igraph as ig

from hedonic.experiments.config import PAPER_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.methods import (
    METHODS,
    method_dependency_identity,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    quality_overlapping_cpm,
)
from hedonic.experiments.overlapping.robustness import audit_cover
from hedonic.experiments.overlapping.snap import (
    ANALYSIS_GRAPH_POLICY,
    canonicalize_cover,
    cover_sha256,
    graph_sha256,
)
from hedonic.utils import sample_uniform_ints


LOCK_PATH = Path(__file__).resolve().parents[4] / "configs" / "overlapping-paper-protocol.lock.json"
IDENTITY_SCHEMA_VERSION = 3
REQUIRED_EXTERNAL_DEPENDENCIES = frozenset({"cpm", "demon"})
REQUIRED_SCIENTIFIC_DEPENDENCIES = frozenset({"numpy", "scipy"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )


def _metrics_hash(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError):
        return None
    return _sha256_bytes(encoded)


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _repo_root_for_lock(lock_path: Path) -> Path:
    parent = lock_path.resolve().parent
    return parent.parent if parent.name == "configs" else parent


def _parse_protocol_lock(encoded: bytes) -> dict[str, Any]:
    lock = json.loads(encoded)
    if int(lock.get("schema_version", -1)) != IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported protocol lock schema: {lock.get('schema_version')}")
    identities = lock.get("dataset_content_identities")
    if identities is not None and not isinstance(identities, dict):
        raise ValueError("Protocol dataset_content_identities must be an object")
    for dataset_key, identity in (identities or {}).items():
        if not isinstance(identity, dict):
            raise ValueError(f"Invalid dataset identity for {dataset_key}")
        for field in ("graph_sha256", "ground_truth_cover_sha256"):
            digest = identity.get(field)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(
                    f"Protocol dataset identity {dataset_key}.{field} must be "
                    "exactly 64 lowercase hexadecimal characters"
                )
    return lock


def _protocol_lock_snapshot(path: Path) -> tuple[bytes, dict[str, Any], str]:
    encoded = path.read_bytes()
    return encoded, _parse_protocol_lock(encoded), _sha256_bytes(encoded)


def load_protocol_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return _protocol_lock_snapshot(path)[1]


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _distribution_tree_sha256(name: str) -> str | None:
    """Hash every installed file declared by a distribution's wheel metadata."""
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    digest = hashlib.sha256()
    files = distribution.files or []
    try:
        for item in sorted(files, key=lambda value: str(value)):
            path = Path(distribution.locate_file(item))
            if not path.is_file():
                continue
            relative = str(item).replace("\\", "/")
            digest.update(relative.encode())
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    except OSError:
        return None
    return digest.hexdigest()


def _imported_module_identity(
    distribution_name: str, module_name: str
) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if source is None:
            raise OSError("imported module has no file")
        source_path = Path(source).resolve()
        distribution_root = Path(distribution.locate_file("")).resolve()
        try:
            relative_path = source_path.relative_to(distribution_root).as_posix()
        except ValueError:
            relative_path = None
        belongs = any(
            Path(distribution.locate_file(item)).resolve() == source_path
            for item in (distribution.files or [])
        )
        source_sha256 = _sha256_file(source_path)
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError):
        relative_path = None
        belongs = False
        source_sha256 = None
    return {
        "module": module_name,
        "path": relative_path,
        "sha256": source_sha256,
        "belongs_to_distribution": belongs,
    }


def _igraph_version() -> str | None:
    try:
        import igraph
    except ImportError:
        return None
    value = getattr(igraph, "__version__", None)
    return str(value) if value is not None else None


def _resolve_path(repo_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _uv_lock_package_sha256(encoded: bytes | None, distribution: str) -> str | None:
    """Hash the canonical uv.lock package entry for a distribution.

    The digest is deliberately independent of the host platform: the package
    entry contains the version, dependency set, sdist, and all locked wheels.
    The selected wheel's hash is still preserved inside that canonical entry.
    """
    if encoded is None:
        return None
    try:
        packages = tomllib.loads(encoded.decode("utf-8")).get("package", [])
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    wanted = _normalise_distribution_name(distribution)
    for package in packages:
        if _normalise_distribution_name(package.get("name", "")) == wanted:
            return _canonical_hash(package)
    return None


@cache
def _current_experiment_identity_from_snapshot(
    lock_path_text: str, encoded_lock: bytes
) -> dict[str, Any]:
    """Build identity from one immutable lock snapshot."""
    lock_path = Path(lock_path_text)
    lock = json.loads(encoded_lock)
    if not isinstance(lock, dict):
        raise ValueError("Identity lock must be a JSON object")
    lock_sha256 = _sha256_bytes(encoded_lock)
    strict_paper_lock = (
        int(lock.get("schema_version", -1)) == IDENTITY_SCHEMA_VERSION
        and "run_protocol_version" in lock
    )
    external_lock = lock.get("external_dependencies")
    scientific_lock = lock.get("scientific_dependencies")
    external_fields = {
        "distribution", "version", "package_lock_sha256",
        "distribution_tree_sha256", "implementation_module",
        "implementation_attribute", "implementation_path",
        "implementation_sha256",
    }
    scientific_fields = {
        "distribution", "version", "package_lock_sha256",
        "distribution_tree_sha256", "module", "module_path", "module_sha256",
    }
    external_lock_complete = (
        (not strict_paper_lock and external_lock in (None, {}))
        or (
            isinstance(external_lock, dict)
            and (
                set(external_lock) == REQUIRED_EXTERNAL_DEPENDENCIES
                if strict_paper_lock
                else True
            )
            and all(
                isinstance(value, dict) and external_fields.issubset(value)
                for value in external_lock.values()
            )
        )
    )
    scientific_lock_complete = (
        isinstance(scientific_lock, dict)
        and (
            set(scientific_lock) == REQUIRED_SCIENTIFIC_DEPENDENCIES
            if strict_paper_lock
            else bool(scientific_lock)
        )
        and all(
            isinstance(value, dict) and scientific_fields.issubset(value)
            for value in scientific_lock.values()
        )
    )
    repo_root = _repo_root_for_lock(lock_path)
    tracked_lock = lock.get("tracked_files")
    tracked_lock = tracked_lock if isinstance(tracked_lock, dict) else {}
    files = {
        relative: _sha256_file(repo_root / relative)
        for relative in tracked_lock
    }
    dependency = lock.get("lucas_igraph", {})
    lucas_fields = {
        "distribution", "version", "igraph_version", "uv_lock_path",
        "uv_lock_sha256", "package_lock_sha256", "distribution_tree_sha256",
        "module", "module_path", "module_sha256",
    }
    lucas_lock_complete = isinstance(dependency, dict) and lucas_fields.issubset(
        dependency
    )
    distribution = str(dependency.get("distribution", "lucas-igraph"))
    uv_lock_path = _resolve_path(repo_root, dependency.get("uv_lock_path", "uv.lock"))
    expected_version = dependency.get("version")
    expected_igraph_version = dependency.get("igraph_version")
    expected_uv_lock_sha256 = dependency.get("uv_lock_sha256")
    expected_package_lock_sha256 = dependency.get("package_lock_sha256")
    actual_version = _distribution_version(distribution)
    actual_igraph_version = _igraph_version()
    try:
        encoded_uv_lock = uv_lock_path.read_bytes() if uv_lock_path else None
    except OSError:
        encoded_uv_lock = None
    actual_uv_lock_sha256 = (
        _sha256_bytes(encoded_uv_lock) if encoded_uv_lock is not None else None
    )
    actual_package_lock_sha256 = _uv_lock_package_sha256(
        encoded_uv_lock, distribution
    )
    actual_distribution_tree_sha256 = _distribution_tree_sha256(distribution)
    lucas_expected_module = {
        "module": dependency.get("module"),
        "path": dependency.get("module_path"),
        "sha256": dependency.get("module_sha256"),
    }
    lucas_actual_module_identity = _imported_module_identity(
        distribution, str(dependency.get("module", "igraph._igraph"))
    )
    lucas_actual_module = {
        key: lucas_actual_module_identity.get(key) for key in lucas_expected_module
    }
    module_matches_lock = (
        lucas_actual_module == lucas_expected_module
        and lucas_actual_module_identity.get("belongs_to_distribution") is True
    )
    external_dependencies: dict[str, dict[str, Any]] = {}
    for method, external in (
        external_lock if isinstance(external_lock, dict) else {}
    ).items():
        external_distribution = str(external["distribution"])
        runtime_identity = method_dependency_identity(str(method)) or {}
        external_actual_version = runtime_identity.get("version")
        external_actual_package_sha256 = _uv_lock_package_sha256(
            encoded_uv_lock, external_distribution
        )
        external_actual_tree_sha256 = _distribution_tree_sha256(
            external_distribution
        )
        expected_implementation = {
            key: external.get(key)
            for key in (
                "implementation_module",
                "implementation_attribute",
                "implementation_path",
                "implementation_sha256",
            )
        }
        actual_implementation = {
            key: runtime_identity.get(key)
            for key in expected_implementation
        }
        implementation_matches_lock = (
            actual_implementation == expected_implementation
            and runtime_identity.get("implementation_belongs_to_distribution")
            is True
        )
        external_dependencies[str(method)] = {
            "distribution": external_distribution,
            "expected_version": external.get("version"),
            "actual_version": external_actual_version,
            "version_matches_lock": (
                external_actual_version == external.get("version")
            ),
            "expected_package_lock_sha256": external.get(
                "package_lock_sha256"
            ),
            "actual_package_lock_sha256": external_actual_package_sha256,
            "package_lock_matches_lock": (
                external_actual_package_sha256
                == external.get("package_lock_sha256")
            ),
            "expected_distribution_tree_sha256": external.get(
                "distribution_tree_sha256"
            ),
            "actual_distribution_tree_sha256": external_actual_tree_sha256,
            "distribution_tree_matches_lock": (
                external_actual_tree_sha256
                == external.get("distribution_tree_sha256")
            ),
            "expected_implementation": expected_implementation,
            "actual_implementation": actual_implementation,
            "implementation_belongs_to_distribution": runtime_identity.get(
                "implementation_belongs_to_distribution"
            ),
            "implementation_matches_lock": implementation_matches_lock,
        }

    scientific_dependencies: dict[str, dict[str, Any]] = {}
    for name, scientific in (
        scientific_lock if isinstance(scientific_lock, dict) else {}
    ).items():
        scientific_distribution = str(scientific["distribution"])
        module_name = str(scientific["module"])
        actual_module = _imported_module_identity(
            scientific_distribution, module_name
        )
        expected_module = {
            "module": module_name,
            "path": scientific.get("module_path"),
            "sha256": scientific.get("module_sha256"),
        }
        actual_module_public = {
            key: actual_module.get(key) for key in expected_module
        }
        actual_scientific_version = _distribution_version(
            scientific_distribution
        )
        actual_scientific_package_sha256 = _uv_lock_package_sha256(
            encoded_uv_lock, scientific_distribution
        )
        actual_scientific_tree_sha256 = _distribution_tree_sha256(
            scientific_distribution
        )
        scientific_dependencies[str(name)] = {
            "distribution": scientific_distribution,
            "expected_version": scientific.get("version"),
            "actual_version": actual_scientific_version,
            "version_matches_lock": (
                actual_scientific_version == scientific.get("version")
            ),
            "expected_package_lock_sha256": scientific.get(
                "package_lock_sha256"
            ),
            "actual_package_lock_sha256": actual_scientific_package_sha256,
            "package_lock_matches_lock": (
                actual_scientific_package_sha256
                == scientific.get("package_lock_sha256")
            ),
            "expected_distribution_tree_sha256": scientific.get(
                "distribution_tree_sha256"
            ),
            "actual_distribution_tree_sha256": actual_scientific_tree_sha256,
            "distribution_tree_matches_lock": (
                actual_scientific_tree_sha256
                == scientific.get("distribution_tree_sha256")
            ),
            "expected_module": expected_module,
            "actual_module": actual_module_public,
            "module_belongs_to_distribution": actual_module.get(
                "belongs_to_distribution"
            ),
            "module_matches_lock": (
                actual_module_public == expected_module
                and actual_module.get("belongs_to_distribution") is True
            ),
        }

    source_path = _resolve_path(repo_root, dependency.get("source_path"))
    actual_source_revision = _git_revision(source_path) if source_path else None
    released_native_sha = dependency.get("released_native_sha")
    source_revision_matches_release = (
        None
        if released_native_sha is None
        else actual_source_revision is None or actual_source_revision == released_native_sha
    )
    version_matches_lock = expected_version is None or actual_version == expected_version
    igraph_version_matches_lock = (
        expected_igraph_version is None or actual_igraph_version == expected_igraph_version
    )
    uv_lock_matches_lock = (
        expected_uv_lock_sha256 is None or actual_uv_lock_sha256 == expected_uv_lock_sha256
    )
    package_lock_matches_lock = (
        expected_package_lock_sha256 is None
        or actual_package_lock_sha256 == expected_package_lock_sha256
    )
    package_identity_matches_lock = all(
        (
            lucas_lock_complete,
            version_matches_lock,
            igraph_version_matches_lock,
            uv_lock_matches_lock,
            package_lock_matches_lock,
            actual_distribution_tree_sha256
            == dependency.get("distribution_tree_sha256"),
            module_matches_lock,
        )
    )
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "protocol_lock_sha256": lock_sha256,
        "protocol_name": lock.get("protocol_name"),
        "run_protocol_version": lock.get("run_protocol_version"),
        "analysis_graph_policy": lock.get("analysis_graph_policy"),
        "tracked_files": files,
        "tracked_files_match_lock": all(
            digest == tracked_lock.get(relative)
            for relative, digest in files.items()
        ),
        "lucas_igraph": {
            "distribution": distribution,
            "expected_version": expected_version,
            "actual_version": actual_version,
            "version_matches_lock": version_matches_lock,
            "expected_igraph_version": expected_igraph_version,
            "actual_igraph_version": actual_igraph_version,
            "igraph_version_matches_lock": igraph_version_matches_lock,
            "uv_lock_path": _portable_path(uv_lock_path) if uv_lock_path else None,
            "expected_uv_lock_sha256": expected_uv_lock_sha256,
            "actual_uv_lock_sha256": actual_uv_lock_sha256,
            "uv_lock_matches_lock": uv_lock_matches_lock,
            "expected_package_lock_sha256": expected_package_lock_sha256,
            "actual_package_lock_sha256": actual_package_lock_sha256,
            "package_lock_matches_lock": package_lock_matches_lock,
            "lock_complete": lucas_lock_complete,
            "expected_distribution_tree_sha256": dependency.get(
                "distribution_tree_sha256"
            ),
            "actual_distribution_tree_sha256": actual_distribution_tree_sha256,
            "distribution_tree_matches_lock": (
                actual_distribution_tree_sha256
                == dependency.get("distribution_tree_sha256")
            ),
            "expected_module": lucas_expected_module,
            "actual_module": lucas_actual_module,
            "module_belongs_to_distribution": lucas_actual_module_identity.get(
                "belongs_to_distribution"
            ),
            "module_matches_lock": module_matches_lock,
            "package_identity_matches_lock": package_identity_matches_lock,
            "released_native_sha": released_native_sha,
            "source_path": str(dependency.get("source_path")) if dependency.get("source_path") else None,
            "actual_source_revision": actual_source_revision,
            "source_revision_matches_release": source_revision_matches_release,
        },
        "external_dependencies": external_dependencies,
        "external_dependency_lock_complete": external_lock_complete,
        "external_dependencies_match_lock": external_lock_complete and all(
            item["version_matches_lock"]
            and item["package_lock_matches_lock"]
            and item["distribution_tree_matches_lock"]
            and item["implementation_matches_lock"]
            for item in external_dependencies.values()
        ),
        "scientific_dependencies": scientific_dependencies,
        "scientific_dependency_lock_complete": scientific_lock_complete,
        "scientific_dependencies_match_lock": scientific_lock_complete and all(
            item["version_matches_lock"]
            and item["package_lock_matches_lock"]
            and item["distribution_tree_matches_lock"]
            and item["module_matches_lock"]
            for item in scientific_dependencies.values()
        ),
    }


def current_experiment_identity(
    lock_path: Path = LOCK_PATH, *, lock_bytes: bytes | None = None
) -> dict[str, Any]:
    """Return identity bound to exactly one protocol-lock byte snapshot."""
    resolved = lock_path.expanduser().resolve()
    encoded = resolved.read_bytes() if lock_bytes is None else bytes(lock_bytes)
    return _current_experiment_identity_from_snapshot(str(resolved), encoded)


def identity_rejection_reasons(existing: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(existing, dict):
        return ["missing_experiment_identity"]
    reasons: list[str] = []
    if existing.get("schema_version") != expected.get("schema_version"):
        reasons.append("experiment_identity.schema_version_mismatch")
    for key in (
        "protocol_lock_sha256", "protocol_name", "run_protocol_version",
        "analysis_graph_policy",
    ):
        if existing.get(key) != expected.get(key):
            reasons.append(f"experiment_identity.{key}_mismatch")
    old_dep = existing.get("lucas_igraph")
    new_dep = expected.get("lucas_igraph")
    if not isinstance(old_dep, dict):
        reasons.append("missing_lucas_igraph_identity")
    elif not isinstance(new_dep, dict):
        reasons.append("missing_expected_lucas_igraph_identity")
    else:
        if old_dep.get("distribution") != new_dep.get("distribution"):
            reasons.append("lucas_igraph_distribution_mismatch")
        comparisons = (
            ("actual_version", "expected_version", "lucas_igraph_version_mismatch"),
            ("actual_igraph_version", "expected_igraph_version", "igraph_version_mismatch"),
            ("actual_uv_lock_sha256", "expected_uv_lock_sha256", "lucas_igraph_lockfile_mismatch"),
            (
                "actual_package_lock_sha256",
                "expected_package_lock_sha256",
                "lucas_igraph_package_lock_mismatch",
            ),
            (
                "actual_distribution_tree_sha256",
                "expected_distribution_tree_sha256",
                "lucas_igraph_distribution_tree_mismatch",
            ),
        )
        package_identity_fields_present = False
        for actual_key, expected_key, reason in comparisons:
            expected_value = new_dep.get(expected_key)
            if expected_value is None:
                continue
            package_identity_fields_present = True
            if old_dep.get(actual_key) != expected_value:
                reasons.append(reason)
        if not package_identity_fields_present:
            reasons.append("missing_lucas_igraph_package_identity")
        if old_dep.get("actual_module") != new_dep.get("expected_module"):
            reasons.append("lucas_igraph_imported_module_mismatch")
        if old_dep.get("module_belongs_to_distribution") is not True:
            reasons.append("lucas_igraph_imported_module_outside_distribution")
    if existing.get("tracked_files") != expected.get("tracked_files"):
        reasons.append("tracked_code_or_config_hash_mismatch")
    if existing.get("external_dependencies") != expected.get(
        "external_dependencies"
    ):
        reasons.append("external_dependency_identity_mismatch")
    if existing.get("scientific_dependencies") != expected.get(
        "scientific_dependencies"
    ):
        reasons.append("scientific_dependency_identity_mismatch")
    return reasons


def dataset_metadata_identity(report: Any) -> dict[str, Any] | None:
    """Hash exact graph/GT content, excluding host paths and cache provenance."""
    if not isinstance(report, dict):
        return None
    identity = report.get("content_identity")
    if not isinstance(identity, dict):
        return None
    return {
        "schema_version": 1,
        "sha256": _canonical_hash(identity),
        "content_identity": identity,
    }


def _parse_ints(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part[1:]:
            start, stop = (int(item) for item in part.split("-", 1))
            result.extend(range(start, stop + (1 if stop >= start else -1), 1 if stop >= start else -1))
        elif part:
            result.append(int(part))
    return list(dict.fromkeys(result))


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record.get("dataset")), str(record.get("cover")),
        str(record.get("method")), int(record.get("seed", -1)),
    )


def _resolution_specs(value: Any) -> list[str | float]:
    result: list[str | float] = []
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "auto":
            result.append("auto")
        elif part.count(":") == 2:
            start_text, stop_text, count_text = part.split(":")
            start, stop, count = float(start_text), float(stop_text), int(count_text)
            if count <= 0:
                raise ValueError("resolution range count must be positive")
            result.extend(
                [start]
                if count == 1
                else [start + (stop - start) * i / (count - 1) for i in range(count)]
            )
        else:
            result.append(float(part))
    if not result:
        raise ValueError("overlapping_paper.resolutions must not be empty")
    return list(dict.fromkeys(result))


def _condition_specs_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = _parse_ints(str(raw["seeds"]))
    return [
        {"job": job["name"], "dataset": job["dataset"], "cover": job["cover"],
         "method": method, "seed": seed, "resolution_spec": resolution}
        for job in raw["jobs"] for method in raw["methods"] for seed in seeds
        for resolution in _resolution_specs(raw["resolutions"])
    ]


def _condition_specs(config_path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for callers outside strict single-read reconcile."""
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))["overlapping_paper"]
    return _condition_specs_from_raw(raw)


def _expected_resolution(
    method: str, resolution_spec: str | float, content: dict[str, Any]
) -> float:
    n, m = int(content["n"]), int(content["m"])
    density = 0.0 if n < 2 else 2.0 * m / (n * (n - 1))
    requested = density if resolution_spec == "auto" else float(resolution_spec)
    multiplier = {
        "hedonic_multiphase": 1.0,
        "hedonic_multiphase_x10": 10.0,
        "hedonic_multiphase_x100": 100.0,
    }.get(method)
    return requested if multiplier is None else min(density * multiplier, 1.0)


def _expected_timeout(raw: dict[str, Any], dataset: str, method: str) -> float:
    limits = [float(raw["timeout_per_run"])]
    for mapping, key in (
        (raw.get("timeout_by_method") or {}, method),
        (raw.get("timeout_by_dataset") or {}, dataset),
    ):
        if key in mapping:
            limits.append(float(mapping[key]))
    return min(limits)


def _seeded_initialization_identity(n: int, k: int, seed: int) -> dict[str, Any]:
    requested = max(1, int(k))
    if requested == 1:
        membership = [0] * int(n)
    else:
        raw = sample_uniform_ints(int(n), requested - 1, int(seed)).tolist()
        unique = sorted(set(raw))
        remap = {old: new for new, old in enumerate(unique)}
        membership = [remap[int(label)] for label in raw]
    return {
        "kind": "seeded_random_disjoint",
        "requested_community_count": requested,
        "realized_community_count": len(set(membership)),
        "n_vertices": len(membership),
        "membership_sha256": _sha256_bytes(
            json.dumps(membership, separators=(",", ":")).encode()
        ),
        "seed": int(seed),
        "shared_across_hedonic_methods": True,
    }


def _float_matches(value: Any, expected: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _metric_value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(value, bool):
        return value is expected
    if isinstance(expected, (int, float)):
        if not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value)) and abs(
                float(value) - float(expected)
            ) <= 1e-12 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return value == expected


def _artifact_root(record_path: Path) -> Path | None:
    current = record_path.parent
    while current != current.parent:
        if current.name == "runs":
            return current.parent
        current = current.parent
    return None


def _read_bound_artifact(
    record_path: Path,
    record: dict[str, Any],
    *,
    prefix: str,
) -> tuple[Any | None, list[str], dict[str, str] | None]:
    reasons: list[str] = []
    reference = record.get(f"{prefix}_artifact")
    expected_file_hash = record.get(f"{prefix}_artifact_sha256")
    expected_content_hash = record.get(f"{prefix}_sha256")
    if not all(isinstance(value, str) and value for value in (
        reference, expected_file_hash, expected_content_hash
    )):
        return None, [f"missing_{prefix}_artifact_identity"], None
    relative = Path(str(reference))
    root = _artifact_root(record_path)
    if root is None or relative.is_absolute() or ".." in relative.parts:
        return None, [f"nonportable_{prefix}_artifact_path"], None
    path = root / relative
    try:
        encoded = path.read_bytes()
    except OSError:
        return None, [f"missing_{prefix}_artifact"], None
    actual_file_hash = _sha256_bytes(encoded)
    if actual_file_hash != expected_file_hash:
        reasons.append(f"{prefix}_artifact_sha256_mismatch")
    try:
        value = json.loads(gzip.decompress(encoded).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None, reasons + [f"invalid_{prefix}_artifact"], None
    if prefix in {"final_cover", "ground_truth_cover"}:
        if not isinstance(value, list):
            reasons.append(f"invalid_{prefix}_artifact")
        else:
            try:
                canonical, _ = canonicalize_cover(value, minimum_size=1)
            except (TypeError, ValueError):
                canonical = []
                reasons.append(f"invalid_{prefix}_artifact")
            if canonical != value:
                reasons.append(f"{prefix}_not_canonical")
            if cover_sha256(canonical) != expected_content_hash:
                reasons.append(f"{prefix}_sha256_mismatch")
            value = canonical
    elif prefix == "analysis_graph":
        try:
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != 1
                or value.get("analysis_graph_policy") != ANALYSIS_GRAPH_POLICY
                or isinstance(value.get("n"), bool)
                or not isinstance(value.get("n"), int)
                or isinstance(value.get("m"), bool)
                or not isinstance(value.get("m"), int)
                or not isinstance(value.get("directed"), bool)
                or not isinstance(value.get("edges"), list)
            ):
                raise ValueError("invalid analysis graph payload")
            edges = []
            for edge in value["edges"]:
                if (
                    not isinstance(edge, list)
                    or len(edge) != 2
                    or any(isinstance(vertex, bool) or not isinstance(vertex, int) for vertex in edge)
                ):
                    raise ValueError("invalid analysis graph edge")
                edges.append((edge[0], edge[1]))
            reconstructed = ig.Graph(
                n=value["n"], edges=edges, directed=value["directed"]
            )
            if (
                reconstructed.ecount() != value["m"]
                or not reconstructed.is_simple()
                or graph_sha256(reconstructed) != expected_content_hash
            ):
                raise ValueError("analysis graph identity mismatch")
            value = reconstructed
        except (TypeError, ValueError, ig.InternalError):
            reasons.append("analysis_graph_content_mismatch")
    else:
        try:
            normalized = [[int(label) for label in labels] for labels in value]
        except (TypeError, ValueError):
            normalized = None
        digest = (
            _sha256_bytes(
                json.dumps(normalized, separators=(",", ":")).encode()
            )
            if normalized is not None
            else None
        )
        if digest != expected_content_hash:
            reasons.append(f"{prefix}_sha256_mismatch")
        if (
            prefix in {"raw_membership", "pre_cleanup_membership"}
            and record.get("raw_membership_hash") != expected_content_hash
        ):
            reasons.append("raw_membership_compatibility_hash_mismatch")
    return value, reasons, {
        "path": relative.as_posix(),
        "artifact_sha256": actual_file_hash,
        "content_sha256": str(expected_content_hash),
    }


def _metrics_rejection_reasons(
    record: dict[str, Any], *, omega: bool, omega_sample_size: int
) -> list[str]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return ["missing_metrics"]
    reasons: list[str] = []
    if _metrics_hash(metrics) != record.get("metrics_sha256"):
        reasons.append("metrics_sha256_mismatch")
    required = (
        "symmetric_best_match_f1",
        "matching_f1",
        "node_micro_f1",
        "size_weighted_community_f1",
        "runtime_seconds",
        "predicted_overlapping_node_fraction",
        "coverage_rate",
        "inclusion_rate",
        "cpm_overlapping_quality",
    )
    if omega:
        required += ("omega",)
    for key in required:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"metrics.{key}_missing_or_non_numeric")
        elif not math.isfinite(float(value)):
            reasons.append(f"metrics.{key}_nonfinite")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                reasons.append(f"metrics.{key}_nonfinite")
    unit_interval_fields = (
        "f1", "symmetric_best_match_f1", "jaccard",
        "symmetric_best_match_jaccard", "precision", "recall",
        "size_weighted_community_f1", "matching_precision",
        "matching_recall", "matching_f1", "matching_mean_weight",
        "node_micro_precision", "node_micro_recall", "node_micro_f1",
        "node_macro_f1", "inclusion_rate", "coverage_rate",
        "overlapping_rate", "distribution_rate",
        "predicted_singleton_fraction", "gt_singleton_fraction",
        "predicted_vertices_covered_fraction",
        "gt_vertices_covered_fraction",
        "predicted_overlapping_node_fraction",
        "gt_overlapping_node_fraction",
        "singleton_fraction", "vertices_covered_fraction",
    )
    for key in unit_interval_fields:
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            reasons.append(f"metrics.{key}_outside_unit_interval")
    runtime = metrics.get("runtime_seconds")
    if isinstance(runtime, (int, float)) and not isinstance(runtime, bool):
        if float(runtime) < 0.0:
            reasons.append("metrics.runtime_seconds_negative")
    if metrics.get("singleton_mode") != "all":
        reasons.append("metrics.singleton_mode_mismatch")
    if metrics.get("matching_weight") != "f1":
        reasons.append("metrics.matching_weight_mismatch")
    for alias, source in {
        "singleton_fraction": "predicted_singleton_fraction",
        "singleton_count": "predicted_singleton_count",
        "vertices_covered_fraction": "predicted_vertices_covered_fraction",
        "average_community_size": "predicted_average_community_size",
        "median_community_size": "predicted_median_community_size",
        "average_memberships_per_vertex": "predicted_average_memberships_per_vertex",
    }.items():
        if metrics.get(alias) != metrics.get(source):
            reasons.append(f"metrics.{alias}_alias_mismatch")
    integer_nonnegative_fields = (
        "n_predicted_comms", "n_gt_comms",
        "predicted_community_count", "gt_community_count",
        "predicted_singleton_count", "gt_singleton_count", "singleton_count",
        "predicted_max_community_size", "gt_max_community_size",
        "predicted_max_memberships_per_vertex", "gt_max_memberships_per_vertex",
        "predicted_overlapping_node_count", "gt_overlapping_node_count",
        "n_matched_communities", "n_unmatched_predicted_comms",
        "n_unmatched_gt_comms", "overlap_pair_count", "overlap_pair_events",
    )
    if omega:
        integer_nonnegative_fields += ("omega_sample_size", "omega_seed")
    for key in integer_nonnegative_fields:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reasons.append(f"metrics.{key}_invalid_nonnegative_integer")
    nonnegative_numeric_fields = (
        "community_count_ratio",
        "predicted_average_community_size", "gt_average_community_size",
        "predicted_median_community_size", "gt_median_community_size",
        "predicted_p95_community_size", "gt_p95_community_size",
        "average_community_size", "median_community_size",
        "predicted_average_memberships_per_vertex",
        "gt_average_memberships_per_vertex",
        "predicted_average_memberships_per_covered_vertex",
        "gt_average_memberships_per_covered_vertex",
        "average_memberships_per_vertex",
    )
    for key in nonnegative_numeric_fields:
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            reasons.append(f"metrics.{key}_invalid_nonnegative_numeric")
    if not isinstance(metrics.get("overlap_pair_events_truncated"), bool):
        reasons.append("metrics.overlap_pair_events_truncated_not_boolean")
    predicted_count = metrics.get("predicted_community_count")
    ground_truth_count = metrics.get("gt_community_count")
    if isinstance(predicted_count, int) and not isinstance(predicted_count, bool):
        if metrics.get("n_predicted_comms") != predicted_count:
            reasons.append("metrics.n_predicted_comms_alias_mismatch")
        predicted_singletons = metrics.get("predicted_singleton_count")
        if (
            isinstance(predicted_singletons, int)
            and not isinstance(predicted_singletons, bool)
            and predicted_singletons > predicted_count
        ):
            reasons.append("metrics.predicted_singleton_count_exceeds_communities")
        predicted_max_memberships = metrics.get(
            "predicted_max_memberships_per_vertex"
        )
        if (
            isinstance(predicted_max_memberships, int)
            and not isinstance(predicted_max_memberships, bool)
            and predicted_max_memberships > predicted_count
        ):
            reasons.append("metrics.predicted_max_memberships_exceeds_communities")
        for key in (
            "predicted_average_memberships_per_vertex",
            "predicted_average_memberships_per_covered_vertex",
            "average_memberships_per_vertex",
        ):
            value = metrics.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > predicted_count
            ):
                reasons.append(f"metrics.{key}_exceeds_community_count")
    if isinstance(ground_truth_count, int) and not isinstance(ground_truth_count, bool):
        if metrics.get("n_gt_comms") != ground_truth_count:
            reasons.append("metrics.n_gt_comms_alias_mismatch")
        gt_singletons = metrics.get("gt_singleton_count")
        if (
            isinstance(gt_singletons, int)
            and not isinstance(gt_singletons, bool)
            and gt_singletons > ground_truth_count
        ):
            reasons.append("metrics.gt_singleton_count_exceeds_communities")
        gt_max_memberships = metrics.get("gt_max_memberships_per_vertex")
        if (
            isinstance(gt_max_memberships, int)
            and not isinstance(gt_max_memberships, bool)
            and gt_max_memberships > ground_truth_count
        ):
            reasons.append("metrics.gt_max_memberships_exceeds_communities")
        for key in (
            "gt_average_memberships_per_vertex",
            "gt_average_memberships_per_covered_vertex",
        ):
            value = metrics.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > ground_truth_count
            ):
                reasons.append(f"metrics.{key}_exceeds_community_count")
    if (
        isinstance(predicted_count, int)
        and not isinstance(predicted_count, bool)
        and isinstance(ground_truth_count, int)
        and not isinstance(ground_truth_count, bool)
        and ground_truth_count > 0
        and not _float_matches(
            metrics.get("community_count_ratio"),
            predicted_count / ground_truth_count,
        )
    ):
        reasons.append("metrics.community_count_ratio_mismatch")
    dataset_report = record.get("dataset_report")
    content = (
        dataset_report.get("content_identity") or {}
        if isinstance(dataset_report, dict)
        else {}
    )
    n_vertices = content.get("n")
    if isinstance(n_vertices, int) and not isinstance(n_vertices, bool) and n_vertices >= 0:
        for key in (
            "predicted_max_community_size", "gt_max_community_size",
            "predicted_overlapping_node_count", "gt_overlapping_node_count",
            "predicted_average_community_size", "gt_average_community_size",
            "predicted_median_community_size", "gt_median_community_size",
            "predicted_p95_community_size", "gt_p95_community_size",
            "average_community_size", "median_community_size",
        ):
            value = metrics.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > n_vertices
            ):
                reasons.append(f"metrics.{key}_exceeds_vertex_count")
    pair_count = metrics.get("overlap_pair_count")
    pair_events = metrics.get("overlap_pair_events")
    if (
        isinstance(pair_count, int)
        and not isinstance(pair_count, bool)
        and isinstance(pair_events, int)
        and not isinstance(pair_events, bool)
        and pair_events < pair_count
    ):
        reasons.append("metrics.overlap_pair_events_below_pair_count")
    if (
        isinstance(pair_count, int)
        and not isinstance(pair_count, bool)
        and isinstance(predicted_count, int)
        and not isinstance(predicted_count, bool)
        and pair_count > predicted_count * max(0, predicted_count - 1) // 2
    ):
        reasons.append("metrics.overlap_pair_count_exceeds_community_pairs")
    if str(record.get("method", "")).startswith("hedonic_"):
        predicted_max_memberships = metrics.get(
            "predicted_max_memberships_per_vertex"
        )
        detector_cap = record.get("max_memberships")
        if (
            isinstance(predicted_max_memberships, int)
            and not isinstance(predicted_max_memberships, bool)
            and isinstance(detector_cap, int)
            and not isinstance(detector_cap, bool)
            and predicted_max_memberships > detector_cap
        ):
            reasons.append("metrics.predicted_memberships_exceed_detector_cap")
    if metrics.get("cpm_overlapping_quality_status") != "computed":
        reasons.append("metrics.cpm_overlapping_quality_status_mismatch")
    if omega:
        if metrics.get("omega_method") != "sampled_pairwise":
            reasons.append("metrics.omega_method_mismatch")
        if metrics.get("omega_sample_size") != omega_sample_size:
            reasons.append("metrics.omega_sample_size_mismatch")
        if metrics.get("omega_seed") != record.get("seed"):
            reasons.append("metrics.omega_seed_mismatch")
        omega_value = metrics.get("omega")
        if (
            isinstance(omega_value, (int, float))
            and not isinstance(omega_value, bool)
            and not -1.0 <= float(omega_value) <= 1.0
        ):
            reasons.append("metrics.omega_outside_valid_range")
    if not _float_matches(record.get("runtime_seconds"), metrics.get("runtime_seconds", math.nan)):
        reasons.append("runtime_seconds_metric_mismatch")
    return reasons


def _canonical_projection_from_memberships(
    memberships: Any, n_vertices: int
) -> tuple[list[list[int]], dict[str, Any]] | None:
    if not isinstance(memberships, list) or len(memberships) != n_vertices:
        return None
    communities: dict[int, list[int]] = {}
    try:
        for vertex, labels in enumerate(memberships):
            normalized = [int(label) for label in labels]
            if not normalized or len(normalized) != len(set(normalized)):
                return None
            for label in normalized:
                if label < 0:
                    return None
                communities.setdefault(label, []).append(vertex)
        labeled_bodies = list(communities.values())
        canonical, validation = canonicalize_cover(
            labeled_bodies,
            n_vertices=n_vertices,
            minimum_size=1,
        )
        duplicate_bodies = int(validation["duplicate_communities_removed"])
        return canonical, {
            "strategy": "unique_community_body_projection_v1",
            "exact_labeled_community_count": len(labeled_bodies),
            "canonical_unique_body_count": len(canonical),
            "duplicate_community_bodies_removed": duplicate_bodies,
            "projection_changed": duplicate_bodies > 0,
            "equilibrium_target": "exact_labeled_final_memberships",
            "canonical_scoring_projection_certified_as_equilibrium": False,
        }
    except (TypeError, ValueError):
        return None


def reconcile(
    artifact_dir: Path,
    config_path: Path,
    lock_path: Path = LOCK_PATH,
    *,
    plan_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Strictly reconcile every locked condition without modifying artifacts."""
    lock_path = lock_path.expanduser().resolve()
    encoded_lock, lock, lock_sha256 = _protocol_lock_snapshot(lock_path)
    identity = current_experiment_identity(lock_path, lock_bytes=encoded_lock)
    artifact_dir = artifact_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    plan_path = artifact_dir / "orchestration" / "plan.json"
    try:
        encoded_plan = plan_path.read_bytes() if plan_bytes is None else bytes(plan_bytes)
        plan = json.loads(encoded_plan)
        if not isinstance(plan, dict):
            raise TypeError("plan must be a JSON object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        encoded_plan = b""
        plan = {}
    plan_sha256 = _sha256_bytes(encoded_plan) if encoded_plan else None
    encoded_config = config_path.read_bytes()
    config_sha256 = _sha256_bytes(encoded_config)
    raw_config = tomllib.loads(encoded_config.decode("utf-8"))["overlapping_paper"]
    plan_jobs = {str(job.get("name")): job for job in plan.get("jobs", [])}
    records: dict[
        tuple[str, str, str, int],
        list[tuple[Path, dict[str, Any], str]],
    ] = {}
    for path in sorted((artifact_dir / "shards").glob("*/runs/**/*.json")):
        try:
            encoded = path.read_bytes()
            record = json.loads(encoded)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        records.setdefault(_record_key(record), []).append(
            (path, record, _sha256_bytes(encoded))
        )

    specs = _condition_specs_from_raw(raw_config)
    if len(specs) != int(lock["expected_conditions"]):
        raise ValueError(
            f"Locked protocol expects {lock['expected_conditions']} conditions; config defines {len(specs)}"
        )
    global_reasons: list[str] = []
    if identity.get("tracked_files_match_lock") is not True:
        global_reasons.append("current_tracked_files_mismatch_lock")
    if (
        (identity.get("lucas_igraph") or {}).get(
            "package_identity_matches_lock"
        )
        is not True
    ):
        global_reasons.append("current_lucas_igraph_mismatch_lock")
    if identity.get("external_dependencies_match_lock") is not True:
        global_reasons.append("current_external_dependencies_mismatch_lock")
    if identity.get("scientific_dependencies_match_lock") is not True:
        global_reasons.append("current_scientific_dependencies_mismatch_lock")
    locked_config = lock.get("protocol_config")
    if isinstance(locked_config, dict):
        repo_root = _repo_root_for_lock(lock_path)
        expected_config_path = _resolve_path(repo_root, locked_config.get("path"))
        if expected_config_path is None or config_path != expected_config_path.resolve():
            global_reasons.append("protocol_config_path_mismatch")
        if config_sha256 != locked_config.get("sha256"):
            global_reasons.append("protocol_config_sha256_mismatch")
    else:
        global_reasons.append("missing_locked_protocol_config")
    if plan.get("config_sha256") != config_sha256:
        global_reasons.append("plan_config_sha256_mismatch")
    if plan.get("profile") != raw_config.get("profile", "full"):
        global_reasons.append("plan_profile_mismatch")
    if int(plan.get("max_nodes", -1)) != int(raw_config.get("max_nodes", 0)):
        global_reasons.append("plan_max_nodes_mismatch")
    if list(plan.get("methods") or []) != list(raw_config.get("methods") or []):
        global_reasons.append("plan_methods_mismatch")
    if str(plan.get("seeds")) != str(raw_config.get("seeds")):
        global_reasons.append("plan_seeds_mismatch")
    if str(plan.get("resolutions")) != str(raw_config.get("resolutions")):
        global_reasons.append("plan_resolutions_mismatch")
    global_reasons.extend(
        f"plan.{reason}"
        for reason in identity_rejection_reasons(
            plan.get("experiment_identity"), identity
        )
    )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        expected_shard = (artifact_dir / "shards" / spec["job"]).resolve()
        candidates = [
            item
            for item in records.get(
                (spec["dataset"], spec["cover"], spec["method"], spec["seed"]),
                [],
            )
            if item[0].resolve().is_relative_to(expected_shard)
        ]
        if not candidates:
            rows.append({**spec, "present": False, "admissible": False,
                         "rejection_reasons": list(dict.fromkeys(global_reasons + ["missing_record"]))})
            continue
        job = plan_jobs.get(spec["job"], {})
        plan_content = job.get("dataset_content_identity")
        matching_candidates = []
        for candidate_path, candidate, candidate_sha256 in candidates:
            candidate_report = candidate.get("dataset_report")
            candidate_content = (
                candidate_report.get("content_identity")
                if isinstance(candidate_report, dict)
                else None
            )
            if isinstance(candidate_content, dict):
                expected_candidate_resolution = _expected_resolution(
                    spec["method"], spec["resolution_spec"], candidate_content
                )
                if _float_matches(candidate.get("resolution"), expected_candidate_resolution):
                    matching_candidates.append(
                        (candidate_path, candidate, candidate_sha256)
                    )
        selected = matching_candidates or candidates
        path, record, record_sha256 = max(
            selected, key=lambda item: str(item[1].get("created_at", ""))
        )
        reasons = list(global_reasons)
        reasons.extend(
            identity_rejection_reasons(record.get("experiment_identity"), identity)
        )
        if record.get("profile") != raw_config.get("profile", "full"):
            reasons.append("profile_mismatch")
        if record.get("dataset") != spec["dataset"] or record.get("cover") != spec["cover"]:
            reasons.append("condition_dataset_or_cover_mismatch")
        if record.get("method") != spec["method"] or record.get("seed") != spec["seed"]:
            reasons.append("condition_method_or_seed_mismatch")
        dataset_report = record.get("dataset_report")
        content = (
            dataset_report.get("content_identity")
            if isinstance(dataset_report, dict)
            else None
        )
        if not isinstance(content, dict):
            reasons.append("missing_dataset_content_identity")
            if record.get("dataset_metadata_identity") is None:
                reasons.append("missing_dataset_metadata_identity")
            content = {}
        else:
            expected_resolution = _expected_resolution(
                spec["method"], spec["resolution_spec"], content
            )
            if not _float_matches(record.get("resolution"), expected_resolution):
                reasons.append(
                    f"effective_resolution_mismatch:expected={expected_resolution}:"
                    f"actual={record.get('resolution')}"
                )
            if plan_content != content:
                reasons.append("dataset_content_identity_plan_mismatch")
            locked_contents = lock.get("dataset_content_identities") or {}
            locked_content = locked_contents.get(f"{spec['dataset']}/{spec['cover']}")
            if locked_content is None:
                reasons.append("missing_locked_dataset_content_identity")
            elif locked_content != content:
                reasons.append("dataset_content_identity_lock_mismatch")
            expected_dataset_metadata_identity = dataset_metadata_identity(
                dataset_report
            )
            if record.get("dataset_metadata_identity") is None:
                reasons.append("missing_dataset_metadata_identity")
            elif record.get("dataset_metadata_identity") != expected_dataset_metadata_identity:
                reasons.append("dataset_metadata_identity_mismatch")

        analysis_graph = (
            dataset_report.get("analysis_graph")
            if isinstance(dataset_report, dict) else None
        )
        expected_analysis_policy = lock["analysis_graph_policy"]
        if not isinstance(analysis_graph, dict):
            reasons.append("missing_analysis_graph_identity")
        elif analysis_graph.get("policy") != expected_analysis_policy:
            reasons.append(
                "analysis_graph_policy_mismatch:"
                f"expected={expected_analysis_policy}:actual={analysis_graph.get('policy')}"
            )
        max_nodes = int(raw_config.get("max_nodes", 0))
        bounded = dataset_report.get("bounded_subgraph") if isinstance(dataset_report, dict) else None
        if max_nodes > 0:
            if not isinstance(bounded, dict) or bounded.get("max_nodes") != max_nodes:
                reasons.append("bounded_subgraph_max_nodes_mismatch")
            if content and int(content.get("n", max_nodes + 1)) > max_nodes:
                reasons.append("bounded_subgraph_vertex_count_exceeds_cap")

        expected_memory = (job.get("memory") or {}).get("detector_memory_limit_bytes")
        job_memory = job.get("memory") or {}
        if job_memory.get("ground_truth_community_count") != content.get(
            "ground_truth_community_count"
        ):
            reasons.append("plan_ground_truth_community_count_mismatch")
        if job_memory.get("ground_truth_max_memberships_per_node") != content.get(
            "ground_truth_max_memberships_per_node"
        ):
            reasons.append("plan_ground_truth_membership_cap_mismatch")
        if not isinstance(expected_memory, int) or expected_memory <= 0:
            reasons.append("missing_plan_detector_memory_limit")
        elif record.get("memory_limit_bytes") != expected_memory:
            reasons.append(
                f"memory_limit_bytes_mismatch:expected={expected_memory}:actual={record.get('memory_limit_bytes')}"
            )
        detector_memory = record.get("detector_memory")
        if not isinstance(detector_memory, dict):
            reasons.append("missing_detector_memory_provenance")
        else:
            if detector_memory.get("limit_bytes") != expected_memory:
                reasons.append("detector_memory.limit_bytes_mismatch")
            if detector_memory.get("enforcement") != "process_tree_rss_polling":
                reasons.append("detector_memory.enforcement_mismatch")
            if detector_memory.get("result_transport") != "temporary_pickle_file":
                reasons.append("detector_memory.result_transport_mismatch")
            peak = detector_memory.get("observed_peak_rss_bytes")
            if isinstance(peak, (int, float)) and isinstance(expected_memory, int) and peak > expected_memory:
                reasons.append("detector_memory.completed_peak_exceeds_limit")
        expected_timeout = _expected_timeout(raw_config, spec["dataset"], spec["method"])
        if not _float_matches(record.get("timeout_seconds"), expected_timeout):
            reasons.append(
                f"timeout_seconds_mismatch:expected={expected_timeout}:actual={record.get('timeout_seconds')}"
            )
        overlap = dataset_report.get("overlap_statistics", {}) if isinstance(dataset_report, dict) else {}
        gt_cap = max(
            1, int(content.get("ground_truth_max_memberships_per_node", 1))
        )
        if overlap.get("max_memberships_per_node") != gt_cap:
            reasons.append("ground_truth_cap_report_content_mismatch")
        report_community_count = (
            dataset_report.get("number_of_communities")
            if isinstance(dataset_report, dict)
            else None
        )
        if report_community_count != content.get(
            "ground_truth_community_count"
        ):
            reasons.append("ground_truth_community_count_report_content_mismatch")
        configured_cap = raw_config.get("max_memberships")
        expected_memberships = max(
            2,
            min(gt_cap, int(configured_cap) if configured_cap is not None else gt_cap),
        )
        if job.get("max_memberships") != expected_memberships:
            reasons.append("plan_max_memberships_mismatch")
        if record.get("max_memberships") != expected_memberships:
            reasons.append(
                f"max_memberships_mismatch:expected={expected_memberships}:actual={record.get('max_memberships')}"
            )
        if record.get("ground_truth_max_memberships") != gt_cap:
            reasons.append("ground_truth_max_memberships_mismatch")

        is_hedonic = spec["method"].startswith("hedonic_")
        expected_initialization = (
            _seeded_initialization_identity(
                int(content.get("n", 0)),
                int(content.get("ground_truth_community_count", 0)),
                spec["seed"],
            )
            if is_hedonic and content
            else None
        )
        if record.get("initialization") != expected_initialization:
            reasons.append("initialization_identity_mismatch")
        expected_flags = {
            "n_iterations": -1 if is_hedonic else None,
            "local_move_only": False,
            "allow_isolation": is_hedonic,
            "ensure_equilibrium": is_hedonic,
        }
        for flag, expected_value in expected_flags.items():
            if record.get(flag) != expected_value:
                reasons.append(
                    f"{flag}_mismatch:expected={expected_value}:actual={record.get(flag)}"
                )
        method_parameters = (record.get("method_metadata") or {}).get("parameters")
        expected_method_parameters = dict(METHODS[spec["method"]].parameters)
        if method_parameters != expected_method_parameters:
            reasons.append("method_parameters_mismatch")
        if (record.get("method_metadata") or {}).get(
            "dependency"
        ) != method_dependency_identity(spec["method"]):
            reasons.append("method_dependency_identity_mismatch")
        if is_hedonic:
            if record.get("equilibrium_status") != "verified_independent_audit":
                reasons.append("equilibrium_status_not_verified")
            if isinstance(method_parameters, dict):
                for flag in ("n_iterations", "local_move_only", "allow_isolation", "ensure_equilibrium"):
                    if method_parameters.get(flag) != expected_flags[flag]:
                        reasons.append(f"method_parameters.{flag}_mismatch")

        run_options = record.get("run_options") or {}
        for option in ("omega", "omega_sample_size"):
            if run_options.get(option) != raw_config.get(option):
                reasons.append(
                    f"run_options.{option}_mismatch:expected={raw_config.get(option)}:actual={run_options.get(option)}"
                )
        expected_external_policy = spec["method"] in set(
            raw_config.get("not_rerun_external_methods") or []
        )
        if run_options.get("external_baseline_policy") != expected_external_policy:
            reasons.append("run_options.external_baseline_policy_mismatch")
        if int(record.get("protocol_version", -1)) != int(lock["run_protocol_version"]):
            reasons.append("run_protocol_version_mismatch")
        if record.get("status") != "completed":
            reasons.append(f"status_not_completed:{record.get('status', 'missing')}")
        reasons.extend(
            _metrics_rejection_reasons(
                record,
                omega=bool(raw_config.get("omega")),
                omega_sample_size=int(raw_config.get("omega_sample_size", 100_000)),
            )
        )
        verified_artifacts: dict[str, dict[str, str]] = {}
        analysis_graph, graph_reasons, graph_identity = _read_bound_artifact(
            path, record, prefix="analysis_graph"
        )
        reasons.extend(graph_reasons)
        if graph_identity is not None:
            verified_artifacts["analysis_graph"] = graph_identity
        ground_truth_cover, gt_reasons, gt_identity = _read_bound_artifact(
            path, record, prefix="ground_truth_cover"
        )
        reasons.extend(gt_reasons)
        if gt_identity is not None:
            verified_artifacts["ground_truth_cover"] = gt_identity
        if record.get("analysis_graph_sha256") != content.get("graph_sha256"):
            reasons.append("analysis_graph_artifact_content_identity_mismatch")
        if record.get("ground_truth_cover_sha256") != content.get(
            "ground_truth_cover_sha256"
        ):
            reasons.append("ground_truth_artifact_content_identity_mismatch")
        final_cover, artifact_reasons, final_cover_identity = _read_bound_artifact(
            path, record, prefix="final_cover"
        )
        reasons.extend(artifact_reasons)
        if final_cover_identity is not None:
            verified_artifacts["final_cover"] = final_cover_identity
        if isinstance(final_cover, list) and content:
            try:
                canonicalize_cover(
                    final_cover,
                    n_vertices=int(content["n"]),
                    minimum_size=1,
                )
            except (TypeError, ValueError):
                reasons.append("final_cover_vertex_out_of_range")
            metrics = record.get("metrics") or {}
            predicted_count = len(final_cover)
            ground_truth_count = content.get("ground_truth_community_count")
            if metrics.get("n_predicted_comms") != predicted_count:
                reasons.append("metrics.n_predicted_comms_final_cover_mismatch")
            if metrics.get("predicted_community_count") != predicted_count:
                reasons.append("metrics.predicted_community_count_final_cover_mismatch")
            if metrics.get("n_gt_comms") != ground_truth_count:
                reasons.append("metrics.n_gt_comms_ground_truth_mismatch")
            if metrics.get("gt_community_count") != ground_truth_count:
                reasons.append("metrics.gt_community_count_mismatch")
            matched = metrics.get("n_matched_communities")
            if (
                isinstance(matched, bool)
                or not isinstance(matched, int)
                or matched < 0
                or matched > min(predicted_count, int(ground_truth_count or 0))
            ):
                reasons.append("metrics.n_matched_communities_invalid")
            else:
                if metrics.get("n_unmatched_predicted_comms") != predicted_count - matched:
                    reasons.append("metrics.unmatched_predicted_count_mismatch")
                if metrics.get("n_unmatched_gt_comms") != int(ground_truth_count) - matched:
                    reasons.append("metrics.unmatched_ground_truth_count_mismatch")
        metrics = record.get("metrics")
        if (
            isinstance(analysis_graph, ig.Graph)
            and isinstance(ground_truth_cover, list)
            and isinstance(final_cover, list)
            and isinstance(metrics, dict)
        ):
            try:
                recomputed_metrics = evaluate_cover(
                    final_cover,
                    ground_truth_cover,
                    analysis_graph.vcount(),
                    compute_omega=bool(raw_config.get("omega")),
                    omega_sample_size=int(
                        raw_config.get("omega_sample_size", 100_000)
                    ),
                    omega_seed=spec["seed"],
                )
                recomputed_metrics["cpm_overlapping_quality"] = (
                    quality_overlapping_cpm(
                        analysis_graph,
                        final_cover,
                        float(record["resolution"]),
                    )
                )
                recomputed_metrics["cpm_overlapping_quality_status"] = "computed"
                recomputed_metrics["runtime_seconds"] = record.get(
                    "runtime_seconds"
                )
            except (KeyError, TypeError, ValueError):
                recomputed_metrics = None
                reasons.append("metric_recomputation_failed")
            if isinstance(recomputed_metrics, dict):
                if set(metrics) != set(recomputed_metrics):
                    reasons.append("metrics_key_set_mismatch_recomputed")
                for key, expected_value in recomputed_metrics.items():
                    if not _metric_value_matches(
                        metrics.get(key), expected_value
                    ):
                        reasons.append(f"metrics.{key}_recomputed_mismatch")
        else:
            reasons.append("metric_recomputation_artifacts_unavailable")
        certificate = record.get("equilibrium_certificate")
        independent_audit_recomputed = False
        if is_hedonic:
            _pre_cleanup, pre_cleanup_reasons, pre_cleanup_identity = _read_bound_artifact(
                path, record, prefix="pre_cleanup_membership"
            )
            reasons.extend(pre_cleanup_reasons)
            if pre_cleanup_identity is not None:
                verified_artifacts["pre_cleanup_membership"] = pre_cleanup_identity
            _final_membership, final_membership_reasons, final_membership_identity = _read_bound_artifact(
                path, record, prefix="final_membership"
            )
            reasons.extend(final_membership_reasons)
            if final_membership_identity is not None:
                verified_artifacts["final_membership"] = final_membership_identity
            if isinstance(final_cover, list) and content:
                projection = _canonical_projection_from_memberships(
                    _final_membership, int(content.get("n", 0))
                )
                if projection is None:
                    reasons.append("invalid_exact_final_memberships")
                elif projection[0] != final_cover:
                    reasons.append("final_cover_membership_content_mismatch")
                else:
                    expected_projection = {
                        **projection[1],
                        "exact_final_membership_sha256": record.get(
                            "final_membership_sha256"
                        ),
                        "canonical_scoring_cover_sha256": record.get(
                            "final_cover_sha256"
                        ),
                    }
                    if record.get("final_membership_projection") != expected_projection:
                        reasons.append("final_membership_projection_mismatch")
                    normalization = (record.get("method_metadata") or {}).get(
                        "normalization"
                    ) or {}
                    if normalization.get(
                        "duplicate_communities_removed"
                    ) != projection[1]["duplicate_community_bodies_removed"]:
                        reasons.append("projection_normalization_duplicate_count_mismatch")
            if not isinstance(certificate, dict):
                reasons.append("missing_independent_equilibrium_certificate")
            else:
                if certificate.get("status") != "verified":
                    reasons.append("independent_equilibrium_not_verified")
                if certificate.get("is_local_equilibrium_at_resolution") is not True:
                    reasons.append("independent_equilibrium_flag_false")
                if not _float_matches(certificate.get("gamma"), record.get("resolution")):
                    reasons.append("equilibrium_certificate_gamma_mismatch")
                if certificate.get("max_memberships") != expected_memberships:
                    reasons.append("equilibrium_certificate_cap_mismatch")
                if certificate.get("allow_isolation") is not True:
                    reasons.append("equilibrium_certificate_policy_mismatch")
                if certificate.get("final_membership_sha256") != record.get(
                    "final_membership_sha256"
                ):
                    reasons.append(
                        "equilibrium_certificate_final_membership_mismatch"
                    )
                if certificate.get("canonical_scoring_cover_sha256") != record.get(
                    "final_cover_sha256"
                ):
                    reasons.append("equilibrium_certificate_scoring_cover_mismatch")
                if certificate.get("certificate_target") != (
                    "exact_labeled_final_memberships"
                ):
                    reasons.append("equilibrium_certificate_target_mismatch")
                if certificate.get(
                    "canonical_scoring_projection_certified_as_equilibrium"
                ) is not False:
                    reasons.append("equilibrium_certificate_projection_claim_mismatch")
                if certificate.get("auditor") != (
                    "hedonic.experiments.overlapping.robustness.audit_cover"
                ):
                    reasons.append("equilibrium_certificate_auditor_mismatch")
                expected_auditor_sha256 = identity.get("tracked_files", {}).get(
                    "src/hedonic/experiments/overlapping/robustness.py"
                )
                if certificate.get("auditor_source_sha256") != expected_auditor_sha256:
                    reasons.append("equilibrium_certificate_auditor_source_mismatch")
                if certificate.get("independent_of_native_stop") is not True:
                    reasons.append("equilibrium_certificate_not_independent")
                if certificate.get("analysis_graph_policy") != expected_analysis_policy:
                    reasons.append("equilibrium_certificate_graph_policy_mismatch")
                if certificate.get("analysis_graph_sha256") != content.get("graph_sha256"):
                    reasons.append("equilibrium_certificate_graph_sha256_mismatch")
                if certificate.get("n_vertices_scored") != content.get("n"):
                    reasons.append("equilibrium_certificate_vertex_count_mismatch")
                if certificate.get("stable_fraction") != 1.0:
                    reasons.append("equilibrium_certificate_incomplete_stability")
                for field in (
                    "max_positive_regret", "max_stability_tolerance",
                    "audit_runtime_seconds",
                ):
                    value = certificate.get(field)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        reasons.append(f"equilibrium_certificate.{field}_missing_or_nonfinite")
                max_regret = certificate.get("max_positive_regret")
                max_tolerance = certificate.get("max_stability_tolerance")
                if (
                    isinstance(max_regret, (int, float))
                    and not isinstance(max_regret, bool)
                    and isinstance(max_tolerance, (int, float))
                    and not isinstance(max_tolerance, bool)
                    and math.isfinite(float(max_regret))
                    and math.isfinite(float(max_tolerance))
                    and (
                        float(max_regret) < 0.0
                        or float(max_tolerance) < 0.0
                        or float(max_regret) > float(max_tolerance) + 1e-15
                    )
                ):
                    reasons.append("equilibrium_certificate_regret_exceeds_tolerance")
                tolerance = certificate.get("tolerance")
                if tolerance != {"atol": 1e-10, "rtol": 1e-9}:
                    reasons.append("equilibrium_certificate_tolerance_mismatch")
                if isinstance(analysis_graph, ig.Graph) and isinstance(
                    _final_membership, list
                ):
                    try:
                        fresh_audit = audit_cover(
                            analysis_graph,
                            _final_membership,
                            max_memberships=expected_memberships,
                            allow_isolation=True,
                            gamma=float(record["resolution"]),
                            atol=1e-10,
                            rtol=1e-9,
                            compute_intervals=False,
                            dense=False,
                        )
                        independent_audit_recomputed = True
                    except (KeyError, TypeError, ValueError):
                        fresh_audit = None
                        reasons.append("independent_equilibrium_recomputation_failed")
                    if isinstance(fresh_audit, dict):
                        if fresh_audit.get(
                            "is_local_equilibrium_at_resolution"
                        ) is not True:
                            reasons.append(
                                "recomputed_independent_equilibrium_flag_false"
                            )
                        for audit_field, certificate_field in (
                            (
                                "stable_fraction_at_resolution",
                                "stable_fraction",
                            ),
                            (
                                "max_positive_regret_at_resolution",
                                "max_positive_regret",
                            ),
                            (
                                "max_stability_tolerance_at_resolution",
                                "max_stability_tolerance",
                            ),
                        ):
                            if not _metric_value_matches(
                                certificate.get(certificate_field),
                                fresh_audit.get(audit_field),
                            ):
                                reasons.append(
                                    "equilibrium_certificate_"
                                    f"{certificate_field}_recomputed_mismatch"
                                )
                        if fresh_audit.get("n_vertices_scored") != content.get(
                            "n"
                        ):
                            reasons.append(
                                "recomputed_equilibrium_vertex_count_mismatch"
                            )
                else:
                    reasons.append("independent_equilibrium_artifacts_unavailable")
        else:
            if record.get("equilibrium_status") != "not_applicable":
                reasons.append("external_equilibrium_status_mismatch")
            if not isinstance(certificate, dict) or certificate.get("status") != "not_applicable":
                reasons.append("external_equilibrium_status_not_applicable")
            elif (
                certificate.get("auditor")
                != "hedonic.experiments.overlapping.robustness.audit_cover"
                or certificate.get("auditor_source_sha256")
                != identity.get("tracked_files", {}).get(
                    "src/hedonic/experiments/overlapping/robustness.py"
                )
            ):
                reasons.append("external_equilibrium_provenance_mismatch")
        rows.append({
            **spec, "present": True, "admissible": not reasons,
            "status": record.get("status"), "resolution": record.get("resolution"),
            "timeout_seconds": record.get("timeout_seconds"),
            "memory_limit_bytes": record.get("memory_limit_bytes"),
            "protocol_version": record.get("protocol_version"),
            "record_path": path.relative_to(artifact_dir).as_posix(),
            "record_sha256": record_sha256,
            "graph_sha256": content.get("graph_sha256"),
            "ground_truth_cover_sha256": content.get("ground_truth_cover_sha256"),
            "final_cover_sha256": record.get("final_cover_sha256"),
            "verified_artifacts": verified_artifacts,
            "independent_audit_recomputed": independent_audit_recomputed,
            "dataset_report_sha256": _canonical_hash(record.get("dataset_report"))
            if isinstance(record.get("dataset_report"), dict) else None,
            "rejection_reasons": list(dict.fromkeys(reasons)),
        })
    reason_counts = Counter(reason for row in rows for reason in row["rejection_reasons"])
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "read_only_audit": True,
        "artifact_dir": _portable_path(artifact_dir),
        "config_path": _portable_path(config_path),
        "protocol_lock_path": _portable_path(lock_path),
        "protocol_lock_sha256": lock_sha256,
        "plan_path": _portable_path(plan_path),
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "experiment_identity": identity,
        "expected_conditions": len(rows),
        "present_records": sum(bool(row["present"]) for row in rows),
        "admissible_records": sum(bool(row["admissible"]) for row in rows),
        "rejected_records": sum(bool(row["present"] and not row["admissible"]) for row in rows),
        "missing_records": sum(not bool(row["present"]) for row in rows),
        "global_rejection_reasons": list(dict.fromkeys(global_reasons)),
        "ready_for_publication": (
            raw_config.get("profile") in {"standard", "full"}
            and bool(rows)
            and all(bool(row["admissible"]) for row in rows)
        ),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [key for key in report["rows"][0] if key != "rejection_reasons"] + ["rejection_reasons"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({**row, "rejection_reasons": ";".join(row["rejection_reasons"])})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only audit of the public overlapping benchmark identity")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PAPER_ARTIFACTS_DIR / "equilibrium_v2" / "full",
        help=(
            "Paper artifact directory (default: "
            "artifacts/papers/overlapping_communities/equilibrium_v2/full)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hedonic.toml"),
    )
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; writes a sibling CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.artifact_dir = expand_path(args.artifact_dir)
    args.config = expand_path(args.config)
    args.lock = expand_path(args.lock)
    if args.output:
        args.output = expand_path(args.output)
    report = reconcile(args.artifact_dir, args.config, args.lock)
    if args.output:
        _write_report(report, args.output)
    print(json.dumps({key: report[key] for key in (
        "expected_conditions", "present_records", "admissible_records",
        "rejected_records", "missing_records", "rejection_reason_counts",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
