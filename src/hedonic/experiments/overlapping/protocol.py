"""Locked protocol identity and read-only evidence reconciliation.

This module deliberately does not load a graph or run a detector.  It gives
the overlapping-paper artifacts a durable identity and explains, field by
field, whether every requested condition is admissible under that identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
import tomllib
from collections import Counter
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any


LOCK_PATH = Path(__file__).resolve().parents[4] / "configs" / "overlapping-paper-protocol.lock.json"
IDENTITY_SCHEMA_VERSION = 2


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


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def load_protocol_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if int(lock.get("schema_version", -1)) != IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported protocol lock schema: {lock.get('schema_version')}")
    return lock


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


def _uv_lock_package_sha256(path: Path | None, distribution: str) -> str | None:
    """Hash the canonical uv.lock package entry for a distribution.

    The digest is deliberately independent of the host platform: the package
    entry contains the version, dependency set, sdist, and all locked wheels.
    The selected wheel's hash is still preserved inside that canonical entry.
    """
    if path is None:
        return None
    try:
        packages = tomllib.loads(path.read_text(encoding="utf-8")).get("package", [])
    except (OSError, tomllib.TOMLDecodeError):
        return None
    wanted = _normalise_distribution_name(distribution)
    for package in packages:
        if _normalise_distribution_name(package.get("name", "")) == wanted:
            return _canonical_hash(package)
    return None


@cache
def current_experiment_identity(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    """Return reproducible code/config/dependency identity for new records."""
    lock = load_protocol_lock(lock_path)
    repo_root = lock_path.resolve().parents[1]
    files = {
        relative: _sha256_file(repo_root / relative)
        for relative in lock.get("tracked_files", {})
    }
    dependency = lock.get("lucas_igraph", {})
    distribution = str(dependency.get("distribution", "lucas-igraph"))
    uv_lock_path = _resolve_path(repo_root, dependency.get("uv_lock_path", "uv.lock"))
    expected_version = dependency.get("version")
    expected_igraph_version = dependency.get("igraph_version")
    expected_uv_lock_sha256 = dependency.get("uv_lock_sha256")
    expected_package_lock_sha256 = dependency.get("package_lock_sha256")
    actual_version = _distribution_version(distribution)
    actual_igraph_version = _igraph_version()
    actual_uv_lock_sha256 = _sha256_file(uv_lock_path) if uv_lock_path else None
    actual_package_lock_sha256 = _uv_lock_package_sha256(uv_lock_path, distribution)

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
            version_matches_lock,
            igraph_version_matches_lock,
            uv_lock_matches_lock,
            package_lock_matches_lock,
        )
    )
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "protocol_lock_sha256": _sha256_file(lock_path),
        "protocol_name": lock["protocol_name"],
        "run_protocol_version": lock["run_protocol_version"],
        "analysis_graph_policy": lock["analysis_graph_policy"],
        "tracked_files": files,
        "tracked_files_match_lock": all(
            digest == lock["tracked_files"].get(relative)
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
            "package_identity_matches_lock": package_identity_matches_lock,
            "released_native_sha": released_native_sha,
            "source_path": str(dependency.get("source_path")) if dependency.get("source_path") else None,
            "actual_source_revision": actual_source_revision,
            "source_revision_matches_release": source_revision_matches_release,
        },
    }


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
    if existing.get("tracked_files") != expected.get("tracked_files"):
        reasons.append("tracked_code_or_config_hash_mismatch")
    return reasons


def dataset_metadata_identity(report: Any) -> dict[str, Any] | None:
    """Hash the loader's ID-safe graph/cover report without exposing raw data."""
    if not isinstance(report, dict):
        return None
    return {"sha256": _canonical_hash(report), "report": report}


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


def _condition_specs(config_path: Path) -> list[dict[str, Any]]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))["overlapping_paper"]
    seeds = _parse_ints(str(raw["seeds"]))
    return [
        {"job": job["name"], "dataset": job["dataset"], "cover": job["cover"],
         "method": method, "seed": seed, "resolution_spec": str(raw["resolutions"])}
        for job in raw["jobs"] for method in raw["methods"] for seed in seeds
    ]


def reconcile(
    artifact_dir: Path,
    config_path: Path,
    lock_path: Path = LOCK_PATH,
) -> dict[str, Any]:
    """Inventory one current record per locked condition without modifying artifacts."""
    lock = load_protocol_lock(lock_path)
    identity = current_experiment_identity(lock_path)
    plan_path = artifact_dir / "orchestration" / "plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        plan = {}
    plan_jobs = {str(job.get("name")): job for job in plan.get("jobs", [])}
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))["overlapping_paper"]
    records: dict[tuple[str, str, str, int], list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted((artifact_dir / "shards").glob("*/runs/**/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        records.setdefault(_record_key(record), []).append((path, record))

    specs = _condition_specs(config_path)
    if len(specs) != int(lock["expected_conditions"]):
        raise ValueError(
            f"Locked protocol expects {lock['expected_conditions']} conditions; config defines {len(specs)}"
        )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        candidates = records.get((spec["dataset"], spec["cover"], spec["method"], spec["seed"]), [])
        if not candidates:
            rows.append({**spec, "present": False, "admissible": False,
                         "rejection_reasons": ["missing_record"]})
            continue
        path, record = max(candidates, key=lambda item: str(item[1].get("created_at", "")))
        reasons = identity_rejection_reasons(record.get("experiment_identity"), identity)
        job = plan_jobs.get(spec["job"], {})
        expected_memory = (job.get("memory") or {}).get("detector_memory_limit_bytes")
        if expected_memory is not None and record.get("memory_limit_bytes") != expected_memory:
            reasons.append(
                f"memory_limit_bytes_mismatch:expected={expected_memory}:actual={record.get('memory_limit_bytes')}"
            )
        method_timeout = (raw_config.get("timeout_by_method") or {}).get(spec["method"])
        dataset_timeout = (raw_config.get("timeout_by_dataset") or {}).get(spec["dataset"])
        expected_timeout = min(float(value) for value in (
            raw_config["timeout_per_run"], method_timeout, dataset_timeout
        ) if value is not None)
        try:
            timeout_matches = abs(float(record.get("timeout_seconds")) - expected_timeout) <= 1e-12
        except (TypeError, ValueError):
            timeout_matches = False
        if not timeout_matches:
            reasons.append(
                f"timeout_seconds_mismatch:expected={expected_timeout}:actual={record.get('timeout_seconds')}"
            )
        expected_memberships = job.get("max_memberships")
        if expected_memberships is not None and record.get("max_memberships") != expected_memberships:
            reasons.append(
                f"max_memberships_mismatch:expected={expected_memberships}:actual={record.get('max_memberships')}"
            )
        run_options = record.get("run_options") or {}
        for option in ("omega", "omega_sample_size"):
            if run_options.get(option) != raw_config.get(option):
                reasons.append(
                    f"run_options.{option}_mismatch:expected={raw_config.get(option)}:actual={run_options.get(option)}"
                )
        if int(record.get("protocol_version", -1)) != int(lock["run_protocol_version"]):
            reasons.append("run_protocol_version_mismatch")
        dataset_report = record.get("dataset_report")
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
        if (
            spec["dataset"] == "wikipedia"
            and isinstance(dataset_report, dict)
            and bool(dataset_report.get("directed"))
            and (
                not isinstance(analysis_graph, dict)
                or analysis_graph.get("policy") != expected_analysis_policy
            )
        ):
            reasons.append("historical_wikipedia_directionality_not_common_undirected_simple")
        if record.get("status") != "completed":
            reasons.append(f"status_not_completed:{record.get('status', 'missing')}")
        if record.get("dataset_metadata_identity") is None:
            reasons.append("missing_dataset_metadata_identity")
        rows.append({
            **spec, "present": True, "admissible": not reasons,
            "status": record.get("status"), "resolution": record.get("resolution"),
            "timeout_seconds": record.get("timeout_seconds"),
            "memory_limit_bytes": record.get("memory_limit_bytes"),
            "protocol_version": record.get("protocol_version"),
            "record_path": str(path), "record_sha256": _sha256_file(path),
            "dataset_report_sha256": _canonical_hash(record.get("dataset_report"))
            if isinstance(record.get("dataset_report"), dict) else None,
            "rejection_reasons": reasons,
        })
    reason_counts = Counter(reason for row in rows for reason in row["rejection_reasons"])
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "read_only_audit": True,
        "artifact_dir": _portable_path(artifact_dir),
        "config_path": _portable_path(config_path),
        "protocol_lock_path": _portable_path(lock_path),
        "protocol_lock_sha256": _sha256_file(lock_path),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "experiment_identity": identity,
        "expected_conditions": len(rows),
        "present_records": sum(bool(row["present"]) for row in rows),
        "admissible_records": sum(bool(row["admissible"]) for row in rows),
        "rejected_records": sum(bool(row["present"] and not row["admissible"]) for row in rows),
        "missing_records": sum(not bool(row["present"]) for row in rows),
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
    parser = argparse.ArgumentParser(description="Read-only audit of the locked overlapping-paper evidence")
    parser.add_argument("--artifact-dir", type=Path, default=Path("docs/papers/overlapping_communities/artifacts/full"))
    parser.add_argument("--config", type=Path, default=Path("configs/hedonic.toml"))
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; writes a sibling CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
