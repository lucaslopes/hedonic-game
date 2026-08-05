"""Locked protocol identity and read-only evidence reconciliation.

This module deliberately does not load a graph or run a detector.  It gives
the overlapping-paper artifacts a durable identity and explains, field by
field, whether every requested condition is admissible under that identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tomllib
from collections import Counter
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any


LOCK_PATH = Path(__file__).resolve().parents[4] / "configs" / "overlapping-paper-protocol.lock.json"
IDENTITY_SCHEMA_VERSION = 1


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


@cache
def current_experiment_identity(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    """Return reproducible code/config/dependency identity for new records."""
    lock = load_protocol_lock(lock_path)
    repo_root = lock_path.resolve().parents[1]
    files = {
        relative: _sha256_file(repo_root / relative)
        for relative in lock.get("tracked_files", {})
    }
    source = repo_root / str(lock["lucas_igraph"]["source_path"])
    actual_revision = _git_revision(source)
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
            "distribution": lock["lucas_igraph"]["distribution"],
            "expected_revision": lock["lucas_igraph"]["revision"],
            "actual_revision": actual_revision,
            "revision_matches_lock": actual_revision == lock["lucas_igraph"]["revision"],
        },
    }


def identity_rejection_reasons(existing: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(existing, dict):
        return ["missing_experiment_identity"]
    reasons: list[str] = []
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
    elif old_dep.get("actual_revision") != new_dep.get("expected_revision"):
        reasons.append("lucas_igraph_revision_mismatch")
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
