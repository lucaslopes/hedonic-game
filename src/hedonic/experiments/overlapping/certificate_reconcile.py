"""Detector-free reconciliation of overlapping ledgers (TKT-07).

This command does not load SNAP graphs and does not invoke detectors unless
``--replay-gt`` is passed.  Classification uses already-written native
outcomes versus independent-audit fields.  Optional membership replay
recomputes tiny DNN exact covers and re-audits persisted GT-v3 memberships
without rewriting frozen ledgers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_EVIDENCE = Path("artifacts/evidence/overlapping_communities")
DEFAULT_GT_RESULTS = Path(
    "artifacts/overlapping/ground_truth_robustness_v3/results.jsonl"
)
DEFAULT_CONTROLLED = DEFAULT_EVIDENCE / "controlled_overlap_v2.json"
DEFAULT_DNN = DEFAULT_EVIDENCE / "dnn_certificate_v2.json"
DEFAULT_RATIONAL_DNN = DEFAULT_EVIDENCE / "dual_witnesses_verified_1.0.0.6.json"
DEFAULT_EQV2 = Path(
    "artifacts/papers/overlapping_communities/equilibrium_v2/full/coverage_report.json"
)
DEFAULT_EQV2_RESULTS = DEFAULT_EQV2.parent / "results.jsonl"
DEFAULT_PAPER_SUMMARY = DEFAULT_EQV2.parent / "paper_summary.csv"
DEFAULT_TEX_FRAGMENT_DIR = DEFAULT_EQV2.parent
PAPER_HEDONIC_REGRET_BOUND = 1.8e-15
_DATASET_ORDER = ("amazon", "dblp", "livejournal", "youtube", "wikipedia")
_DATASET_LABEL = {
    "amazon": "Amazon",
    "dblp": "DBLP",
    "livejournal": "LiveJournal",
    "youtube": "YouTube",
    "wikipedia": "Wikipedia",
}
_BENCHMARK_METHODS = (
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "cpm",
    "demon",
)
_CONTROLLED_ROWS = (
    ("singleton", "local", "Singleton"),
    ("singleton", "multi-phase", "Singleton"),
    ("neutral-disjoint", "local", "Neutral disjoint"),
    ("neutral-disjoint", "multi-phase", "Neutral disjoint"),
    ("gt-primary", "local", "GT-primary$^\\dagger$"),
    ("gt-primary", "multi-phase", "GT-primary$^\\dagger$"),
)
_DNN_DISPLAY = {
    "path4": "Path (4,2,2)",
    "bow_tie5": "Bow tie (5,2,2)",
    "weighted_bridge5": "Weighted bridge (5,3,2)",
}
# Student $t_{0.975, n-1}$ for the 24-graph and 3-seed clustered intervals.
_TCRIT = {3: 4.302652729911275, 24: 2.0686577645122027}
DEFAULT_LOCKED = DEFAULT_EVIDENCE / "locked_benchmark_manifest.json"
DEFAULT_POSTRELEASE = DEFAULT_EVIDENCE / "postrelease_protocol_audit_1.0.0.2.json"
DEFAULT_HISTORICAL = DEFAULT_EVIDENCE / "protocol_audit.json"
DEFAULT_NATIVE_DIFFERENTIAL = DEFAULT_EVIDENCE / "native_differential.jsonl"
DEFAULT_REPLAY_SUMMARY = DEFAULT_EVIDENCE / "membership_replay_summary.json"
DEFAULT_FEE_V1 = DEFAULT_EVIDENCE / "unit_l2_cpm_fee_v1.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_PROTOCOL_LOCK = REPO_ROOT / "configs" / "overlapping-paper-protocol.lock.json"
GT_PROTOCOL_LOCK = REPO_ROOT / "configs" / "overlapping-ground-truth-protocol.lock.json"
LEIDEN_PIN_ARCHIVE = DEFAULT_EVIDENCE / "source_archives" / "leiden.c.99d6fd99"
DEFAULT_MAIN_TEX = (
    REPO_ROOT / "docs" / "papers" / "overlapping_communities" / "main.tex"
)
DEFAULT_MAIN_PDF = (
    REPO_ROOT / "docs" / "papers" / "overlapping_communities" / "main.pdf"
)
DEFAULT_TEX_PDF_SNAPSHOT = DEFAULT_EVIDENCE / "tex_pdf_snapshot_2026-09-30.md"
DEFAULT_SOURCE_MANIFEST = DEFAULT_EVIDENCE / "source_manifest.json"
DEFAULT_PUBLIC_RELEASE = DEFAULT_EVIDENCE / "public_release_verification.json"
EXPECTED_LEIDEN_C_SHA256 = (
    "355814ae905c27d9f0e17847550711057d9d873f7d5d0aaf09631c73b520f30d"
)
# Producer identity of the frozen ledgers verified by Gate H. The live wrapper
# may move on (hedonic.Game.HEDONIC_ALGORITHM_IDENTITY); the package keeps
# this string verbatim in HISTORICAL_ALGORITHM_IDENTITIES.
EXPECTED_HEDONIC_ALGORITHM_IDENTITY = (
    "community_hedonic/lucas-igraph-1.0.0.3/interrupt-unsupported"
)
# SHA-256 of the frozen lock *files*. Do not update these to chase live wrapper
# bytes; a new publication ledger needs a new lock, not an in-place rewrite.
FROZEN_LOCK_SHA256 = {
    "overlapping-paper-equilibrium-v2": (
        "e7ff181556a9d68d7e455fc92db4ebd2256e27fe42bc6eac62eba69c9ea839a9"
    ),
    "overlapping-ground-truth-robustness-v3": (
        "985eaa121a22aa62f4b2c7929abe2a5931457342d3d79daf31d86f84152fe2bb"
    ),
}
# Live research-wrapper files that postdate the frozen producer locks. All other
# tracked files must still match. Expanding this set requires a review note.
# Files that changed after a lock was frozen and are guarded by another part
# of the identity check rather than by their own bytes: the dependency pins
# (pyproject.toml, uv.lock) are compared through the lucas_igraph and
# scientific-dependency identities, the loaders and method registry (snap.py,
# methods.py) through the frozen dataset content identities and method
# dependency identities of every record, and plans are documentation.  They
# postdate both locks (lucas-igraph 1.0.0.4/1.0.0.5 pins, the unified method
# registry, the durable front door, the ground-truth spectrum).
_POST_LOCK_GUARDED_ELSEWHERE = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "src/hedonic/experiments/overlapping/methods.py",
        "src/hedonic/experiments/overlapping/snap.py",
    }
)

POST_LOCK_WRAPPER_FILES = {
    "overlapping-paper-equilibrium-v2": _POST_LOCK_GUARDED_ELSEWHERE | frozenset(
        {
            "src/hedonic/Game.py",
            "src/hedonic/experiments/CLI.py",
            "src/hedonic/experiments/overlapping/benchmark.py",
            "src/hedonic/experiments/overlapping/metrics.py",
            "src/hedonic/experiments/overlapping/reproduce_paper.py",
            "src/hedonic/experiments/overlapping/robustness.py",
        }
    ),
    "overlapping-ground-truth-robustness-v3": _POST_LOCK_GUARDED_ELSEWHERE | frozenset(
        {
            "AGENTS.md",
            "docs/overlapping_ground_truth_reverse_engineering_plan.md",
            "src/hedonic/Game.py",
            "src/hedonic/experiments/CLI.py",
            "src/hedonic/experiments/overlapping/metrics.py",
            "src/hedonic/experiments/overlapping/robustness.py",
        }
    ),
}

# This is the public release identity observed in the PyPI JSON API on
# 2026-09-13.  It is deliberately separate from the frozen producer and
# replay identities below: publishing 1.0.0.4/0.1.1 does not relabel a
# 1.0.0.2 ledger or a 1.0.0.3 replay.  The corresponding receipt is kept under
# artifacts/evidence/overlapping_communities/public_release_verification.json.
EXPECTED_PUBLIC_RELEASES = {
    "lucas-igraph": {
        "version": "1.0.0.4",
        "requires_python": ">=3.9",
        "release_url": "https://pypi.org/project/lucas-igraph/1.0.0.4/",
        "json_url": "https://pypi.org/pypi/lucas-igraph/1.0.0.4/json",
        "files": {
            "lucas_igraph-1.0.0.4-cp39-abi3-macosx_10_15_x86_64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "187a41688038fbda679778d52953a098bb4c9b0da3657ac31338da743b5f9a55",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-macosx_11_0_arm64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "68472abd91f3568f7366a12ac68a28f377d552e6c9a9f0ed1ecaec31e0f0d47c",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-manylinux_2_28_aarch64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "54b6a5b75c438b175c17bf71a3abede421a3f875179882fbf9c7e7b9c6695036",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-manylinux_2_28_x86_64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "2fac0b1a00f6070b7734a02750f602d5930a2b4205b909f194cf01691bd2954c",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-musllinux_1_2_aarch64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "6d7d28735fa8bbdfce2ddd953cf5800b1d7b7b58a846114307ed2b4e94377d9b",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-musllinux_1_2_x86_64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "3bd567654df888fc735048328c5a7c83a677915e6f7146a6abb58c1f13e3cf8a",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-win32.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "a6b387620d466f8fe3e0450f404a4935ae6d9a124efccc5a2aeedef59bd6833e",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-win_amd64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "58758cc28219f1f8a5917370cdc9e1e3e0c8ec443afed37c440929d562d04b69",
            },
            "lucas_igraph-1.0.0.4-cp39-abi3-win_arm64.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "9afe7f92c503e3e2c2f4e94c5e733e3040d580aa2f2a676cfd6ca680050a2c4c",
            },
            "lucas_igraph-1.0.0.4.tar.gz": {
                "packagetype": "sdist",
                "sha256": "56b38e065e09e75749a601ffc34aab629ee2ca92a862afef92471b85bba7719f",
            },
        },
    },
    "hedonic": {
        "version": "0.1.1",
        "requires_python": ">=3.12",
        "release_url": "https://pypi.org/project/hedonic/0.1.1/",
        "json_url": "https://pypi.org/pypi/hedonic/0.1.1/json",
        "requires_dist": "lucas-igraph==1.0.0.4",
        "files": {
            "hedonic-0.1.1-py3-none-any.whl": {
                "packagetype": "bdist_wheel",
                "sha256": "08d920ce365f0c230e657d4efe16663be2bc8d7eb939d3a8135bab2aa0434db5",
            },
            "hedonic-0.1.1.tar.gz": {
                "packagetype": "sdist",
                "sha256": "c3f483e006c034278e80fd214206c0c4b5f66cdff4f102ebe88338daa73e61c7",
            },
        },
    },
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_pypi_json(url: str) -> dict[str, Any]:
    """Fetch one immutable-version PyPI JSON response for an opt-in check."""

    from urllib.request import Request, urlopen

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hedonic-gate-h/1 (release provenance check)",
        },
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed PyPI URL
        return json.load(response)


def verify_public_release_provenance(
    path: Path = DEFAULT_PUBLIC_RELEASE,
    *,
    online: bool = False,
) -> dict[str, Any]:
    """Verify the published 1.0.0.4/0.1.1 receipt and its identity boundary.

    The checked-in receipt is an immutable observation of the official PyPI
    JSON API.  Gate H uses that offline receipt so the reader path remains
    reproducible without network access.  ``online=True`` additionally fetches
    each immutable-version endpoint and compares every filename, digest,
    package type, and non-yanked flag with the receipt.
    """

    info = _file_record(path)
    if not path.is_file():
        return {
            **info,
            "present": False,
            "checks": {"receipt_present": False},
            "online": online,
            "ok": False,
        }

    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **info,
            "present": True,
            "checks": {"receipt_readable": False},
            "error": f"{type(exc).__name__}: {exc}",
            "online": online,
            "ok": False,
        }

    checks: dict[str, bool] = {
        "receipt_present": True,
        "schema_version": payload.get("schema_version") == 1,
        "source_is_pypi_json_api": payload.get("source") == "PyPI JSON API",
        "observed_date": str(payload.get("observed_utc", "")).startswith(
            "2026-09-13"
        ),
    }
    project_reports: dict[str, Any] = {}
    projects = payload.get("projects") or {}
    for project, expected in EXPECTED_PUBLIC_RELEASES.items():
        observed = projects.get(project) or {}
        observed_files = observed.get("files") or []
        files_by_name = {
            row.get("filename"): row
            for row in observed_files
            if isinstance(row, dict) and row.get("filename")
        }
        project_checks: dict[str, bool] = {
            "version": observed.get("version") == expected["version"],
            "latest_version_at_observation": observed.get(
                "latest_version_at_observation"
            )
            == expected["version"],
            "requires_python": observed.get("requires_python")
            == expected["requires_python"],
            "release_url": observed.get("release_url") == expected["release_url"],
            "json_url": observed.get("json_url") == expected["json_url"],
            "published": observed.get("published") is True,
            "yanked": observed.get("yanked") is False,
            "all_files_non_yanked": observed.get("all_files_non_yanked") is True,
            "file_names": set(files_by_name) == set(expected["files"]),
            "file_count": len(observed_files) == len(expected["files"]),
            "wheel_count": observed.get("wheel_count")
            == sum(
                row["packagetype"] == "bdist_wheel"
                for row in expected["files"].values()
            ),
            "sdist_count": observed.get("sdist_count")
            == sum(
                row["packagetype"] == "sdist"
                for row in expected["files"].values()
            ),
        }
        file_checks: dict[str, bool] = {}
        for filename, expected_file in expected["files"].items():
            actual = files_by_name.get(filename) or {}
            key = filename.replace(".", "_").replace("-", "_")
            file_checks[f"{key}_sha256"] = (
                actual.get("sha256") == expected_file["sha256"]
            )
            file_checks[f"{key}_packagetype"] = (
                actual.get("packagetype") == expected_file["packagetype"]
            )
            file_checks[f"{key}_not_yanked"] = actual.get("yanked") is False
        project_checks.update(file_checks)
        project_reports[project] = {
            "expected_version": expected["version"],
            "observed_version": observed.get("version"),
            "expected_file_count": len(expected["files"]),
            "observed_file_count": len(observed_files),
            "checks": project_checks,
            "ok": all(project_checks.values()),
        }
        checks[f"{project}_release"] = project_reports[project]["ok"]

    distinction = payload.get("distinction") or {}
    distinction_checks = {
        "historical_ledger_producer_lucas_igraph": distinction.get(
            "historical_ledger_producer_lucas_igraph"
        )
        == "1.0.0.2",
        "historical_replay_lucas_igraph": distinction.get(
            "historical_replay_lucas_igraph"
        )
        == "1.0.0.3",
        "historical_replay_hedonic": distinction.get("historical_replay_hedonic")
        == "0.1.0",
        "published_not_ledger_producer": distinction.get(
            "published_release_used_to_produce_displayed_ledgers"
        )
        is False,
        "published_not_replay_producer": distinction.get(
            "published_release_used_to_replay_displayed_ledgers"
        )
        is False,
        "new_protocol_lock_required": distinction.get(
            "new_protocol_lock_required_for_rerun"
        )
        is True,
    }
    checks.update({f"distinction_{name}": value for name, value in distinction_checks.items()})

    online_report: dict[str, Any] = {"requested": online, "ok": True}
    if online:
        online_checks: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for project, expected in EXPECTED_PUBLIC_RELEASES.items():
            try:
                remote = _fetch_pypi_json(expected["json_url"])
                remote_files = {
                    row.get("filename"): row
                    for row in remote.get("urls", [])
                    if isinstance(row, dict) and row.get("filename")
                }
                online_checks[f"{project}_version"] = (
                    remote.get("info", {}).get("version") == expected["version"]
                )
                online_checks[f"{project}_requires_python"] = (
                    remote.get("info", {}).get("requires_python")
                    == expected["requires_python"]
                )
                online_checks[f"{project}_file_names"] = set(remote_files) == set(
                    expected["files"]
                )
                for filename, expected_file in expected["files"].items():
                    actual = remote_files.get(filename) or {}
                    key = filename.replace(".", "_").replace("-", "_")
                    online_checks[f"{project}_{key}_sha256"] = (
                        actual.get("digests", {}).get("sha256")
                        == expected_file["sha256"]
                    )
                    online_checks[f"{project}_{key}_packagetype"] = (
                        actual.get("packagetype") == expected_file["packagetype"]
                    )
                    online_checks[f"{project}_{key}_not_yanked"] = (
                        actual.get("yanked") is False
                    )
            except Exception as exc:  # network and HTTP errors are report data
                errors[project] = f"{type(exc).__name__}: {exc}"
        online_report = {
            "requested": True,
            "checks": online_checks,
            "errors": errors,
            "ok": not errors and all(online_checks.values()),
        }

    checks["online_release_check"] = online_report["ok"]
    return {
        **info,
        "present": True,
        "source": payload.get("source"),
        "observed_utc": payload.get("observed_utc"),
        "projects": project_reports,
        "distinction": distinction,
        "checks": checks,
        "online": online_report,
        "ok": all(checks.values()),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def reconcile_ground_truth(path: Path) -> dict[str, Any]:
    """Classify the 3,840-cell v3 ledger from ``results.jsonl`` only."""
    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "expected": 3840}
    status = Counter()
    equilibrium = Counter()
    groups: dict[tuple[str, str, str, str], int] = defaultdict(int)
    keys: list[str] = []
    native_returns = 0
    certified = 0
    terminal = 0
    missing_hash = 0
    for record in _iter_jsonl(path):
        keys.append(str(record.get("condition_key")))
        st = str(record.get("status"))
        status[st] += 1
        equilibrium[str(record.get("equilibrium_status"))] += 1
        dataset = str(record.get("dataset"))
        phase = str(record.get("phase"))
        policy = str(record.get("action_policy") or record.get("policy"))
        groups[(dataset, phase, policy, st)] += 1
        if st == "completed":
            native_returns += 1
            certified += 1
        elif st == "completed_non_equilibrium":
            native_returns += 1
        elif st == "unsupported_cleanup":
            terminal += 1
        if st in {"completed", "completed_non_equilibrium"} and not record.get(
            "final_membership_hash"
        ):
            missing_hash += 1
    unique = len(set(keys))
    return {
        **info,
        "present": True,
        "family": "ground_truth_robustness_v3",
        "expected": 3840,
        "observed": len(keys),
        "unique_condition_keys": unique,
        "duplicate_keys": len(keys) - unique,
        "status_counts": dict(status),
        "equilibrium_status_counts": dict(equilibrium),
        "native_returns": native_returns,
        "independently_certified": certified,
        "uncertified_native_returns": status.get("completed_non_equilibrium", 0),
        "registered_terminals": terminal,
        "missing_final_membership_hash": missing_hash,
        "complete_key_coverage": unique == 3840 and len(keys) == 3840,
        "groups": [
            {
                "dataset": dataset,
                "phase": phase,
                "action_policy": policy,
                "status": st,
                "count": count,
                "native_return": st in {"completed", "completed_non_equilibrium"},
                "independently_certified": st == "completed",
                "terminal": st == "unsupported_cleanup",
            }
            for (dataset, phase, policy, st), count in sorted(groups.items())
        ],
        "producer_note": (
            "Ledger records lucas-igraph 1.0.0.2 / hedonic 0.0.10; "
            "status completed vs completed_non_equilibrium is the "
            "independent audit classification, not a native self-report."
        ),
    }


def reconcile_controlled(path: Path) -> dict[str, Any]:
    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "expected": 864}
    payload = _load_json(path)
    statuses = Counter()
    n_runs = 0
    for instance in payload.get("instances", []):
        for run in instance.get("runs", []):
            n_runs += 1
            statuses[str(run.get("status") or run.get("error") or "unknown")] += 1
    n_expected = sum(
        int(row.get("n_expected", 0)) for row in payload.get("summary", [])
    )
    n_completed = sum(
        int(row.get("n_completed", 0)) for row in payload.get("summary", [])
    )
    environment = payload.get("environment") or {}
    identity = (payload.get("implementation_identity") or {}).get("environment") or {}
    return {
        **info,
        "present": True,
        "family": "controlled_overlap_v2",
        "expected": 864,
        "observed_runs": n_runs,
        "summary_n_expected": n_expected,
        "summary_n_completed": n_completed,
        "status_counts": dict(statuses),
        "complete": n_runs == 864 and statuses.get("completed", 0) == 864,
        "construction_label": payload.get("construction_label"),
        "producer": environment or identity,
        "recorded_ensure_equilibrium": (payload.get("experiment_design") or {}).get(
            "ensure_equilibrium"
        ),
        "n_iterations": (payload.get("experiment_design") or {}).get("n_iterations"),
        "note": (
            "LFR-derived controlled overlap, not canonical overlapping LFR. "
            "Cells are detector outcomes with a five-second timeout; they are "
            "not independent unit-ℓ₂ certificates."
        ),
    }


def reconcile_dnn(path: Path) -> dict[str, Any]:
    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "expected_instances": 3}
    payload = _load_json(path)
    instances = []
    for result in payload.get("results", []):
        chain = result.get("certificate_chain") or {}
        instances.append(
            {
                "name": (result.get("instance") or {}).get("name"),
                "n_algorithm_runs": len(result.get("algorithm_runs") or []),
                "algorithm_to_exact_gap": chain.get("algorithm_to_exact_gap"),
                "valid_cover_to_repaired_dual_outer_gap": chain.get(
                    "valid_cover_to_repaired_dual_outer_gap"
                ),
                "dnn_status": (result.get("dnn") or {}).get("status"),
                "interpretation": chain.get("interpretation"),
            }
        )
    return {
        **info,
        "present": True,
        "family": "dnn_certificate_v2",
        "expected_instances": 3,
        "observed_instances": len(instances),
        "instances": instances,
        "producer": payload.get("environment"),
        "purpose": payload.get("purpose"),
        "complete": len(instances) == 3,
        "note": (
            "Tiny-instance DNN outer calibration. Dual witnesses are numerical "
            "SCS certificates after repair, not a claim of general tightness."
        ),
    }


def reconcile_coverage_json(
    path: Path,
    *,
    family: str,
    expected: int = 125,
) -> dict[str, Any]:
    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "family": family, "expected": expected}
    payload = _load_json(path)
    rows = payload.get("rows") or []
    statuses = Counter(str(row.get("status")) for row in rows)
    methods = (
        Counter(str(row.get("method")) for row in rows)
        if rows and "method" in rows[0]
        else {}
    )
    completed = int(
        payload.get("completed")
        or payload.get("completed_records")
        or statuses.get("completed", 0)
    )
    present = int(
        payload.get("present_records")
        or payload.get("expected_records")
        or len(rows)
        or 0
    )
    return {
        **info,
        "present": True,
        "family": family,
        "expected": int(
            payload.get("expected_conditions")
            or payload.get("expected_records")
            or expected
        ),
        "observed_rows": len(rows) or present,
        "completed": completed,
        "admissible_records": payload.get("admissible_records"),
        "missing_records": payload.get("missing_records"),
        "status_counts": dict(payload.get("status_counts") or statuses),
        "method_counts": dict(methods) if methods else None,
        "interpretation": payload.get("interpretation"),
        "complete_plan": completed == expected
        and (payload.get("missing_records") in (0, None)),
    }


def reconcile_paper_results(path: Path) -> dict[str, Any]:
    """Classify the 125-row equilibrium-v2 ``results.jsonl`` certificates."""
    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "expected": 125, "observed": 0}
    methods: Counter[str] = Counter()
    cert_status: Counter[str] = Counter()
    keys: list[tuple[Any, ...]] = []
    hedonic_verified = 0
    external_not_applicable = 0
    hedonic_not_verified = 0
    missing_certificate = 0
    regrets: list[float] = []
    incomplete_stability = 0
    vertices_scored: set[int] = set()
    for record in _iter_jsonl(path):
        method = str(record.get("method"))
        methods[method] += 1
        keys.append(
            (
                record.get("dataset"),
                record.get("cover"),
                method,
                record.get("seed"),
                record.get("requested_resolution"),
            )
        )
        cert = record.get("equilibrium_certificate") or {}
        status = str(cert.get("status") if cert else "missing")
        cert_status[status] += 1
        if method.startswith("hedonic_"):
            verified = (
                status == "verified"
                and bool(cert.get("is_local_equilibrium_at_resolution"))
                and str(cert.get("certificate_target"))
                == "exact_labeled_final_memberships"
                and str(cert.get("auditor"))
                == "hedonic.experiments.overlapping.robustness.audit_cover"
                and bool(cert.get("independent_of_native_stop"))
                and int(cert.get("n_vertices_scored") or 0) == 3000
                and float(cert.get("stable_fraction") or 0) == 1.0
                and cert.get("max_positive_regret") is not None
                and float(cert["max_positive_regret"]) <= PAPER_HEDONIC_REGRET_BOUND
            )
            if verified:
                hedonic_verified += 1
                regrets.append(float(cert["max_positive_regret"]))
                vertices_scored.add(int(cert["n_vertices_scored"]))
            elif not cert:
                missing_certificate += 1
            else:
                hedonic_not_verified += 1
                if (
                    int(cert.get("n_vertices_scored") or 0) != 3000
                    or float(cert.get("stable_fraction") or 0) != 1.0
                ):
                    incomplete_stability += 1
        elif status == "not_applicable":
            external_not_applicable += 1
        elif not cert:
            missing_certificate += 1
    unique = len(set(keys))
    return {
        **info,
        "present": True,
        "expected": 125,
        "observed": len(keys),
        "unique_condition_keys": unique,
        "duplicate_keys": len(keys) - unique,
        "method_counts": dict(methods),
        "certificate_status_counts": dict(cert_status),
        "hedonic_verified": hedonic_verified,
        "hedonic_not_verified": hedonic_not_verified,
        "external_not_applicable": external_not_applicable,
        "missing_certificate": missing_certificate,
        "incomplete_stability": incomplete_stability,
        "max_positive_regret": max(regrets) if regrets else None,
        "vertices_scored": sorted(vertices_scored),
        "complete_certificates": (
            len(keys) == 125
            and unique == 125
            and hedonic_verified == 75
            and external_not_applicable == 50
            and hedonic_not_verified == 0
            and missing_certificate == 0
        ),
        "note": (
            "75 Hedonic rows carry independent exact-state certificates; "
            "50 CPM/DEMON rows are not_applicable. Detector completion "
            "(125/125) is not itself a Nash certificate."
        ),
    }


def build_manifest(
    *,
    gt_results: Path,
    controlled: Path,
    dnn: Path,
    equilibrium_v2: Path,
    locked: Path,
    postrelease: Path,
    historical: Path,
    equilibrium_v2_results: Path | None = None,
) -> dict[str, Any]:
    gt = reconcile_ground_truth(gt_results)
    ctl = reconcile_controlled(controlled)
    tiny = reconcile_dnn(dnn)
    eqv2 = reconcile_coverage_json(
        equilibrium_v2, family="overlapping_paper_equilibrium_v2"
    )
    results_path = equilibrium_v2_results
    if results_path is None:
        results_path = equilibrium_v2.parent / "results.jsonl"
    paper_results = reconcile_paper_results(results_path)
    eqv2["results_jsonl"] = paper_results
    eqv2["hedonic_verified"] = paper_results.get("hedonic_verified")
    eqv2["external_not_applicable"] = paper_results.get("external_not_applicable")
    eqv2["complete_certificates"] = bool(paper_results.get("complete_certificates"))
    eqv2["max_positive_regret"] = paper_results.get("max_positive_regret")
    eqv2["certificate_status_counts"] = paper_results.get("certificate_status_counts")
    locked_rep = reconcile_coverage_json(
        locked, family="locked_benchmark_historical_resource"
    )
    post = reconcile_coverage_json(
        postrelease, family="postrelease_protocol_audit_1.0.0.2"
    )
    hist = reconcile_coverage_json(historical, family="protocol_audit_historical_full")
    families = {
        "ground_truth_robustness_v3": gt,
        "controlled_overlap_v2": ctl,
        "dnn_certificate_v2": tiny,
        "overlapping_paper_equilibrium_v2": eqv2,
        "locked_benchmark_historical_resource": locked_rep,
        "postrelease_protocol_audit_1.0.0.2": post,
        "protocol_audit_historical_full": hist,
    }
    publication_125 = eqv2 if eqv2.get("present") else locked_rep
    return {
        "schema_version": SCHEMA_VERSION,
        "detector_free": True,
        "do_not_rewrite_producer_version_strings": True,
        "families": families,
        "publication_selection": {
            "gt_v3": {
                "expected": 3840,
                "observed": gt.get("observed"),
                "native_returns": gt.get("native_returns"),
                "independently_certified": gt.get("independently_certified"),
                "uncertified_native_returns": gt.get("uncertified_native_returns"),
                "registered_terminals": gt.get("registered_terminals"),
                "complete_key_coverage": gt.get("complete_key_coverage"),
            },
            "controlled_v2": {
                "expected": 864,
                "observed_runs": ctl.get("observed_runs"),
                "complete": ctl.get("complete"),
            },
            "paper_benchmark_125": {
                "selected_family": publication_125.get("family"),
                "expected": 125,
                "completed": publication_125.get("completed"),
                "hedonic_verified": publication_125.get("hedonic_verified"),
                "external_not_applicable": publication_125.get(
                    "external_not_applicable"
                ),
                "complete_certificates": publication_125.get("complete_certificates"),
                "max_positive_regret": publication_125.get("max_positive_regret"),
                "note": (
                    "Manuscript comparison uses the bounded equilibrium-v2 "
                    "125-condition ledger. 75 Hedonic rows are independent "
                    "exact-state certificates; 50 CPM/DEMON rows have no "
                    "equilibrium claim. Historical full and 1.0.0.2 "
                    "post-release audits are separate families and must not "
                    "be mixed into the v2 tables."
                ),
            },
            "tiny_dnn": {
                "expected_instances": 3,
                "observed_instances": tiny.get("observed_instances"),
                "complete": tiny.get("complete"),
            },
        },
        "readiness": {
            "gt_v3_accounted": bool(gt.get("complete_key_coverage")),
            "controlled_v2_accounted": bool(ctl.get("complete")),
            "tiny_dnn_accounted": bool(tiny.get("complete")),
            "selected_125_present": bool(publication_125.get("present")),
            "selected_125_certificates": bool(
                publication_125.get("complete_certificates")
            ),
            "no_silent_gt_exclusion": bool(
                gt.get("observed") == gt.get("expected")
                and gt.get("duplicate_keys") == 0
            ),
        },
    }


def coverage_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt = manifest["families"]["ground_truth_robustness_v3"]
    for group in gt.get("groups") or []:
        rows.append(
            {
                "family": "ground_truth_robustness_v3",
                "dataset": group["dataset"],
                "phase": group["phase"],
                "action_policy": group["action_policy"],
                "status": group["status"],
                "count": group["count"],
                "native_return": group["native_return"],
                "independently_certified": group["independently_certified"],
                "terminal": group["terminal"],
            }
        )
    ctl = manifest["families"]["controlled_overlap_v2"]
    for status, count in (ctl.get("status_counts") or {}).items():
        rows.append(
            {
                "family": "controlled_overlap_v2",
                "dataset": "",
                "phase": "",
                "action_policy": "",
                "status": status,
                "count": count,
                "native_return": status == "completed",
                "independently_certified": False,
                "terminal": status in {"timeout", "failed"},
            }
        )
    for family_name in (
        "overlapping_paper_equilibrium_v2",
        "locked_benchmark_historical_resource",
        "postrelease_protocol_audit_1.0.0.2",
        "protocol_audit_historical_full",
        "dnn_certificate_v2",
    ):
        family = manifest["families"][family_name]
        if family_name == "overlapping_paper_equilibrium_v2" and family.get(
            "complete_certificates"
        ):
            rows.append(
                {
                    "family": family_name,
                    "dataset": "",
                    "phase": "",
                    "action_policy": "",
                    "status": "hedonic_verified",
                    "count": family.get("hedonic_verified"),
                    "native_return": True,
                    "independently_certified": True,
                    "terminal": False,
                }
            )
            rows.append(
                {
                    "family": family_name,
                    "dataset": "",
                    "phase": "",
                    "action_policy": "",
                    "status": "external_not_applicable",
                    "count": family.get("external_not_applicable"),
                    "native_return": True,
                    "independently_certified": False,
                    "terminal": False,
                }
            )
            continue
        for status, count in (family.get("status_counts") or {}).items():
            rows.append(
                {
                    "family": family_name,
                    "dataset": "",
                    "phase": "",
                    "action_policy": "",
                    "status": status,
                    "count": count,
                    "native_return": status == "completed",
                    "independently_certified": False,
                    "terminal": status
                    in {"timeout", "oom", "memory_limit", "unsupported_cleanup"},
                }
            )
        if family_name == "dnn_certificate_v2":
            rows.append(
                {
                    "family": family_name,
                    "dataset": "",
                    "phase": "",
                    "action_policy": "",
                    "status": "instances",
                    "count": family.get("observed_instances") or 0,
                    "native_return": False,
                    "independently_certified": True,
                    "terminal": False,
                }
            )
    return rows


def _dot(value: float, places: int) -> str:
    text = f"{float(value):.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _pct(value: float, places: int) -> str:
    return rf"{100.0 * float(value):.{places}f}\%"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _student_interval(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = _mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    half = _TCRIT[n] * se
    return mean, mean - half, mean + half


def _controlled_matching_cells(
    payload: dict[str, Any],
) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], list[float]]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    node_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for instance in payload["instances"]:
        for run in instance["runs"]:
            if run.get("status") != "completed":
                continue
            if str(run.get("max_memberships_spec")) != "2":
                continue
            if float(run.get("resolution_multiplier") or 0) != 1.0:
                continue
            phase = "local" if run["local_move_only"] else "multi-phase"
            key = (str(run["initialization"]), phase)
            buckets[key].append(float(run["matching_f1"]))
            node_buckets[key].append(float(run["node_micro_f1"]))
    return buckets, node_buckets


def _controlled_paired_deltas(payload: dict[str, Any]) -> dict[str, Any]:
    graph_deltas: list[float] = []
    seed_local: dict[int, list[float]] = defaultdict(list)
    seed_multi: dict[int, list[float]] = defaultdict(list)
    res_deltas: list[float] = []
    for instance in payload["instances"]:
        seed = int(instance["construction"]["seed_requested"])
        local = multi = res1 = res10 = None
        for run in instance["runs"]:
            if run.get("status") != "completed":
                continue
            if str(run.get("max_memberships_spec")) != "2":
                continue
            if (
                run["initialization"] == "singleton"
                and float(run.get("resolution_multiplier") or 0) == 1.0
            ):
                if run["local_move_only"]:
                    local = float(run["matching_f1"])
                else:
                    multi = float(run["matching_f1"])
            if (
                run["initialization"] == "neutral-disjoint"
                and not run["local_move_only"]
            ):
                multiplier = float(run.get("resolution_multiplier") or 0)
                if multiplier == 1.0:
                    res1 = float(run["matching_f1"])
                elif multiplier == 10.0:
                    res10 = float(run["matching_f1"])
        if local is None or multi is None or res1 is None or res10 is None:
            continue
        delta = multi - local
        graph_deltas.append(delta)
        seed_local[seed].append(local)
        seed_multi[seed].append(multi)
        res_deltas.append(res10 - res1)
    seed_deltas = [
        _mean(seed_multi[seed]) - _mean(seed_local[seed]) for seed in sorted(seed_local)
    ]
    graph_mean, graph_lo, graph_hi = _student_interval(graph_deltas)
    seed_mean, seed_lo, seed_hi = _student_interval(seed_deltas)
    return {
        "n_graphs": len(graph_deltas),
        "phase_delta": graph_mean,
        "phase_ci_graph_low": graph_lo,
        "phase_ci_graph_high": graph_hi,
        "phase_ci_seed_low": seed_lo,
        "phase_ci_seed_high": seed_hi,
        "resolution_delta": _mean(res_deltas),
    }


def _load_paper_summary_rows(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["dataset"], row["method"])] = row
    return rows


def _metadata_overlap_by_dataset(results_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in _iter_jsonl(results_path):
        dataset = str(record["dataset"])
        if dataset in values:
            continue
        values[dataset] = float(record["gt_overlapping_node_fraction"])
        if len(values) == len(_DATASET_ORDER):
            break
    return values


def _dnn_display_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    exact_hits = 0
    trials = 0
    best_of_three = 0
    for result in payload["results"]:
        name = result["instance"]["name"]
        chain = result["certificate_chain"]
        exact = float(chain["exact_valid_cover_objective"])
        best = float(chain["best_algorithm_objective"])
        alg_gap = float(chain["algorithm_to_exact_gap"])
        outer = float(chain["valid_cover_to_repaired_dual_outer_gap"])
        candidates = int(result["exact_valid_cover_optimum"]["candidate_count"])
        bound = float(chain["repaired_dual_upper_bound"])
        hits = sum(
            1
            for run in result["algorithm_runs"]
            if float(run.get("algorithm_to_exact_gap") or 0) == 0.0
        )
        exact_hits += hits
        trials += len(result["algorithm_runs"])
        if alg_gap == 0.0:
            best_of_three += 1
        rows.append(
            {
                "name": name,
                "label": _DNN_DISPLAY[name],
                "candidates": candidates,
                "exact": exact,
                "best": best,
                "alg_gap": alg_gap,
                "outer": outer,
                "bound": bound,
            }
        )
    return {
        "rows": rows,
        "exact_hits": exact_hits,
        "trials": trials,
        "best_of_three": best_of_three,
        "n_instances": len(rows),
    }


def _gt_analysis_units(gt_root: Path) -> list[dict[str, Any]]:
    directory = gt_root / "ground_truth"
    if not directory.is_dir():
        return []
    order = {name: index for index, name in enumerate(_DATASET_ORDER)}
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        policies = payload.get("policies") or {}
        rows.append(
            {
                "dataset": str(payload["dataset"]),
                "n_vertices": int(payload["n_vertices"]),
                "n_edges": int(payload["n_edges"]),
                "n_communities": int(
                    policies["fixed_labels"]["at_density"]["n_communities"]
                ),
                "cap": int(payload["effective_max_memberships"]),
                "overlap": float(
                    payload["reference"]["accuracy"]["gt_overlapping_node_fraction"]
                ),
                "fixed": float(
                    policies["fixed_labels"]["endpoint"]["robust_fraction_gamma_0_1"]
                ),
                "open": float(
                    policies["open_labels"]["endpoint"]["robust_fraction_gamma_0_1"]
                ),
            }
        )
    rows.sort(key=lambda row: order.get(row["dataset"], 99))
    return rows


def _perturbation_ranges(gt_results: Path) -> dict[float, tuple[float, float]]:
    if not gt_results.is_file():
        return {}
    buckets: dict[float, list[float]] = defaultdict(list)
    for record in _iter_jsonl(gt_results):
        target = record.get("perturbation_target_distance")
        distance = record.get("perturbation_distance")
        if target is None or distance is None:
            continue
        target_f = float(target)
        if target_f <= 0:
            continue
        buckets[round(target_f, 5)].append(float(distance))
    return {target: (min(values), max(values)) for target, values in buckets.items()}


def write_displayed_manuscript_tables(
    *,
    paper_summary: Path,
    paper_results: Path,
    controlled: Path,
    dnn: Path,
    output_dirs: Iterable[Path],
    gt_root: Path | None = None,
    gt_results: Path | None = None,
) -> dict[str, Path]:
    """Write TeX fragments whose numbers are computed from locked ledgers."""
    destinations = [path for path in output_dirs if path is not None]
    if not (
        paper_summary.is_file()
        and paper_results.is_file()
        and controlled.is_file()
        and dnn.is_file()
    ):
        return {}
    gt_root = gt_root or DEFAULT_GT_RESULTS.parent
    gt_results = gt_results or DEFAULT_GT_RESULTS
    summary = _load_paper_summary_rows(paper_summary)
    metadata_overlap = _metadata_overlap_by_dataset(paper_results)
    payload = _load_json(controlled)
    matching, node_micro = _controlled_matching_cells(payload)
    contrasts = _controlled_paired_deltas(payload)
    dnn_info = _dnn_display_rows(_load_json(dnn))
    gt_units = _gt_analysis_units(gt_root)
    perturb = _perturbation_ranges(gt_results)

    macros: list[str] = [
        "% Generated by hedonic-exp overlapping-certificate-reconcile. Do not edit.",
        rf"\newcommand{{\DNNExactHits}}{{{dnn_info['exact_hits']}}}",
        rf"\newcommand{{\DNNExactTrials}}{{{dnn_info['trials']}}}",
        rf"\newcommand{{\DNNBestOfThree}}{{{dnn_info['best_of_three']}}}",
        rf"\newcommand{{\DNNInstanceCount}}{{{dnn_info['n_instances']}}}",
    ]
    for row in dnn_info["rows"]:
        prefix = {
            "path4": "DNNPath",
            "bow_tie5": "DNNBow",
            "weighted_bridge5": "DNNBridge",
        }[row["name"]]
        macros.append(rf"\newcommand{{\{prefix}Exact}}{{{_dot(row['exact'], 6)}}}")
        macros.append(rf"\newcommand{{\{prefix}Best}}{{{_dot(row['best'], 6)}}}")
        macros.append(rf"\newcommand{{\{prefix}Bound}}{{{_dot(row['bound'], 6)}}}")
        macros.append(rf"\newcommand{{\{prefix}OuterGap}}{{{_dot(row['outer'], 6)}}}")
        macros.append(
            rf"\newcommand{{\{prefix}OuterPct}}{{{_pct(row['outer'] / row['exact'], 3)}}}"
        )
        macros.append(rf"\newcommand{{\{prefix}Candidates}}{{{row['candidates']:,}}}")
    for dataset in _DATASET_ORDER:
        label = _DATASET_LABEL[dataset].replace(" ", "")
        hedonic = float(
            summary[(dataset, "hedonic_multiphase_x100")][
                "mean_predicted_overlapping_node_fraction"
            ]
        )
        macros.append(
            rf"\newcommand{{\HedonicXHundredOverlap{label}}}{{{_pct(hedonic, 2)}}}"
        )
        macros.append(
            rf"\newcommand{{\MetadataOverlap{label}}}{{{_pct(metadata_overlap[dataset], 1)}}}"
        )
        f1 = float(summary[(dataset, "hedonic_multiphase")]["mean_matching_f1"])
        macros.append(rf"\newcommand{{\BenchmarkMatchingFOne{label}}}{{{_dot(f1, 4)}}}")
    start_macros = {
        ("singleton", "local"): "ControlledSingletonLocal",
        ("singleton", "multi-phase"): "ControlledSingletonMulti",
        ("neutral-disjoint", "local"): "ControlledNeutralLocal",
        ("neutral-disjoint", "multi-phase"): "ControlledNeutralMulti",
        ("gt-primary", "local"): "ControlledGTLocal",
        ("gt-primary", "multi-phase"): "ControlledGTMulti",
    }
    for key, prefix in start_macros.items():
        macros.append(rf"\newcommand{{\{prefix}Match}}{{{_mean(matching[key]):.4f}}}")
        macros.append(rf"\newcommand{{\{prefix}Node}}{{{_mean(node_micro[key]):.4f}}}")
    macros.extend(
        [
            rf"\newcommand{{\ControlledSingletonPhaseDelta}}{{{contrasts['phase_delta']:.4f}}}",
            rf"\newcommand{{\ControlledSingletonPhaseCILow}}{{{contrasts['phase_ci_graph_low']:.4f}}}",
            rf"\newcommand{{\ControlledSingletonPhaseCIHigh}}{{{contrasts['phase_ci_graph_high']:.4f}}}",
            rf"\newcommand{{\ControlledNeutralResDelta}}{{{abs(contrasts['resolution_delta']):.4f}}}",
            rf"\newcommand{{\ControlledGraphCount}}{{{contrasts['n_graphs']}}}",
        ]
    )
    for row in gt_units:
        label = _DATASET_LABEL[row["dataset"]].replace(" ", "")
        macros.append(
            rf"\newcommand{{\GT{label}FixedRobust}}{{{_dot(row['fixed'], 3)}}}"
        )
        macros.append(rf"\newcommand{{\GT{label}OpenRobust}}{{{_dot(row['open'], 3)}}}")
    perturb_names = {
        0.005: "Half",
        0.02: "Two",
        0.05: "Five",
    }
    for target, name in perturb_names.items():
        if target not in perturb:
            continue
        low, high = perturb[target]
        macros.append(rf"\newcommand{{\GTPerturb{name}Low}}{{{100.0 * low:.4f}}}")
        macros.append(rf"\newcommand{{\GTPerturb{name}High}}{{{100.0 * high:.4f}}}")

    dnn_lines = [
        "% Generated by hedonic-exp overlapping-certificate-reconcile. Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Tiny exact/DNN calibration.  ``Algorithm gap'' compares the best",
        r"detector return with the exact valid-cover optimum; ``outer gap'' compares the",
        r"exact optimum with the repaired DNN upper bound.}",
        r"\label{tab:dnn}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Instance & candidates & exact & best & alg. gap & outer gap\\",
        r"\midrule",
    ]
    for row in dnn_info["rows"]:
        if abs(row["outer"]) < 1e-10:
            outer = r"$<10^{-10}$"
        else:
            outer = _dot(row["outer"], 6)
        alg = "0" if row["alg_gap"] == 0.0 else _dot(row["alg_gap"], 6)
        dnn_lines.append(
            " & ".join(
                [
                    row["label"],
                    f"{row['candidates']:,}",
                    _dot(row["exact"], 6),
                    _dot(row["best"], 6),
                    alg,
                    outer,
                ]
            )
            + r" \\"
        )
    dnn_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    controlled_lines = [
        "% Generated by hedonic-exp overlapping-certificate-reconcile. Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selected paired means from the 864-cell LFR-derived controlled",
        r"diagnostic at numeric cap two and density.  GT-primary uses the supplied cover and is",
        r"therefore supervised.}",
        r"\label{tab:controlled}",
        r"\scriptsize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Start & phase & matching $F_1$ & node-micro $F_1$\\",
        r"\midrule",
    ]
    for init, phase, label in _CONTROLLED_ROWS:
        key = (init, phase)
        controlled_lines.append(
            " & ".join(
                [
                    label,
                    phase,
                    _dot(_mean(matching[key]), 4),
                    _dot(_mean(node_micro[key]), 4),
                ]
            )
            + r" \\"
        )
    controlled_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\parbox{0.95\linewidth}{\scriptsize $^\dagger$ Uses the supplied cover and is supervised.}",
            r"\end{table}",
            "",
        ]
    )

    bench_lines = [
        "% Generated by hedonic-exp overlapping-certificate-reconcile. Do not edit.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{One-to-one matching $F_1$ on the five induced analysis graphs.  H$\times r$",
        r"denotes multi-phase Hedonic at $\gamma=r\,d(G)$; CPM is clique percolation,",
        r"not the fractional objective.  Values are means over five seeds.}",
        r"\label{tab:benchmark-f1}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & H$\times1$ & H$\times10$ & H$\times100$ & CPM & DEMON\\",
        r"\midrule",
    ]
    for dataset in _DATASET_ORDER:
        cells = [_DATASET_LABEL[dataset]]
        for method in _BENCHMARK_METHODS:
            cells.append(_dot(float(summary[(dataset, method)]["mean_matching_f1"]), 4))
        bench_lines.append(" & ".join(cells) + r" \\")
    bench_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    files = {
        "displayed_result_macros.tex": "\n".join(macros) + "\n",
        "dnn_calibration_table.tex": "\n".join(dnn_lines),
        "controlled_slice_table.tex": "\n".join(controlled_lines),
        "benchmark_matching_f1_table.tex": "\n".join(bench_lines),
    }
    if gt_units:
        unit_lines = [
            "% Generated by hedonic-exp overlapping-certificate-reconcile. Do not edit.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Locked primary analysis units.  Counts are after deterministic",
            r"bounded induction and common projection; metadata groups are the canonical",
            r"unique member sets.}",
            r"\label{tab:datasets}",
            r"\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Dataset & $n$ & $m$ & groups & $M$ & GT overlap\\",
            r"\midrule",
        ]
        for row in gt_units:
            unit_lines.append(
                " & ".join(
                    [
                        _DATASET_LABEL[row["dataset"]],
                        f"{row['n_vertices']:,}",
                        f"{row['n_edges']:,}",
                        f"{row['n_communities']:,}",
                        str(row["cap"]),
                        f"{row['overlap']:.3f}",
                    ]
                )
                + r" \\"
            )
        unit_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        files["analysis_units_table.tex"] = "\n".join(unit_lines)
    written: dict[str, Path] = {}
    for directory in destinations:
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            path = directory / name
            path.write_text(text, encoding="utf-8")
            written[name] = path
    return written


def classify_frozen_protocol_lock(
    path: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compare a frozen protocol lock to the live tree without rewriting it."""

    root = root or REPO_ROOT
    info = _file_record(path)
    if not path.is_file():
        return {
            **info,
            "present": False,
            "ok": False,
            "lock_bytes_match_freeze": False,
            "unexpected_drift": [],
            "allowed_wrapper_drift": [],
            "missing": [str(path)],
            "matching": [],
            "note": "frozen lock file is absent",
        }
    payload = _load_json(path)
    protocol_name = str(payload.get("protocol_name") or "")
    expected_lock = FROZEN_LOCK_SHA256.get(protocol_name)
    allowed = POST_LOCK_WRAPPER_FILES.get(protocol_name, frozenset())
    matching: list[str] = []
    allowed_drift: list[dict[str, str]] = []
    unexpected: list[dict[str, str]] = []
    missing: list[str] = []
    for relative, expected in (payload.get("tracked_files") or {}).items():
        live = root / relative
        if not live.is_file():
            missing.append(relative)
            continue
        actual = hashlib.sha256(live.read_bytes()).hexdigest()
        if actual == expected:
            matching.append(relative)
            continue
        row = {
            "path": relative,
            "lock_sha256": expected,
            "live_sha256": actual,
        }
        if relative in allowed:
            allowed_drift.append(row)
        else:
            unexpected.append(row)
    lock_bytes_match = bool(expected_lock) and info["sha256"] == expected_lock
    ok = lock_bytes_match and not missing and not unexpected
    return {
        **info,
        "present": True,
        "protocol_name": protocol_name,
        "expected_lock_sha256": expected_lock,
        "lock_bytes_match_freeze": lock_bytes_match,
        "matching": matching,
        "allowed_wrapper_drift": allowed_drift,
        "unexpected_drift": unexpected,
        "missing": missing,
        "do_not_rewrite_lock": True,
        "ok": ok,
        "note": (
            "Historical producer identity stays frozen. Live wrapper files in "
            "allowed_wrapper_drift postdate the lock; other tracked files must match."
        ),
    }


def verify_dnn_dual_witnesses(
    path: Path,
    *,
    reenumerate_exact: bool = True,
) -> dict[str, Any]:
    """Re-check frozen tiny DNN duals from stored y/S and the locked instances.

    This does not call CVXPY/SCS.  Arithmetic is floating-point, matching the
    ledger qualification, not interval-arithmetic.
    """

    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "ok": False, "instances": []}
    import numpy as np

    from hedonic.experiments.overlapping.dnn_certificate import (
        LOCKED_INSTANCES,
        LockedInstance,
        adjacency_matrix,
        cover_objective,
        exact_valid_cover_optimum,
        instance_sha256,
    )

    payload = _load_json(path)
    reports: list[dict[str, Any]] = []
    all_ok = True
    for result in payload.get("results") or []:
        raw_instance = result.get("instance") or {}
        name = str(raw_instance.get("name") or "")
        locked = LOCKED_INSTANCES.get(name)
        dnn = result.get("dnn") or {}
        chain = result.get("certificate_chain") or {}
        dual = dnn.get("dual_upper_checks") or {}
        exact_payload = result.get("exact_valid_cover_optimum") or {}
        row: dict[str, Any] = {"name": name, "ok": False}
        if locked is None:
            row["error"] = "unknown instance"
            reports.append(row)
            all_ok = False
            continue
        reconstructed = LockedInstance(
            name=str(raw_instance["name"]),
            n_vertices=int(raw_instance["n_vertices"]),
            edges=tuple(tuple(edge) for edge in raw_instance["edges"]),
            edge_weights=tuple(
                float(weight) for weight in raw_instance["edge_weights"]
            ),
            vertex_weights=tuple(
                float(weight) for weight in raw_instance["vertex_weights"]
            ),
            resolution=float(raw_instance["resolution"]),
            max_labels=int(raw_instance["max_labels"]),
            max_memberships=int(raw_instance["max_memberships"]),
        )
        stored_sha = result.get("instance_sha256")
        live_sha = instance_sha256(locked)
        json_sha = instance_sha256(reconstructed)
        spec_ok = stored_sha == live_sha and json_sha == live_sha
        memberships = exact_payload.get("memberships_by_vertex") or []
        stored_exact = float(chain.get("exact_valid_cover_objective"))
        recomputed_exact, _, _ = cover_objective(locked, memberships)
        enumerated = None
        if reenumerate_exact:
            enumerated = float(exact_valid_cover_optimum(locked)["objective"])
        total_edge_weight = float(sum(locked.edge_weights))
        vertex_weights = np.asarray(locked.vertex_weights, dtype=float)
        coefficient = (
            adjacency_matrix(locked)
            - locked.resolution * np.outer(vertex_weights, vertex_weights)
        ) / (2.0 * total_edge_weight)
        gram = np.asarray(dnn.get("gram"), dtype=float)
        y_value = np.asarray(dual.get("raw_diagonal_dual"), dtype=float)
        s_value = np.maximum(
            np.asarray(dual.get("raw_nonnegative_dual_clipped"), dtype=float),
            0.0,
        )
        n = locked.n_vertices
        residual = np.diag(y_value) - coefficient - s_value
        raw_min = float(np.min(np.linalg.eigvalsh(residual)))
        roundoff_guard = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(residual, ord=2)))
        )
        recomputed_shift = float(max(0.0, -raw_min) + roundoff_guard)
        stored_shift = float(dual.get("uniform_repair_shift"))
        repaired_z = residual + stored_shift * np.eye(n)
        repaired_min = float(np.min(np.linalg.eigvalsh(repaired_z)))
        stored_bound = float(dual.get("repaired_dual_upper_bound"))
        independent_bound = float(np.sum(y_value) + n * stored_shift)
        primal = float(np.sum(coefficient * gram))
        gram_min = float(np.min(np.linalg.eigvalsh(gram)))
        diag_error = float(np.max(np.abs(np.diag(gram) - 1.0)))
        entrywise_min = float(np.min(gram))
        tolerance = float(chain.get("verification_tolerance") or 1e-7)
        exact_matches_stored = abs(recomputed_exact - stored_exact) <= tolerance
        enumerated_matches = (
            enumerated is None or abs(enumerated - stored_exact) <= tolerance
        )
        bound_matches = abs(independent_bound - stored_bound) <= 1e-12
        dual_feasible = (
            repaired_min >= -tolerance and float(np.min(s_value)) >= -tolerance
        )
        exact_le_bound = stored_exact <= stored_bound + tolerance
        primal_le_bound = primal <= stored_bound + tolerance
        chain_bound = float(chain.get("repaired_dual_upper_bound"))
        ok = all(
            (
                spec_ok,
                exact_matches_stored,
                enumerated_matches,
                bound_matches,
                dual_feasible,
                exact_le_bound,
                primal_le_bound,
                abs(chain_bound - stored_bound) <= 1e-12,
                gram_min >= -tolerance,
                diag_error <= 1e-8,
                entrywise_min >= -tolerance,
            )
        )
        row.update(
            {
                "ok": ok,
                "instance_sha256_matches_locked_spec": spec_ok,
                "stored_exact_objective": stored_exact,
                "recomputed_exact_from_memberships": recomputed_exact,
                "reenumerated_exact_objective": enumerated,
                "stored_repaired_dual_upper_bound": stored_bound,
                "independent_repaired_dual_upper_bound": independent_bound,
                "primal_gram_objective": primal,
                "stored_repair_shift": stored_shift,
                "recomputed_repair_shift": recomputed_shift,
                "repaired_psd_minimum_eigenvalue": repaired_min,
                "gram_minimum_eigenvalue": gram_min,
                "gram_diagonal_max_error": diag_error,
                "qualification": dual.get("qualification"),
                "exact_le_repaired_dual": exact_le_bound,
            }
        )
        reports.append(row)
        all_ok = all_ok and ok
    return {
        **info,
        "present": True,
        "ok": all_ok and len(reports) == 3,
        "expected_instances": 3,
        "observed_instances": len(reports),
        "instances": reports,
        "qualification": (
            "floating-point feasible-dual check of stored SCS witnesses; "
            "not interval-arithmetic or a claim of general tightness"
        ),
    }


def verify_tex_pdf_snapshot(
    snapshot_path: Path = DEFAULT_TEX_PDF_SNAPSHOT,
    tex_path: Path = DEFAULT_MAIN_TEX,
    pdf_path: Path = DEFAULT_MAIN_PDF,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    """Require the snapshot hashes to match the live TeX/PDF pair (TKT-09)."""

    import re

    snapshot = _file_record(snapshot_path)
    tex = _file_record(tex_path)
    pdf = _file_record(pdf_path)
    source_manifest = _file_record(source_manifest_path)
    text = snapshot_path.read_text(encoding="utf-8") if snapshot_path.is_file() else ""
    tex_match = re.search(
        r"`docs/papers/overlapping_communities/main\.tex`\s*\|\s*`([0-9a-f]{64})`",
        text,
    )
    pdf_match = re.search(
        r"`docs/papers/overlapping_communities/main\.pdf`\s*\|\s*`([0-9a-f]{64})`",
        text,
    )
    page_match = re.search(r"`page_count=(\d+)`", text)
    overfull_match = re.search(r"`overfull_boxes=(\d+)`", text)
    undefined_match = re.search(r"`undefined_references=(\d+)`", text)
    snapshot_tex = tex_match.group(1) if tex_match else None
    snapshot_pdf = pdf_match.group(1) if pdf_match else None
    declared_page_count = int(page_match.group(1)) if page_match else None
    edition = {}
    if source_manifest_path.is_file():
        payload = _load_json(source_manifest_path)
        edition = payload.get("edition_map") or {}
    checks = {
        "snapshot_present": snapshot_path.is_file(),
        "tex_present": tex_path.is_file(),
        "pdf_present": pdf_path.is_file(),
        "snapshot_tex_matches_live": snapshot_tex == tex.get("sha256"),
        "snapshot_pdf_matches_live": snapshot_pdf == pdf.get("sha256"),
        "source_manifest_tex_matches_live": edition.get("current_main_tex_sha256")
        == tex.get("sha256"),
        "source_manifest_pdf_matches_live": edition.get("pdf_sha256")
        == pdf.get("sha256"),
        "positive_declared_page_count": bool(
            declared_page_count is not None and declared_page_count > 0
        ),
        "no_overfull_boxes": bool(
            overfull_match is not None and int(overfull_match.group(1)) == 0
        ),
        "no_undefined_references": bool(
            undefined_match is not None and int(undefined_match.group(1)) == 0
        ),
    }
    return {
        "snapshot": snapshot,
        "tex": tex,
        "pdf": pdf,
        "source_manifest": source_manifest,
        "snapshot_tex_sha256": snapshot_tex,
        "snapshot_pdf_sha256": snapshot_pdf,
        "declared_page_count": declared_page_count,
        "checks": checks,
        "ok": all(checks.values()),
    }


def verify_wrapper_contract() -> dict[str, Any]:
    """Executable TKT-06 truth table for the published wrapper."""

    import inspect

    import igraph as ig

    from hedonic import Game
    from hedonic.Game import (
        HEDONIC_ALGORITHM_IDENTITY,
        HISTORICAL_ALGORITHM_IDENTITIES,
    )

    params = inspect.signature(Game.community_hedonic).parameters
    graph = Game(ig.Graph([(0, 1)], directed=False))
    rows: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        rows.append({"name": name, "ok": ok, "detail": detail})

    record(
        "ensure_equilibrium_absent",
        "ensure_equilibrium" not in params and "local_move_only" in params,
    )
    record(
        "n_iterations_default_negative",
        params["n_iterations"].default == -1,
    )
    record(
        "frozen_producer_identity_published",
        HISTORICAL_ALGORITHM_IDENTITIES.get("lucas-igraph-1.0.0.3")
        == EXPECTED_HEDONIC_ALGORITHM_IDENTITY,
        f"live wrapper identity: {HEDONIC_ALGORITHM_IDENTITY}",
    )
    empty = Game(ig.Graph(n=2))
    try:
        empty.community_hedonic(max_memberships=1, resolution=0.5)
        record("reject_W0", False, "did not raise")
    except ValueError:
        record("reject_W0", True)
    try:
        graph.community_hedonic(ensure_equilibrium=True)
        record("reject_ensure_equilibrium", False, "did not raise")
    except TypeError:
        record("reject_ensure_equilibrium", True)
    cover = graph.community_hedonic(
        resolution=1.0,
        max_memberships=2,
        n_iterations=-1,
        local_move_only=True,
    )
    record(
        "stamps_live_identity",
        getattr(cover, "_hedonic_algorithm_identity", None)
        == HEDONIC_ALGORITHM_IDENTITY,
    )
    record(
        "raw_memberships_on_negative_iterations",
        hasattr(cover, "_hedonic_raw_memberships")
        and len(cover._hedonic_raw_memberships) == 2,
    )
    nested = graph._as_overlapping_init([0, 0])
    record("flat_nested_init", nested == [[0], [0]])
    ok = all(row["ok"] for row in rows)
    return {"ok": ok, "rows": rows}


def verify_metric_fixtures() -> dict[str, Any]:
    """Executable TKT-08 matching / micro / canonicalization fixtures."""

    from hedonic.experiments.overlapping.metrics import (
        SCORE_DEFINITION_VERSION,
        node_membership_multilabel_metrics,
        one_to_one_community_metrics,
        symmetric_best_match_f1,
    )

    matching = one_to_one_community_metrics(
        [[0, 1]], [[0, 1], [2, 3]], matching_weight="jaccard"
    )
    micro = node_membership_multilabel_metrics([[0, 1], [2]], [[0, 1], [1, 2]])
    duplicate = one_to_one_community_metrics([[0, 1], [0, 1]], [[0, 1]])
    rows = [
        {
            "name": "score_definition_version",
            "ok": SCORE_DEFINITION_VERSION == "canonical_unique_vertex_set_cover_v1",
        },
        {
            "name": "unmatched_gt_jaccard_f1",
            "ok": abs(float(matching["matching_f1"]) - 2.0 / 3.0) < 1e-12,
            "value": matching["matching_f1"],
        },
        {
            "name": "node_micro_f1",
            "ok": abs(float(micro["node_micro_f1"]) - 6.0 / 7.0) < 1e-12,
            "value": micro["node_micro_f1"],
        },
        {
            "name": "canonical_duplicate_bodies_collapse",
            "ok": abs(float(duplicate["matching_f1"]) - 1.0) < 1e-12
            and abs(symmetric_best_match_f1([[0, 1], [0, 1]], [[0, 1]]) - 1.0) < 1e-12,
        },
    ]
    return {"ok": all(row["ok"] for row in rows), "rows": rows}


def verify_native_differential_artifact(
    path: Path = DEFAULT_NATIVE_DIFFERENTIAL,
) -> dict[str, Any]:
    """Require the frozen CE1 native differential not to certify the trap."""

    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "ok": False, "records": 0}
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    ce1 = [
        row
        for row in records
        if row.get("case") == "ce1_weighted" and row.get("local_move_only") is True
    ]
    escaped = all(
        row.get("escaped_trap") is True
        and row.get("native_certified_false_on_trap") is True
        for row in ce1
    )
    oracles_agree = all(
        row.get("oracles_agree") is True for row in records if "oracles_agree" in row
    )
    ok = bool(ce1) and escaped and oracles_agree and len(records) >= 8
    return {
        **info,
        "present": True,
        "ok": ok,
        "records": len(records),
        "ce1_weighted_local_rows": len(ce1),
        "escaped_trap": escaped,
        "oracles_agree": oracles_agree,
    }


def verify_membership_replay_summary(
    path: Path = DEFAULT_REPLAY_SUMMARY,
) -> dict[str, Any]:
    """Require frozen replay accounting without rewriting ledgers."""

    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "ok": False}
    payload = _load_json(path)
    gt = payload.get("ground_truth_robustness_v3") or {}
    paper = payload.get("overlapping_paper_equilibrium_v2") or {}
    dnn = payload.get("dnn") or {}
    checks = {
        "gt_agreements": gt.get("agreements") == 3835,
        "gt_disagreements": gt.get("disagreements") == 0,
        "gt_skipped_terminal": gt.get("skipped_terminal") == 5,
        "paper_agreements": paper.get("agreements") == 75,
        "paper_disagreements": paper.get("disagreements") == 0,
        "dnn_agreements": dnn.get("agreements") == 3,
        "dnn_disagreements": dnn.get("disagreements") == 0,
        "no_rewrite": payload.get("rewrites_frozen_ledgers") is False,
    }
    return {**info, "present": True, "ok": all(checks.values()), "checks": checks}


def verify_fee_v1_artifact(
    path: Path = DEFAULT_FEE_V1,
    *,
    recomputed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the separately versioned TKT-16 report against a fresh run."""

    from hedonic.experiments.overlapping.unit_l2_oracle import fee_v1_check

    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "ok": False}
    payload = _load_json(path)
    expected = recomputed or fee_v1_check()
    identities = payload.get("identities") or {}
    expected_identities = expected.get("identities") or {}
    stored_rows = (payload.get("ablation") or {}).get("rows") or []
    expected_rows = (expected.get("ablation") or {}).get("rows") or []

    def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
        path_report = row.get("path") or {}
        r10 = row.get("r10") or {}
        return (
            row.get("tau"),
            row.get("mean_memberships"),
            row.get("state"),
            path_report.get("ok"),
            path_report.get("accepted_moves"),
            r10.get("full_equilibrium", True),
            r10.get("fresh_private_singleton_available", True),
            r10.get("cardinality_bound", True),
        )

    checks = {
        "objective_version": payload.get("objective_version") == "unit-l2-cpm-fee-v1",
        "separate_from_manuscript": payload.get("not_the_manuscript_model") is True,
        "does_not_reuse_certificates": payload.get(
            "does_not_reuse_original_certificates"
        )
        is True,
        "report_passed": payload.get("ok") is True,
        "checks_match_recomputation": payload.get("checks") == expected.get("checks"),
        "identity_residuals_within_tolerance": (
            identities.get("cases") == expected_identities.get("cases")
            and identities.get("seed") == expected_identities.get("seed")
            and max(
                float(identities.get("max_delta_error", math.inf)),
                float(identities.get("max_pair_delta_error", math.inf)),
                float(identities.get("max_prefix_error", math.inf)),
            )
            < 1e-10
        ),
        "units_match_recomputation": payload.get("units", {}).get("checks")
        == expected.get("units", {}).get("checks"),
        "ablation_match_recomputation": [row_signature(row) for row in stored_rows]
        == [row_signature(row) for row in expected_rows],
    }
    return {**info, "present": True, "checks": checks, "ok": all(checks.values())}


def verify_rational_dnn_duals(
    path: Path = DEFAULT_RATIONAL_DNN,
) -> dict[str, Any]:
    """Independently reconstruct the versioned exact rational DNN duals."""

    from hedonic.experiments.overlapping.dnn_rational_verify import (
        verify_manifest_payload,
    )

    info = _file_record(path)
    if not path.is_file():
        return {**info, "present": False, "ok": False}
    verification = verify_manifest_payload(_load_json(path))
    return {**info, "present": True, **verification}


def verify_gate_h_reader_path(
    *,
    manifest: dict[str, Any],
    dnn_path: Path,
    rational_dnn_path: Path = DEFAULT_RATIONAL_DNN,
    tex_fragment_dir: Path,
    public_release_path: Path = DEFAULT_PUBLIC_RELEASE,
    online_release_check: bool = False,
    paper_lock: Path = PAPER_PROTOCOL_LOCK,
    gt_lock: Path = GT_PROTOCOL_LOCK,
    leiden_archive: Path = LEIDEN_PIN_ARCHIVE,
    reenumerate_exact: bool = True,
) -> dict[str, Any]:
    """Machine-check the reader path that Gate H requires."""

    from hedonic.Game import (
        HEDONIC_ALGORITHM_IDENTITY,
        HISTORICAL_ALGORITHM_IDENTITIES,
    )

    paper_lock_report = classify_frozen_protocol_lock(paper_lock)
    gt_lock_report = classify_frozen_protocol_lock(gt_lock)
    duals = verify_dnn_dual_witnesses(dnn_path, reenumerate_exact=reenumerate_exact)
    rational_duals = verify_rational_dnn_duals(rational_dnn_path)
    from hedonic.experiments.overlapping.unit_l2_oracle import (
        fee_v1_check,
        proof_reconstruction,
    )

    proofs = proof_reconstruction()
    fee_v1 = fee_v1_check()
    fee_v1_artifact = verify_fee_v1_artifact(recomputed=fee_v1)
    snapshot = verify_tex_pdf_snapshot()
    public_release = verify_public_release_provenance(
        public_release_path, online=online_release_check
    )
    wrapper = verify_wrapper_contract()
    metric_fixtures = verify_metric_fixtures()
    native_diff = verify_native_differential_artifact()
    replay_summary = verify_membership_replay_summary()
    families = manifest.get("families") or {}
    gt = families.get("ground_truth_robustness_v3") or {}
    controlled = families.get("controlled_overlap_v2") or {}
    paper = families.get("overlapping_paper_equilibrium_v2") or {}
    ledgers = {
        "gt_3840": {
            "observed": gt.get("observed"),
            "certified": gt.get("independently_certified"),
            "uncertified_native_returns": gt.get("uncertified_native_returns"),
            "terminals": gt.get("registered_terminals"),
            "ok": (
                gt.get("observed") == 3840
                and gt.get("independently_certified") == 1995
                and gt.get("uncertified_native_returns") == 1840
                and gt.get("registered_terminals") == 5
            ),
        },
        "controlled_864": {
            "summary_n_completed": controlled.get("summary_n_completed"),
            "complete": controlled.get("complete"),
            "not_nash_certificates": True,
            "ok": controlled.get("complete") is True
            and controlled.get("summary_n_completed") == 864,
        },
        "paper_125": {
            "hedonic_verified": paper.get("hedonic_verified"),
            "external_not_applicable": paper.get("external_not_applicable"),
            "ok": (
                paper.get("hedonic_verified") == 75
                and paper.get("external_not_applicable") == 50
            ),
        },
        "tiny_dnn": duals,
    }
    macros_path = tex_fragment_dir / "displayed_result_macros.tex"
    macros = macros_path.read_text(encoding="utf-8") if macros_path.is_file() else ""
    fragment_ok = all(
        (
            macros_path.is_file(),
            r"\newcommand{\ControlledSingletonPhaseDelta}{0.1864}" in macros,
            r"\newcommand{\ControlledSingletonPhaseCILow}{0.1365}" in macros,
            r"\newcommand{\GTPerturbTwoLow}{1.9499}" in macros,
            (tex_fragment_dir / "dnn_calibration_table.tex").is_file(),
            (tex_fragment_dir / "controlled_slice_table.tex").is_file(),
            (tex_fragment_dir / "benchmark_matching_f1_table.tex").is_file(),
            (tex_fragment_dir / "analysis_units_table.tex").is_file(),
        )
    )
    leiden_sha = _sha256(leiden_archive)
    frozen_identity = HISTORICAL_ALGORITHM_IDENTITIES.get("lucas-igraph-1.0.0.3")
    identity = {
        "algorithm": frozen_identity,
        "algorithm_ok": frozen_identity == EXPECTED_HEDONIC_ALGORITHM_IDENTITY,
        "live_algorithm": HEDONIC_ALGORITHM_IDENTITY,
        "live_wrapper_is_frozen_producer": HEDONIC_ALGORITHM_IDENTITY
        == EXPECTED_HEDONIC_ALGORITHM_IDENTITY,
        "leiden_c_sha256": leiden_sha,
        "leiden_c_ok": leiden_sha == EXPECTED_LEIDEN_C_SHA256,
    }
    checks = {
        "paper_lock": paper_lock_report["ok"],
        "gt_lock": gt_lock_report["ok"],
        "dnn_dual_witnesses": duals["ok"],
        "rational_dnn_dual_witnesses": rational_duals["ok"],
        "ledgers_gt": ledgers["gt_3840"]["ok"],
        "ledgers_controlled": ledgers["controlled_864"]["ok"],
        "ledgers_paper": ledgers["paper_125"]["ok"],
        "displayed_fragments": fragment_ok,
        "algorithm_identity": identity["algorithm_ok"],
        "leiden_pin": identity["leiden_c_ok"],
        "short_proofs": proofs["ok"],
        "tex_pdf_snapshot": snapshot["ok"],
        "wrapper_contract": wrapper["ok"],
        "metric_fixtures": metric_fixtures["ok"],
        "native_differential": native_diff["ok"],
        "membership_replay_summary": replay_summary["ok"],
        "fee_v1": fee_v1["ok"],
        "fee_v1_artifact": fee_v1_artifact["ok"],
        "public_release_provenance": public_release["ok"],
    }
    declared_limitations = [
        "No named human-coauthor sign-off is claimed; the retained short proofs have an independent executable reconstruction.",
        "Overlapping interruption is outside the supported wrapper API; the frozen pin leaves 16 FINALLY entries under the ASan interrupt fixture.",
        "The frozen 1.0.0.3 C collision ABI remains boolean; the published 1.0.0.4 opt-in diagnostic ABI records exact collision counts, while original-space projection guards and independent collision-list fixtures remain authoritative.",
        "Darwin LeakSanitizer was unavailable and the published 1.0.0.3 wheels were not sanitizer builds; the published 1.0.0.4 artifacts were not sanitizer builds either.",
        "PyPI publishes lucas-igraph 1.0.0.4 (nine ABI3 wheels plus sdist) and hedonic 0.1.1 (wheel plus sdist); those releases are a current public identity, not producers of the frozen 1.0.0.2 ledgers or 1.0.0.3 replay.",
        "A new protocol lock and post-release rerun remain required before attributing any displayed result to lucas-igraph 1.0.0.4/hedonic 0.1.1.",
        "TKT-04 published-wheel validation covers macOS arm64 only and found a directed-copy regression in hedonic 0.1.1; code fix ddef924e68f0cf0278c93dfe46c8a07ecc508047 and public-safe 0.1.2 release metadata 358f5fe are prepared on astra/tkt04-direction-fix but remain unmerged and unpublished. Linux, Windows, macOS x86_64, and uploaded-wheel sanitizer checks remain open.",
        "TKT-07 v4 source scan accounts for 4,302 raw-membership artifacts and 3,840 rows; the bounded pilot audits 4 GT states and 8 controlled cells. A corrected streamed-slice memory qualification completed 4/4 cap-9/cap-51 cases with 25,739,264-byte maximum fresh-child ru_maxrss, but Darwin rejected hard RLIMIT_AS/DATA lowering; the full 1,840 plus 864 successor workloads remain unlaunched and require a new full-size protocol/authorization.",
        "TKT-10 has an executable and adversarial finite proof audit, but no named human-coauthor sign-off is claimed.",
        "TKT-11–14 have runnable fail-closed scaffolds and bounded smokes; the authors' LFRbenchmarks source is commit-pinned and passes a one-graph native pilot, while the full programs remain unlaunched and outside the manuscript claims.",
        "TKT-17 has a semantics-locked cached oracle path and report (3.20x median on the registered case); the rebuilding reference path remains default and no broad runtime claim is made.",
        "TKT-19 now has a separate public-allocation-v0.1 mechanism prototype and exhaustive information-loss fixtures; it contributes no data or result to this manuscript and makes no fairness, truthfulness, equilibrium, or empirical-benefit claim.",
        "Live wrapper/dependency tree hashes postdate the frozen 125-condition producer lock; historical lock bytes are intentionally preserved.",
        "The live wrapper identity (community_hedonic/lucas-igraph-1.0.0.5) is not the frozen 1.0.0.3 producer; Gate H verifies the frozen producer string the package still publishes, and results of the live wrapper are not evidence for the frozen ledgers.",
        "The 864 controlled cells are detector outcomes, not Nash certificates.",
        "TKT-11–14 remain outside the manuscript claim set; their fail-closed runners and bounded smokes do not substitute for the unlaunched full programs.",
        "TKT-17 retains the slower checked reference as the default; the opt-in cached path has strict equivalence evidence and a scoped benchmark, not a broad runtime-complexity claim.",
    ]
    ok = all(checks.values())
    blocking_failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "ok": ok,
        "checks": checks,
        "locks": {"paper": paper_lock_report, "ground_truth": gt_lock_report},
        "ledgers": ledgers,
        "displayed_fragments": {
            "tex_fragment_dir": str(tex_fragment_dir),
            "ok": fragment_ok,
        },
        "identity": identity,
        "proofs": proofs,
        "tex_pdf_snapshot": snapshot,
        "wrapper_contract": wrapper,
        "metric_fixtures": metric_fixtures,
        "native_differential": native_diff,
        "membership_replay_summary": replay_summary,
        "fee_v1": fee_v1,
        "fee_v1_artifact": fee_v1_artifact,
        "public_release": public_release,
        "rational_dnn_dual_witnesses": rational_duals,
        "blocking_failures": blocking_failures,
        "declared_limitations": declared_limitations,
        "residuals": declared_limitations,
        "gate_h_complete": ok and not blocking_failures,
        "note": (
            "Gate H is complete when every executable reader-path check passes. "
            "Declared limitations are accepted fallback boundaries from the plan, "
            "not hidden certificates or unsupported claims."
        ),
    }


def write_outputs(manifest: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "status_coverage.csv"
    json_path = output_dir / "certificate_manifest.json"
    rows = coverage_rows(manifest)
    fieldnames = [
        "family",
        "dataset",
        "phase",
        "action_policy",
        "status",
        "count",
        "native_return",
        "independently_certified",
        "terminal",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status_coverage_csv": csv_path, "certificate_manifest": json_path}


def attach_existing_membership_replay(
    manifest: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Keep prior replay results when reconcile is detector-free."""
    if manifest.get("membership_replay"):
        return manifest
    json_path = output_dir / "certificate_manifest.json"
    summary_path = output_dir / "membership_replay_summary.json"
    previous = None
    if json_path.is_file():
        previous = json.loads(json_path.read_text(encoding="utf-8")).get(
            "membership_replay"
        )
    if previous is None and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
    if previous is not None:
        manifest["membership_replay"] = previous
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detector-free classification of the 3840/864/125/tiny overlapping "
            "ledgers into status_coverage.csv and certificate_manifest.json"
        )
    )
    parser.add_argument("--gt-results", type=Path, default=DEFAULT_GT_RESULTS)
    parser.add_argument("--controlled", type=Path, default=DEFAULT_CONTROLLED)
    parser.add_argument("--dnn", type=Path, default=DEFAULT_DNN)
    parser.add_argument(
        "--rational-dnn",
        type=Path,
        default=DEFAULT_RATIONAL_DNN,
        help="versioned exact rational DNN companion verified by Gate H",
    )
    parser.add_argument("--equilibrium-v2", type=Path, default=DEFAULT_EQV2)
    parser.add_argument(
        "--equilibrium-v2-results",
        type=Path,
        default=None,
        help="125-row results.jsonl (default: sibling of --equilibrium-v2)",
    )
    parser.add_argument("--locked", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--postrelease", type=Path, default=DEFAULT_POSTRELEASE)
    parser.add_argument("--historical-audit", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="directory for status_coverage.csv and certificate_manifest.json",
    )
    parser.add_argument(
        "--replay-gt",
        action="store_true",
        help=(
            "re-audit persisted GT-v3 memberships with audit_cover; "
            "does not rewrite the frozen ledger"
        ),
    )
    parser.add_argument(
        "--replay-dnn",
        action="store_true",
        help="recompute tiny DNN exact covers with the independent unit-ℓ₂ oracle",
    )
    parser.add_argument(
        "--replay-paper",
        action="store_true",
        help=(
            "re-audit persisted equilibrium-v2 Hedonic memberships against "
            "local shard analysis graphs; does not rewrite the frozen ledger"
        ),
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=DEFAULT_EQV2.parent,
        help="equilibrium-v2 artifact root (shards/, results.jsonl)",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_RESULTS.parent,
        help="ground-truth-v3 artifact root (runs/, raw_memberships/, results.jsonl)",
    )
    parser.add_argument("--networks-dir", type=Path, default=None)
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=0,
        help="optional stratified cap on GT replay rows; 0 means all pending rows",
    )
    parser.add_argument(
        "--no-replay-resume",
        action="store_true",
        help="rewrite membership_replay.jsonl instead of appending",
    )
    parser.add_argument("--max-nodes", type=int, default=3000)
    parser.add_argument(
        "--paper-summary",
        type=Path,
        default=DEFAULT_PAPER_SUMMARY,
        help="locked equilibrium-v2 paper_summary.csv used for displayed tables",
    )
    parser.add_argument(
        "--tex-fragment-dir",
        type=Path,
        default=DEFAULT_TEX_FRAGMENT_DIR,
        help="directory for generated manuscript table/macro fragments",
    )
    parser.add_argument(
        "--public-release",
        type=Path,
        default=DEFAULT_PUBLIC_RELEASE,
        help="immutable PyPI release-observation receipt used by Gate H",
    )
    parser.add_argument(
        "--online-release-check",
        action="store_true",
        help=(
            "when used with --gate-h-check, re-fetch the pinned PyPI JSON "
            "endpoints and compare every published filename and SHA-256"
        ),
    )
    parser.add_argument(
        "--no-tex-fragments",
        action="store_true",
        help="skip writing displayed TeX fragments from locked ledgers",
    )
    parser.add_argument(
        "--gate-h-check",
        action="store_true",
        help=(
            "after classification, verify frozen locks, independent DNN dual "
            "witnesses, ledger counts, algorithm identity, displayed TeX, "
            "short proofs, wrapper/metric contracts, the TeX/PDF snapshot "
            "hashes, and the separate fee-v1 identities; exit 1 if those "
            "checks fail. Does not rewrite protocol locks."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(
        gt_results=args.gt_results,
        controlled=args.controlled,
        dnn=args.dnn,
        equilibrium_v2=args.equilibrium_v2,
        equilibrium_v2_results=args.equilibrium_v2_results,
        locked=args.locked,
        postrelease=args.postrelease,
        historical=args.historical_audit,
    )
    attach_existing_membership_replay(manifest, args.output_dir)
    paths = write_outputs(manifest, args.output_dir)
    summary: dict[str, Any] = {
        "status_coverage_csv": str(paths["status_coverage_csv"]),
        "certificate_manifest": str(paths["certificate_manifest"]),
        "readiness": manifest["readiness"],
        "publication_selection": manifest["publication_selection"],
    }
    if not args.no_tex_fragments:
        fragments = write_displayed_manuscript_tables(
            paper_summary=args.paper_summary,
            paper_results=args.equilibrium_v2_results
            or (args.equilibrium_v2.parent / "results.jsonl"),
            controlled=args.controlled,
            dnn=args.dnn,
            gt_root=args.gt_root,
            gt_results=args.gt_results,
            output_dirs=(args.tex_fragment_dir, args.output_dir),
        )
        if fragments:
            summary["displayed_tex_fragments"] = {
                name: str(path) for name, path in fragments.items()
            }
    if args.replay_gt or args.replay_dnn or args.replay_paper:
        from hedonic.experiments.overlapping.membership_replay import (
            run_membership_replay,
        )

        replay = run_membership_replay(
            gt_root=args.gt_root,
            dnn_path=args.dnn,
            output_dir=args.output_dir,
            networks_dir=args.networks_dir,
            replay_gt=bool(args.replay_gt),
            replay_dnn=bool(args.replay_dnn),
            replay_paper=bool(args.replay_paper),
            paper_root=args.paper_root,
            paper_results=args.equilibrium_v2_results
            or (args.equilibrium_v2.parent / "results.jsonl"),
            limit=args.replay_limit or None,
            resume=not args.no_replay_resume,
            max_nodes=args.max_nodes,
        )
        manifest["membership_replay"] = replay
        paths["certificate_manifest"].write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        summary["membership_replay"] = replay
    if args.gate_h_check:
        gate = verify_gate_h_reader_path(
            manifest=manifest,
            dnn_path=args.dnn,
            rational_dnn_path=args.rational_dnn,
            tex_fragment_dir=args.tex_fragment_dir,
            public_release_path=args.public_release,
            online_release_check=args.online_release_check,
        )
        gate_path = args.output_dir / "gate_h_check.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        summary["gate_h_check"] = {
            "ok": gate["ok"],
            "checks": gate["checks"],
            "path": str(gate_path),
            "gate_h_complete": gate["gate_h_complete"],
            "residuals": gate["residuals"],
        }
        print(json.dumps(summary, indent=2))
        return 0 if gate["ok"] else 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
