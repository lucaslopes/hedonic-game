"""Pinned, experiment-only overlapping baselines (Astra TKT-13).

The public benchmark needs a fair *manageable* comparator set without making
optional packages a hidden prerequisite.  This module therefore provides
small, inspectable adapters for SLPA, DEMON, KCP, and a scoped Chen-style
set-valued-game replica, together with trivial/disjoint controls.  Each
adapter declares its implementation identity, parameter budget, coverage
convention, and whether a dependency was available.  A missing package is
represented as an explicit ``MethodUnavailable``/``failed`` outcome by
callers; no singleton completion or ground-truth cap is silently injected.
Local SLPA/DEMON replicas are compatibility smoke fixtures only.  Standard
baseline runs fail closed unless a maintained external implementation is
actually imported, so a replica can never be reported as a pinned algorithm.
"""

from __future__ import annotations

import contextlib
import argparse
import hashlib
import importlib.metadata
import io
import inspect
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import igraph as ig

from hedonic.experiments.overlapping.metrics import partition_to_cover_lists
from hedonic.experiments.overlapping.overlap_lfr import (
    OFFICIAL_LFRBENCHMARKS_COMMIT,
    OFFICIAL_LFRBENCHMARKS_TREE,
    condition_grid,
    environment_receipt,
    graph_sha256,
)
from hedonic.experiments.overlapping.robustness import canonicalize_cover


BASELINE_PROTOCOL_VERSION = "baseline-set-v2"
BASELINE_TUNING_PROTOCOL_VERSION = "baseline-tuning-v1"

# This is the prospective test configuration.  It is deliberately a plain
# JSON-compatible object so the exact bytes can be hashed into every launch
# receipt and result row *before* any detector is executed.  The values are
# method parameters, not values selected after looking at a graph's planted
# cover or its result.
BASELINE_PROSPECTIVE_TUNING: dict[str, Any] = {
    "protocol_version": BASELINE_TUNING_PROTOCOL_VERSION,
    "selection_policy": "frozen_before_test_scoring",
    "methods": {
        "slpa": {"iterations": 20, "threshold": 0.10},
        "demon": {"epsilon": 0.25, "min_community_size": 2},
        "kcp": {"clique_size": 3, "max_communities": 50_000},
        "cpm": {"max_memberships": 1, "local_move_only": False},
        "chen": {
            "iterations": 8,
            "membership_cost": 0.0,
            "include_self": False,
            "action_space_scope": "current_singleton_current_plus_one_label",
            "complete_set_valued_best_response": False,
        },
        "disjoint": {"max_memberships": 1, "local_move_only": True},
        "singleton": {},
        "grand_coalition": {},
        "components": {},
    },
    "stochastic_methods": ["slpa", "demon", "chen"],
    "deterministic_methods": ["kcp", "cpm", "disjoint", "singleton", "grand_coalition", "components"],
    "seed_plan": (
        "stochastic methods use optimizer seeds 0,1,2 in standard and 0 in smoke; "
        "deterministic methods use optimizer seed 0; detector seed is the low "
        "31 bits of sha256(graph_hash:graph_seed:method:optimizer_seed)"
    ),
    "information_budget": "graph topology only; no planted cover, cap tuning, or warm start",
}
# DEMON and NetworkX are installed in the main experiments environment.  CDlib
# is deliberately isolated: its dependency on ``python-igraph`` would install
# a second distribution providing the top-level ``igraph`` module and could
# overwrite the project's lucas-igraph extension.  The dedicated environment
# under ``tools/slpa_env`` is locked separately and invoked through its JSON
# worker, so the standard SLPA comparator is pinned without changing the core
# runtime environment.
PINNED_EXTERNAL_VERSIONS: dict[str, str | None] = {
    "demon": "2.0.6",
    "networkx": "3.6.1",
    "cdlib": "0.4.0",
}
SLPA_PROTOCOL_VERSION = "cdlib-slpa-isolated-v1"
SLPA_DIRECT_PINS: dict[str, str] = {
    "cdlib": "0.4.0",
    "networkx": "3.6.1",
    "numpy": "2.3.3",
    "python-igraph": "1.0.0",
}


class MethodUnavailable(RuntimeError):
    """Raised when an external baseline cannot be executed."""


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_pinned(distribution: str) -> str:
    actual = _installed_version(distribution)
    expected = PINNED_EXTERNAL_VERSIONS.get(distribution)
    if expected is None:
        raise MethodUnavailable(f"{distribution} has no version pin in the experiment lock")
    if actual != expected:
        raise MethodUnavailable(
            f"{distribution} version {actual or 'missing'} is not pinned to {expected}"
        )
    return actual


@dataclass(frozen=True)
class BaselineAdapter:
    name: str
    family: str
    implementation: str
    parameters: dict[str, Any]
    stochastic: bool
    information_budget: str
    runner: Callable[[ig.Graph, int, float, int, dict[str, Any]], list[list[int]]]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with the protocol's canonical encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably append one checkpoint event.

    Baseline rows can be expensive or unavailable, so losing a whole manifest
    at the end of a run is not an acceptable persistence model.  Each event is
    fsynced before the next detector invocation; terminal JSON is only a
    materialized, reload-checked view of these append-only streams.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read baseline checkpoint {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid baseline checkpoint {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"baseline checkpoint {path}:{line_number} is not an object")
        rows.append(dict(value))
    return rows


def _checkpoint_paths(destination: Path) -> tuple[Path, Path]:
    """Return stable append-only row and graph sidecars for a manifest path."""
    return (
        destination.with_name(destination.name + ".rows.jsonl"),
        destination.with_name(destination.name + ".graphs.jsonl"),
    )


def _baseline_row_key(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    try:
        graph_hash = str(row["graph_hash"])
        method = str(row["method"])
        # v1 manifests used the detector seed as their only seed field.  Keep
        # those manifests reloadable while v2 rows bind optimizer_seed and the
        # derived detector seed separately.
        optimizer_seed = int(row.get("optimizer_seed", row.get("seed", 0)))
    except (KeyError, TypeError, ValueError):
        return None
    if not graph_hash or not method:
        return None
    return graph_hash, method, optimizer_seed


def _baseline_graph_key(record: Mapping[str, Any]) -> str | None:
    value = record.get("graph_hash")
    return str(value) if value else None


def _method_seed(*, graph_hash: str, graph_seed: int, method: str, optimizer_seed: int) -> int:
    encoded = f"{graph_hash}:{int(graph_seed)}:{method}:{int(optimizer_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & 0x7FFFFFFF


def _condition_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return (
            abs(float(left["mixing"]) - float(right["mixing"])) <= 1e-12
            and abs(float(left["overlap_fraction"]) - float(right["overlap_fraction"])) <= 1e-12
            and int(left.get("overlap_multiplicity", 1)) == int(right.get("overlap_multiplicity", 1))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _resolve_archive(ledger_path: Path, record: Mapping[str, Any]) -> Path:
    graph_hash = str(record.get("graph_hash") or "")
    if not graph_hash:
        raise ValueError("canonical baseline graph record has no graph_hash")
    value = record.get("graph_path") or record.get("archive")
    archive = Path(str(value)).expanduser() if value else Path("graphs") / f"{graph_hash}.json"
    if not archive.is_absolute():
        archive = ledger_path.parent / archive
    return archive.resolve()


def load_baseline_graph_ledger(
    ledger: str | Path,
    *,
    profile: str = "standard",
    max_graphs: int | None = None,
) -> list[dict[str, Any]]:
    """Load and strictly verify the canonical TKT-11 graph ledger.

    Standard baseline rows must consume the graph archives generated by the
    official LFRbenchmarks adapter.  This loader validates the source policy,
    every graph/cover hash, archive schema, condition/index identity, and the
    full 28 × 30 registered grid when no bounded ``max_graphs`` is requested.
    A bounded prefix is allowed only as a disposable pilot/recovery fixture;
    it still validates the complete input JSON before selecting that prefix.
    """
    profile = str(profile).lower()
    if profile not in {"pilot", "standard"}:
        raise ValueError("canonical baseline ledger loading requires pilot or standard profile")
    if max_graphs is not None and int(max_graphs) < 1:
        raise ValueError("max_graphs must be positive when supplied")
    ledger_path = Path(ledger).expanduser().resolve()
    if not ledger_path.is_file():
        raise ValueError(f"canonical baseline graph ledger is missing: {ledger_path}")
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load canonical baseline graph ledger {ledger_path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("canonical baseline graph ledger must be a non-empty JSON list")

    conditions = condition_grid()
    loaded: list[dict[str, Any]] = []
    seen_condition_keys: set[tuple[int, int]] = set()
    seen_graph_hashes: set[str] = set()
    for position, record_value in enumerate(payload):
        if not isinstance(record_value, dict):
            raise ValueError(f"canonical baseline graph record {position} is not an object")
        record = dict(record_value)
        status = str(record.get("status", "completed"))
        if status != "completed":
            raise ValueError(
                f"canonical baseline graph ledger contains non-completed record {position}: {status}"
            )
        condition = record.get("condition")
        if not isinstance(condition, dict):
            raise ValueError(f"canonical baseline graph record {position} has no condition object")
        condition_index = next(
            (index for index, expected in enumerate(conditions) if _condition_matches(condition, expected)),
            None,
        )
        if condition_index is None:
            raise ValueError(f"canonical baseline graph record {position} has an unregistered condition")
        try:
            declared_condition_index = int(record.get("condition_index", condition_index))
            graph_index = int(record["graph_index_within_condition"])
            graph_seed = int(record["graph_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"canonical baseline graph record {position} has invalid indices/seeds") from exc
        if declared_condition_index != condition_index or graph_index < 0:
            raise ValueError(f"canonical baseline graph record {position} condition/index mismatch")
        condition_key = (condition_index, graph_index)
        if condition_key in seen_condition_keys:
            raise ValueError(f"duplicate canonical baseline graph condition/index: {condition_key}")
        seen_condition_keys.add(condition_key)

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"canonical baseline graph record {position} has no metadata receipt")
        policy = str(record.get("generator_policy") or metadata.get("generator_policy") or "")
        if policy != "official_binary_required":
            raise ValueError(
                "canonical baseline graph ledger is not an official LFRbenchmarks record: "
                f"condition={condition_index} graph={graph_index} policy={policy!r}"
            )
        if str(metadata.get("official_source_commit") or "") != OFFICIAL_LFRBENCHMARKS_COMMIT:
            raise ValueError("canonical baseline graph source commit does not match the pinned LFRbenchmarks commit")
        if str(metadata.get("official_source_tree") or "") != OFFICIAL_LFRBENCHMARKS_TREE:
            raise ValueError("canonical baseline graph source tree does not match the pinned LFRbenchmarks tree")
        if metadata.get("canonical_evidence") is not True or metadata.get("metadata_free_detection") is not True:
            raise ValueError("canonical baseline graph is not marked metadata-free canonical evidence")
        if metadata.get("cover_is_detector_input") is not False:
            raise ValueError("canonical baseline graph cover is not explicitly held out from detector input")

        graph_hash_value = str(record.get("graph_hash") or "")
        cover_hash_value = str(record.get("cover_hash") or "")
        if not graph_hash_value or not cover_hash_value:
            raise ValueError(f"canonical baseline graph record {position} has missing graph/cover hash")
        if graph_hash_value in seen_graph_hashes:
            raise ValueError(f"duplicate canonical baseline graph hash: {graph_hash_value}")
        archive = _resolve_archive(ledger_path, record)
        if not archive.is_file():
            raise ValueError(f"canonical baseline graph archive is missing: {archive}")
        try:
            archive_payload = json.loads(archive.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load canonical baseline graph archive {archive}") from exc
        if not isinstance(archive_payload, dict):
            raise ValueError(f"canonical baseline graph archive must be an object: {archive}")
        archive_metadata = archive_payload.get("metadata")
        if not isinstance(archive_metadata, dict):
            raise ValueError(f"canonical baseline graph archive has no metadata: {archive}")
        try:
            n_vertices = int(archive_metadata["n"])
            raw_edges = archive_payload["edges"]
            raw_cover = archive_payload["cover"]
            if n_vertices < 2 or not isinstance(raw_edges, list) or not isinstance(raw_cover, list):
                raise ValueError("n, edges, or cover has invalid type")
            edges: list[tuple[int, int]] = []
            edge_keys: set[tuple[int, int]] = set()
            for edge in raw_edges:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValueError("edge is not a two-element sequence")
                first, second = map(int, edge)
                if first == second or not (0 <= first < n_vertices and 0 <= second < n_vertices):
                    raise ValueError("edge endpoint is invalid")
                key = tuple(sorted((first, second)))
                if key in edge_keys:
                    raise ValueError("duplicate edge in graph archive")
                edge_keys.add(key)
                edges.append(key)
            cover = canonicalize_cover(raw_cover, n_vertices=n_vertices)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed canonical baseline graph archive {archive}: {exc}") from exc
        graph = ig.Graph(n=n_vertices, edges=edges, directed=False)
        actual_graph_hash = graph_sha256(graph)
        actual_cover_hash = _canonical_sha256(cover)
        # The project cover hash includes the vertex universe; import the
        # canonical helper instead of relying on archive ordering.
        from hedonic.experiments.overlapping.robustness import cover_hash

        actual_cover_hash = cover_hash(cover, n_vertices)
        if actual_graph_hash != graph_hash_value or actual_graph_hash != str(archive_metadata.get("graph_hash") or ""):
            raise ValueError(f"canonical baseline graph hash mismatch for {archive}")
        expected_cover_hashes = {
            cover_hash_value,
            str(archive_metadata.get("cover_hash") or ""),
            str(archive_payload.get("cover_hash") or ""),
        }
        expected_cover_hashes.discard("")
        if expected_cover_hashes and any(actual_cover_hash != expected for expected in expected_cover_hashes):
            raise ValueError(f"canonical baseline cover hash mismatch for {archive}")
        seen_graph_hashes.add(graph_hash_value)
        loaded.append(
            {
                "graph": graph,
                "cover": cover,
                "graph_id": f"m{float(condition['mixing']):.3g}_o{float(condition['overlap_fraction']):.3g}_k{int(condition.get('overlap_multiplicity', 1))}_g{graph_index:03d}",
                "condition": {key: condition[key] for key in ("mixing", "overlap_fraction", "overlap_multiplicity")},
                "condition_index": condition_index,
                "graph_index_within_condition": graph_index,
                "graph_seed": graph_seed,
                "graph_hash": graph_hash_value,
                "cover_hash": cover_hash_value,
                "archive": str(archive),
                "metadata": dict(archive_metadata),
                "source_record": record,
            }
        )

    loaded.sort(key=lambda row: (int(row["condition_index"]), int(row["graph_index_within_condition"])))
    if profile == "standard" and max_graphs is None:
        expected_per_condition = 30
        counts = {index: 0 for index in range(len(conditions))}
        for row in loaded:
            counts[int(row["condition_index"])] += 1
        missing = {index: count for index, count in counts.items() if count != expected_per_condition}
        if len(loaded) != len(conditions) * expected_per_condition or missing:
            raise ValueError(
                "canonical baseline graph ledger is incomplete for the registered 840-graph grid: "
                + json.dumps(missing, sort_keys=True)
            )
    if max_graphs is not None:
        loaded = loaded[: int(max_graphs)]
    if not loaded:
        raise ValueError("canonical baseline graph ledger contains no selected graphs")
    return loaded


def _load_tuning_config(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return json.loads(json.dumps(BASELINE_PROSPECTIVE_TUNING))
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load baseline tuning config {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("methods"), dict):
        raise ValueError("baseline tuning config must be an object with a methods object")
    if str(payload.get("protocol_version") or "") != BASELINE_TUNING_PROTOCOL_VERSION:
        raise ValueError(
            f"baseline tuning config protocol must be {BASELINE_TUNING_PROTOCOL_VERSION}"
        )
    return payload


def _method_plan(
    selected: Sequence[str],
    *,
    tuning_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    configured = tuning_config.get("methods")
    if not isinstance(configured, Mapping):
        raise ValueError("baseline tuning config methods must be an object")
    plan: dict[str, dict[str, Any]] = {}
    for method in selected:
        parameters = configured.get(str(method))
        if not isinstance(parameters, Mapping):
            raise ValueError(f"baseline tuning config has no parameters for {method}")
        plan[str(method)] = dict(parameters)
    return plan


def _optimizer_seed_plan(profile: str, optimizer_seeds: Sequence[int] | None) -> tuple[int, ...]:
    if optimizer_seeds is None:
        values = (0,) if profile == "smoke" else (0, 1, 2)
    else:
        values = tuple(int(value) for value in optimizer_seeds)
    if not values:
        raise ValueError("optimizer_seeds must contain at least one seed")
    if len(set(values)) != len(values):
        raise ValueError("optimizer_seeds must not contain duplicates")
    return values


def _method_optimizer_seeds(method: str, *, tuning_config: Mapping[str, Any], optimizer_seeds: Sequence[int]) -> tuple[int, ...]:
    stochastic = {str(item) for item in tuning_config.get("stochastic_methods", ())}
    return tuple(int(value) for value in optimizer_seeds) if method in stochastic else (0,)


def _baseline_row_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "failed"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _source_identity(function: Callable[..., Any]) -> dict[str, Any]:
    try:
        source = Path(inspect.getsourcefile(function) or "").resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return {"path": source.name, "sha256": digest}
    except (OSError, TypeError):
        return {"path": None, "sha256": None}


def _module_source_identity() -> dict[str, Any]:
    """Return the exact source identity for the baseline runner module.

    Baseline manifests already bind the interpreter and lock files through
    :func:`environment_receipt`, and bind prospective parameters through the
    tuning hash.  A standard preflight must also prove that the runner itself
    was the current one.  Keep the path absolute, matching the other local
    provenance fields in the receipt, and fail closed when it cannot be read.
    """
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _slpa_environment_root() -> Path:
    """Resolve the isolated SLPA environment without guessing another one."""
    configured = os.environ.get("HEDONIC_SLPA_ENV")
    if configured:
        return Path(configured).expanduser().resolve()
    # baselines.py -> overlapping -> experiments -> hedonic -> src -> repo
    repository_project = Path(__file__).resolve().parents[4] / "tools" / "slpa_env"
    if (repository_project / "pyproject.toml").is_file():
        return repository_project
    configured_cache = os.environ.get(
        "HEDONIC_CODESEG_CACHE", "~/.cache/hedonic/codeseg"
    )
    return Path(configured_cache).expanduser().resolve() / "cdlib_env"


def slpa_environment_receipt() -> dict[str, Any]:
    """Return the exact files/pins needed by the external SLPA worker.

    This is intentionally read-only.  A missing project, lock, worker, or
    both execution paths (the materialized project interpreter and ``uv``)
    is represented as ``ready=False``; callers must not infer that an
    unpinned ambient CDlib installation is suitable.
    """
    root = _slpa_environment_root()
    project = root / "pyproject.toml"
    lock = root / "uv.lock"
    worker = root / "slpa_worker.py"
    # Prefer an already materialized interpreter from this locked project.
    # ``uv run`` can be unable to read a host-level cache in restricted
    # environments even when the project's venv is complete.  The worker
    # verifies every direct pin after startup, so using this interpreter does
    # not turn an unpinned ambient CDlib import into evidence.
    project_python = root / ".venv" / "bin" / "python"
    python_ready = project_python.is_file() and os.access(project_python, os.X_OK)
    uv = shutil.which("uv")
    files = {
        "project": {"path": str(project), "sha256": _sha256_file(project)},
        "lock": {"path": str(lock), "sha256": _sha256_file(lock)},
        "worker": {"path": str(worker), "sha256": _sha256_file(worker)},
    }
    lock_is_materialized = False
    if lock.is_file():
        try:
            lock_is_materialized = not lock.read_text(encoding="utf-8").startswith(
                "# Generated by hedonic codeseg-setup"
            )
        except OSError:
            lock_is_materialized = False
    ready = bool(
        (python_ready or (uv and lock_is_materialized))
        and all(item["sha256"] for item in files.values())
    )
    return {
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "root": str(root),
        "uv": str(Path(uv).resolve()) if uv else None,
        "project_python": str(project_python) if python_ready else None,
        "execution_preference": (
            "project_venv"
            if python_ready
            else "uv_locked"
            if uv and lock_is_materialized
            else None
        ),
        "files": files,
        "direct_pins": dict(SLPA_DIRECT_PINS),
        "lock_is_materialized": lock_is_materialized,
        "ready": ready,
        "isolation": "dedicated uv project; never imported in the lucas-igraph process",
    }


def _last_json_line(stdout: str) -> dict[str, Any]:
    """Decode the worker's terminal JSON despite third-party import chatter."""
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise MethodUnavailable("isolated SLPA worker emitted no JSON result")


def _run_slpa_worker(
    graph: ig.Graph,
    *,
    iterations: int,
    threshold: float,
    seed: int,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Invoke CDlib 0.4.0 in the dedicated locked environment."""
    receipt = slpa_environment_receipt()
    if not receipt["ready"]:
        missing = [
            key for key, value in receipt["files"].items() if not value.get("sha256")
        ]
        if not receipt.get("uv"):
            missing.append("uv executable")
        raise MethodUnavailable(
            "pinned SLPA environment unavailable: " + ", ".join(missing)
        )
    root = Path(receipt["root"])
    worker = root / "slpa_worker.py"
    request = {
        "n": int(graph.vcount()),
        "edges": [[int(left), int(right)] for left, right in graph.get_edgelist()],
        "iterations": int(iterations),
        "threshold": float(threshold),
        "seed": int(seed),
    }
    project_python = receipt.get("project_python")
    if project_python:
        command = [str(project_python), str(worker)]
        execution_mode = "project_venv"
    else:
        command = [
            str(receipt["uv"]),
            "run",
            "--project",
            str(root),
            "--locked",
            "python",
            str(worker),
        ]
        execution_mode = "uv_locked"
    worker_environment = os.environ.copy()
    # CDlib imports matplotlib while loading optional algorithms.  Keep its
    # cache outside the repository and avoid failures caused by a read-only
    # host-level ~/.matplotlib directory.
    matplotlib_config = Path(tempfile.gettempdir()) / "hedonic-slpa-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    worker_environment["MPLCONFIGDIR"] = str(matplotlib_config)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=str(root.parent.parent),
            input=json.dumps(request, sort_keys=True),
            text=True,
            capture_output=True,
            env=worker_environment,
            timeout=float(timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MethodUnavailable(f"isolated SLPA worker could not complete: {exc}") from exc
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise MethodUnavailable(
            f"isolated SLPA worker exited {process.returncode}: "
            + (detail[-1] if detail else "no diagnostic")
        )
    payload = _last_json_line(process.stdout)
    communities = payload.get("communities")
    worker_metadata = payload.get("metadata")
    if not isinstance(communities, list) or not isinstance(worker_metadata, dict):
        raise MethodUnavailable("isolated SLPA worker returned an invalid payload")
    actual_versions = {
        "cdlib": worker_metadata.get("cdlib_version"),
        "networkx": worker_metadata.get("networkx_version"),
        "numpy": worker_metadata.get("numpy_version"),
        "python-igraph": worker_metadata.get("python_igraph_version"),
    }
    mismatches = {
        name: (actual_versions.get(name), expected)
        for name, expected in SLPA_DIRECT_PINS.items()
        if actual_versions.get(name) != expected
    }
    if mismatches:
        raise MethodUnavailable(f"SLPA worker dependency drift: {mismatches}")
    cover = canonicalize_cover(
        [sorted(int(vertex) for vertex in community) for community in communities],
        graph.vcount(),
    )
    return cover, {
        "method": "slpa",
        "family": "label_propagation",
        "implementation": "cdlib.algorithms.slpa",
        "implementation_kind": "external_pinned_adapter",
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "dependency": {
            "distribution": "cdlib",
            "version": actual_versions["cdlib"],
            "direct_pins": dict(SLPA_DIRECT_PINS),
            "lock_sha256": receipt["files"]["lock"]["sha256"],
        },
        "iterations": int(iterations),
        "threshold": float(threshold),
        "seed": int(seed),
        "seed_support": False,
        "rng_semantics": "numpy.random.seed(seed) in isolated worker",
        "graph_semantics": "undirected, unweighted NetworkX conversion; no planted cover/cap",
        "warm_start": False,
        "metadata_free": True,
        "availability": "external",
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "worker_seconds": elapsed,
        "worker_command": command,
        "execution_mode": execution_mode,
        "worker_source_identity": receipt["files"]["worker"],
        "environment_identity": receipt,
        "source_identity": _source_identity(_run_slpa_worker),
    }


def slpa_preflight(*, timeout_seconds: float = 90.0) -> dict[str, Any]:
    """Exercise the locked SLPA environment on a disposable four-cycle."""
    receipt = slpa_environment_receipt()
    result: dict[str, Any] = {
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "generated_at_unix": time.time(),
        "status": "blocked" if not receipt["ready"] else "ready",
        "environment": receipt,
        "fixture": {"n": 4, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]},
        "full_grid_launched": False,
        "recovery_contract": "canonical cover reload check is required before publication",
    }
    if not receipt["ready"]:
        result["reason"] = "locked SLPA project, worker, or executable is missing"
        return result
    graph = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)], directed=False)
    try:
        cover, metadata = _run_slpa_worker(
            graph,
            iterations=2,
            threshold=0.1,
            seed=7,
            timeout_seconds=timeout_seconds,
        )
    except MethodUnavailable as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)
    else:
        result["status"] = "completed"
        result["cover"] = cover
        result["metadata"] = metadata
        result["reload_check"] = bool(isinstance(cover, list) and cover == canonicalize_cover(cover, 4))
        result["recovery_check"] = {
            "cover_reloaded_and_canonicalized": result["reload_check"],
            "rows_persisted": False,
            "note": "baseline-study resume remains separately tested; this is disposable preflight only",
        }
    return result


def link_clustering_environment_receipt() -> dict[str, Any]:
    """Describe the isolated CDlib environment used by Link Communities.

    CDlib exposes Ahn--Bagrow--Lehmann Link Communities under the
    ``hierarchical_link_community`` name, not ``link_clustering``.  Keep that
    API detail explicit and reuse the same locked project as SLPA without
    importing CDlib into the lucas-igraph process.
    """
    receipt = slpa_environment_receipt()
    worker = Path(receipt["root"]) / "link_clustering_worker.py"
    files = dict(receipt.get("files", {}))
    files["link_clustering_worker"] = {
        "path": str(worker),
        "sha256": _sha256_file(worker),
    }
    ready = bool(receipt.get("ready") and files["link_clustering_worker"]["sha256"])
    return {
        **receipt,
        "files": files,
        "ready": ready,
        "worker": str(worker),
        "api": "cdlib.algorithms.hierarchical_link_community",
        "output_conversion": "edge communities projected to unique node sets",
    }


def _run_link_clustering_worker(
    graph: ig.Graph,
    *,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Invoke CDlib HLC in the isolated, pinned environment."""
    receipt = link_clustering_environment_receipt()
    if not receipt["ready"]:
        missing = [
            key for key, value in receipt["files"].items() if not value.get("sha256")
        ]
        if not receipt.get("uv") and not receipt.get("project_python"):
            missing.append("uv executable or project Python")
        raise MethodUnavailable(
            "pinned Link Clustering environment unavailable: " + ", ".join(missing)
        )
    root = Path(receipt["root"])
    worker = root / "link_clustering_worker.py"
    project_python = receipt.get("project_python")
    if project_python:
        command = [str(project_python), str(worker)]
        execution_mode = "project_venv"
    else:
        command = [
            str(receipt["uv"]),
            "run",
            "--project",
            str(root),
            "--locked",
            "python",
            str(worker),
        ]
        execution_mode = "uv_locked"
    worker_environment = os.environ.copy()
    matplotlib_config = Path(tempfile.gettempdir()) / "hedonic-slpa-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    worker_environment["MPLCONFIGDIR"] = str(matplotlib_config)
    request = {
        "n": int(graph.vcount()),
        "edges": [[int(left), int(right)] for left, right in graph.get_edgelist()],
    }
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=str(root.parent.parent),
            input=json.dumps(request, sort_keys=True),
            text=True,
            capture_output=True,
            env=worker_environment,
            timeout=float(timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MethodUnavailable(
            f"isolated Link Clustering worker could not complete: {exc}"
        ) from exc
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise MethodUnavailable(
            f"isolated Link Clustering worker exited {process.returncode}: "
            + (detail[-1] if detail else "no diagnostic")
        )
    payload = _last_json_line(process.stdout)
    communities = payload.get("communities")
    worker_metadata = payload.get("metadata")
    if not isinstance(communities, list) or not isinstance(worker_metadata, dict):
        raise MethodUnavailable("isolated Link Clustering worker returned an invalid payload")
    actual_versions = {
        "cdlib": worker_metadata.get("cdlib_version"),
        "networkx": worker_metadata.get("networkx_version"),
    }
    mismatches = {
        name: (actual_versions.get(name), expected)
        for name, expected in {
            "cdlib": PINNED_EXTERNAL_VERSIONS["cdlib"],
            "networkx": PINNED_EXTERNAL_VERSIONS["networkx"],
        }.items()
        if actual_versions.get(name) != expected
    }
    if mismatches:
        raise MethodUnavailable(f"Link Clustering worker dependency drift: {mismatches}")
    cover = canonicalize_cover(
        [sorted(int(vertex) for vertex in community) for community in communities],
        graph.vcount(),
    )
    return cover, {
        "method": "link_clustering",
        "family": "edge_link_communities",
        "implementation": "cdlib.algorithms.hierarchical_link_community",
        "implementation_kind": "external_pinned_adapter",
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "dependency": {
            "distribution": "cdlib",
            "version": actual_versions["cdlib"],
            "direct_pins": {
                "cdlib": PINNED_EXTERNAL_VERSIONS["cdlib"],
                "networkx": PINNED_EXTERNAL_VERSIONS["networkx"],
            },
            "lock_sha256": receipt["files"]["lock"]["sha256"],
        },
        "seed": None,
        "seed_support": False,
        "graph_semantics": "undirected, unweighted NetworkX conversion; no planted cover/cap",
        "warm_start": False,
        "metadata_free": True,
        "availability": "external",
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "worker_seconds": elapsed,
        "worker_command": command,
        "execution_mode": execution_mode,
        "worker_source_identity": receipt["files"]["link_clustering_worker"],
        "environment_identity": receipt,
        "source_identity": _source_identity(_run_link_clustering_worker),
        "output_conversion": worker_metadata.get("output_conversion"),
    }


def run_link_clustering(
    graph: ig.Graph,
    *,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run CDlib's pinned Ahn--Bagrow--Lehmann Link Communities adapter."""
    return _run_link_clustering_worker(graph, timeout_seconds=timeout_seconds)


def _networkx_graph(graph: ig.Graph):
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - experiments extra
        raise MethodUnavailable("networkx is not installed") from exc
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.vcount()))
    nx_graph.add_edges_from((int(a), int(b)) for a, b in graph.get_edgelist() if a != b)
    return nx_graph


def _run_native_disjoint(
    graph: ig.Graph,
    *,
    resolution: float | None,
    seed: int,
    local_move_only: bool,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run the native disjoint CPM/Leiden control on graph-only input.

    ``cpm`` and ``disjoint`` used to be aliases for NetworkX KCP, which made
    the prospective baseline table silently report a clique-percolation run
    under a Constant-Potts/disjoint name.  Keep KCP in its own adapter and
    expose the native cap-one reduction explicitly.  A singleton start is
    supplied by the wrapper, so neither the planted cover nor a post-hoc cap
    enters this control.
    """
    from hedonic import Game

    selected_resolution = float(graph.density() if resolution is None else resolution)
    started = time.perf_counter()
    result = Game(graph).community_hedonic(
        max_memberships=1,
        resolution=selected_resolution,
        local_move_only=bool(local_move_only),
        n_iterations=-1,
        allow_isolation=True,
        seed=int(seed),
    )
    elapsed = time.perf_counter() - started
    cover = partition_to_cover_lists(result)
    return cover, {
        "method": "disjoint" if local_move_only else "cpm",
        "family": "disjoint_hedonic" if local_move_only else "disjoint_cpm",
        "implementation": "hedonic.Game.community_hedonic",
        "implementation_kind": "native_pinned_adapter",
        "local_move_only": bool(local_move_only),
        "max_memberships": 1,
        "resolution": selected_resolution,
        "seed": int(seed),
        "n_iterations": -1,
        "allow_isolation": True,
        "warm_start": False,
        "metadata_free": True,
        "information_budget": "graph topology only; singleton initialization; no planted cover/cap tuning",
        "runtime_seconds": elapsed,
        "native_algorithm_identity": getattr(result, "_hedonic_algorithm_identity", None),
        "native_token_preflight": getattr(result, "_hedonic_token_preflight", None),
        "source_identity": _source_identity(_run_native_disjoint),
    }


def run_disjoint(
    graph: ig.Graph,
    *,
    resolution: float | None = None,
    seed: int = 0,
    local_move_only: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run a graph-only native cap-one disjoint control.

    ``local_move_only=True`` is the lightweight disjoint control.  Set it to
    ``False`` for the full native CPM/Leiden phase.  Both variants retain the
    strategic partition and never receive a planted cover or oracle cap.
    """
    return _run_native_disjoint(
        graph,
        resolution=resolution,
        seed=seed,
        local_move_only=bool(local_move_only),
    )


def _cover_from_label_rows(rows: Sequence[Sequence[int]]) -> list[list[int]]:
    communities: dict[int, list[int]] = {}
    for vertex, labels in enumerate(rows):
        for label in sorted({int(value) for value in labels}):
            if label < 0:
                continue
            communities.setdefault(label, []).append(vertex)
    return canonicalize_cover(list(communities.values()), n_vertices=len(rows))


def _run_external_slpa(
    graph: ig.Graph,
    *,
    iterations: int,
    threshold: float,
    seed: int,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Use CDlib 0.4.0 through the isolated, locked worker."""
    return _run_slpa_worker(
        graph,
        iterations=iterations,
        threshold=threshold,
        seed=seed,
        timeout_seconds=float(timeout_seconds),
    )


def run_slpa(
    graph: ig.Graph,
    *,
    iterations: int = 20,
    threshold: float = 0.10,
    seed: int = 0,
    allow_replica: bool = True,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run pinned external SLPA or an explicit smoke-only replica.

    The local fallback follows the usual SLPA memory rule: a listener samples
    one label from each neighbour proportional to that speaker's label memory,
    then stores the modal sampled label.  Labels with relative frequency below
    ``threshold`` are omitted.  No warm start or planted cover is accepted.
    Set ``allow_replica=False`` for standard/paper runs; missing or unsuitable
    CDlib support then raises :class:`MethodUnavailable`.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        return _run_external_slpa(
            graph,
            iterations=iterations,
            threshold=threshold,
            seed=seed,
            timeout_seconds=float(timeout_seconds),
        )
    except MethodUnavailable as exc:
        if not allow_replica:
            raise
        external_failure_reason = str(exc)
    rng = random.Random(int(seed))
    neighbours = [list(map(int, graph.neighbors(vertex))) for vertex in range(graph.vcount())]
    memories: list[dict[int, int]] = [{vertex: 1} for vertex in range(graph.vcount())]
    for _ in range(int(iterations)):
        order = list(range(graph.vcount()))
        rng.shuffle(order)
        for listener in order:
            received: list[int] = []
            for speaker in neighbours[listener]:
                memory = memories[speaker]
                labels = list(memory)
                weights = list(memory.values())
                if labels:
                    received.append(rng.choices(labels, weights=weights, k=1)[0])
            if not received:
                continue
            counts: dict[int, int] = {}
            for label in received:
                counts[label] = counts.get(label, 0) + 1
            best = max(counts, key=lambda label: (counts[label], -int(label)))
            memories[listener][best] = memories[listener].get(best, 0) + 1
    cover: list[list[int]] = []
    for vertex, memory in enumerate(memories):
        total = sum(memory.values())
        retained = [label for label, count in memory.items() if count / total >= threshold]
        if not retained:
            retained = [max(memory, key=lambda label: (memory[label], -int(label)))]
        for label in retained:
            while len(cover) <= label:
                cover.append([])
            cover[label].append(vertex)
    return canonicalize_cover(cover, graph.vcount()), {
        "method": "slpa",
        "family": "label_propagation",
        "implementation": "slpa_compatibility_replica_v1",
        "implementation_kind": "compatibility_smoke_fixture",
        "iterations": int(iterations),
        "threshold": float(threshold),
        "seed": int(seed),
        "warm_start": False,
        "metadata_free": True,
        "availability": "replica",
        "external_failure_reason": external_failure_reason,
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "source_identity": _source_identity(run_slpa),
    }


def run_demon(
    graph: ig.Graph,
    *,
    epsilon: float = 0.25,
    min_community_size: int = 2,
    seed: int = 0,
    allow_replica: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run DEMON when installed, or an explicitly labelled smoke replica.

    The fallback is intentionally labelled as a compatibility fixture in
    metadata; it is not silently reported as the external package.  Set
    ``allow_replica=False`` for standard/paper runs to fail closed when the
    external ``demon`` package is unavailable.  Both variants use graph-only
    information and preserve uncovered vertices in diagnostics.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if min_community_size < 1:
        raise ValueError("min_community_size must be positive")
    try:
        from demon.alg.Demon import Demon
    except ImportError:
        Demon = None
    if Demon is not None:
        version = _installed_version("demon")
        if not allow_replica:
            _require_pinned("demon")
        random.seed(int(seed))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = Demon(
                graph=_networkx_graph(graph),
                epsilon=float(epsilon),
                min_community_size=int(min_community_size),
            ).execute()
        cover = canonicalize_cover([list(map(int, community)) for community in result], graph.vcount())
        implementation = "demon.Demon"
        availability = "external"
        implementation_kind = (
            "external_pinned_adapter"
            if PINNED_EXTERNAL_VERSIONS.get("demon") == version
            else "external_version_recorded_not_pinned"
        )
    else:
        if not allow_replica:
            raise MethodUnavailable("pinned external DEMON requires the demon package")
        # Lightweight deterministic local expansion: each ego neighbourhood is
        # closed under vertices having at least (1-epsilon) of the ego's
        # neighbours, then duplicate bodies are canonicalized.
        rng = random.Random(int(seed))
        cover = []
        for vertex in range(graph.vcount()):
            ego = set(map(int, graph.neighbors(vertex))) | {vertex}
            if len(ego) < min_community_size:
                continue
            for candidate in sorted(tuple(ego)):
                neighbours = set(map(int, graph.neighbors(candidate)))
                support = len(neighbours & ego) / max(1, len(neighbours))
                if support + 1e-12 < 1.0 - epsilon:
                    ego.discard(candidate)
            if len(ego) >= min_community_size:
                cover.append(sorted(ego))
        cover = canonicalize_cover(cover, graph.vcount())
        implementation = "demon_compatibility_replica_v1"
        implementation_kind = "compatibility_smoke_fixture"
        availability = "replica"
    return cover, {
        "method": "demon",
        "family": "local_expansion",
        "implementation": implementation,
        "implementation_kind": implementation_kind,
        "epsilon": float(epsilon),
        "min_community_size": int(min_community_size),
        "seed": int(seed),
        "warm_start": False,
        "metadata_free": True,
        "availability": availability,
        "dependency": {"distribution": "demon", "version": _installed_version("demon")},
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "source_identity": _source_identity(run_demon),
    }


def run_kcp(
    graph: ig.Graph,
    *,
    clique_size: int = 3,
    max_communities: int = 50_000,
    seed: int = 0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run NetworkX's k-clique percolation (KCP) baseline."""
    if clique_size < 2:
        raise ValueError("clique_size must be at least 2")
    if max_communities < 1:
        raise ValueError("max_communities must be positive")
    networkx_version = _installed_version("networkx")
    from networkx.algorithms.community.kclique import k_clique_communities

    communities: list[list[int]] = []
    for community in k_clique_communities(_networkx_graph(graph), int(clique_size)):
        communities.append(sorted(int(vertex) for vertex in community))
        if len(communities) >= int(max_communities):
            break
    return canonicalize_cover(communities, graph.vcount()), {
        "method": "kcp",
        "family": "clique_percolation",
        "implementation": "networkx.algorithms.community.kclique.k_clique_communities",
        "implementation_kind": "external_pinned_adapter" if networkx_version == PINNED_EXTERNAL_VERSIONS.get("networkx") else "external_version_recorded_not_pinned",
        "dependency": {"distribution": "networkx", "version": networkx_version, "pinned_version": PINNED_EXTERNAL_VERSIONS.get("networkx")},
        "clique_size": int(clique_size),
        "max_communities": int(max_communities),
        "seed": int(seed),
        "stochastic": False,
        "warm_start": False,
        "metadata_free": True,
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "uncovered_vertices_policy": "retain uncovered count; no singleton append",
        "source_identity": _source_identity(run_kcp),
    }


def _chen_utility(
    graph: ig.Graph,
    memberships: Sequence[set[int]],
    vertex: int,
    labels: set[int],
    *,
    gamma: float,
    membership_cost: float,
    include_self: bool,
) -> float:
    if not labels:
        return -float("inf")
    neighbours = graph.neighbors(vertex)
    score = 0.0
    for other in neighbours:
        shared = len(labels.intersection(memberships[int(other)]))
        score += 1.0 if shared else 0.0
        score -= float(gamma) * shared / max(1, len(labels))
    if include_self:
        score -= float(gamma) * len(labels) / max(1, 2 * graph.ecount())
    return score - float(membership_cost) * max(0, len(labels) - 1)


def run_chen(
    graph: ig.Graph,
    *,
    max_memberships: int = 4,
    gamma: float | None = None,
    membership_cost: float = 0.0,
    iterations: int = 8,
    seed: int = 0,
    include_self: bool = False,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run a scoped Chen-style set-valued game replica.

    The replica uses graph-neighbour shared-label reward and a documented
    linear membership cost.  Self terms are excluded by default (the printed
    Chen equations require a diagonal correction when they are included), and
    empty label sets are illegal.  Candidate actions are restricted to the
    current singleton and current-plus-one-neighbour-label sets.  This is a
    manageable conceptual comparator, not a complete set-valued best-response
    solver and not a claim of bit-level equivalence to the 2010 implementation.
    """
    if max_memberships < 1 or iterations < 1:
        raise ValueError("max_memberships and iterations must be positive")
    if membership_cost < 0:
        raise ValueError("membership_cost must be non-negative")
    selected_gamma = float(graph.density() if gamma is None else gamma)
    rng = random.Random(int(seed))
    memberships = [{vertex} for vertex in range(graph.vcount())]
    # Candidate labels come from neighbours plus one anonymous fresh label.
    fresh = graph.vcount()
    for _ in range(int(iterations)):
        changed = False
        order = list(range(graph.vcount()))
        rng.shuffle(order)
        for vertex in order:
            candidates = {label for other in graph.neighbors(vertex) for label in memberships[int(other)]}
            candidates.update(memberships[vertex])
            if not candidates:
                candidates = {fresh}
            current = set(memberships[vertex])
            options = [current]
            for label in sorted(candidates):
                if label in current and len(current) == 1:
                    continue
                options.append({label})
                if len(current) < int(max_memberships):
                    options.append(current | {label})
            best = max(
                options,
                key=lambda labels: (
                    _chen_utility(
                        graph,
                        memberships,
                        vertex,
                        labels,
                        gamma=selected_gamma,
                        membership_cost=membership_cost,
                        include_self=include_self,
                    ),
                    -len(labels),
                    tuple(-int(label) for label in sorted(labels)),
                ),
            )
            if best != current:
                memberships[vertex] = set(best)
                changed = True
        if not changed:
            break
    # Replace any anonymous/private labels with contiguous IDs solely for
    # output serialization; no body is merged and no GT labels are used.
    cover = _cover_from_label_rows([sorted(row) for row in memberships])
    return cover, {
        "method": "chen",
        "family": "set_valued_game",
        "implementation": "chen_style_restricted_replica_v1",
        "implementation_kind": "scoped_closest_work_replica",
        "action_space_scope": "current/singleton/current-plus-one candidate labels",
        "complete_set_valued_best_response": False,
        "closest_work_reproduction": False,
        "max_memberships": int(max_memberships),
        "gamma": selected_gamma,
        "membership_cost": float(membership_cost),
        "iterations": int(iterations),
        "seed": int(seed),
        "self_term_convention": "excluded" if not include_self else "included_with_diagonal_penalty",
        "empty_label_convention": "illegal; every vertex retains at least one label",
        "warm_start": False,
        "metadata_free": True,
        "information_budget": "graph topology only; no planted cover/cap tuning",
        "source_identity": _source_identity(run_chen),
    }


def run_singleton(graph: ig.Graph, **_kwargs: Any) -> tuple[list[list[int]], dict[str, Any]]:
    return [[vertex] for vertex in range(graph.vcount())], {
        "method": "singleton",
        "family": "control",
        "implementation": "deterministic_singleton_control_v1",
        "stochastic": False,
        "metadata_free": True,
    }


def run_grand_coalition(graph: ig.Graph, **_kwargs: Any) -> tuple[list[list[int]], dict[str, Any]]:
    return ([list(range(graph.vcount()))] if graph.vcount() else []), {
        "method": "grand_coalition",
        "family": "control",
        "implementation": "deterministic_grand_coalition_control_v1",
        "stochastic": False,
        "metadata_free": True,
    }


def run_components(graph: ig.Graph, **_kwargs: Any) -> tuple[list[list[int]], dict[str, Any]]:
    return [list(map(int, component)) for component in graph.components()], {
        "method": "components",
        "family": "control",
        "implementation": "igraph_connected_components",
        "stochastic": False,
        "metadata_free": True,
    }


def run_baseline(
    name: str,
    graph: ig.Graph,
    *,
    cap: int = 4,
    resolution: float | None = None,
    seed: int = 0,
    parameters: dict[str, Any] | None = None,
    allow_replicas: bool = True,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Execute one named baseline and attach a stable timing receipt."""
    normalized = str(name).strip().lower()
    if normalized not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")
    params = dict(parameters or {})
    params.setdefault("max_memberships", int(cap))
    params.setdefault("gamma", resolution)
    params.setdefault("seed", int(seed))
    params.setdefault("allow_replicas", bool(allow_replicas))
    started = time.perf_counter()
    adapter = BASELINES[normalized]
    selected_resolution = graph.density() if resolution is None else resolution
    cover, metadata = adapter.runner(graph, int(cap), float(selected_resolution), int(seed), params)
    metadata = {
        **metadata,
        "adapter_name": adapter.name,
        "adapter_parameters": dict(adapter.parameters),
        "stochastic": bool(adapter.stochastic),
        "information_budget": adapter.information_budget,
        "runtime_seconds": time.perf_counter() - started,
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "allow_replicas": bool(allow_replicas),
        "environment": environment_receipt(),
    }
    return cover, metadata


def _slpa_adapter(graph, _cap, _resolution, seed, params):
    return run_slpa(
        graph,
        iterations=int(params.get("iterations", 20)),
        threshold=float(params.get("threshold", 0.10)),
        seed=seed,
        allow_replica=bool(params.get("allow_replicas", True)),
        timeout_seconds=float(params.get("timeout_seconds", 90.0)),
    )


def _demon_adapter(graph, _cap, _resolution, seed, params):
    return run_demon(
        graph,
        epsilon=float(params.get("epsilon", 0.25)),
        min_community_size=int(params.get("min_community_size", 2)),
        seed=seed,
        allow_replica=bool(params.get("allow_replicas", True)),
    )


def _kcp_adapter(graph, _cap, _resolution, seed, params):
    if not bool(params.get("allow_replicas", True)):
        _require_pinned("networkx")
    return run_kcp(graph, clique_size=int(params.get("clique_size", 3)), max_communities=int(params.get("max_communities", 50_000)), seed=seed)


def _cpm_adapter(graph, _cap, resolution, seed, params):
    return _run_native_disjoint(
        graph,
        resolution=resolution,
        seed=seed,
        local_move_only=False,
    )


def _disjoint_adapter(graph, _cap, resolution, seed, params):
    return _run_native_disjoint(
        graph,
        resolution=resolution,
        seed=seed,
        local_move_only=bool(params.get("local_move_only", True)),
    )


def _chen_adapter(graph, cap, resolution, seed, params):
    return run_chen(graph, max_memberships=int(params.get("max_memberships", cap)), gamma=resolution, membership_cost=float(params.get("membership_cost", 0.0)), iterations=int(params.get("iterations", 8)), seed=seed, include_self=bool(params.get("include_self", False)))


def _singleton_adapter(graph, _cap, _resolution, seed, params):
    return run_singleton(graph, seed=seed)


def _grand_adapter(graph, _cap, _resolution, seed, params):
    return run_grand_coalition(graph, seed=seed)


def _components_adapter(graph, _cap, _resolution, seed, params):
    return run_components(graph, seed=seed)


BASELINES: dict[str, BaselineAdapter] = {
    "slpa": BaselineAdapter("slpa", "label_propagation", "cdlib.algorithms.slpa via locked isolated worker; replica only for smoke", {"iterations": 20, "threshold": 0.10}, True, "graph topology only", _slpa_adapter),
    "demon": BaselineAdapter("demon", "local_expansion", "demon.Demon (replica only for smoke)", {"epsilon": 0.25, "min_community_size": 2}, True, "graph topology only", _demon_adapter),
    "kcp": BaselineAdapter("kcp", "clique_percolation", "networkx k_clique_communities", {"clique_size": 3, "max_communities": 50_000}, False, "graph topology only", _kcp_adapter),
    "cpm": BaselineAdapter("cpm", "disjoint_cpm", "hedonic.Game.community_hedonic (cap-one full Leiden)", {"max_memberships": 1, "local_move_only": False}, False, "graph topology only; singleton initialization", _cpm_adapter),
    "disjoint": BaselineAdapter("disjoint", "disjoint_hedonic", "hedonic.Game.community_hedonic (cap-one local moving)", {"max_memberships": 1, "local_move_only": True}, False, "graph topology only; singleton initialization", _disjoint_adapter),
    "chen": BaselineAdapter("chen", "set_valued_game", "chen_style_restricted_replica_v1", {"max_memberships": 4, "iterations": 8, "include_self": False, "membership_cost": 0.0}, True, "graph topology only", _chen_adapter),
    "singleton": BaselineAdapter("singleton", "control", "deterministic_singleton_control_v1", {}, False, "graph topology only", _singleton_adapter),
    "grand_coalition": BaselineAdapter("grand_coalition", "control", "deterministic_grand_coalition_control_v1", {}, False, "graph topology only", _grand_adapter),
    "components": BaselineAdapter("components", "control", "igraph_connected_components", {}, False, "graph topology only", _components_adapter),
}

BASELINE_METHODS = tuple(BASELINES)


def baseline_manifest(names: Sequence[str] | None = None) -> dict[str, Any]:
    selected = BASELINE_METHODS if names is None else tuple(str(name).lower() for name in names)
    unknown = sorted(set(selected) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown baseline(s): {', '.join(unknown)}")
    result: dict[str, Any] = {}
    for name in selected:
        adapter = BASELINES[name]
        dependency = None
        slpa_environment = None
        if name == "slpa":
            slpa_environment = slpa_environment_receipt()
            dependency = {
                "distribution": "cdlib",
                "version": PINNED_EXTERNAL_VERSIONS["cdlib"],
                "environment": slpa_environment["root"],
                "lock_sha256": slpa_environment["files"]["lock"]["sha256"],
            }
        elif name in {"demon", "kcp"}:
            distribution = "demon" if name == "demon" else "networkx"
            try:
                dependency = {"distribution": distribution, "version": importlib.metadata.version(distribution)}
            except importlib.metadata.PackageNotFoundError:
                dependency = {"distribution": distribution, "version": None}
        result[name] = {
            "name": adapter.name,
            "family": adapter.family,
            "implementation": adapter.implementation,
            "parameters": dict(adapter.parameters),
            "stochastic": adapter.stochastic,
            "information_budget": adapter.information_budget,
            "dependency": dependency,
            "pinned_dependency_version": (
                PINNED_EXTERNAL_VERSIONS.get(str(dependency.get("distribution")))
                if dependency
                else None
            ),
            "dependency_pin_status": (
                "pinned"
                if dependency
                and (
                    name == "slpa"
                    and slpa_environment is not None
                    and slpa_environment["ready"]
                    and dependency.get("version") == PINNED_EXTERNAL_VERSIONS.get("cdlib")
                    or name != "slpa"
                    and PINNED_EXTERNAL_VERSIONS.get(str(dependency.get("distribution"))) is not None
                    and dependency.get("version") == PINNED_EXTERNAL_VERSIONS.get(str(dependency.get("distribution")))
                )
                else "not_pinned"
                if dependency
                else "not_applicable"
            ),
            "external_required_for_standard": name in {"slpa", "demon"},
            "replica_allowed_for_smoke": name in {"slpa", "demon", "chen"},
            "implementation_scope": (
                "scoped candidate action set; not complete set-valued best response"
                if name == "chen"
                else "external adapter only for standard; compatibility replica is smoke-only"
                if name in {"slpa", "demon"}
                else "native cap-one CPM/Leiden control"
                if name in {"cpm", "disjoint"}
                else "maintained networkx adapter or deterministic control"
            ),
            "slpa_environment": slpa_environment,
            "coverage_policy": "common vertex universe; uncovered vertices counted, never silently appended",
        }
    return {
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "methods": result,
        "environment": environment_receipt(),
    }


def _baseline_graph_event(
    source: Mapping[str, Any],
    expected_graph_keys: Sequence[tuple[str, str, int]],
) -> dict[str, Any]:
    """Build the committed graph-completion event for a baseline source.

    Rows and graph events are separate append-only streams.  In particular,
    an interruption immediately after the last method row must still produce
    the graph event on resume; otherwise a terminal manifest could claim a
    complete graph while its graph checkpoint was empty.
    """
    metadata = source.get("metadata")
    return {
        "graph_id": source.get("graph_id"),
        "condition": source.get("condition"),
        "condition_index": source.get("condition_index"),
        "graph_index_within_condition": source.get("graph_index_within_condition"),
        "graph_hash": str(source["graph_hash"]),
        "cover_hash": str(source["cover_hash"]),
        "graph_seed": int(source["graph_seed"]),
        "input_archive": source.get("archive"),
        "generator_policy": metadata.get("generator_policy") if isinstance(metadata, Mapping) else None,
        "row_keys": [list(key) for key in sorted(expected_graph_keys)],
        "status": "completed",
    }


def run_baseline_study(
    *,
    output: str | Path,
    profile: str = "smoke",
    methods: Sequence[str] | None = None,
    graph_seed: int = 0,
    cap: int = 4,
    timeout_seconds: float | None = None,
    memory_limit_bytes: int | None = None,
    allow_replicas: bool | None = None,
    resume: bool = False,
    graph_ledger: str | Path | None = None,
    max_graphs: int | None = None,
    optimizer_seeds: Sequence[int] | None = None,
    tuning_config: Mapping[str, Any] | str | Path | None = None,
    stop_after_rows: int | None = None,
) -> dict[str, Any]:
    """Run the baseline set with bounded workers and explicit availability.

    Smoke permits the labelled compatibility replicas for SLPA/DEMON.  The
    standard profile requires an official TKT-11 ``graphs.json`` ledger and
    never synthesizes a compatibility graph.  Every graph/method/optimizer
    row is appended and fsynced before the next detector invocation; terminal
    JSON is a reload-checked view of those sidecars.  The tuning object and
    seed plan are hashed into the manifest and every row before any detector
    is run.  ``stop_after_rows`` exists only for disposable recovery tests.
    """
    from hedonic.experiments.overlapping.execution import run_in_subprocess
    from hedonic.experiments.overlapping.overlap_lfr import generate_overlapping_lfr

    profile = str(profile).lower()
    if profile not in {"smoke", "standard"}:
        raise ValueError("profile must be smoke or standard")
    if int(cap) < 1:
        raise ValueError("cap must be positive")
    if timeout_seconds is None:
        timeout_seconds = 5.0 if profile == "smoke" else 60.0
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if memory_limit_bytes is None:
        memory_limit_bytes = (1 << 30) if profile == "smoke" else (4 << 30)
    if int(memory_limit_bytes) <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    if max_graphs is not None and int(max_graphs) < 1:
        raise ValueError("max_graphs must be positive when supplied")
    if stop_after_rows is not None and int(stop_after_rows) < 1:
        raise ValueError("stop_after_rows must be positive when supplied")
    replicas = profile == "smoke" if allow_replicas is None else bool(allow_replicas)
    selected = tuple(
        str(name).strip().lower()
        for name in (methods or ("slpa", "demon", "kcp", "chen", "cpm", "disjoint", "singleton", "grand_coalition", "components"))
    )
    unknown = sorted(set(selected) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown baseline(s): {', '.join(unknown)}")
    tuning = _load_tuning_config(tuning_config)
    plan = _method_plan(selected, tuning_config=tuning)
    optimizer_plan = _optimizer_seed_plan(profile, optimizer_seeds)
    tuning_hash = _canonical_sha256(tuning)
    source_identity = _module_source_identity()
    tuning_config_path = (
        str(Path(tuning_config).expanduser().resolve())
        if isinstance(tuning_config, (str, Path))
        else None
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_path, graphs_path = _checkpoint_paths(destination)

    # Load/validate canonical input before creating a compatibility graph or
    # invoking a detector.  A bounded prefix is a disposable qualification;
    # an unbounded standard launch requires the registered 840 graph ledger.
    selected_graphs: list[dict[str, Any]] = []
    blocked_reason: str | None = None
    ledger_path = Path(graph_ledger).expanduser().resolve() if graph_ledger is not None else None
    ledger_sha256: str | None = None
    if profile == "standard":
        if ledger_path is None:
            blocked_reason = "canonical_graph_ledger_required"
        else:
            try:
                selected_graphs = load_baseline_graph_ledger(
                    ledger_path, profile=profile, max_graphs=max_graphs
                )
                ledger_sha256 = _sha256_file(ledger_path)
            except ValueError as exc:
                blocked_reason = f"invalid_canonical_graph_ledger: {exc}"
    elif ledger_path is not None:
        # Pilot/standard are the only canonical profiles today.  Refuse a
        # ledger on smoke rather than accidentally mixing policies.
        raise ValueError("graph_ledger is only valid for the standard canonical profile")
    else:
        instance = generate_overlapping_lfr(
            n=80,
            average_degree=10,
            max_degree=30,
            min_community=10,
            max_community=20,
            mixing=0.3,
            overlap_fraction=0.3,
            overlap_multiplicity=2,
            seed=int(graph_seed),
        )
        selected_graphs = [{
            "graph": instance.graph,
            "cover": instance.cover,
            "graph_id": "compatibility_g000",
            "condition": {"mixing": 0.3, "overlap_fraction": 0.3, "overlap_multiplicity": 2},
            "condition_index": None,
            "graph_index_within_condition": 0,
            "graph_seed": int(graph_seed),
            "graph_hash": instance.graph_hash,
            "cover_hash": instance.cover_hash,
            "archive": None,
            "metadata": dict(instance.metadata),
        }]

    existing_rows: list[dict[str, Any]] = []
    existing_graph_events: list[dict[str, Any]] = []
    if resume:
        if rows_path.is_file():
            existing_rows = _load_jsonl(rows_path)
        elif destination.is_file():
            try:
                prior = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot resume corrupt baseline manifest: {destination}") from exc
            if not isinstance(prior, dict) or not isinstance(prior.get("rows"), list):
                raise ValueError("cannot resume baseline manifest without a rows list")
            if not all(isinstance(row, dict) for row in prior["rows"]):
                raise ValueError("cannot resume baseline manifest with non-object rows")
            existing_rows = [dict(row) for row in prior["rows"]]
            for row in existing_rows:
                _append_jsonl(rows_path, row)
        if graphs_path.is_file():
            existing_graph_events = _load_jsonl(graphs_path)
        elif destination.is_file():
            try:
                prior = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot resume corrupt baseline manifest: {destination}") from exc
            prior_graphs = prior.get("graphs", []) if isinstance(prior, dict) else []
            if not isinstance(prior_graphs, list) or not all(isinstance(row, dict) for row in prior_graphs):
                raise ValueError("cannot resume baseline manifest with invalid graphs")
            existing_graph_events = [dict(row) for row in prior_graphs]
            for row in existing_graph_events:
                _append_jsonl(graphs_path, row)
    elif rows_path.exists() or graphs_path.exists():
        raise ValueError("output already contains append-only baseline checkpoints; pass --resume")

    rows = [dict(row) for row in existing_rows]
    seen_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = _baseline_row_key(row)
        if key is None:
            raise ValueError("baseline checkpoint contains a row without graph_hash/method/optimizer_seed")
        if key in seen_rows:
            raise ValueError(f"cannot resume baseline checkpoint with duplicate row key: {key}")
        seen_rows[key] = row
    graph_events = [dict(row) for row in existing_graph_events]
    seen_graphs: dict[str, dict[str, Any]] = {}
    for record in graph_events:
        key = _baseline_graph_key(record)
        if key is None:
            raise ValueError("baseline graph checkpoint contains a record without graph_hash")
        if key in seen_graphs:
            raise ValueError(f"cannot resume baseline checkpoint with duplicate graph key: {key}")
        seen_graphs[key] = record

    row_expected_keys = {
        (str(source["graph_hash"]), str(method), int(optimizer_seed))
        for source in selected_graphs
        for method in selected
        for optimizer_seed in _method_optimizer_seeds(
            str(method), tuning_config=tuning, optimizer_seeds=optimizer_plan
        )
    }
    run_identity_payload = {
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "profile": profile,
        "methods": list(selected),
        "cap": int(cap),
        "allow_replicas": bool(replicas),
        "timeout_seconds": float(timeout_seconds),
        "memory_limit_bytes": int(memory_limit_bytes),
        "graph_ledger": str(ledger_path) if ledger_path else None,
        "graph_ledger_sha256": ledger_sha256,
        "graph_hashes": [str(source["graph_hash"]) for source in selected_graphs],
        "optimizer_seeds": list(optimizer_plan),
        "tuning_config_sha256": tuning_hash,
        "tuning_config": tuning,
        "source_identity": source_identity,
        "tuning_config_path": tuning_config_path,
    }
    run_identity = _canonical_sha256(run_identity_payload)
    if resume and destination.is_file():
        try:
            prior_identity = json.loads(destination.read_text(encoding="utf-8")).get("run_identity")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot resume corrupt baseline manifest: {destination}") from exc
        if prior_identity and str(prior_identity) != run_identity:
            raise ValueError("resume configuration does not match the persisted baseline run identity")

    started = time.perf_counter()
    rows_reloaded = len(rows)
    processed_rows = 0
    interrupted = False

    def materialize(status: str, *, blocked: str | None = None) -> dict[str, Any]:
        latest_rows = [seen_rows[key] for key in sorted(seen_rows)]
        latest_graphs = [seen_graphs[key] for key in sorted(seen_graphs)]
        reloaded_rows = _load_jsonl(rows_path) if rows_path.is_file() else []
        reloaded_graphs = _load_jsonl(graphs_path) if graphs_path.is_file() else []
        if reloaded_rows != rows:
            raise ValueError("baseline row checkpoint reload differs from in-memory append-only rows")
        if reloaded_graphs != graph_events:
            raise ValueError("baseline graph checkpoint reload differs from in-memory append-only graphs")
        expected_keys = row_expected_keys
        present_keys = set(seen_rows)
        completed_graphs = sum(
            1
            for source in selected_graphs
            if {
                (str(source["graph_hash"]), str(method), int(seed))
                for method in selected
                for seed in _method_optimizer_seeds(str(method), tuning_config=tuning, optimizer_seeds=optimizer_plan)
            }.issubset(present_keys)
        )
        manifest = {
            "schema_version": 2,
            "protocol_version": BASELINE_PROTOCOL_VERSION,
            "run_identity": run_identity,
            "profile": profile,
            "study_status": status,
            "blocked_reason": blocked,
            "methods": list(selected),
            "cap": int(cap),
            "timeout_seconds": float(timeout_seconds),
            "memory_limit_bytes": int(memory_limit_bytes),
            "allow_replicas": bool(replicas),
            "graph_seed": int(graph_seed) if profile == "smoke" else None,
            "graph_ledger": str(ledger_path) if ledger_path else None,
            "graph_ledger_sha256": ledger_sha256,
            "canonical_input_policy": (
                "official TKT-11 LFRbenchmarks graph archives"
                if profile == "standard"
                else "explicit compatibility smoke fixture"
            ),
            "graph_count": len(selected_graphs),
            "graph_hashes": [str(source["graph_hash"]) for source in selected_graphs],
            "graph_generator_policy": (
                "official_binary_required" if profile == "standard" else "compatibility_smoke_fixture"
            ),
            "metadata_free": True,
            "optimizer_seeds": list(optimizer_plan),
            "seed_plan": tuning.get("seed_plan"),
            "tuning_config": tuning,
            "tuning_config_sha256": tuning_hash,
            "tuning_config_identity": {
                "path": tuning_config_path,
                "sha256": tuning_hash,
            },
            "source_identity": source_identity,
            "prospective_tuning_bound_before_scoring": True,
            "resource_policy": "spawned worker; wall-clock timeout; posthoc peak RSS; failures retained",
            "standard_external_policy": "SLPA/DEMON replicas fail closed unless pinned external implementation is available",
            "environment": environment_receipt(),
            "resume": bool(resume),
            "checkpoint": {
                "rows_jsonl": str(rows_path),
                "graphs_jsonl": str(graphs_path),
                "append_only": True,
                "row_event_count": len(rows),
                "graph_event_count": len(graph_events),
            },
            # Canonical inputs are checked before entering the detector loop.
            # Keep an explicit count so a blocked preflight cannot be
            # mistaken for a zero-row completed study.
            "detector_invocations": len(rows),
            "detector_work_started": bool(rows),
            "recovery": {
                "resume_requested": bool(resume),
                "rows_reloaded": int(rows_reloaded),
                "graphs_reloaded": len(existing_graph_events),
                "reload_check": bool(
                    reloaded_rows == rows
                    and reloaded_graphs == graph_events
                    and ((not latest_rows and not latest_graphs) or (rows_path.is_file() and graphs_path.is_file()))
                ),
                "duplicate_keys": False,
                "no_duplicate_rows_after_resume": len(seen_rows) == len(rows),
                "no_duplicate_graphs_after_resume": len(seen_graphs) == len(graph_events),
                "expected_rows": len(expected_keys),
                "rows_persisted": len(rows),
                "completed_graphs": completed_graphs,
            },
            "baseline_manifest": baseline_manifest(selected),
            "graphs": latest_graphs,
            "rows": latest_rows,
            "status_counts": _baseline_row_status_counts(latest_rows),
            "failure_policy": "retain method failures and dependency availability; no synthetic cover",
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(destination, manifest)
        return manifest

    if blocked_reason is not None:
        # Fail closed before any compatibility graph or detector work.
        return materialize("blocked", blocked=blocked_reason)

    for source in selected_graphs:
        graph_hash_value = str(source["graph_hash"])
        expected_graph_keys = {
            (graph_hash_value, str(method), int(seed))
            for method in selected
            for seed in _method_optimizer_seeds(str(method), tuning_config=tuning, optimizer_seeds=optimizer_plan)
        }
        if expected_graph_keys.issubset(seen_rows):
            # A disposable stop hook may fire immediately after the final row
            # and before the graph-sidecar event.  Complete that sidecar
            # transaction on resume instead of silently skipping it.
            if graph_hash_value not in seen_graphs:
                graph_record = _baseline_graph_event(source, expected_graph_keys)
                _append_jsonl(graphs_path, graph_record)
                graph_events.append(graph_record)
                seen_graphs[graph_hash_value] = graph_record
            continue
        graph = source["graph"]
        gamma = float(graph.density())
        for name in selected:
            for optimizer_seed in _method_optimizer_seeds(
                str(name), tuning_config=tuning, optimizer_seeds=optimizer_plan
            ):
                key = (graph_hash_value, str(name), int(optimizer_seed))
                if key in seen_rows:
                    continue
                detector_seed = _method_seed(
                    graph_hash=graph_hash_value,
                    graph_seed=int(source["graph_seed"]),
                    method=str(name),
                    optimizer_seed=int(optimizer_seed),
                )
                started_row = time.perf_counter()
                outcome = run_in_subprocess(
                    run_baseline,
                    str(name),
                    graph,
                    cap=int(cap),
                    resolution=gamma,
                    seed=detector_seed,
                    parameters=dict(plan[str(name)]),
                    allow_replicas=replicas,
                    timeout_seconds=timeout_seconds,
                )
                successful = outcome.status == "ok" and isinstance(outcome.payload, tuple)
                status = "completed" if successful else str(outcome.status)
                error = outcome.error
                if (
                    successful
                    and memory_limit_bytes is not None
                    and outcome.peak_rss_bytes is not None
                    and int(outcome.peak_rss_bytes) > int(memory_limit_bytes)
                ):
                    status = "memory_limit_exceeded"
                    successful = False
                    error = (
                        f"worker peak RSS {int(outcome.peak_rss_bytes)} exceeds limit "
                        f"{int(memory_limit_bytes)}"
                    )
                cover = outcome.payload[0] if successful else None
                metadata = outcome.payload[1] if successful else None
                row = {
                    "graph_id": source.get("graph_id"),
                    "condition": source.get("condition"),
                    "condition_index": source.get("condition_index"),
                    "graph_index_within_condition": source.get("graph_index_within_condition"),
                    "graph_hash": graph_hash_value,
                    "cover_hash": str(source["cover_hash"]),
                    "graph_seed": int(source["graph_seed"]),
                    "method": str(name),
                    "optimizer_seed": int(optimizer_seed),
                    "seed": int(detector_seed),
                    "cap": int(cap),
                    "gamma": gamma,
                    "tuning_config_sha256": tuning_hash,
                    "status": status,
                    "runtime_seconds": float(outcome.runtime_seconds or (time.perf_counter() - started_row)),
                    "peak_rss_bytes": outcome.peak_rss_bytes,
                    "memory_limit_bytes": int(memory_limit_bytes),
                    "predicted_cover": cover,
                    "metadata": metadata,
                    "error": error,
                    "input_archive": source.get("archive"),
                    "metadata_free_detection": True,
                }
                _append_jsonl(rows_path, row)
                rows.append(row)
                seen_rows[key] = row
                processed_rows += 1
                materialize("running")
                if stop_after_rows is not None and processed_rows >= int(stop_after_rows):
                    interrupted = True
                    break
            if interrupted:
                break
        if interrupted:
            break
        graph_record = _baseline_graph_event(source, expected_graph_keys)
        _append_jsonl(graphs_path, graph_record)
        graph_events.append(graph_record)
        seen_graphs[graph_hash_value] = graph_record
        materialize("running")

    expected_complete = row_expected_keys.issubset(set(seen_rows))
    if interrupted:
        final_status = "interrupted"
    elif expected_complete:
        final_status = "completed" if all(
            str(row.get("status")) == "completed" for row in rows
        ) else "completed_with_failures"
    else:
        final_status = "completed_with_failures"
    return materialize(final_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fair overlapping baseline adapters (Astra TKT-13)")
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlapping/baselines/baselines.json"))
    parser.add_argument("--methods", default=None, help="comma-separated baseline names")
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument(
        "--graph-ledger",
        type=Path,
        default=None,
        help="completed official TKT-11 graphs.json; required for standard",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="bounded disposable prefix for standard qualification; omit for all 840 graphs",
    )
    parser.add_argument(
        "--optimizer-seeds",
        default=None,
        help="comma-separated stochastic optimizer seeds (standard default: 0,1,2)",
    )
    parser.add_argument(
        "--tuning-config",
        type=Path,
        default=None,
        help="JSON prospective baseline parameters, bound before detector scoring",
    )
    parser.add_argument(
        "--stop-after-rows",
        type=int,
        default=None,
        help="disposable interruption hook for recovery tests; do not use for production",
    )
    parser.add_argument("--cap", type=int, default=4, help="maximum memberships for cap-aware adapters")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--allow-replicas", action="store_true", help="allow SLPA/DEMON compatibility replicas (smoke only)")
    parser.add_argument("--resume", action="store_true", help="resume method rows from an existing manifest")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check the locked CDlib SLPA environment and run a disposable four-cycle",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help="optional JSON path for the machine-readable SLPA preflight receipt",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_methods:
        for name in BASELINE_METHODS:
            print(name)
        return 0
    if args.preflight:
        receipt = slpa_preflight(timeout_seconds=float(args.timeout_seconds or 90.0))
        if args.preflight_output is not None:
            _atomic_json(args.preflight_output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0 if receipt.get("status") == "completed" else 2
    methods = None if args.methods is None else tuple(item.strip().lower() for item in args.methods.split(",") if item.strip())
    optimizer_seeds = (
        None
        if args.optimizer_seeds is None
        else tuple(int(item.strip()) for item in args.optimizer_seeds.split(",") if item.strip())
    )
    memory = None if args.memory_limit_gb is None else int(float(args.memory_limit_gb) * 1024**3)
    if args.allow_replicas and args.profile != "smoke":
        raise SystemExit("--allow-replicas is restricted to the smoke compatibility profile")
    manifest = run_baseline_study(
        output=args.output,
        profile=args.profile,
        methods=methods,
        graph_seed=args.graph_seed,
        graph_ledger=args.graph_ledger,
        max_graphs=args.max_graphs,
        optimizer_seeds=optimizer_seeds,
        tuning_config=args.tuning_config,
        stop_after_rows=args.stop_after_rows,
        cap=args.cap,
        timeout_seconds=args.timeout_seconds,
        memory_limit_bytes=memory,
        allow_replicas=True if args.allow_replicas else None,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "BASELINES",
    "BASELINE_METHODS",
    "BASELINE_PROSPECTIVE_TUNING",
    "BASELINE_TUNING_PROTOCOL_VERSION",
    "BaselineAdapter",
    "MethodUnavailable",
    "baseline_manifest",
    "load_baseline_graph_ledger",
    "environment_receipt",
    "run_baseline_study",
    "build_parser",
    "main",
    "run_baseline",
    "run_chen",
    "run_components",
    "run_disjoint",
    "run_demon",
    "run_grand_coalition",
    "run_kcp",
    "run_link_clustering",
    "run_singleton",
    "run_slpa",
    "link_clustering_environment_receipt",
    "slpa_environment_receipt",
    "slpa_preflight",
]
