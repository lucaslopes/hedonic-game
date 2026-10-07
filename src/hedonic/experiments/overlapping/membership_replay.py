"""Read-only replay of saved memberships against the paper auditor.

This module does not invoke detectors and does not rewrite frozen ledgers.
GT-v3 replay loads the same bounded SNAP graphs used by the original grid
and re-runs ``audit_cover`` on persisted raw memberships.  Tiny DNN replay
recomputes stored exact covers with the independent unit-ℓ₂ oracle.
Equilibrium-v2 paper replay uses local shard analysis graphs and
persisted Hedonic memberships; it does not download SNAP archives.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hedonic.experiments.config import NETWORKS_DIR
from hedonic.experiments.overlapping import unit_l2_oracle as oracle

DEFAULT_GT_ROOT = Path("artifacts/overlapping/ground_truth_robustness_v3")
DEFAULT_DNN = Path(
    "artifacts/evidence/overlapping_communities/dnn_certificate_v2.json"
)
DEFAULT_PAPER_ROOT = Path(
    "artifacts/papers/overlapping_communities/equilibrium_v2/full"
)
REGRET_ATOL = 1e-8
REGRET_RTOL = 1e-6
OBJECTIVE_ATOL = 1e-10
OBJECTIVE_RTOL = 1e-8


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)


def cover_to_memberships(
    cover_by_label: Iterable[Iterable[int]], n_vertices: int
) -> list[list[int]]:
    memberships: list[list[int]] = [[] for _ in range(n_vertices)]
    for label, community in enumerate(cover_by_label):
        for vertex in community:
            memberships[int(vertex)].append(int(label))
    if any(not row for row in memberships):
        raise ValueError("cover leaves an uncovered vertex")
    return memberships


def replay_dnn_ledger(path: Path) -> dict[str, Any]:
    """Recompute each tiny exact cover with the independent NumPy oracle."""
    info = {
        "path": str(path),
        "present": path.is_file(),
        "family": "dnn_certificate_v2",
        "agreements": 0,
        "disagreements": 0,
        "instances": [],
    }
    if not path.is_file():
        return info
    payload = json.loads(path.read_text(encoding="utf-8"))
    for result in payload.get("results") or []:
        instance = result.get("instance") or {}
        exact = result.get("exact_valid_cover_optimum") or result.get("exact") or {}
        name = instance.get("name")
        n_vertices = int(instance["n_vertices"])
        edges = [
            (int(edge[0]), int(edge[1]), float(weight))
            for edge, weight in zip(instance["edges"], instance["edge_weights"])
        ]
        adjacency = oracle.make_symmetric_graph(n_vertices, edges)
        weights = np.asarray(instance["vertex_weights"], dtype=float)
        gamma = float(instance["resolution"])
        label_count = int(instance["max_labels"])
        memberships = [list(map(int, row)) for row in exact["memberships_by_vertex"]]
        phi_tilde = oracle.unnormalized_potential(
            adjacency, weights, gamma, memberships, label_count
        )
        total_weight = float(sum(instance["edge_weights"]))
        replayed = phi_tilde / total_weight
        stored = float(exact["objective"])
        agree = _close(
            replayed, stored, atol=OBJECTIVE_ATOL, rtol=OBJECTIVE_RTOL
        )
        row = {
            "name": name,
            "stored_objective": stored,
            "replayed_normalized_potential": replayed,
            "agree": agree,
        }
        info["instances"].append(row)
        if agree:
            info["agreements"] += 1
        else:
            info["disagreements"] += 1
    info["complete"] = info["disagreements"] == 0 and info["agreements"] == 3
    return info


def _run_path(gt_root: Path, dataset: str, cover: str, condition_key: str) -> Path:
    return gt_root / "runs" / f"{dataset}-{cover}" / f"{condition_key}.json"


def _load_done_keys(jsonl_path: Path) -> set[str]:
    if not jsonl_path.is_file():
        return set()
    done: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("condition_key")
            if key:
                done.add(str(key))
    return done


def _select_jsonl_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    if not limit or limit <= 0 or limit >= len(rows):
        return rows
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("dataset")), str(row.get("status")))].append(row)
    selected: list[dict[str, Any]] = []
    remaining = {key: list(value) for key, value in buckets.items()}
    while remaining and len(selected) < limit:
        empty: list[tuple[str, str]] = []
        for key in sorted(remaining):
            if len(selected) >= limit:
                break
            group = remaining[key]
            if not group:
                empty.append(key)
                continue
            selected.append(group.pop(0))
        for key in empty:
            remaining.pop(key, None)
    return selected


def replay_gt_v3(
    *,
    gt_root: Path,
    networks_dir: Path,
    output_jsonl: Path,
    limit: int | None = None,
    resume: bool = True,
    max_nodes: int = 3000,
    dense: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Replay persisted GT-v3 memberships with ``audit_cover``.

    Frozen ``results.jsonl`` and run JSON files are read-only.  Disagreements
    are recorded in ``output_jsonl``; they never rewrite the ledger.
    """
    from hedonic.experiments.overlapping.ground_truth_data import (
        load_prepared_dataset,
    )
    from hedonic.experiments.overlapping.ground_truth_robustness import (
        _load_membership_artifact,
    )
    from hedonic.experiments.overlapping.robustness import audit_cover

    results_path = gt_root / "results.jsonl"
    summary: dict[str, Any] = {
        "family": "ground_truth_robustness_v3",
        "gt_root": str(gt_root),
        "results": str(results_path),
        "output": str(output_jsonl),
        "limit": limit,
        "dense": dense,
        "max_nodes": max_nodes,
        "graph_identity_ok": {},
        "status_counts": {},
        "replayed": 0,
        "agreements": 0,
        "disagreements": 0,
        "skipped_terminal": 0,
        "skipped_missing": 0,
        "errors": 0,
        "complete_unconditional": False,
    }
    if not results_path.is_file():
        summary["error"] = "missing results.jsonl"
        return summary

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = _load_done_keys(output_jsonl) if resume else set()
    pending = [row for row in rows if str(row.get("condition_key")) not in done]
    pending = _select_jsonl_rows(pending, limit=limit)
    graphs: dict[tuple[str, str, str], Any] = {}
    audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def graph_for(dataset: str, cover: str, policy: str, expected_identity: str):
        key = (dataset, cover, policy)
        if key not in graphs:
            prepared = load_prepared_dataset(
                dataset,
                cover_variant=cover,
                data_root=networks_dir,
                policy=policy,  # type: ignore[arg-type]
                max_nodes=None if smoke else max_nodes,
                smoke=smoke,
            )
            graphs[key] = prepared
            summary["graph_identity_ok"][f"{dataset}:{cover}:{policy}"] = (
                prepared.graph_identity == expected_identity
            )
        prepared = graphs[key]
        if prepared.graph_identity != expected_identity:
            raise RuntimeError(
                f"graph identity mismatch for {dataset}: "
                f"{prepared.graph_identity} != {expected_identity}"
            )
        return prepared

    with output_jsonl.open("a" if resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            condition_key = str(row["condition_key"])
            dataset = str(row["dataset"])
            cover = str(row.get("cover") or "top5000")
            stored_status = str(row["status"])
            record: dict[str, Any] = {
                "condition_key": condition_key,
                "dataset": dataset,
                "stored_status": stored_status,
                "stored_equilibrium_status": row.get("equilibrium_status"),
                "stored_max_positive_regret": row.get("max_positive_regret"),
            }
            if stored_status == "unsupported_cleanup":
                record["replay_status"] = "skipped_terminal"
                summary["skipped_terminal"] += 1
                handle.write(json.dumps(record) + "\n")
                continue
            run_path = _run_path(gt_root, dataset, cover, condition_key)
            if not run_path.is_file():
                record["replay_status"] = "missing_run"
                record["run_path"] = str(run_path)
                summary["skipped_missing"] += 1
                handle.write(json.dumps(record) + "\n")
                continue
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
                condition = run.get("condition") or {}
                membership_hash = run.get("final_membership_hash") or row.get(
                    "final_membership_hash"
                )
                memberships = (
                    _load_membership_artifact(gt_root, str(membership_hash))
                    if isinstance(membership_hash, str)
                    else None
                )
                if memberships is None:
                    record["replay_status"] = "missing_membership"
                    summary["skipped_missing"] += 1
                    handle.write(json.dumps(record) + "\n")
                    continue
                prepared = graph_for(
                    dataset,
                    cover,
                    str(condition.get("completion_policy") or "covered-induced"),
                    str(condition.get("graph_identity") or row.get("graph_identity")),
                )
                selected = (run.get("robustness") or {}).get("selected_policy") or {}
                cap = int(condition.get("max_memberships") or selected.get("max_memberships"))
                gamma = float(condition.get("gamma") if condition.get("gamma") is not None else row["gamma"])
                allow_isolation = bool(
                    condition.get("allow_isolation")
                    if "allow_isolation" in condition
                    else row.get("action_policy") == "open_labels"
                )
                atol = float(condition.get("robustness_atol") or 1e-10)
                rtol = float(condition.get("robustness_rtol") or 1e-9)
                cache_key = (
                    prepared.graph_identity,
                    str(membership_hash),
                    cap,
                    allow_isolation,
                    gamma,
                    atol,
                    rtol,
                    dense,
                )
                audit = audit_cache.get(cache_key)
                if audit is None:
                    audit = audit_cover(
                        prepared.dataset.graph,
                        memberships,
                        max_memberships=cap,
                        allow_isolation=allow_isolation,
                        gamma=gamma,
                        interval=(0.0, 1.0),
                        atol=atol,
                        rtol=rtol,
                        dense=dense,
                    )
                    audit_cache[cache_key] = audit
                replay_eq = bool(audit["is_local_equilibrium_at_resolution"])
                replay_status = (
                    "completed" if replay_eq else "completed_non_equilibrium"
                )
                stored_regret = row.get("max_positive_regret")
                replay_regret = float(audit["max_positive_regret_at_resolution"])
                status_agree = replay_status == stored_status
                regret_agree = stored_regret is None or _close(
                    replay_regret,
                    float(stored_regret),
                    atol=REGRET_ATOL,
                    rtol=REGRET_RTOL,
                )
                agree = status_agree and regret_agree
                record.update(
                    {
                        "replay_status": replay_status,
                        "replay_max_positive_regret": replay_regret,
                        "status_agree": status_agree,
                        "regret_agree": regret_agree,
                        "agree": agree,
                        "graph_identity": prepared.graph_identity,
                    }
                )
                summary["replayed"] += 1
                if agree:
                    summary["agreements"] += 1
                else:
                    summary["disagreements"] += 1
            except Exception as exc:  # noqa: BLE001 — ledger replay must record faults
                record["replay_status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                summary["errors"] += 1
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(
                    f"[replay-gt] {index}/{len(pending)} "
                    f"agree={summary['agreements']} "
                    f"disagree={summary['disagreements']} "
                    f"err={summary['errors']}",
                    flush=True,
                )

    counts = Counter()
    agreed = 0
    disagreed = 0
    if output_jsonl.is_file():
        with output_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                status = str(record.get("replay_status", "unknown"))
                counts[status] += 1
                if status == "skipped_terminal":
                    continue
                if record.get("agree") is True:
                    agreed += 1
                elif record.get("agree") is False or status == "error":
                    disagreed += 1
    summary["status_counts"] = dict(counts)
    summary["elapsed_seconds"] = time.monotonic() - started
    expected = len(rows)
    accounted = sum(counts.values())
    summary["expected"] = expected
    summary["accounted"] = accounted
    summary["replayed"] = int(counts.get("completed", 0)) + int(
        counts.get("completed_non_equilibrium", 0)
    )
    summary["agreements"] = agreed
    summary["disagreements"] = disagreed
    summary["skipped_terminal"] = int(counts.get("skipped_terminal", 0))
    summary["skipped_missing"] = int(counts.get("missing_run", 0)) + int(
        counts.get("missing_membership", 0)
    )
    summary["errors"] = int(counts.get("error", 0))
    summary["complete_unconditional"] = (
        accounted == expected
        and disagreed == 0
        and summary["errors"] == 0
        and summary["skipped_missing"] == 0
    )
    return summary


def _load_gzip_json(path: Path) -> Any:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _paper_shard_root(paper_root: Path, record: dict[str, Any]) -> Path:
    return paper_root / "shards" / f"{record['dataset']}-{record['cover']}"


def replay_paper_125(
    *,
    paper_root: Path,
    results_path: Path,
    output_jsonl: Path,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Re-audit persisted Hedonic memberships on local shard graphs."""
    import igraph as ig

    from hedonic import Game
    from hedonic.experiments.overlapping.robustness import audit_cover

    info: dict[str, Any] = {
        "path": str(results_path),
        "paper_root": str(paper_root),
        "present": results_path.is_file(),
        "family": "overlapping_paper_equilibrium_v2",
        "rewrites_frozen_ledgers": False,
        "expected_hedonic": 75,
        "accounted": 0,
        "agreements": 0,
        "disagreements": 0,
        "skipped_external": 0,
        "skipped_missing": 0,
        "errors": 0,
        "max_replayed_regret": None,
    }
    if not results_path.is_file():
        return info
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    graph_cache: dict[str, Any] = {}
    done: set[str] = set()
    if resume:
        done = _load_done_keys(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_jsonl.is_file() else "w"
    hedonic_rows = [row for row in rows if str(row.get("method", "")).startswith("hedonic_")]
    external_rows = [row for row in rows if not str(row.get("method", "")).startswith("hedonic_")]
    info["skipped_external"] = len(external_rows)
    selected = hedonic_rows if not limit else hedonic_rows[: int(limit)]
    max_regret = 0.0
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        for record in selected:
            key = "|".join(
                str(record.get(field))
                for field in ("dataset", "cover", "method", "seed", "requested_resolution")
            )
            if key in done:
                info["accounted"] += 1
                continue
            shard = _paper_shard_root(paper_root, record)
            graph_rel = record.get("analysis_graph_artifact")
            mem_rel = record.get("final_membership_artifact")
            cert = record.get("equilibrium_certificate") or {}
            row: dict[str, Any] = {
                "condition_key": key,
                "dataset": record.get("dataset"),
                "method": record.get("method"),
                "seed": record.get("seed"),
            }
            if not isinstance(graph_rel, str) or not isinstance(mem_rel, str):
                row["status"] = "missing_artifact"
                info["skipped_missing"] += 1
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue
            graph_path = shard / graph_rel
            mem_path = shard / mem_rel
            if str(graph_path) not in graph_cache:
                payload = _load_gzip_json(graph_path)
                graph_cache[str(graph_path)] = Game(
                    ig.Graph(
                        n=int(payload["n"]),
                        edges=[(int(u), int(v)) for u, v in payload["edges"]],
                        directed=bool(payload.get("directed")),
                    )
                )
            graph = graph_cache[str(graph_path)]
            memberships = _load_gzip_json(mem_path)
            audit = audit_cover(
                graph,
                memberships,
                max_memberships=int(record["max_memberships"]),
                allow_isolation=bool(record.get("allow_isolation")),
                gamma=float(record["resolution"]),
                compute_intervals=False,
                dense=False,
            )
            replayed_eq = bool(audit["is_local_equilibrium_at_resolution"])
            replayed_regret = float(audit["max_positive_regret_at_resolution"])
            stored_eq = bool(cert.get("is_local_equilibrium_at_resolution"))
            stored_regret = float(cert.get("max_positive_regret"))
            agree = replayed_eq == stored_eq and _close(
                replayed_regret,
                stored_regret,
                atol=REGRET_ATOL,
                rtol=REGRET_RTOL,
            )
            row.update(
                {
                    "status": "agree" if agree else "disagree",
                    "stored_equilibrium": stored_eq,
                    "replayed_equilibrium": replayed_eq,
                    "stored_regret": stored_regret,
                    "replayed_regret": replayed_regret,
                    "n_vertices_scored": audit["n_vertices_scored"],
                }
            )
            info["accounted"] += 1
            max_regret = max(max_regret, replayed_regret)
            if agree:
                info["agreements"] += 1
            else:
                info["disagreements"] += 1
            handle.write(json.dumps(row) + "\n")
            handle.flush()
    info["max_replayed_regret"] = max_regret if info["accounted"] else None
    info["complete"] = (
        info["disagreements"] == 0
        and info["errors"] == 0
        and info["skipped_missing"] == 0
        and (info["agreements"] == 75 if not limit else info["disagreements"] == 0)
    )
    return info


def _summarize_gt_replay_jsonl(path: Path) -> dict[str, Any]:
    """Rebuild a GT replay summary from an existing JSONL without rerunning."""
    counts: Counter[str] = Counter()
    agreed = 0
    disagreed = 0
    skipped_terminal = 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            n += 1
            status = str(record.get("replay_status") or record.get("stored_status") or "unknown")
            counts[status] += 1
            if record.get("stored_status") == "unsupported_cleanup" or status == "skipped_terminal":
                skipped_terminal += 1
                continue
            if record.get("agree") is True:
                agreed += 1
            elif record.get("agree") is False:
                disagreed += 1
    return {
        "path": str(path),
        "family": "ground_truth_robustness_v3",
        "rewrites_frozen_ledgers": False,
        "accounted": n,
        "agreements": agreed,
        "disagreements": disagreed,
        "skipped_terminal": skipped_terminal,
        "status_counts": dict(counts),
        "reconstructed_from_jsonl": True,
        "complete_unconditional": n == 3840 and disagreed == 0,
    }


def write_replay_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_membership_replay(
    *,
    gt_root: Path = DEFAULT_GT_ROOT,
    dnn_path: Path = DEFAULT_DNN,
    output_dir: Path,
    networks_dir: Path | None = None,
    replay_gt: bool = False,
    replay_dnn: bool = False,
    replay_paper: bool = False,
    paper_root: Path | None = None,
    paper_results: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
    max_nodes: int = 3000,
    smoke: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "detector_free": True,
        "rewrites_frozen_ledgers": False,
    }
    summary_path = output_dir / "membership_replay_summary.json"
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if isinstance(previous, dict):
            for key in ("dnn", "ground_truth_robustness_v3", "overlapping_paper_equilibrium_v2"):
                if key in previous and key not in payload:
                    payload[key] = previous[key]
    if replay_dnn:
        payload["dnn"] = replay_dnn_ledger(dnn_path)
    if replay_gt:
        payload["ground_truth_robustness_v3"] = replay_gt_v3(
            gt_root=gt_root,
            networks_dir=Path(networks_dir or NETWORKS_DIR),
            output_jsonl=output_dir / "membership_replay.jsonl",
            limit=limit,
            resume=resume,
            max_nodes=max_nodes,
            smoke=smoke,
        )
    if replay_paper:
        root = Path(paper_root or DEFAULT_PAPER_ROOT)
        results = Path(paper_results or (root / "results.jsonl"))
        payload["overlapping_paper_equilibrium_v2"] = replay_paper_125(
            paper_root=root,
            results_path=results,
            output_jsonl=output_dir / "paper_membership_replay.jsonl",
            limit=limit,
            resume=resume,
        )
    gt_jsonl = output_dir / "membership_replay.jsonl"
    if "ground_truth_robustness_v3" not in payload and gt_jsonl.is_file():
        payload["ground_truth_robustness_v3"] = _summarize_gt_replay_jsonl(gt_jsonl)
    write_replay_summary(output_dir / "membership_replay_summary.json", payload)
    return payload
