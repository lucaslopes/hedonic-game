"""TOML-driven, tmux-orchestrated reproduction of the overlapping SNAP paper.

The lower-level ``overlapping-benchmark`` command owns one output directory and
therefore deliberately runs its conditions serially.  This module parallelizes
only independent dataset/cover *shards*.  Every shard has a separate output
tree, which keeps the benchmark's manifests, logs, summaries, and atomic run
records free of concurrent-writer races.  A coordinator then merges the run
records, regenerates paper plots and tables, and (when the full protocol has
no retryable records) compiles the manuscript.

All normal experiment parameters live in ``[overlapping_paper]`` in the TOML
file.  The public one-command entry point is::

    hedonic-exp reproduce-overlapping-paper

The default repository config is deliberately conservative: it chooses at
most the configured memory-safe number of concurrent dataset workers.  See
``configs/hedonic.toml`` for the full protocol and all fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments import config as experiment_config
from hedonic.experiments.overlapping import benchmark
from hedonic.experiments.overlapping.methods import METHODS
from hedonic.experiments.overlapping.snap import network_names


PAPER_SCHEMA_VERSION = 1
DEFAULT_METHODS = (
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "cpm",
    "demon",
)
DEFAULT_JOBS = (
    ("amazon", "top5000"),
    ("dblp", "top5000"),
    ("livejournal", "top5000"),
    ("youtube", "top5000"),
    ("wikipedia", "all"),
)
TERMINAL_STATUSES = {"completed", "unavailable", "skipped", "data_unavailable"}
RETRYABLE_STATUSES = {"timeout", "error"}
_DATASET_WEIGHT = {
    "dblp": 1,
    "amazon": 2,
    "youtube": 3,
    "livejournal": 5,
    "wikipedia": 5,
}
_PAPER_METRICS = (
    "symmetric_best_match_f1",
    "matching_f1",
    "node_membership_micro_f1",
    "size_weighted_f1",
    "omega",
    "runtime_seconds",
    "predicted_overlapping_node_fraction",
    "coverage_rate",
    "inclusion_rate",
    "cpm_overlapping_quality",
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        result = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        result = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"overlapping_paper.{field} must be a string or TOML array")
    if not result:
        raise ValueError(f"overlapping_paper.{field} must not be empty")
    return result


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"overlapping_paper.{field} must be a non-empty path")
    return Path(os.path.expanduser(str(value))).expanduser().resolve()


def _physical_memory_bytes() -> int | None:
    """Return installed RAM without introducing an optional dependency."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return page_size * pages
    except (AttributeError, OSError, ValueError):
        pass
    # macOS exposes this even when POSIX sysconf does not.
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        value = int(output)
        return value if value > 0 else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _resolve_workers(value: Any, *, jobs: int, memory_gb_per_worker: float, cap: int) -> int:
    if jobs < 1:
        raise ValueError("At least one paper job is required")
    cpu_count = os.cpu_count() or 1
    if cap < 1:
        raise ValueError("overlapping_paper.max_parallel_cap must be >= 1")
    memory = _physical_memory_bytes()
    memory_cap = jobs
    if memory is not None and memory_gb_per_worker > 0:
        per_worker = int(memory_gb_per_worker * (1024**3))
        memory_cap = max(1, memory // per_worker)

    if isinstance(value, str) and value.strip().lower() == "auto":
        return max(1, min(jobs, cpu_count, cap, memory_cap))
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "overlapping_paper.max_parallel_workers must be a positive integer or 'auto'"
        ) from exc
    if workers < 1:
        raise ValueError("overlapping_paper.max_parallel_workers must be >= 1")
    if workers > cpu_count:
        raise ValueError(
            f"Requested {workers} workers but only {cpu_count} logical CPUs are available"
        )
    if workers > cap:
        raise ValueError(
            f"Requested {workers} workers but max_parallel_cap is {cap}; raise the cap explicitly"
        )
    if workers > memory_cap:
        required = workers * memory_gb_per_worker
        raise ValueError(
            f"Requested {workers} workers needs at least {required:g} GiB according to "
            "memory_gb_per_worker; lower workers or revise the explicit safety budget"
        )
    return min(workers, jobs)


def _parse_jobs(raw: Any) -> list[dict[str, Any]]:
    source = raw if raw is not None else [
        {"dataset": dataset, "cover": cover} for dataset, cover in DEFAULT_JOBS
    ]
    if not isinstance(source, list) or not source:
        raise ValueError("overlapping_paper.jobs must be a non-empty TOML array of tables")
    valid_datasets = set(network_names())
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError("Each overlapping_paper.jobs item must be a TOML table")
        dataset = str(item.get("dataset", "")).strip().lower()
        cover = str(item.get("cover", "")).strip().lower()
        if dataset not in valid_datasets:
            raise ValueError(f"Unknown paper job dataset: {dataset!r}")
        if cover not in {"all", "top5000"}:
            raise ValueError(f"Paper job {dataset!r} has invalid cover {cover!r}")
        if dataset == "wikipedia" and cover != "all":
            raise ValueError("Wikipedia only has the supplied 'all' category cover")
        name = str(item.get("name") or f"{dataset}-{cover}").strip()
        if not name or name in seen:
            raise ValueError("Paper job names must be unique and non-empty")
        seen.add(name)
        weight = item.get("weight", _DATASET_WEIGHT.get(dataset, 1))
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Paper job {name!r} has invalid weight") from exc
        if weight <= 0:
            raise ValueError(f"Paper job {name!r} weight must be positive")
        jobs.append({"name": name, "dataset": dataset, "cover": cover, "weight": weight, "index": index})
    return jobs


def _assign_jobs(jobs: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Greedily balance the known large SNAP graphs across worker windows."""
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda item: (-float(item["weight"]), item["index"])):
        target = min(range(workers), key=lambda index: (loads[index], index))
        assignments[target].append(job)
        loads[target] += float(job["weight"])
    return assignments


def _load_options(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    toml = experiment_config.load_config_file(args.config, search_cwd=True, apply=True)
    config_path = experiment_config.get_loaded_config_path()
    if config_path is None:
        raise ValueError(
            "No TOML configuration found. Run from the repository root or pass --config."
        )
    raw = toml.get("overlapping_paper")
    if not isinstance(raw, dict):
        raise ValueError(
            f"{config_path} needs an [overlapping_paper] table for this command"
        )
    methods = _as_list(raw.get("methods", list(DEFAULT_METHODS)), "methods")
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown configured methods: {', '.join(unknown)}")
    if "hedonic_local" in methods:
        raise ValueError(
            "The paper protocol excludes hedonic_local; use the three hedonic_multiphase "
            "resolution variants with cpm,demon"
        )
    if "hedonic_multiphase" not in methods:
        raise ValueError("The paper protocol must include hedonic_multiphase")

    profile = str(raw.get("profile", "full")).strip().lower()
    if profile not in benchmark.PROFILE_DEFAULTS:
        raise ValueError("overlapping_paper.profile must be smoke, standard, or full")
    jobs = _parse_jobs(raw.get("jobs"))
    memory_gb = float(raw.get("memory_gb_per_worker", 24))
    if memory_gb <= 0:
        raise ValueError("overlapping_paper.memory_gb_per_worker must be positive")
    cap = int(raw.get("max_parallel_cap", 2))
    worker_setting = args.workers if args.workers is not None else raw.get("max_parallel_workers", "auto")
    workers = _resolve_workers(
        worker_setting, jobs=len(jobs), memory_gb_per_worker=memory_gb, cap=cap
    )
    if args.session is not None:
        session = args.session
    else:
        session = str(raw.get("tmux_session", "hedonic-overlapping-paper")).strip()
    if not session:
        raise ValueError("overlapping_paper.tmux_session must be non-empty")
    if any(character.isspace() for character in session) or ":" in session:
        raise ValueError("tmux session names must not contain whitespace or ':'")

    data_root = _path(raw.get("data_root", experiment_config.NETWORKS_DIR), "data_root")
    output_dir = _path(raw.get("output_dir"), "output_dir")
    paper_dir = _path(raw.get("paper_dir"), "paper_dir")
    if not (paper_dir / "main.tex").is_file():
        raise ValueError(f"overlapping_paper.paper_dir has no main.tex: {paper_dir}")
    expected_artifact_dir = (paper_dir / "artifacts" / "full").resolve()
    if output_dir != expected_artifact_dir:
        raise ValueError(
            "overlapping_paper.output_dir must be paper_dir/artifacts/full so main.tex "
            "can consume the generated status, tables, and plots"
        )
    benchmark._safe_output_dir(output_dir, data_root)

    retries = int(raw.get("retry_attempts", 1))
    if retries < 0:
        raise ValueError("overlapping_paper.retry_attempts must be >= 0")
    poll_seconds = float(raw.get("poll_seconds", 10))
    if poll_seconds <= 0:
        raise ValueError("overlapping_paper.poll_seconds must be positive")
    max_nodes = int(raw.get("max_nodes", 0))
    timeout = float(raw.get("timeout_per_run", 3600))
    if timeout <= 0:
        raise ValueError("overlapping_paper.timeout_per_run must be positive")

    options = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "config_path": str(config_path),
        "profile": profile,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "paper_dir": str(paper_dir),
        "tmux_session": session,
        "methods": methods,
        "seeds": str(raw.get("seeds", "0-4")),
        "resolutions": str(raw.get("resolutions", "auto")),
        "timeout_per_run": timeout,
        "max_nodes": max_nodes,
        "omega": bool(raw.get("omega", True)),
        "omega_sample_size": int(raw.get("omega_sample_size", 100_000)),
        "resume": bool(raw.get("resume", True)),
        "retry_attempts": retries,
        "compile_paper": bool(raw.get("compile_paper", True)),
        "poll_seconds": poll_seconds,
        "memory_gb_per_worker": memory_gb,
        "max_parallel_cap": cap,
        "workers": workers,
        "jobs": jobs,
    }
    return options, config_path


def _plan_path(output_dir: Path) -> Path:
    return output_dir / "orchestration" / "plan.json"


def _state_path(output_dir: Path, worker_index: int) -> Path:
    return output_dir / "orchestration" / "workers" / f"worker_{worker_index}.json"


def _make_plan(options: dict[str, Any]) -> dict[str, Any]:
    workers = int(options["workers"])
    assignments = _assign_jobs(list(options["jobs"]), workers)
    return {
        **options,
        "plan_id": uuid.uuid4().hex,
        "started_at": _timestamp(),
        "assignments": assignments,
    }


def _write_new_plan(options: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(options["output_dir"])
    plan = _make_plan(options)
    _write_json(_plan_path(output_dir), plan)
    for index, assignment in enumerate(plan["assignments"]):
        _write_json(
            _state_path(output_dir, index),
            {
                "plan_id": plan["plan_id"],
                "worker_index": index,
                "status": "pending",
                "jobs": [job["name"] for job in assignment],
                "updated_at": _timestamp(),
            },
        )
    return plan


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    if plan is None:
        raise ValueError(f"Paper orchestration plan is missing or invalid: {path}")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(f"Paper orchestration plan has no assignments: {path}")
    return plan


def _benchmark_argv(plan: dict[str, Any], job: dict[str, Any], output_dir: Path) -> list[str]:
    argv = [
        "--profile",
        str(plan["profile"]),
        "--data_root",
        str(plan["data_root"]),
        "--datasets",
        str(job["dataset"]),
        "--cover",
        str(job["cover"]),
        "--methods",
        ",".join(plan["methods"]),
        "--seeds",
        str(plan["seeds"]),
        "--resolutions",
        str(plan["resolutions"]),
        "--timeout_per_run",
        str(plan["timeout_per_run"]),
        "--max_nodes",
        str(plan["max_nodes"]),
        "--output_dir",
        str(output_dir),
        "--no-plots",
    ]
    if bool(plan.get("resume", True)):
        argv.append("--resume")
    if bool(plan.get("omega", True)):
        argv.extend(["--omega", "--omega_sample_size", str(plan["omega_sample_size"])])
    return argv


def _job_retryable_records(
    shard_dir: Path, plan: dict[str, Any], job: dict[str, Any]
) -> int:
    return sum(
        1
        for record in benchmark._completed_records(shard_dir)
        if (
            record.get("status") in RETRYABLE_STATUSES
            and record.get("dataset") == job["dataset"]
            and record.get("cover") == job["cover"]
            and record.get("method") in set(plan["methods"])
        )
    )


def run_worker(plan: dict[str, Any], worker_index: int) -> int:
    """Execute one independent, memory-bounded worker assignment."""
    assignments = plan["assignments"]
    if worker_index < 0 or worker_index >= len(assignments):
        raise ValueError(f"Worker index {worker_index} is outside this plan")
    output_dir = Path(plan["output_dir"])
    state = {
        "plan_id": plan["plan_id"],
        "worker_index": worker_index,
        "status": "running",
        "jobs": [job["name"] for job in assignments[worker_index]],
        "completed_jobs": [],
        "failures": [],
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    _write_json(_state_path(output_dir, worker_index), state)

    for job in assignments[worker_index]:
        shard_dir = output_dir / "shards" / str(job["name"])
        attempts = 0
        try:
            while True:
                code = benchmark.main(_benchmark_argv(plan, job, shard_dir))
                retryable = _job_retryable_records(shard_dir, plan, job)
                if code == 0 and (retryable == 0 or attempts >= int(plan["retry_attempts"])):
                    break
                attempts += 1
                if attempts > int(plan["retry_attempts"]):
                    break
            if code != 0:
                raise RuntimeError(f"benchmark returned exit code {code}")
            state["completed_jobs"].append(
                {"name": job["name"], "attempts": attempts + 1, "retryable_records": retryable}
            )
        except BaseException as exc:
            state["failures"].append({"name": job["name"], "error": f"{type(exc).__name__}: {exc}"})
        state["updated_at"] = _timestamp()
        _write_json(_state_path(output_dir, worker_index), state)

    state["status"] = "completed" if not state["failures"] else "error"
    state["finished_at"] = _timestamp()
    state["updated_at"] = state["finished_at"]
    _write_json(_state_path(output_dir, worker_index), state)
    return 0 if state["status"] == "completed" else 1


def _selected_record(record: dict[str, Any], plan: dict[str, Any]) -> bool:
    return (
        record.get("method") in set(plan["methods"])
        and any(
            record.get("dataset") == job["dataset"] and record.get("cover") == job["cover"]
            for job in plan["jobs"]
        )
    )


def _collect_shard_records(plan: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = Path(plan["output_dir"])
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for job in plan["jobs"]:
        shard_dir = output_dir / "shards" / str(job["name"])
        for record in benchmark._completed_records(shard_dir):
            if not _selected_record(record, plan):
                continue
            key = (
                record.get("dataset"),
                record.get("cover"),
                record.get("method"),
                record.get("seed"),
                record.get("resolution"),
            )
            if key not in seen:
                seen.add(key)
                records.append(record)
    return sorted(
        records,
        key=lambda record: (
            str(record.get("dataset")),
            str(record.get("cover")),
            str(record.get("method")),
            int(record.get("seed", 0)),
            float(record.get("resolution", 0)),
        ),
    )


def _expected_count(plan: dict[str, Any]) -> int:
    seeds = benchmark.parse_seeds(str(plan["seeds"])) or []
    resolutions = benchmark.parse_resolutions(str(plan["resolutions"])) or []
    return len(plan["jobs"]) * len(plan["methods"]) * len(seeds) * len(resolutions)


def _audit_records(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("status", "unknown")) for record in records)
    expected = _expected_count(plan)
    retryable = sum(counts[status] for status in RETRYABLE_STATUSES)
    seeds = benchmark.parse_seeds(str(plan["seeds"])) or []
    expected_bases = {
        (job["dataset"], job["cover"], method, seed)
        for job in plan["jobs"]
        for method in plan["methods"]
        for seed in seeds
    }
    expected_per_base = len(benchmark.parse_resolutions(str(plan["resolutions"])) or [])
    observed_bases = Counter(
        (record.get("dataset"), record.get("cover"), record.get("method"), record.get("seed"))
        for record in records
    )
    missing_bases = sorted(
        "/".join(map(str, base))
        for base in expected_bases
        if observed_bases[base] != expected_per_base
    )
    return {
        "expected_records": expected,
        "observed_records": len(records),
        "status_counts": dict(sorted(counts.items())),
        "missing_records": max(0, expected - len(records)),
        "missing_or_incomplete_conditions": missing_bases,
        "retryable_records": retryable,
        "non_completed_records": expected - counts["completed"],
        "ready_for_paper": (
            plan.get("profile") == "full"
            and len(records) == expected
            and not missing_bases
            and counts["completed"] == expected
        ),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _paper_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record.get("dataset")),
                str(record.get("cover")),
                str(record.get("method")),
                float(record.get("resolution", 0)),
            )
        ].append(record)
    result: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        dataset, cover, method, resolution = key
        row: dict[str, Any] = {
            "dataset": dataset,
            "cover": cover,
            "method": method,
            "resolution": resolution,
            "n_records": len(group),
        }
        statuses = Counter(str(record.get("status", "unknown")) for record in group)
        for status, count in sorted(statuses.items()):
            row[f"n_{status}"] = count
        completed = [record for record in group if record.get("status") == "completed"]
        row["n_completed"] = len(completed)
        for metric in _PAPER_METRICS:
            values = [
                value
                for record in completed
                for value in [_numeric((record.get("metrics") or {}).get(metric))]
                if value is not None
            ]
            mean, ci = _mean_ci(values)
            if mean is not None:
                row[f"mean_{metric}"] = mean
                row[f"ci95_{metric}"] = ci
                row[f"n_{metric}"] = len(values)
        result.append(row)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("&", "\\&")


def _tex_method(method: str) -> str:
    return {
        "hedonic_multiphase": r"\methodmulti{}",
        "hedonic_multiphase_x10": r"\methodmultiTen{}",
        "hedonic_multiphase_x100": r"\methodmultiHundred{}",
        "cpm": r"\methodcpm{}",
        "demon": r"\methoddemon{}",
    }.get(method, r"\texttt{" + _tex_escape(method) + "}")


def _format_metric(row: dict[str, Any], metric: str, *, seconds: bool = False) -> str:
    value = row.get(f"mean_{metric}")
    if not isinstance(value, (int, float)):
        return "--"
    ci = row.get(f"ci95_{metric}")
    if seconds:
        if value < 1:
            text = f"{value:.3f}"
            ci_text = f"{float(ci):.3f}" if isinstance(ci, (int, float)) else None
        else:
            text = f"{value:.1f}"
            ci_text = f"{float(ci):.1f}" if isinstance(ci, (int, float)) else None
    else:
        text = f"{value:.3f}"
        ci_text = f"{float(ci):.3f}" if isinstance(ci, (int, float)) else None
    return text if ci_text is None else f"{text} $\\pm$ {ci_text}"


def _write_paper_tex(output_dir: Path, plan: dict[str, Any], audit: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    complete_rows = [row for row in rows if int(row.get("n_completed", 0)) > 0]
    status_counts = audit["status_counts"]
    lines = [
        "% Generated by hedonic-exp reproduce-overlapping-paper. Do not edit.",
        r"\subsection{Ground-truth recovery}",
        (
            "The reproducibility runner aggregated "
            f"{audit['observed_records']} of {audit['expected_records']} planned run records "
            "from the immutable per-condition caches.  Entries are mean $\\pm$ 95\\% "
            "normal-approximation confidence interval across completed seeds; status counts "
            "remain explicit rather than being silently discarded."
        ),
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full SNAP recovery results generated from cached run records. $n$ is the number of completed seeds; \texttt{--} means no completed estimate.}",
        r"\label{tab:full-recovery}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $n$ & Best-match $F_1$ & Matched $F_1$ & Node micro-$F_1$ & Omega & Runtime (s) \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = _tex_escape(str(row["dataset"]))
        lines.append(
            " & ".join(
                [
                    dataset,
                    _tex_method(str(row["method"])),
                    str(row.get("n_completed", 0)),
                    _format_metric(row, "symmetric_best_match_f1"),
                    _format_metric(row, "matching_f1"),
                    _format_metric(row, "node_membership_micro_f1"),
                    _format_metric(row, "omega"),
                    _format_metric(row, "runtime_seconds", seconds=True),
                ]
            )
            + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            (
                r"\par\smallskip\footnotesize Status counts across all planned records: "
                + ", ".join(f"{_tex_escape(status)}={count}" for status, count in sorted(status_counts.items()))
                + "."
            ),
            r"\end{table*}",
            r"\subsection{Overlap structure and CPM quality}",
            (
                "The machine-readable paper summary includes coverage, inclusion, predicted overlap, "
                "community-size, membership, and CPM-quality diagnostics for every completed condition. "
                "CPM quality is deliberately marked unavailable where full-network computation is skipped."
            ),
            r"\begin{figure*}[t]",
            r"  \centering",
            r"  \includegraphics[width=.96\textwidth]{accuracy_by_dataset.pdf}",
            r"  \caption{Symmetric best-match recovery by dataset and method; regenerated from the full cached runs.}",
            r"  \label{fig:full-accuracy}",
            r"\end{figure*}",
            r"\begin{figure*}[t]",
            r"  \centering",
            r"  \includegraphics[width=.96\textwidth]{overlap_structure.pdf}",
            r"  \caption{Predicted overlapping-node fractions by dataset and method.}",
            r"  \label{fig:full-overlap}",
            r"\end{figure*}",
            r"\subsection{Runtime and completion}",
            (
                "Runtime figures report detector wall-clock time for completed conditions. "
                "Timeouts, errors, unavailable baselines, and intentionally skipped conditions are retained "
                "in the artifact manifest and are not converted into artificial runtimes."
            ),
            r"\begin{figure*}[t]",
            r"  \centering",
            r"  \includegraphics[width=.96\textwidth]{runtime_by_dataset.pdf}",
            r"  \caption{Completed-condition detector runtime by dataset and method.}",
            r"  \label{fig:full-runtime}",
            r"\end{figure*}",
        ]
    )
    if not complete_rows:
        lines.insert(2, r"\textbf{No completed full-network records are available yet.}")
    (output_dir / "paper_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_status(output_dir: Path, audit: dict[str, Any]) -> None:
    if audit["ready_for_paper"]:
        state = r"\smokeresultsfalse"
        label = "complete full protocol"
    else:
        state = r"\smokeresultstrue"
        label = "incomplete or non-full protocol"
    (output_dir / "paper_status.tex").write_text(
        "% Generated by hedonic-exp reproduce-overlapping-paper.\n"
        f"% State: {label}.\n{state}\n",
        encoding="utf-8",
    )


def _compile_paper(plan: dict[str, Any], audit: dict[str, Any]) -> int:
    if not bool(plan.get("compile_paper")) or not bool(audit["ready_for_paper"]):
        return 0
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        print("[paper] latexmk is unavailable; artifacts were generated but PDF was not compiled.", file=sys.stderr)
        return 1
    paper_dir = Path(plan["paper_dir"])
    proc = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=paper_dir,
        text=True,
    )
    return int(proc.returncode)


def finalize(plan: dict[str, Any]) -> int:
    """Merge shard caches and materialize aggregate, plot, and paper artifacts."""
    output_dir = Path(plan["output_dir"])
    records = _collect_shard_records(plan)
    audit = _audit_records(plan, records)
    rows = benchmark._write_results(output_dir, records)
    benchmark._write_summary(output_dir, rows)
    plots = benchmark._write_plots(output_dir, rows)
    paper_rows = _paper_summary(records)
    _write_csv(output_dir / "paper_summary.csv", paper_rows)
    paper_manifest = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "plan_id": plan["plan_id"],
        "config_path": plan["config_path"],
        "methods": plan["methods"],
        "jobs": plan["jobs"],
        "audit": audit,
        "artifacts": {
            "results_csv": str(output_dir / "results.csv.gz"),
            "summary_csv": str(output_dir / "summary.csv"),
            "paper_summary_csv": str(output_dir / "paper_summary.csv"),
            "paper_results_tex": str(output_dir / "paper_results.tex"),
            "plots": plots,
        },
    }
    _write_json(output_dir / "paper_manifest.json", paper_manifest)
    # Keep the conventional top-level manifest name for tooling that already
    # understands benchmark artifacts; paper_manifest.json remains the more
    # explicit stable name for this orchestration layer.
    _write_json(output_dir / "manifest.json", paper_manifest)
    _write_paper_tex(output_dir, plan, audit, paper_rows)
    _write_paper_status(output_dir, audit)
    compile_code = _compile_paper(plan, audit)
    if compile_code != 0:
        # Do not leave a status file that asks TeX to use full results after a
        # failed render. The cached numerical artifacts remain intact.
        (output_dir / "paper_status.tex").write_text(
            "% Generated full artifacts, but manuscript compilation failed.\n"
            "\\smokeresultstrue\n",
            encoding="utf-8",
        )
        return compile_code
    print(f"[finalize] records={audit['observed_records']}/{audit['expected_records']}")
    print(f"[finalize] status_counts={json.dumps(audit['status_counts'], sort_keys=True)}")
    print(f"[finalize] artifacts: {output_dir}")
    if not audit["ready_for_paper"]:
        print(
            "[finalize] full-paper switch remains disabled: every expected condition must complete; fix unavailable/error/timeout records, then rerun with --finalize.",
            file=sys.stderr,
        )
    return 0


def run_coordinator(plan: dict[str, Any]) -> int:
    """Wait for tmux workers, then run the sole aggregate/paper writer."""
    output_dir = Path(plan["output_dir"])
    worker_count = len(plan["assignments"])
    print(f"[coordinator] waiting for {worker_count} worker windows", flush=True)
    while True:
        states = [_read_json(_state_path(output_dir, index)) for index in range(worker_count)]
        final = [
            state is not None
            and state.get("plan_id") == plan["plan_id"]
            and state.get("status") in {"completed", "error"}
            for state in states
        ]
        if all(final):
            break
        time.sleep(float(plan["poll_seconds"]))
    return finalize(plan)


def _tmux_command(plan_path: Path, action: str, worker_index: int | None = None) -> str:
    argv = [sys.executable, "-m", "hedonic.experiments.CLI", "reproduce-overlapping-paper", "--plan", str(plan_path)]
    if action == "worker":
        argv.extend(["--worker-index", str(worker_index)])
    else:
        argv.append("--coordinator")
    return "exec " + shlex.join(argv)


def launch_tmux(plan: dict[str, Any]) -> int:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required; install it or use --no-tmux for a foreground debug run")
    session = str(plan["tmux_session"])
    exists = subprocess.run(
        ["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    if exists:
        raise RuntimeError(
            f"tmux session {session!r} already exists; attach to it or set overlapping_paper.tmux_session to a new name"
        )
    plan_path = _plan_path(Path(plan["output_dir"])).resolve()
    for index in range(len(plan["assignments"])):
        target = f"worker-{index + 1}"
        command = _tmux_command(plan_path, "worker", index)
        if index == 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", target, command], check=True)
        else:
            subprocess.run(["tmux", "new-window", "-d", "-t", session, "-n", target, command], check=True)
        subprocess.run(["tmux", "set-window-option", "-t", f"{session}:{target}", "remain-on-exit", "on"], check=True)
    subprocess.run(
        ["tmux", "new-window", "-d", "-t", session, "-n", "coordinator", _tmux_command(plan_path, "coordinator")],
        check=True,
    )
    subprocess.run(["tmux", "set-window-option", "-t", f"{session}:coordinator", "remain-on-exit", "on"], check=True)
    print(f"[launch] tmux session: {session}")
    print(f"[launch] attach with: tmux attach -t {session}")
    print(f"[launch] plan: {plan_path}")
    return 0


def run_foreground(plan: dict[str, Any]) -> int:
    """Debug/test mode without tmux; intentionally serial and race-free."""
    code = 0
    for index in range(len(plan["assignments"])):
        code = max(code, run_worker(plan, index))
    finalized = finalize(plan)
    return max(code, finalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce docs/papers/overlapping_communities/main.tex from the "
            "[overlapping_paper] TOML protocol. Launches RAM-bounded tmux workers, "
            "merges their caches, regenerates figures/tables, and compiles only a complete full run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="TOML config (default: configs/hedonic.toml from cwd)")
    parser.add_argument("--workers", type=int, help="Override TOML worker count after CPU/RAM safety checks")
    parser.add_argument("--session", help="Override TOML tmux session name")
    parser.add_argument("--no-tmux", action="store_true", help="Foreground serial debug/test run; normal reproduction uses tmux")
    parser.add_argument("--dry-run", action="store_true", help="Validate TOML and write the balanced worker plan without running it")
    parser.add_argument("--finalize", action="store_true", help="Merge existing shard caches and regenerate paper artifacts only")
    parser.add_argument("--plan", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan:
        plan = _load_plan(Path(args.plan).expanduser().resolve())
        if args.worker_index is not None:
            return run_worker(plan, args.worker_index)
        if args.coordinator:
            return run_coordinator(plan)
        if args.finalize:
            return finalize(plan)
        raise ValueError("Internal --plan requires --worker-index, --coordinator, or --finalize")
    if args.worker_index is not None or args.coordinator:
        raise ValueError("Internal worker/coordinator actions require --plan")

    options, _config_path = _load_options(args)
    plan = _write_new_plan(options)
    if args.dry_run:
        print(json.dumps({"plan": str(_plan_path(Path(plan["output_dir"]))), "workers": plan["workers"], "assignments": plan["assignments"]}, indent=2))
        return 0
    if args.finalize:
        return finalize(plan)
    if args.no_tmux:
        return run_foreground(plan)
    return launch_tmux(plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
