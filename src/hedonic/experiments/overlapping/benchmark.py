"""Reproducible multi-network overlapping-community benchmark CLI.

``hedonic-exp overlapping-benchmark`` is intentionally self-contained at the
experiment layer: it loads the five supported SNAP covers, delegates methods
through adapters, writes resumable run records, and emits aggregate tables and
plots.  It never writes inside a raw network directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments.config import OUTPUT_DIR
from hedonic.experiments.overlapping.methods import (
    METHODS,
    effective_resolution,
    method_availability,
    resolve_methods,
    run_method,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    quality_overlapping_cpm,
)
from hedonic.experiments.overlapping.snap import (
    DEFAULT_NETWORKS_DIR,
    SnapDataset,
    SnapLoadError,
    UnsupportedCoverVariant,
    bounded_induced_dataset,
    load_snap_dataset,
    network_names,
    print_dataset_report,
    smoke_dataset,
)


SCHEMA_VERSION = 1
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "cover": "all",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": 64,
        "timeout_per_run": 30.0,
        "plots": True,
    },
    "standard": {
        "cover": "top5000",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": 3_000,
        "timeout_per_run": 180.0,
        "plots": True,
    },
    "full": {
        "cover": "top5000",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": None,
        "timeout_per_run": 3_600.0,
        "plots": True,
    },
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _parse_csv(value: str | None, valid: Iterable[str], option: str) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = set(valid)
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {option}: {', '.join(unknown)}; choose from {', '.join(sorted(allowed))}"
        )
    return values


def parse_seeds(value: str | None) -> list[int] | None:
    """Parse ``0-4`` and comma-list seed syntax without a third-party helper."""
    if value is None:
        return None
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            step = 1 if stop >= start else -1
            values.extend(range(start, stop + step, step))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("--seeds must contain at least one integer")
    return list(dict.fromkeys(values))


def parse_resolutions(value: str | None) -> list[str | float] | None:
    """Parse ``auto``, comma values, or inclusive ``start:stop:count`` grids."""
    if value is None:
        return None
    result: list[str | float] = []
    for part in value.split(","):
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
            if count == 1:
                result.append(start)
            else:
                result.extend(start + (stop - start) * i / (count - 1) for i in range(count))
        else:
            result.append(float(part))
    if not result:
        raise ValueError("--resolutions must contain at least one value")
    return list(dict.fromkeys(result))


def _resolution_label(resolution: float) -> str:
    return f"{resolution:.12g}".replace("-", "m").replace(".", "p")


def _run_path(
    output_dir: Path,
    dataset: str,
    cover: str,
    method: str,
    seed: int,
    resolution: float,
) -> Path:
    return (
        output_dir
        / "runs"
        / dataset
        / cover
        / method
        / f"seed_{seed}"
        / f"resolution_{_resolution_label(resolution)}.json"
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _append_log(output_dir: Path, message: str) -> None:
    """Persist concise orchestration events alongside the printed progress."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "benchmark.log").open("a", encoding="utf-8") as f:
        f.write(f"{_timestamp()} {message}\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _worker(
    queue,
    method_name: str,
    graph,
    max_memberships: int,
    resolution: float,
    seed: int,
    memory_limit_bytes: int | None,
) -> None:
    """Subprocess entry point used for enforceable per-run wall-clock limits."""
    memory: dict[str, Any] = {"limit_bytes": memory_limit_bytes}
    try:
        cover, method_meta = run_method(
            METHODS[method_name],
            graph,
            max_memberships=max_memberships,
            resolution=resolution,
            seed=seed,
        )
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # macOS reports bytes; Linux reports KiB.
            memory["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        except (AttributeError, ValueError):  # pragma: no cover - platform-specific
            pass
        queue.put(
            {"status": "ok", "cover": cover, "method_meta": method_meta, "memory": memory}
        )
    except BaseException as exc:  # child errors must reach a resumable run record
        queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "memory": memory,
            }
        )


def _process_rss_bytes(pid: int) -> int | None:
    """Return current RSS for an isolated detector process, if observable."""
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(output) * 1024 if output else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _run_with_timeout(
    method_name: str,
    graph,
    *,
    max_memberships: int,
    resolution: float,
    seed: int,
    timeout_seconds: float | None,
    memory_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Run a detector in an isolated process, terminating genuine timeouts."""
    if timeout_seconds is None or timeout_seconds <= 0:
        try:
            cover, method_meta = run_method(
                METHODS[method_name],
                graph,
                max_memberships=max_memberships,
                resolution=resolution,
                seed=seed,
            )
            return {"status": "ok", "cover": cover, "method_meta": method_meta}
        except BaseException as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    # ``fork`` avoids copying bounded igraph inputs through a second pickle on
    # Unix; Windows falls back to spawn.  The result still crosses a queue.
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else "spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            queue,
            method_name,
            graph,
            max_memberships,
            resolution,
            seed,
            memory_limit_bytes,
        ),
    )
    started = time.monotonic()
    process.start()
    peak_rss_bytes = 0
    while process.is_alive():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            process.terminate()
            process.join(5)
            return {
                "status": "timeout",
                "runtime_seconds": elapsed,
                "timeout_seconds": timeout_seconds,
                "memory": {
                    "limit_bytes": memory_limit_bytes,
                    "enforcement": "parent_rss_monitor" if memory_limit_bytes else None,
                    "observed_peak_rss_bytes": peak_rss_bytes,
                },
            }
        rss_bytes = _process_rss_bytes(process.pid)
        if rss_bytes is not None:
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            if memory_limit_bytes is not None and rss_bytes > memory_limit_bytes:
                process.terminate()
                process.join(5)
                return {
                    "status": "memory_limit",
                    "runtime_seconds": elapsed,
                    "memory": {
                        "limit_bytes": memory_limit_bytes,
                        "enforcement": "parent_rss_monitor",
                        "observed_peak_rss_bytes": peak_rss_bytes,
                    },
                }
        process.join(min(0.25, max(0.01, timeout_seconds - elapsed)))
    elapsed = time.monotonic() - started
    try:
        packet = queue.get(timeout=2)
    except Exception:
        return {
            "status": "error",
            "runtime_seconds": elapsed,
            "error": f"Detector subprocess exited without a result (exit={process.exitcode})",
        }
    packet["runtime_seconds"] = elapsed
    memory = packet.get("memory")
    if not isinstance(memory, dict):
        memory = {}
        packet["memory"] = memory
    memory.update(
        {
            "limit_bytes": memory_limit_bytes,
            "enforcement": "parent_rss_monitor" if memory_limit_bytes else None,
            "observed_peak_rss_bytes": peak_rss_bytes,
        }
    )
    return packet


def _record(
    *,
    dataset: str,
    cover: str,
    method: str,
    seed: int,
    resolution: float,
    status: str,
    profile: str,
    dataset_report: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _timestamp(),
        "dataset": dataset,
        "cover": cover,
        "method": method,
        "seed": seed,
        "resolution": resolution,
        "status": status,
        "profile": profile,
        "dataset_report": dataset_report,
        **extra,
    }


def _completed_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runs = output_dir / "runs"
    if not runs.is_dir():
        return records
    for path in sorted(runs.rglob("*.json")):
        record = _read_json(path)
        if record is not None:
            record["run_path"] = str(path)
            records.append(record)
    return records


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key not in {"metrics", "dataset_report", "method_metadata", "traceback"}
    }
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        row.update(metrics)
    method_metadata = record.get("method_metadata")
    if isinstance(method_metadata, dict):
        row["method_parameters"] = json.dumps(
            method_metadata.get("parameters", {}), sort_keys=True
        )
    return row


def _write_results(output_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_flatten_record(record) for record in records]
    jsonl = output_dir / "results.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    columns = sorted({key for row in rows for key in row})
    with gzip.open(output_dir / "results.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("dataset"), row.get("cover"), row.get("method"), row.get("resolution"), row.get("status"))
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, cover, method, resolution, status), group in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "dataset": dataset,
            "cover": cover,
            "method": method,
            "resolution": resolution,
            "status": status,
            "n_runs": len(group),
        }
        numeric_keys = {
            key
            for row in group
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group if isinstance(row.get(key), (int, float))]
            if values:
                summary_row[f"mean_{key}"] = sum(values) / len(values)
                summary_row[f"min_{key}"] = min(values)
                summary_row[f"max_{key}"] = max(values)
        summary_rows.append(summary_row)
    return {"schema_version": SCHEMA_VERSION, "groups": summary_rows}, summary_rows


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary, summary_rows = _summary(rows)
    _write_json(output_dir / "summary.json", summary)
    columns = sorted({key for row in summary_rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def _plot_metric(
    rows: list[dict[str, Any]], *, metric: str, title: str, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    usable = [
        row
        for row in rows
        if row.get("status") == "completed" and isinstance(row.get(metric), (int, float))
    ]
    if not usable:
        return False
    labels = [f"{row['dataset']}\n{row['method']}" for row in usable]
    values = [float(row[metric]) for row in usable]
    fig, axis = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_title(title)
    axis.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    requested = [
        ("symmetric_best_match_f1", "Recovery accuracy by dataset", "accuracy_by_dataset"),
        ("runtime_seconds", "Runtime by dataset", "runtime_by_dataset"),
        ("predicted_overlapping_node_fraction", "Predicted overlap structure", "overlap_structure"),
        ("matching_f1", "One-to-one method comparison", "method_comparison"),
    ]
    written: list[str] = []
    for metric, title, stem in requested:
        if _plot_metric(rows, metric=metric, title=title, path=plots_dir / stem):
            written.extend([str(plots_dir / f"{stem}.png"), str(plots_dir / f"{stem}.pdf")])
    return written


def _safe_output_dir(output_dir: Path, data_root: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    protected_inputs = [
        data_root / name
        for name in ("Amazon", "DBLP", "LiveJournal", "Youtube", "YouTube", "Wikipedia")
    ]
    if output_dir == data_root or any(
        output_dir == protected or protected in output_dir.parents
        for protected in protected_inputs
    ):
        raise ValueError("--output_dir must not overwrite a raw SNAP dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_dataset_status_records(
    *,
    output_dir: Path,
    dataset: str,
    cover: str,
    methods,
    seeds: list[int],
    resolutions: list[str | float],
    profile: str,
    status: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Write resumable records when data/cover selection prevents a run."""
    records: list[dict[str, Any]] = []
    for adapter in methods:
        for seed in seeds:
            for requested_resolution in resolutions:
                resolution = 0.0 if requested_resolution == "auto" else float(requested_resolution)
                record = _record(
                    dataset=dataset,
                    cover=cover,
                    method=adapter.name,
                    seed=seed,
                    resolution=resolution,
                    status=status,
                    profile=profile,
                    reason=reason,
                )
                path = _run_path(output_dir, dataset, cover, adapter.name, seed, resolution)
                _write_json(path, record)
                records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible overlapping-community benchmark for Amazon, DBLP, "
            "LiveJournal, YouTube, and Wikipedia SNAP covers. Standard runs "
            "are bounded induced subgraphs; use --profile full to disable that cap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", help="SNAP root (default: HEDONIC_NETWORKS_DIR or ~/Databases/Hedonic/Networks)")
    parser.add_argument("--datasets", help="Comma list: amazon,dblp,livejournal,youtube,wikipedia")
    parser.add_argument("--cover", choices=("all", "top5000"), help="Supplied GT cover variant; Wikipedia provides all categories only")
    parser.add_argument("--methods", help="Comma-list of methods; use --list-methods for details")
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="standard")
    parser.add_argument("--seeds", help="Comma/range syntax, e.g. 0-4")
    parser.add_argument("--resolutions", help="auto, comma floats, or start:stop:count")
    parser.add_argument("--output_dir", help="Artifact root (default: ~/Databases/Hedonic/experiments/snap_benchmark)")
    parser.add_argument("--resume", action="store_true", help="Skip existing per-run records")
    parser.add_argument("--timeout_per_run", type=float, help="Hard wall-clock limit per detector run (seconds; <=0 disables)")
    parser.add_argument("--max_nodes", type=int, help="Deterministic GT-informed induced-subgraph cap; <=0 disables")
    parser.add_argument(
        "--max_memberships",
        type=int,
        help=(
            "Cap memberships per vertex for hedonic methods; the default is the "
            "maximum ground-truth memberships of any node, not the number of communities"
        ),
    )
    parser.add_argument(
        "--memory_limit_gb",
        type=float,
        help=(
            "RSS limit for each isolated detector process; <=0 disables. A parent monitor "
            "terminates an over-limit child. The paper scheduler supplies a graph-aware value."
        ),
    )
    parser.add_argument("--omega", action="store_true", help="Compute memory-safe sampled Omega")
    parser.add_argument("--omega_sample_size", type=int, default=100_000)
    parser.add_argument("--plots", dest="plots", action="store_true", default=None)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--list-networks", action="store_true", help="List supported overlapping SNAP datasets and exit")
    parser.add_argument("--list-methods", action="store_true", help="List adapters and availability then exit")
    parser.add_argument("--dry-run", action="store_true", help="Inspect/validate selected data and write no detector runs")
    return parser


def _print_networks() -> None:
    print("Overlapping SNAP benchmarks:")
    print("  amazon      product cover: all, top5000")
    print("  dblp        co-authorship cover: all, top5000")
    print("  livejournal social-community cover: all, top5000")
    print("  youtube     channel cover: all, top5000")
    print("  wikipedia   wiki-topcats category cover: all (directed graph)")
    print("Excluded: email-Eu-core, Cora, and PubMed have disjoint labels; DBLP_CLI is output only.")


def _print_methods() -> None:
    print("Overlapping benchmark methods:")
    for name, info in method_availability().items():
        state = "available" if info["available"] else f"unavailable: {info['reason']}"
        print(f"  {name:20} {state}")
        print(f"    {info['family']} | {info['implementation']}")
        print(f"    parameters={info['parameters']} | {info['scalability']}")
        if info["install_requirement"]:
            print(f"    install: {info['install_requirement']}")


def _effective_options(args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILE_DEFAULTS[args.profile]
    datasets = _parse_csv(args.datasets, network_names(), "dataset") or profile["datasets"]
    methods = _parse_csv(args.methods, METHODS, "method") or profile["methods"]
    if args.max_memberships is not None and args.max_memberships < 1:
        raise ValueError("--max_memberships must be >= 1")
    if args.memory_limit_gb is not None and args.memory_limit_gb <= 0:
        raise ValueError("--memory_limit_gb must be positive when supplied")
    return {
        "datasets": datasets,
        "methods": methods,
        "cover": args.cover or profile["cover"],
        "seeds": parse_seeds(args.seeds) or profile["seeds"],
        "resolutions": parse_resolutions(args.resolutions) or profile["resolutions"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else profile["max_nodes"],
        "max_memberships": args.max_memberships,
        "memory_limit_bytes": (
            int(args.memory_limit_gb * (1024**3))
            if args.memory_limit_gb is not None
            else None
        ),
        "timeout_per_run": (
            args.timeout_per_run
            if args.timeout_per_run is not None
            else profile["timeout_per_run"]
        ),
        "plots": args.plots if args.plots is not None else profile["plots"],
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the configured suite. Public for small-fixture tests and scripts."""
    options = _effective_options(args)
    selected_methods = resolve_methods(options["methods"])
    data_root = Path(args.data_root).expanduser() if args.data_root else DEFAULT_NETWORKS_DIR
    default_output = Path(OUTPUT_DIR) / "snap_benchmark"
    output_dir = _safe_output_dir(
        Path(args.output_dir).expanduser() if args.output_dir else default_output,
        data_root,
    )
    availability = method_availability()
    _append_log(output_dir, f"start profile={args.profile} datasets={','.join(options['datasets'])}")
    _write_json(output_dir / "method_availability.json", availability)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _timestamp(),
        "command": "hedonic-exp overlapping-benchmark",
        "profile": args.profile,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "options": {**options, "omega": args.omega, "omega_sample_size": args.omega_sample_size},
        "method_availability": availability,
        "datasets": {},
        "run_status_counts": {},
    }
    _write_json(output_dir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    for dataset_name in options["datasets"]:
        try:
            dataset = (
                smoke_dataset(dataset_name, cover_variant=options["cover"])
                if args.profile == "smoke"
                else load_snap_dataset(
                    dataset_name,
                    cover_variant=options["cover"],
                    data_root=data_root,
                )
            )
            dataset = bounded_induced_dataset(dataset, options["max_nodes"])
            print_dataset_report(dataset.report)
            manifest["datasets"][dataset_name] = {"status": "loaded", "report": dataset.report}
        except UnsupportedCoverVariant as exc:
            manifest["datasets"][dataset_name] = {"status": "skipped", "reason": str(exc)}
            records.extend(
                _write_dataset_status_records(
                    output_dir=output_dir,
                    dataset=dataset_name,
                    cover=options["cover"],
                    methods=selected_methods,
                    seeds=options["seeds"],
                    resolutions=options["resolutions"],
                    profile=args.profile,
                    status="skipped",
                    reason=str(exc),
                )
            )
            _append_log(output_dir, f"dataset={dataset_name} status=skipped reason={exc}")
            continue
        except SnapLoadError as exc:
            manifest["datasets"][dataset_name] = {"status": "data_unavailable", "reason": str(exc)}
            print(f"[dataset] {dataset_name}: unavailable: {exc}", file=sys.stderr)
            records.extend(
                _write_dataset_status_records(
                    output_dir=output_dir,
                    dataset=dataset_name,
                    cover=options["cover"],
                    methods=selected_methods,
                    seeds=options["seeds"],
                    resolutions=options["resolutions"],
                    profile=args.profile,
                    status="data_unavailable",
                    reason=str(exc),
                )
            )
            _append_log(output_dir, f"dataset={dataset_name} status=data_unavailable reason={exc}")
            continue

        if args.dry_run:
            continue
        ground_truth_membership_cap = max(
            1,
            int(
                dataset.report.get("overlap_statistics", {}).get(
                    "max_memberships_per_node", 1
                )
            ),
        )
        configured_membership_cap = options["max_memberships"]
        max_memberships = max(
            2,
            min(
                ground_truth_membership_cap,
                configured_membership_cap
                if configured_membership_cap is not None
                else ground_truth_membership_cap,
            ),
        )
        manifest["datasets"][dataset_name]["membership_capacity"] = {
            "ground_truth_max_memberships_per_node": ground_truth_membership_cap,
            "configured_cap": configured_membership_cap,
            "effective_max_memberships": max_memberships,
        }
        for requested_resolution in options["resolutions"]:
            requested_value = (
                dataset.graph.density()
                if requested_resolution == "auto"
                else float(requested_resolution)
            )
            for adapter in selected_methods:
                resolution = effective_resolution(adapter, dataset.graph, requested_value)
                for seed in options["seeds"]:
                    path = _run_path(output_dir, dataset.name, options["cover"], adapter.name, seed, resolution)
                    if args.resume and path.is_file():
                        existing = _read_json(path)
                        if existing is not None and existing.get("status") in {
                            "completed",
                            "unavailable",
                            "skipped",
                            "data_unavailable",
                            "memory_limit",
                        } and existing.get("max_memberships") == max_memberships:
                            existing["run_path"] = str(path)
                            records.append(existing)
                            print(f"[resume] {dataset.name}/{adapter.name}/seed={seed}/resolution={resolution:.6g}")
                            continue
                    if not availability[adapter.name]["available"]:
                        record = _record(
                            dataset=dataset.name,
                            cover=options["cover"],
                            method=adapter.name,
                            seed=seed,
                            resolution=resolution,
                            status="unavailable",
                            profile=args.profile,
                            dataset_report=dataset.report,
                            reason=availability[adapter.name]["reason"],
                            install_requirement=availability[adapter.name]["install_requirement"],
                        )
                    else:
                        print(
                            f"[run] {dataset.name}/{adapter.name} seed={seed} "
                            f"resolution={resolution:.6g}",
                            flush=True,
                        )
                        outcome = _run_with_timeout(
                            adapter.name,
                            dataset.graph,
                            max_memberships=max_memberships,
                            resolution=resolution,
                            seed=seed,
                            timeout_seconds=options["timeout_per_run"],
                            memory_limit_bytes=options["memory_limit_bytes"],
                        )
                        if outcome["status"] == "ok":
                            cover = outcome.pop("cover")
                            method_meta = outcome.pop("method_meta")
                            detector_memory = outcome.pop("memory", None)
                            metrics = evaluate_cover(
                                cover,
                                dataset.cover,
                                dataset.graph.vcount(),
                                compute_omega=args.omega,
                                omega_sample_size=args.omega_sample_size,
                                omega_seed=seed,
                            )
                            if dataset.graph.vcount() <= 20_000:
                                metrics["cpm_overlapping_quality"] = quality_overlapping_cpm(
                                    dataset.graph, cover, resolution
                                )
                                metrics["cpm_overlapping_quality_status"] = "computed"
                            else:
                                metrics["cpm_overlapping_quality"] = None
                                metrics["cpm_overlapping_quality_status"] = "skipped_large_graph"
                            metrics["runtime_seconds"] = outcome.get(
                                "runtime_seconds", method_meta["runtime_seconds"]
                            )
                            record = _record(
                                dataset=dataset.name,
                                cover=options["cover"],
                                method=adapter.name,
                                seed=seed,
                                resolution=resolution,
                                status="completed",
                                profile=args.profile,
                                dataset_report=dataset.report,
                                max_memberships=max_memberships,
                                ground_truth_max_memberships=ground_truth_membership_cap,
                                memory_limit_bytes=options["memory_limit_bytes"],
                                detector_memory=detector_memory,
                                n_iterations=-1,
                                only_local_moving=bool(
                                    adapter.parameters.get("only_local_moving", False)
                                ),
                                allow_isolation=bool(
                                    adapter.parameters.get("allow_isolation", False)
                                ),
                                method_metadata=method_meta,
                                metrics=metrics,
                            )
                        else:
                            record = _record(
                                dataset=dataset.name,
                                cover=options["cover"],
                                method=adapter.name,
                                seed=seed,
                                resolution=resolution,
                                status=outcome.pop("status"),
                                profile=args.profile,
                                dataset_report=dataset.report,
                                max_memberships=max_memberships,
                                ground_truth_max_memberships=ground_truth_membership_cap,
                                memory_limit_bytes=options["memory_limit_bytes"],
                                **outcome,
                            )
                    _write_json(path, record)
                    _append_log(
                        output_dir,
                        f"dataset={dataset.name} method={adapter.name} seed={seed} "
                        f"resolution={resolution:.12g} status={record['status']}",
                    )
                    record["run_path"] = str(path)
                    records.append(record)
    # Reading the run tree includes resumed and previously complete points even
    # when a current invocation selected only a subset.
    all_records = _completed_records(output_dir)
    rows = _write_results(output_dir, all_records)
    _write_summary(output_dir, rows)
    plot_paths = _write_plots(output_dir, rows) if options["plots"] else []
    status_counts = CounterLike(record.get("status", "unknown") for record in all_records)
    manifest["run_status_counts"] = dict(status_counts)
    manifest["finished_at"] = _timestamp()
    manifest["artifacts"] = {
        "manifest": str(output_dir / "manifest.json"),
        "runs": str(output_dir / "runs"),
        "results_jsonl": str(output_dir / "results.jsonl"),
        "results_csv": str(output_dir / "results.csv.gz"),
        "summary_json": str(output_dir / "summary.json"),
        "summary_csv": str(output_dir / "summary.csv"),
        "method_availability": str(output_dir / "method_availability.json"),
        "logs": str(output_dir / "logs"),
        "plots": plot_paths,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_log(output_dir, f"finished status_counts={json.dumps(manifest['run_status_counts'], sort_keys=True)}")
    print(f"[done] Benchmark artifacts: {output_dir}")
    for name, path in manifest["artifacts"].items():
        if name != "plots":
            print(f"  {name}: {path}")
    return 0


class CounterLike(dict):
    """Tiny serializable Counter replacement, avoiding another import alias."""

    def __init__(self, values: Iterable[str]) -> None:
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_networks:
        _print_networks()
        return 0
    if args.list_methods:
        _print_methods()
        return 0
    try:
        return run_benchmark(args)
    except (ValueError, SnapLoadError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
