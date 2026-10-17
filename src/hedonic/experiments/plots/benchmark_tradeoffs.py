"""Chapter 6 joint plots: accuracy against detection time and accuracy against cover robustness.

One row per network, two panels per row, one marker per method. Every seed is a faint point; the solid marker is the
mean over seeds with a 95% Student-t interval on both axes. Accuracy and time come from the benchmark records
(``hedonic run paper`` / ``paper-hoc``); robustness comes from the audit of the same covers, which those runs write
to ``RUN_DIR/robustness`` (``--robustness``; ``hedonic run audit`` writes the same ledger for a run that lacks it, and
``--audit`` names such a folder): the share of the resolution range [1e-4, 1], on a log axis, at which a vertex's memberships are
a best response, averaged over vertices, in the ``cover`` profile that every method has.

    hedonic-exp overlapping-tradeoffs --benchmark RUN_DIR [RUN_DIR ...] [--audit AUDIT_DIR [AUDIT_DIR ...]]
    hedonic-exp overlapping-tradeoffs --smoke --output-dir artifacts/overlapping/benchmark_tradeoffs/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from hedonic.experiments.config import OUTPUT_DIR, expand_path  # noqa: E402
from hedonic.experiments.overlapping import cover_audit  # noqa: E402
from hedonic.experiments.overlapping import quickstart as qs  # noqa: E402

DEFAULT_METRIC = "f1"
MEASURES = {"log": "stable_measure_log", "linear": "stable_measure_linear"}
MEASURE_LABEL = {"log": "stable share of the log-resolution range [1e-4, 1]",
                 "linear": "stable share of the resolution range [0, 1]"}
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h"]
METRIC_LABEL = dict(qs.COLUMNS)
RUNNER_TO_KEY = {method.runner: method.key for method in qs.METHODS + qs.CONTROLS}  # controls: supervised references
ROW_FIELDS = ["network", "seed", "method", "accuracy", "seconds", "robustness"]
AGG_FIELDS = ["network", "method", "n", "accuracy_mean", "accuracy_ci95", "seconds_mean", "seconds_ci95",
              "n_robustness", "robustness_mean", "robustness_ci95"]


# --------------------------------------------------------------------------- data
def load_benchmark(folders: Sequence[Path], metric: str, networks: Sequence[str] | None = None,
                   methods: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Accuracy and detection time of every completed unit; the first folder holding a unit wins.

    Returns the rows and the number of units skipped because they did not complete or lack the metric.
    """
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    skipped = 0
    for folder in folders:
        for path in sorted(folder.glob("*/seed*/runs/*/*.json")):
            key = RUNNER_TO_KEY.get(path.stem)
            if key is None or (methods and key not in methods) or (networks and path.parent.name not in networks):
                continue
            record = json.loads(path.read_text())
            unit = (str(record.get("dataset") or path.parent.name), int(record.get("seed", -1)), key)
            if unit in rows:
                continue
            value = (record.get("metrics") or {}).get(metric)
            if record.get("status") != "completed" or value is None or record.get("detection_seconds") is None:
                skipped += 1
                continue
            rows[unit] = {"network": unit[0], "seed": unit[1], "method": key, "accuracy": float(value),
                          "seconds": float(record["detection_seconds"]), "robustness": None}
    return list(rows.values()), skipped


def load_robustness(folders: Sequence[Path], measure: str, policy: str) -> dict[tuple[str, int, str], float]:
    """The cover-profile robustness of every audited unit, by (network, seed, method)."""
    field = MEASURES[measure]
    out: dict[tuple[str, int, str], float] = {}
    for folder in folders:
        ledger = folder / cover_audit.RESULTS_NAME
        if not ledger.is_file():  # a benchmark run folder: its own audit stage keeps the ledger in robustness/
            ledger = folder / cover_audit.STAGE_DIR / cover_audit.RESULTS_NAME
        for record in cover_audit.read_records(ledger).values():
            if record.get("status") != "completed" or record.get("profile") != "cover":
                continue
            out.setdefault((record["network"], int(record["seed"]), record["method"]),
                           float(record["policies"][policy][field]))
    return out


def join(rows: list[dict[str, Any]], robustness: dict[tuple[str, int, str], float]) -> list[dict[str, Any]]:
    for row in rows:
        row["robustness"] = robustness.get((row["network"], row["seed"], row["method"]))
    return rows


def ci95(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(student_t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and 95% interval over seeds for each network and method (methods in the catalogue's order)."""
    order = {method.key: index for index, method in enumerate(qs.METHODS + qs.CONTROLS)}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["network"], row["method"]), []).append(row)
    out = []
    for (network, method), members in sorted(groups.items(), key=lambda item: (item[0][0], order[item[0][1]])):
        accuracy = [m["accuracy"] for m in members]
        seconds = [m["seconds"] for m in members]
        robust = [m["robustness"] for m in members if m["robustness"] is not None]
        out.append({"network": network, "method": method, "n": len(members),
                    "accuracy_mean": statistics.fmean(accuracy), "accuracy_ci95": ci95(accuracy),
                    "seconds_mean": statistics.fmean(seconds), "seconds_ci95": ci95(seconds),
                    "n_robustness": len(robust), "robustness_mean": statistics.fmean(robust) if robust else None,
                    "robustness_ci95": ci95(robust)})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- figure
def _bar(ci: float | None) -> float:
    return 0.0 if ci is None else ci


def plot(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], *, metric: str, measure: str, policy: str,
         output: Path, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    networks = sorted({row["network"] for row in rows}, key=list(qs.NETWORKS).index)
    present = [method for method in qs.METHODS + qs.CONTROLS if any(row["method"] == method.key for row in rows)]
    style = {method.key: (plt.get_cmap("tab10")(index % 10), MARKERS[index % len(MARKERS)])
             for index, method in enumerate(qs.METHODS)}
    style.update({method.key: ("black", "*") for method in qs.CONTROLS})  # supervised references stand apart
    fig, axes = plt.subplots(len(networks), 2, figsize=(10.5, 3.9 * len(networks)), squeeze=False)
    accuracy_label = f"Accuracy ({METRIC_LABEL.get(metric, metric)})"
    for r, network in enumerate(networks):
        for method in present:
            colour, marker = style[method.key]
            seeds = [row for row in rows if row["network"] == network and row["method"] == method.key]
            agg = next((a for a in aggregates if a["network"] == network and a["method"] == method.key), None)
            if not seeds or agg is None:
                continue
            axes[r][0].scatter([s["seconds"] for s in seeds], [s["accuracy"] for s in seeds], s=14, alpha=0.3,
                               color=colour, marker=marker, linewidths=0)
            axes[r][0].errorbar(agg["seconds_mean"], agg["accuracy_mean"], xerr=_bar(agg["seconds_ci95"]),
                                yerr=_bar(agg["accuracy_ci95"]), color=colour, marker=marker, markersize=8,
                                markeredgecolor="black", markeredgewidth=0.6, capsize=2.5, linewidth=1.0,
                                label=method.label)
            audited = [s for s in seeds if s["robustness"] is not None]
            if audited and agg["robustness_mean"] is not None:
                axes[r][1].scatter([s["robustness"] for s in audited], [s["accuracy"] for s in audited], s=14,
                                   alpha=0.3, color=colour, marker=marker, linewidths=0)
                axes[r][1].errorbar(agg["robustness_mean"], statistics.fmean(s["accuracy"] for s in audited),
                                    xerr=_bar(agg["robustness_ci95"]), yerr=_bar(ci95([s["accuracy"] for s in audited])),
                                    color=colour, marker=marker, markersize=8, markeredgecolor="black",
                                    markeredgewidth=0.6, capsize=2.5, linewidth=1.0)
        label = qs.NETWORKS[network][0]
        axes[r][0].set_xscale("log")
        axes[r][0].set_xlabel("Detection time (s, log scale)")
        axes[r][1].set_xlabel(f"Robustness ({policy.replace('_', ' ')}):\n{MEASURE_LABEL[measure]}")
        axes[r][1].set_xlim(-0.02, 1.02)
        for ax in axes[r]:
            ax.set_ylabel(accuracy_label)
            ax.grid(alpha=0.25, linewidth=0.5)
        axes[r][0].set_title(f"{label}: accuracy vs detection time", fontsize=10)
        axes[r][1].set_title(f"{label}: accuracy vs cover robustness", fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, max(1, len(labels))), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07 if len(networks) == 1 else 0.04, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = output / f"benchmark_tradeoffs.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- CLI
def smoke_rows() -> list[dict[str, Any]]:
    """Synthetic rows (no benchmark needed): two networks, four methods, five seeds."""
    rng = random.Random(0)
    rows = []
    for network in ("dblp", "amazon"):
        for method, (acc, sec, rob) in {"hedonic": (0.45, 40, 0.80), "codeseg": (0.40, 6, 0.55),
                                        "clique_percolation": (0.42, 9, 0.35), "bigclam": (0.30, 580, 0.25)}.items():
            for seed in range(5):
                rows.append({"network": network, "seed": seed, "method": method,
                             "accuracy": acc + rng.gauss(0, 0.01), "seconds": sec * math.exp(rng.gauss(0, 0.08)),
                             "robustness": min(1.0, max(0.0, rob + rng.gauss(0, 0.02)))})
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-tradeoffs",
        description="Chapter 6 joint plots: accuracy vs detection time and accuracy vs cover robustness.")
    parser.add_argument("--benchmark", nargs="+", metavar="RUN_DIR", help="benchmark run folders (accuracy and time)")
    parser.add_argument("--audit", nargs="+", metavar="AUDIT_DIR",
                        help="extra cover-audit folders (robustness) for units whose benchmark run has no robustness/ "
                             "ledger of its own; the ledgers inside the --benchmark folders are always read")
    parser.add_argument("--networks", help="comma list (default: those in the benchmark)")
    parser.add_argument("--methods", help="comma list of method keys (default: all in the benchmark)")
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=[key for key, _ in qs.COLUMNS],
                        help=f"accuracy metric (default {DEFAULT_METRIC}: symmetric best-match F1)")
    parser.add_argument("--measure", default="log", choices=sorted(MEASURES),
                        help="robustness measure: log (default) or linear stable share of the resolution range")
    parser.add_argument("--policy", default=cover_audit.PRIMARY_POLICY,
                        choices=[name for name, _ in cover_audit.POLICIES], help="action space of the audit")
    parser.add_argument("--formats", default="pdf,png", help="figure formats (default pdf,png)")
    parser.add_argument("--output-dir", help="folder for the figure and CSVs (default <OUTPUT_DIR>/overlapping/benchmark_tradeoffs)")
    parser.add_argument("--smoke", action="store_true", help="draw synthetic rows; needs no benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = expand_path(args.output_dir) if args.output_dir else OUTPUT_DIR / "overlapping" / "benchmark_tradeoffs"
    formats = [fmt for fmt in args.formats.split(",") if fmt]
    if args.smoke:
        rows, skipped = smoke_rows(), 0
    else:
        if not args.benchmark:
            print("hedonic-exp overlapping-tradeoffs: --benchmark RUN_DIR is required (or use --smoke)", file=sys.stderr)
            return 2
        networks = [n for n in (args.networks or "").split(",") if n] or None
        methods = [m for m in (args.methods or "").split(",") if m] or None
        rows, skipped = load_benchmark([expand_path(f) for f in args.benchmark], args.metric, networks, methods)
        folders = [expand_path(f) for f in [*args.benchmark, *(args.audit or [])]]  # a run's own ledger first
        join(rows, load_robustness(folders, args.measure, args.policy))
    if not rows:
        print("hedonic-exp overlapping-tradeoffs: no completed benchmark unit found", file=sys.stderr)
        return 1
    aggregates = aggregate(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "benchmark_tradeoffs_rows.csv", rows, ROW_FIELDS)
    write_csv(output / "benchmark_tradeoffs.csv", aggregates, AGG_FIELDS)
    written = plot(rows, aggregates, metric=args.metric, measure=args.measure, policy=args.policy, output=output,
                   formats=formats)
    unaudited = sum(row["robustness"] is None for row in rows)
    print(f"{len(rows)} units over {len({r['network'] for r in rows})} networks and {len({r['method'] for r in rows})} "
          f"methods; {skipped} skipped (not completed); {unaudited} without an audit")
    for path in written + [output / "benchmark_tradeoffs.csv", output / "benchmark_tradeoffs_rows.csv"]:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
