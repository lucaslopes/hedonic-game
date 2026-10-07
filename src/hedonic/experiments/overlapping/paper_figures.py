"""Publication-oriented plots for the overlapping-community paper.

The benchmark records are intentionally verbose and are useful for auditing,
but plotting one bar per record produces a very wide, unreadable figure.  This
module keeps the experiment layer responsible for the aggregation while
rendering compact small multiples with a shared, colorblind-safe palette.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASET_ORDER = ("amazon", "dblp", "livejournal", "youtube", "wikipedia")
DATASET_LABELS = {
    "amazon": "Amazon",
    "dblp": "DBLP",
    "livejournal": "LiveJournal",
    "youtube": "YouTube",
    "wikipedia": "Wikipedia",
}

METHOD_ORDER = (
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "cpm",
    "demon",
)
METHOD_LABELS = {
    "hedonic_multiphase": "Hedonic 1x",
    "hedonic_multiphase_x10": "Hedonic 10x",
    "hedonic_multiphase_x100": "Hedonic 100x",
    "cpm": "Clique perc.",
    "demon": "DEMON",
}

# Okabe-Ito colors: the method identity stays stable across every panel.
METHOD_COLORS = {
    "hedonic_multiphase": "#0072B2",
    "hedonic_multiphase_x10": "#009E73",
    "hedonic_multiphase_x100": "#E69F00",
    "cpm": "#D55E00",
    "demon": "#CC79A7",
}
INK = "#243447"
GRID = "#D9E2EC"
MUTED = "#52606D"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _completed(row: dict[str, Any]) -> bool:
    return str(row.get("status", "completed")) == "completed"


def _stats(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, str], tuple[float, float, int]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not _completed(row):
            continue
        value = _number(row.get(metric))
        dataset = str(row.get("dataset", "")).lower()
        method = str(row.get("method", "")).lower()
        if value is not None and dataset and method:
            grouped[(dataset, method)].append(value)
    result: dict[tuple[str, str], tuple[float, float, int]] = {}
    for key, values in grouped.items():
        mean = statistics.fmean(values)
        ci = 0.0
        if len(values) > 1:
            ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        result[key] = (mean, ci, len(values))
    return result


def _save_figure(fig: Any, plt: Any, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [str(path.with_suffix(".png")), str(path.with_suffix(".pdf"))]


def _plot_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _method_list(stats: dict[tuple[str, str], tuple[float, float, int]]) -> list[str]:
    present = {method for _, method in stats}
    return [method for method in METHOD_ORDER if method in present]


def _small_multiples(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    path: Path,
    x_label: str,
    bounded: bool = True,
    log_x: bool = False,
    show_ground_truth_overlap: bool = False,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    stats = _stats(rows, metric)
    methods = _method_list(stats)
    if not methods:
        return []
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(7.25, 4.35),
            sharex=False,
            sharey=True,
            squeeze=False,
        )
        axes_flat = [axis for row_axes in axes for axis in row_axes]
        values = [mean for mean, _, _ in stats.values()]
        upper = max((mean + ci for mean, ci, _ in stats.values()), default=1.0)
        for index, dataset in enumerate(DATASET_ORDER):
            axis = axes_flat[index]
            present = [method for method in methods if (dataset, method) in stats]
            y_positions = list(range(len(present)))
            for y, method in zip(y_positions, present):
                mean, ci, _ = stats[(dataset, method)]
                lower_error = min(ci, max(mean - 1e-9, 1e-9)) if log_x else ci
                axis.errorbar(
                    mean,
                    y,
                    xerr=[[lower_error], [ci]],
                    fmt="o",
                    markersize=4.5,
                    markeredgewidth=0.6,
                    markeredgecolor="white",
                    color=METHOD_COLORS.get(method, MUTED),
                    ecolor=METHOD_COLORS.get(method, MUTED),
                    elinewidth=1.0,
                    capsize=2.2,
                    zorder=3,
                )
            axis.set_title(DATASET_LABELS.get(dataset, dataset.title()), pad=4)
            axis.set_yticks(y_positions)
            axis.set_yticklabels([METHOD_LABELS.get(method, method) for method in present])
            axis.invert_yaxis()
            axis.grid(axis="x", color=GRID)
            axis.grid(axis="y", visible=False)
            axis.set_axisbelow(True)
            if bounded:
                axis.set_xlim(0, min(1.0, max(1.0, upper * 1.08)))
            elif log_x:
                positive = [value for value in values if value > 0]
                if positive:
                    axis.set_xscale("log")
                    axis.set_xlim(min(positive) * 0.55, max(positive) * 1.9)
            if show_ground_truth_overlap:
                reference = next(
                    (
                        _number(row.get("gt_overlapping_node_fraction"))
                        for row in rows
                        if str(row.get("dataset", "")).lower() == dataset
                        and _number(row.get("gt_overlapping_node_fraction")) is not None
                    ),
                    None,
                )
                if reference is not None:
                    axis.axvline(reference, color=INK, linestyle=(0, (3, 2)), linewidth=1.0)
            axis.tick_params(axis="y", length=0, pad=2)
            axis.tick_params(axis="x", pad=2)
            if index // 3 == 1:
                axis.set_xlabel(x_label)
        for axis in axes_flat[len(DATASET_ORDER) :]:
            axis.axis("off")
        fig.suptitle(title, fontsize=10.5, fontweight="bold", y=0.995)
        fig.text(0.995, 0.015, "Dots: mean; whiskers: 95% CI", ha="right", color=MUTED, fontsize=7)
        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.88, wspace=0.34, hspace=0.52)
        return _save_figure(fig, plt, path)


def _overview_panel(
    axis: Any,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    bounded: bool = True,
    overlap_reference: bool = False,
    log_y: bool = False,
    ylabel: str | None = None,
) -> None:
    stats = _stats(rows, metric)
    methods = _method_list(stats)
    offsets = {
        method: (index - (len(METHOD_ORDER) - 1) / 2) * 0.12
        for index, method in enumerate(METHOD_ORDER)
    }
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        for method in methods:
            value = stats.get((dataset, method))
            if value is None:
                continue
            mean, ci, _ = value
            if log_y:
                # Keep the lower whisker strictly positive so the log axis is
                # well-defined even when a normal-approximation CI is wide.
                yerr = [[min(ci, max(mean * 0.8, 1e-12))], [ci]]
            else:
                yerr = ci
            axis.errorbar(
                dataset_index + offsets[method],
                mean,
                yerr=yerr,
                fmt="o",
                markersize=4.1,
                markeredgewidth=0.6,
                markeredgecolor="white",
                color=METHOD_COLORS.get(method, MUTED),
                ecolor=METHOD_COLORS.get(method, MUTED),
                elinewidth=0.9,
                capsize=2.0,
                zorder=3,
            )
    if overlap_reference:
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            reference = next(
                (
                    _number(row.get("gt_overlapping_node_fraction"))
                    for row in rows
                    if str(row.get("dataset", "")).lower() == dataset
                    and _number(row.get("gt_overlapping_node_fraction")) is not None
                ),
                None,
            )
            if reference is not None:
                axis.scatter(
                    dataset_index,
                    reference,
                    marker="_",
                    s=130,
                    linewidths=1.4,
                    color=INK,
                    zorder=4,
                )
    axis.set_title(title, pad=4, fontweight="bold")
    axis.set_xlim(-0.5, len(DATASET_ORDER) - 0.5)
    axis.set_xticks(range(len(DATASET_ORDER)))
    axis.set_xticklabels([DATASET_LABELS[name] for name in DATASET_ORDER], rotation=24, ha="right")
    axis.grid(axis="y", color=GRID)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    if bounded:
        axis.set_ylim(0, 1.02)
    elif log_y:
        positive = [mean for mean, _, _ in stats.values() if mean > 0]
        if positive:
            axis.set_yscale("log")
            axis.set_ylim(min(positive) * 0.55, max(positive) * 1.9)
    if ylabel:
        axis.set_ylabel(ylabel, labelpad=3)
    axis.tick_params(axis="x", pad=2)


def _benchmark_overview(rows: list[dict[str, Any]], path: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return []

    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.95), squeeze=False)
        specs = (
            ("symmetric_best_match_f1", "Best-match recovery $F_1$", True, False, False, "Mean score"),
            ("matching_f1", "One-to-one matched $F_1$", True, False, False, None),
            ("omega", "Omega agreement", True, False, False, None),
            ("predicted_overlapping_node_fraction", "Overlapping-node fraction", True, True, False, None),
            ("runtime_seconds", "Detector runtime (log seconds)", False, False, True, "Runtime (s)"),
        )
        axes_flat = [axis for row_axes in axes for axis in row_axes]
        for axis, (metric, title, bounded, reference, log_y, ylabel) in zip(
            axes_flat, specs
        ):
            _overview_panel(
                axis,
                rows,
                metric=metric,
                title=title,
                bounded=bounded,
                overlap_reference=reference,
                log_y=log_y,
                ylabel=ylabel,
            )
        for axis in axes_flat[len(specs) :]:
            axis.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=METHOD_LABELS[method],
            )
            for method in METHOD_ORDER
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="_",
                linestyle="",
                color=INK,
                markersize=10,
                label="Metadata overlap",
            )
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=3,
            frameon=False,
            handletextpad=0.35,
            columnspacing=1.0,
        )
        fig.text(0.5, 0.015, "Dataset", ha="center", color=MUTED, fontsize=8)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.82, wspace=0.30, hspace=0.52)
        return _save_figure(fig, plt, path)


def write_benchmark_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Write compact benchmark plots and return their generated paths."""
    if not rows:
        return []
    plot_dir = output_dir / "plots"
    paths: list[str] = []
    paths.extend(_benchmark_overview(rows, plot_dir / "benchmark_overview"))
    paths.extend(
        _small_multiples(
            rows,
            metric="symmetric_best_match_f1",
            title="Best-match recovery by dataset",
            path=plot_dir / "accuracy_by_dataset",
            x_label="Symmetric best-match $F_1$",
        )
    )
    paths.extend(
        _small_multiples(
            rows,
            metric="matching_f1",
            title="One-to-one recovery by dataset",
            path=plot_dir / "method_comparison",
            x_label="Matched $F_1$",
        )
    )
    paths.extend(
        _small_multiples(
            rows,
            metric="predicted_overlapping_node_fraction",
            title="Predicted overlap by dataset",
            path=plot_dir / "overlap_structure",
            x_label="Overlapping-node fraction",
            show_ground_truth_overlap=True,
        )
    )
    paths.extend(
        _small_multiples(
            rows,
            metric="runtime_seconds",
            title="Detector runtime by dataset",
            path=plot_dir / "runtime_by_dataset",
            x_label="Detector runtime (s, log scale)",
            bounded=False,
            log_x=True,
        )
    )
    return paths


def _write_ground_truth_overview(
    gt_rows: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    # The repair panel is deliberately a single, interpretable protocol
    # slice: unperturbed canonical metadata, open labels, gamma=density, and
    # independently verified returns. The grid contains only one
    # perturbation seed at distance zero, so uncertainty is estimated
    # across the five detector seeds. Pooling policies, gamma multipliers and
    # perturbed starts would
    # turn the figure into a misleading mixture of different interventions.
    per_seed: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        dataset = str(row.get("dataset", "")).lower()
        phase = str(row.get("phase", "")).lower()
        seed_value = _number(row.get("seed"))
        if (
            dataset not in DATASET_ORDER
            or phase not in {"local", "multiphase"}
            or seed_value is None
            or str(row.get("action_policy", "")) != "open_labels"
            or str(row.get("completion_policy", row.get("policy", "")))
            != "covered-induced"
            or _number(row.get("perturbation_target_distance")) != 0.0
            or _number(row.get("resolution_multiplier")) != 1.0
            or str(row.get("status", "")) != "completed"
            or str(row.get("equilibrium_status", "")) != "verified"
        ):
            continue
        for metric in ("accuracy_node_micro_f1", "robust_fraction_gamma_0_1"):
            value = _number(row.get(metric))
            if value is not None:
                per_seed[(dataset, phase, int(seed_value))][metric].append(value)

    repair: dict[tuple[str, str], dict[str, tuple[float, float, int]]] = defaultdict(dict)
    for dataset in DATASET_ORDER:
        for phase in ("local", "multiphase"):
            seed_rows = [
                values
                for (current_dataset, current_phase, _), values in per_seed.items()
                if current_dataset == dataset and current_phase == phase
            ]
            for metric in ("accuracy_node_micro_f1", "robust_fraction_gamma_0_1"):
                values = [
                    statistics.fmean(seed_values[metric])
                    for seed_values in seed_rows
                    if seed_values.get(metric)
                ]
                if len(values) == 5:
                    ci = (
                        # The locked grid has five detector seeds.  Use the
                        # two-sided 95% Student critical value t_(.975,4), not
                        # a large-sample normal approximation.
                        2.7764451052
                        * statistics.stdev(values)
                        / math.sqrt(len(values))
                    )
                    repair[(dataset, phase)][metric] = (
                        statistics.fmean(values),
                        ci,
                        len(values),
                    )

    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), squeeze=False)
        left, right = axes[0]
        ordered_gt_rows = sorted(
            gt_rows,
            key=lambda row: DATASET_ORDER.index(str(row.get("dataset", "")).lower())
            if str(row.get("dataset", "")).lower() in DATASET_ORDER
            else len(DATASET_ORDER),
        )
        labels = [DATASET_LABELS.get(str(row.get("dataset", "")).lower(), str(row.get("dataset"))) for row in ordered_gt_rows]
        x_positions = list(range(len(labels)))
        width = 0.34
        fixed = [_number(row.get("fixed_labels_robust_fraction")) or 0.0 for row in ordered_gt_rows]
        opened = [_number(row.get("open_labels_robust_fraction")) or 0.0 for row in ordered_gt_rows]
        left.bar(
            [x - width / 2 for x in x_positions],
            fixed,
            width,
            color="#0072B2",
            label="Fixed labels",
        )
        left.bar(
            [x + width / 2 for x in x_positions],
            opened,
            width,
            color="#E69F00",
            label="Open isolation",
        )
        left.set_title("Supplied-cover audit", pad=4, fontweight="bold")
        left.set_ylabel("Full-range robust fraction")
        left.set_ylim(0, 1.02)
        left.set_xticks(x_positions)
        left.set_xticklabels(labels, rotation=24, ha="right")
        left.grid(axis="y", color=GRID)
        left.set_axisbelow(True)

        dataset_colors = {
            "amazon": "#0072B2",
            "dblp": "#E69F00",
            "livejournal": "#009E73",
            "youtube": "#CC79A7",
            "wikipedia": "#6B7280",
        }
        for dataset in DATASET_ORDER:
            local = repair.get((dataset, "local"), {})
            multi = repair.get((dataset, "multiphase"), {})
            if not local or not multi:
                continue
            local_x, local_x_ci, _ = local["accuracy_node_micro_f1"]
            local_y, local_y_ci, _ = local["robust_fraction_gamma_0_1"]
            multi_x, multi_x_ci, _ = multi["accuracy_node_micro_f1"]
            multi_y, multi_y_ci, _ = multi["robust_fraction_gamma_0_1"]
            color = dataset_colors[dataset]
            right.annotate(
                "",
                xy=(multi_x, multi_y),
                xytext=(local_x, local_y),
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.2},
                zorder=2,
            )
            for x, y, x_ci, y_ci, marker in (
                (local_x, local_y, local_x_ci, local_y_ci, "o"),
                (multi_x, multi_y, multi_x_ci, multi_y_ci, "^")
            ):
                right.errorbar(
                    x,
                    y,
                    xerr=x_ci,
                    yerr=y_ci,
                    fmt=marker,
                    markersize=5.0,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    color=color,
                    ecolor=color,
                    capsize=2.0,
                    linewidth=0.9,
                    zorder=3,
                )
            right.text(
                (local_x + multi_x) / 2,
                (local_y + multi_y) / 2,
                DATASET_LABELS[dataset],
                color=color,
                fontsize=7,
                ha="center",
                va="bottom",
            )
        right.set_title(r"Certified repair frontier ($\gamma=$ density)", pad=4, fontweight="bold")
        right.set_xlabel(r"Node micro-$F_1$ vs metadata")
        right.set_ylabel("Full-range robust fraction")
        right.set_xlim(0, 1.02)
        right.set_ylim(0, 1.02)
        right.grid(color=GRID)
        right.set_axisbelow(True)
        left.legend(frameon=False, loc="upper right")
        from matplotlib.lines import Line2D

        right.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="", color=INK, label="Local repair"),
                Line2D([0], [0], marker="^", linestyle="", color=INK, label="Multi-phase repair"),
            ],
            frameon=False,
            loc="lower right",
        )
        fig.text(
            0.995,
            0.015,
            "Arrows: local to multi-phase; whiskers: 95% CI over 5 detector seeds",
            ha="right",
            color=MUTED,
            fontsize=7,
        )
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.90, wspace=0.28)
        return _save_figure(fig, plt, path)


def _write_ground_truth_tradeoff(rows: list[dict[str, Any]], path: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return []

    primary_rows = [
        row
        for row in rows
        if _number(row.get("accuracy_node_micro_f1")) is not None
        and _number(row.get("perturbation_target_distance")) is not None
        and str(row.get("action_policy", "")) == "open_labels"
        and str(row.get("completion_policy", row.get("policy", "")))
        == "covered-induced"
        and _number(row.get("resolution_multiplier")) == 1.0
        and str(row.get("status", ""))
        in {"completed", "completed_non_equilibrium"}
    ]
    if not primary_rows:
        return []

    # At nonzero distances, independently perturbed covers are the structural
    # units: average five detector restarts within each perturbation seed, then
    # estimate a t interval across five perturbed covers. At distance zero
    # there is one cover, so the five detector seeds are the units.
    cells: dict[tuple[str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        dataset = str(row.get("dataset", "")).lower()
        phase = str(row.get("phase", "")).lower()
        distance = _number(row.get("perturbation_target_distance"))
        perturbation_seed = _number(row.get("perturbation_seed"))
        if (
            dataset not in DATASET_ORDER
            or phase not in {"local", "multiphase"}
            or distance is None
            or perturbation_seed is None
        ):
            continue
        cells[(dataset, phase, distance, int(perturbation_seed))].append(row)

    summaries: dict[
        tuple[str, str, float, str], tuple[float, float]
    ] = {}
    for dataset in DATASET_ORDER:
        for phase in ("local", "multiphase"):
            distances = sorted(
                {
                    key[2]
                    for key in cells
                    if key[0] == dataset and key[1] == phase
                }
            )
            for distance in distances:
                perturbation_cells = [
                    cell_rows
                    for (current_dataset, current_phase, current_distance, _), cell_rows
                    in cells.items()
                    if current_dataset == dataset
                    and current_phase == phase
                    and current_distance == distance
                ]
                if distance == 0.0:
                    if len(perturbation_cells) != 1:
                        continue
                    zero_rows = perturbation_cells[0]
                    if (
                        len(zero_rows) != 5
                        or len({_number(row.get("seed")) for row in zero_rows}) != 5
                    ):
                        continue
                    unit_rows = [[row] for row in zero_rows]
                else:
                    if len(perturbation_cells) != 5 or any(
                        len(cell_rows) != 5
                        or len({_number(row.get("seed")) for row in cell_rows}) != 5
                        for cell_rows in perturbation_cells
                    ):
                        continue
                    unit_rows = perturbation_cells
                unit_values = {
                    "certification_rate": [
                        statistics.fmean(
                            1.0 if row.get("status") == "completed" else 0.0
                            for row in cell_rows
                        )
                        for cell_rows in unit_rows
                    ],
                    "accuracy_node_micro_f1": [
                        statistics.fmean(
                            float(row["accuracy_node_micro_f1"])
                            for row in cell_rows
                        )
                        for cell_rows in unit_rows
                    ],
                }
                for metric, values in unit_values.items():
                    if len(values) == 5:
                        summaries[(dataset, phase, distance, metric)] = (
                            statistics.fmean(values),
                            2.7764451052
                            * statistics.stdev(values)
                            / math.sqrt(len(values)),
                        )
    if not summaries:
        return []

    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), squeeze=False)
        left, right = axes[0]
        colors = {
            "amazon": "#0072B2",
            "dblp": "#E69F00",
            "livejournal": "#009E73",
            "youtube": "#CC79A7",
        }
        phase_style = {
            "local": ("o", "-"),
            "multiphase": ("^", "--"),
        }
        for dataset in DATASET_ORDER:
            for phase in ("local", "multiphase"):
                marker, linestyle = phase_style[phase]
                distances = sorted(
                    {
                        key[2]
                        for key in summaries
                        if key[0] == dataset and key[1] == phase
                    }
                )
                if not distances:
                    continue
                for axis, metric in (
                    (left, "certification_rate"),
                    (right, "accuracy_node_micro_f1"),
                ):
                    values = [summaries[(dataset, phase, value, metric)] for value in distances]
                    axis.errorbar(
                        [100.0 * value for value in distances],
                        [value[0] for value in values],
                        yerr=[value[1] for value in values],
                        color=colors.get(dataset, MUTED),
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.0,
                        markersize=4.0,
                        capsize=1.8,
                        alpha=0.9,
                    )
        for axis, title, ylabel in (
            (left, "Certified repair rate", "Independently certified fraction"),
            (right, "Metadata retained after native return", r"Node micro-$F_1$ vs metadata"),
        ):
            axis.set_title(title, pad=4, fontweight="bold")
            axis.set_xlabel("Target labeled-incidence distance (%)")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, 1.02)
            axis.grid(color=GRID)
            axis.set_axisbelow(True)
        legend = [
            Line2D([0], [0], color=colors[dataset], label=DATASET_LABELS[dataset])
            for dataset in colors
        ] + [
            Line2D(
                [0],
                [0],
                color=INK,
                marker=phase_style[phase][0],
                linestyle=phase_style[phase][1],
                label=phase.replace("multiphase", "multi-phase").title(),
            )
            for phase in ("local", "multiphase")
        ]
        fig.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            frameon=False,
            ncol=6,
        )
        fig.text(
            0.995,
            0.015,
            "Open labels, $\\gamma=$ density; t intervals over 5 starts (perturbation seeds, or detector seeds at 0%)",
            ha="right",
            color=MUTED,
            fontsize=7,
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.79, wspace=0.28)
        return _save_figure(fig, plt, path)


def write_ground_truth_plots(
    output_dir: Path,
    gt_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    expected_protocol_lock_sha256: str | None = None,
    expected_condition_axis_keys: set[str] | None = None,
) -> list[str]:
    """Write GT plots only from one complete, identity-bound condition grid."""
    if not gt_rows and not rows:
        return []
    if expected_protocol_lock_sha256 is not None:
        if any(
            row.get("protocol_lock_sha256")
            != expected_protocol_lock_sha256
            for row in [*gt_rows, *rows]
        ):
            return []
    if expected_condition_axis_keys is not None:
        observed = [
            str(row.get("condition_axis_key"))
            for row in rows
            if isinstance(row.get("condition_axis_key"), str)
        ]
        if (
            len(observed) != len(set(observed))
            or set(observed) != expected_condition_axis_keys
        ):
            return []
    plot_dir = output_dir / "plots"
    paths: list[str] = []
    if gt_rows or rows:
        paths.extend(_write_ground_truth_overview(gt_rows, rows, plot_dir / "gt_robustness_overview"))
    if rows:
        paths.extend(
            _write_ground_truth_tradeoff(
                rows, plot_dir / "gt_equilibrium_tradeoff"
            )
        )
    return paths


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _legacy_f1_stats(
    rows: list[dict[str, Any]], method: str
) -> tuple[float, float, int] | None:
    values = [
        _number((row.get(method) or {}).get("f1"))
        for row in rows
        if isinstance(row.get(method), dict)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    ci = 0.0
    if len(values) > 1:
        ci = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return statistics.fmean(values), ci, len(values)


def _write_dblp_subgraph_plot(
    specs: list[tuple[str, str, Path, Path | None]], path: Path
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    loaded: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]] | None]] = []
    for label, small_label, small_path, large_path in specs:
        small = _load_json_list(small_path)
        large = _load_json_list(large_path) if large_path is not None else None
        if small:
            loaded.append((f"{label}\n({small_label}{', n=1000' if large else ''})", small_label, small, large))
    if not loaded:
        return []

    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, len(loaded), figsize=(7.25, 2.85), squeeze=False)
        axes_flat = list(axes[0])
        for axis, (label, small_label, small, large) in zip(axes_flat, loaded):
            n_values = [(small_label, small)]
            if large:
                n_values.append(("n=1000", large))
            x = list(range(len(n_values)))
            width = 0.28
            for offset, method, color, method_label in (
                (-width / 2, "leiden", "#6B7280", "Leiden"),
                (width / 2, "hedonic", "#0072B2", "Hedonic"),
            ):
                means: list[float] = []
                cis: list[float] = []
                for _, rows in n_values:
                    stats = _legacy_f1_stats(rows, method)
                    means.append(stats[0] if stats else float("nan"))
                    cis.append(stats[1] if stats else 0.0)
                axis.errorbar(
                    [position + offset for position in x],
                    means,
                    yerr=cis,
                    fmt="o",
                    color=color,
                    ecolor=color,
                    markersize=4.8,
                    capsize=2.0,
                    linewidth=1.0,
                    label=method_label,
                )
            axis.set_title(label, pad=4, fontweight="bold")
            axis.set_xticks(x)
            axis.set_xticklabels([name for name, _ in n_values])
            axis.set_ylim(0, 0.56)
            axis.set_ylabel("Target-community best-match $F_1$")
            axis.grid(axis="y", color=GRID)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", pad=2)
        if len(loaded) > 1:
            for axis in axes_flat[1:]:
                axis.set_ylabel("")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=2, frameon=False)
        fig.text(0.5, 0.015, "Legacy DBLP subgraph suite; dots are means and whiskers are 95% CIs", ha="center", color=MUTED, fontsize=7)
        fig.subplots_adjust(left=0.08, right=0.995, bottom=0.19, top=0.77, wspace=0.28)
        return _save_figure(fig, plt, path)


def _write_dblp_scaling_plot(points: list[dict[str, Any]], path: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    values = [
        (_number(point.get("n_nodes")), _number(point.get("wallclock_s")))
        for point in points
    ]
    values = [(nodes, seconds) for nodes, seconds in values if nodes and seconds and nodes > 0 and seconds > 0]
    if not values:
        return []
    xs, ys = zip(*values)
    with plt.rc_context(_plot_style()):
        fig, axis = plt.subplots(figsize=(7.25, 3.05))
        axis.loglog(xs, ys, color="#0072B2", linewidth=1.5, marker="o", markersize=4.0)
        axis.scatter([xs[-1]], [ys[-1]], s=42, color="#D55E00", zorder=4, label="Full DBLP endpoint")
        axis.annotate(
            f"{int(xs[-1]):,} vertices\n{ys[-1]:.1f} s",
            xy=(xs[-1], ys[-1]),
            xytext=(-8, 18),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#D55E00",
            arrowprops={"arrowstyle": "-", "color": "#D55E00", "linewidth": 0.7},
        )
        axis.set_xlabel("Vertices in induced DBLP neighborhood (log scale)")
        axis.set_ylabel("Wall-clock seconds (log scale)")
        axis.set_title("Archived full-DBLP multi-phase scaling diagnostic", pad=5, fontweight="bold")
        axis.grid(which="both", color=GRID)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", frameon=False)
        fig.text(0.5, 0.015, "14 nested 1-hop neighborhoods around GT community 1004; multi-phase only", ha="center", color=MUTED, fontsize=7)
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.84)
        return _save_figure(fig, plt, path)


def _write_dblp_resolution_plot(
    smoke_path: Path, legacy_path: Path | None, path: Path
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        smoke = {}
    aggregated = smoke.get("aggregated") if isinstance(smoke, dict) else None
    aggregated = aggregated if isinstance(aggregated, list) else []
    smoke_rows = [
        (
            _number(row.get("resolution")),
            _number(row.get("f1_mean")),
            _number(row.get("f1_ci_half")),
        )
        for row in aggregated
    ]
    smoke_rows = [(x, y, ci or 0.0) for x, y, ci in smoke_rows if x is not None and y is not None]
    legacy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if legacy_path is not None and legacy_path.exists():
        try:
            with legacy_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    community = str(row.get("community_index", ""))
                    x = _number(row.get("resolutions"))
                    y = _number(row.get("fractions"))
                    if community and x is not None and y is not None:
                        legacy[community].append((x, max(y, 1e-5)))
        except OSError:
            legacy = defaultdict(list)
    if not smoke_rows and not legacy:
        return []

    n_panels = 2 if legacy else 1
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, n_panels, figsize=(7.25, 2.95), squeeze=False)
        axes_flat = list(axes[0])
        if smoke_rows:
            axis = axes_flat[0]
            xs, ys, cis = zip(*smoke_rows)
            axis.errorbar(xs, ys, yerr=cis, color="#0072B2", marker="o", markersize=4.5, linewidth=1.4, capsize=2.0)
            axis.set_title("Corrected equilibrium-v2 smoke", pad=4, fontweight="bold")
            axis.set_xlabel("Resolution $\\gamma$")
            axis.set_ylabel("Symmetric best-match $F_1$")
            axis.set_ylim(0.30, 0.60)
            axis.grid(color=GRID)
            axis.set_axisbelow(True)
        if legacy:
            axis = axes_flat[1 if smoke_rows else 0]
            for community, rows in sorted(legacy.items()):
                rows.sort()
                xs, ys = zip(*rows)
                axis.plot(xs, ys, linewidth=1.25, marker="", label=f"Community {community}")
            axis.set_yscale("log")
            axis.set_ylim(8e-6, 1.1)
            axis.set_title("Archived full-DBLP spectrum", pad=4, fontweight="bold")
            axis.set_xlabel("Resolution $\\gamma$")
            axis.set_ylabel("Stable fraction (log scale)")
            axis.grid(which="both", color=GRID)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, fontsize=7)
        fig.text(0.5, 0.015, "Left: v2 F1 smoke check; right: legacy full-DBLP robustness/fraction export", ha="center", color=MUTED, fontsize=7)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.82, wspace=0.32)
        return _save_figure(fig, plt, path)


def write_dblp_diagnostic_plots(
    output_dir: Path,
    *,
    legacy_root: Path | None = None,
    full_scale_path: Path | None = None,
    smoke_resolution_path: Path | None = None,
    legacy_resolution_path: Path | None = None,
) -> list[str]:
    """Write the DBLP diagnostics that complement the locked SNAP scorecard.

    These inputs are deliberately explicit and remain separate from the
    five-network paper ledger: the subgraph suite is archived pre-v2 evidence,
    while the full-DBLP scaling export predates the protocol identity lock.
    """
    root = _project_root()
    legacy_root = legacy_root or root / "artifacts/legacy/hedonic-overlapping-results"
    full_scale_path = full_scale_path or root / "artifacts/legacy/overlapping-scale/complexity_scale.json"
    smoke_resolution_path = smoke_resolution_path or root / "artifacts/evidence/overlapping_communities/resolution_f1_equilibrium_v2_smoke/resolution_f1.json"
    legacy_resolution_path = legacy_resolution_path or Path.home() / "Databases/Hedonic/Networks/DBLP/results/resolution_spectra.csv"
    plot_dir = output_dir / "plots"
    paths: list[str] = []
    subgraph_specs = [
        ("Random 1-hop", "n=200", legacy_root / "subgraph_L1_n200.json", legacy_root / "subgraph_L1_n1000.json"),
        ("GT overlap $\\geq 2$", "n=200", legacy_root / "subgraph_overlapping_n200.json", legacy_root / "subgraph_overlap_n1000.json"),
        ("Overlap coeff. $\\geq .2$", "n=200", legacy_root / "subgraph_overlap_ratio02_n200.json", legacy_root / "subgraph_overlap_ratio02_n1000.json"),
        ("Random 2-hop", "n=100", legacy_root / "subgraph_L2_n100.json", None),
    ]
    paths.extend(_write_dblp_subgraph_plot(subgraph_specs, plot_dir / "dblp_subgraph_recovery"))
    try:
        scale_data = json.loads(full_scale_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        scale_data = {}
    points = scale_data.get("points") if isinstance(scale_data, dict) else None
    if isinstance(points, list):
        paths.extend(_write_dblp_scaling_plot(points, plot_dir / "dblp_scaling"))
    paths.extend(_write_dblp_resolution_plot(smoke_resolution_path, legacy_resolution_path, plot_dir / "dblp_resolution"))
    return paths
