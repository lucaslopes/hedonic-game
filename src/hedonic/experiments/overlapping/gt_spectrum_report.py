"""Figures and generated TeX for the SNAP ground-truth robustness spectrum.

Every number in the TeX bundle and every plotted point comes from the ledger
(``spectrum_audit.csv`` rows and ``results.jsonl`` rows); nothing is typed in by
hand.  There are no confidence bands: the audit of a supplied cover is
deterministic (one graph per network), and detector seeds are restarts on one
graph, not independent networks.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from hedonic.experiments.overlapping.paper_figures import (
    GRID,
    INK,
    MUTED,
    _plot_style,
    _save_figure,
)

LABELS = {
    "amazon": "Amazon", "dblp": "DBLP", "livejournal": "LiveJournal", "youtube": "YouTube",
    "wikipedia": "Wikipedia", "orkut": "Orkut", "friendster": "Friendster",
}
# Okabe-Ito colour per network; cover variant is carried by line style.
COLORS = {
    "amazon": "#E69F00", "dblp": "#0072B2", "livejournal": "#009E73", "youtube": "#D55E00",
    "wikipedia": "#CC79A7", "orkut": "#56B4E9", "friendster": "#7F7F7F",
}
STYLES = {"top5000": "-", "all": (0, (5, 2))}
POLICY_TITLES = {
    "fixed_labels": "Fixed labels (no new community)",
    "open_labels": "Open labels (a fresh singleton community is allowed)",
}
METRIC_LABELS = {
    "f1": "symmetric best-match F1", "omega": "Omega index (sampled)", "node_micro_f1": "vertex-membership micro F1",
    "jaccard": "symmetric best-match Jaccard", "matching_f1": "one-to-one matching F1",
}
LINTHRESH = 1e-4


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    order = list(LABELS)
    return sorted({(r["dataset"], r["cover"]) for r in rows},
                  key=lambda p: (order.index(p[0]) if p[0] in order else 99, p[1]))


def _label(pair: tuple[str, str]) -> str:
    return f"{LABELS.get(pair[0], pair[0])} · {pair[1]}"


def _axes_x(ax) -> None:
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.35)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "0.1", "1"])
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)


def _legend(fig, ax, pairs) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=COLORS.get(p[0], INK), linestyle=STYLES.get(p[1], "-"), lw=1.6,
                      label=_label(p)) for p in pairs]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False,
               bbox_to_anchor=(0.5, 0.05))


def _finish(fig, caption: str) -> None:
    """Reserve a footer: legend above, one-line caption below, both in figure coordinates."""
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=6.8, color=MUTED, wrap=True)


def spectrum_figure(audit_rows: list[dict[str, Any]], path: Path, policies: list[str]) -> list[str]:
    """Principal figure: stable-vertex fraction against resolution, one line per network/cover."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in audit_rows if r.get("row_type", "ground_truth_audit") == "ground_truth_audit"]
    if not rows:
        return []
    pairs = _pairs(rows)
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, len(policies), figsize=(3.9 * len(policies) + 0.6, 4.1), sharey=True, squeeze=False)
        for ax, policy in zip(axes[0], policies):
            for pair in pairs:
                series = sorted(
                    ((r["gamma"], r["stable_fraction"], r["is_nash_equilibrium"], r["density"])
                     for r in rows if (r["dataset"], r["cover"]) == pair and r["action_policy"] == policy),
                )
                if not series:
                    continue
                xs, ys = [s[0] for s in series], [s[1] for s in series]
                colour = COLORS.get(pair[0], INK)
                ax.plot(xs, ys, color=colour, linestyle=STYLES.get(pair[1], "-"), lw=1.5)
                nash = [(x, y) for x, y, is_nash, _ in series if is_nash]
                if nash:
                    ax.scatter(*zip(*nash), s=14, color=colour, edgecolor="white", linewidth=0.4, zorder=4)
                density = series[0][3]
                if xs[0] <= density <= xs[-1]:
                    y_at = float(_interp(xs, ys, density))
                    ax.scatter([density], [y_at], marker="v", s=26, color=colour, edgecolor=INK, linewidth=0.5, zorder=5)
            _axes_x(ax)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(POLICY_TITLES.get(policy, policy), fontsize=8.5)
            ax.set_xlabel("resolution γ (CPM penalty)")
        axes[0][0].set_ylabel("fraction of vertices with no\nprofitable unilateral action")
        _legend(fig, axes[0][0], pairs)
        _finish(fig, "Dots: cover is a Nash equilibrium (every vertex within tolerance). Triangles: γ = graph density. "
                     "Deterministic audit of one graph per network — no confidence band.")
        return _save_figure(fig, plt, path)


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def accuracy_change_figure(result_rows: list[dict[str, Any]], path: Path, policies: list[str],
                           metrics: tuple[str, ...] = ("f1", "omega")) -> list[str]:
    """GT-to-returned-cover accuracy change by seed and resolution (paired with the exact start)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result_rows if r.get("row_type") == "detector_run" and r.get("status") in
            ("completed", "completed_non_equilibrium")]
    metrics = tuple(m for m in metrics if any(_number(r.get(f"delta_{m}")) is not None for r in rows))
    if not rows or not metrics:
        return []
    pairs = _pairs(rows)
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(len(metrics), len(policies), figsize=(3.9 * len(policies) + 0.6, 2.7 * len(metrics) + 1.3),
                                 sharex=True, squeeze=False)
        for i, metric in enumerate(metrics):
            for j, policy in enumerate(policies):
                ax = axes[i][j]
                for pair in pairs:
                    members = [r for r in rows if (r["dataset"], r["cover"]) == pair and r["action_policy"] == policy]
                    by_gamma: dict[float, list[dict[str, Any]]] = defaultdict(list)
                    for r in members:
                        if _number(r.get(f"delta_{metric}")) is not None:
                            by_gamma[float(r["gamma"])].append(r)
                    if not by_gamma:
                        continue
                    colour = COLORS.get(pair[0], INK)
                    xs = sorted(by_gamma)
                    ax.plot(xs, [statistics.fmean(float(r[f"delta_{metric}"]) for r in by_gamma[x]) for x in xs],
                            color=colour, linestyle=STYLES.get(pair[1], "-"), lw=1.3, alpha=0.9)
                    for x in xs:
                        for r in by_gamma[x]:
                            verified = r["status"] == "completed"
                            ax.scatter([x], [float(r[f"delta_{metric}"])], s=9, color=colour if verified else "none",
                                       edgecolor=colour, linewidth=0.6, alpha=0.75, zorder=3)
                _axes_x(ax)
                ax.axhline(0, color=MUTED, lw=0.6)
                if i == 0:
                    ax.set_title(POLICY_TITLES.get(policy, policy), fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"Δ {METRIC_LABELS.get(metric, metric)}\n(returned − ground truth)")
                if i == len(metrics) - 1:
                    ax.set_xlabel("resolution γ")
        _legend(fig, axes[-1][0], pairs)
        _finish(fig, "One point per detector seed (filled: verified equilibrium; hollow: native return with positive regret); "
                     "lines are seed means. Seeds are restarts on one graph, not independent samples.")
        return _save_figure(fig, plt, path)


def accuracy_versus_change_figure(result_rows: list[dict[str, Any]], path: Path, metric: str = "f1") -> list[str]:
    """Accuracy of the returned cover against how far it moved from the exact ground-truth start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result_rows if r.get("row_type") == "detector_run" and r.get("status") in
            ("completed", "completed_non_equilibrium") and _number(r.get(f"final_{metric}")) is not None
            and _number(r.get("changed_vertex_fraction")) is not None]
    if not rows:
        return []
    pairs = _pairs(rows)
    markers = {"fixed_labels": "o", "open_labels": "s"}
    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for pair in pairs:
            for policy, marker in markers.items():
                members = [r for r in rows if (r["dataset"], r["cover"]) == pair and r["action_policy"] == policy]
                for verified in (True, False):
                    part = [r for r in members if (r["status"] == "completed") == verified]
                    if not part:
                        continue
                    colour = COLORS.get(pair[0], INK)
                    ax.scatter([float(r["changed_vertex_fraction"]) for r in part],
                               [float(r[f"final_{metric}"]) for r in part], s=14, marker=marker,
                               color=colour if verified else "none", edgecolor=colour, linewidth=0.7,
                               alpha=0.7 if pair[1] == "top5000" else 0.45)
        ax.set_xlabel("fraction of vertices whose membership set changed")
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} vs the supplied cover")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
        from matplotlib.lines import Line2D

        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS.get(p[0], INK),
                          markeredgecolor=COLORS.get(p[0], INK), markersize=5, label=_label(p)) for p in pairs]
        handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, markersize=5, label="fixed labels"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor=INK, markersize=5, label="open labels"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=INK, markersize=5,
                           label="hollow: positive regret")]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.5)
        fig.tight_layout()
        return _save_figure(fig, plt, path)


# --------------------------------------------------------------------------- tables and macros
def _nash_range(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    gammas = [r["gamma"] for r in rows if r["is_nash_equilibrium"]]
    return (min(gammas), max(gammas)) if gammas else None


def _fmt_gamma(value: float) -> str:
    return "0" if value == 0 else f"{value:.3g}"


def audit_table_tex(audit_rows: list[dict[str, Any]], provenance: list[dict[str, Any]], policies: list[str]) -> str:
    """LaTeX table: per pair, stable fraction at the density and the tolerance-Nash resolution range."""
    lines = [
        "% generated by hedonic-exp overlapping-gt-spectrum; do not edit",
        "\\begin{tabular}{@{}llrrrr" + "rr" * len(policies) + "@{}}",
        "\\toprule",
        "Network & Cover & $n$ & $m$ & $K$ & $M$ & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{p.split('_')[0].capitalize()}}}" for p in policies) + " \\\\",
        "\\cmidrule(lr){7-" + str(6 + 2 * len(policies)) + "}",
        " & & & & & & " + " & ".join("$\\phi_\\rho$ & Nash $\\gamma$" for _ in policies) + " \\\\",
        "\\midrule",
    ]
    for prov in provenance:
        cells = [LABELS.get(prov["dataset"], prov["dataset"]), f"\\texttt{{{prov['cover']}}}",
                 f"{prov['n_vertices']:,}", f"{prov['n_edges']:,}", f"{prov['ground_truth_community_count']:,}", str(prov["cap"])]
        for policy in policies:
            rows = sorted((r for r in audit_rows if r["dataset"] == prov["dataset"] and r["cover"] == prov["cover"]
                           and r["action_policy"] == policy), key=lambda r: r["gamma"])
            if not rows:
                cells += ["--", "--"]
                continue
            at_density = _interp([r["gamma"] for r in rows], [r["stable_fraction"] for r in rows], prov["density"])
            nash = _nash_range(rows)
            cells += [f"{at_density:.3f}", "none" if nash is None else f"[{_fmt_gamma(nash[0])}, {_fmt_gamma(nash[1])}]"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def detector_table_tex(summary_rows: list[dict[str, Any]], policies: list[str]) -> str:
    """LaTeX table of terminal-status denominators and mean paired change per pair and policy."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(row["dataset"], row["cover"], row["action_policy"])].append(row)
    lines = [
        "% generated by hedonic-exp overlapping-gt-spectrum; do not edit",
        "\\begin{tabular}{@{}lllrrrrrr@{}}",
        "\\toprule",
        "Network & Cover & Action space & Runs & Verified & Non-eq. & Timeout/fail & Unchanged & mean $\\Delta$F1 \\\\",
        "\\midrule",
    ]
    for (dataset, cover, policy), members in sorted(
        groups.items(), key=lambda kv: (list(LABELS).index(kv[0][0]) if kv[0][0] in LABELS else 99, kv[0][1], kv[0][2])
    ):
        expected = sum(m["expected_runs"] for m in members)
        returned = sum(m["returned_covers"] for m in members)
        deltas = [(m.get("delta_f1_mean"), m["returned_covers"]) for m in members if m.get("delta_f1_mean") is not None]
        mean_delta = (sum(d * n for d, n in deltas) / sum(n for _, n in deltas)) if deltas else None
        lines.append(" & ".join([
            LABELS.get(dataset, dataset), f"\\texttt{{{cover}}}", policy.split("_")[0], str(expected),
            str(sum(m["verified_equilibria"] for m in members)), str(sum(m["non_equilibrium_returns"] for m in members)),
            str(sum(m["timeouts"] + m["failures"] for m in members)),
            str(sum(m["unchanged_returns"] for m in members)),
            "--" if mean_delta is None or not returned else f"{mean_delta:+.3f}",
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def macros_tex(coverage: dict[str, Any], provenance: list[dict[str, Any]], options: dict[str, Any]) -> str:
    def macro(name: str, value: Any) -> str:
        return f"\\newcommand{{\\GTSpec{name}}}{{{value}}}"

    complete = bool(coverage.get("complete"))
    lines = [
        "% generated by hedonic-exp overlapping-gt-spectrum; do not edit",
        macro("Pairs", len(provenance)),
        macro("Expected", coverage.get("expected_detector_conditions", 0)),
        macro("Recorded", coverage.get("recorded_detector_conditions", 0)),
        macro("Verified", coverage.get("verified_equilibria", 0)),
        macro("NonEq", coverage.get("non_equilibrium_returns", 0)),
        macro("Timeouts", coverage.get("timeouts", 0)),
        macro("Failures", coverage.get("failures_and_unsupported", 0)),
        macro("Seeds", len(options.get("detector_seeds", []))),
        macro("AuditPoints", len(options.get("audit_resolutions", []))),
        macro("DetectorPoints", len(options.get("detector_resolutions", []))),
        macro("MaxNodes", options.get("max_nodes") or "full"),
        macro("Complete", "true" if complete else "false"),
        "\\newif\\ifGTSpecComplete",
        "\\GTSpecComplete" + ("true" if complete else "false"),
        "",
    ]
    return "\n".join(lines)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_all(output_dir: Path, audit_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]],
              summary_rows: list[dict[str, Any]], provenance: list[dict[str, Any]], coverage: dict[str, Any],
              options: dict[str, Any]) -> list[str]:
    """Write figures, tables and macros. Figures need matplotlib (the ``experiments`` extra)."""
    written: list[str] = []
    plot_dir = output_dir / "plots"
    policies = list(options["isolation_policies"])
    try:
        written += spectrum_figure(audit_rows, plot_dir / "gt_spectrum_stable_fraction", policies)
        written += accuracy_change_figure(result_rows, plot_dir / "gt_spectrum_accuracy_change", policies)
        written += accuracy_versus_change_figure(result_rows, plot_dir / "gt_spectrum_accuracy_vs_change")
    except ImportError:
        print("  matplotlib is not installed: figures skipped (pip install 'hedonic[experiments]')")
    bundle = output_dir / "paper"
    tables = output_dir / "tables"
    bundle.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    grid = {"detector_seeds": options.get("detector_seeds", options.get("seeds", [])),
            "audit_resolutions": options["audit_resolutions"], "detector_resolutions": options["detector_resolutions"],
            "max_nodes": options.get("max_nodes")}
    v3_rows = []
    for prov in provenance:
        for policy in policies:
            rows = sorted((r for r in audit_rows if r["dataset"] == prov["dataset"] and r["cover"] == prov["cover"]
                           and r["action_policy"] == policy), key=lambda r: r["gamma"])
            if not rows:
                continue
            nash = _nash_range(rows)
            v3_rows.append({
                "dataset": prov["dataset"], "cover": prov["cover"], "action_policy": policy,
                "n_vertices": prov["n_vertices"], "n_edges": prov["n_edges"], "density": prov["density"],
                "communities": prov["ground_truth_community_count"], "cap": prov["cap"],
                "stable_fraction_at_density": _interp([r["gamma"] for r in rows], [r["stable_fraction"] for r in rows],
                                                      prov["density"]),
                "min_stable_fraction": min(r["stable_fraction"] for r in rows),
                "nash_resolutions_in_grid": sum(bool(r["is_nash_equilibrium"]) for r in rows),
                "nash_gamma_min": None if nash is None else nash[0], "nash_gamma_max": None if nash is None else nash[1],
            })
    _csv(tables / "spectrum_pair_summary.csv", v3_rows)
    written.append(str(tables / "spectrum_pair_summary.csv"))
    contents = {
        bundle / "gt_spectrum_macros.tex": macros_tex(coverage, provenance, grid),
        bundle / "gt_spectrum_audit_table.tex": audit_table_tex(audit_rows, provenance, policies),
        bundle / "gt_spectrum_detector_table.tex": detector_table_tex(summary_rows, policies),
    }
    for path, text in contents.items():
        path.write_text(text, encoding="utf-8")
        written.append(str(path))
    return written
