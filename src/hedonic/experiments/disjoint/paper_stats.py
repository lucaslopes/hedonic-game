"""Summary statistics of the disjoint SBM benchmark, as reported in the paper.

Computes, from a result store or its logical-cell table, every number the
Empirical Analysis of the disjoint manuscript quotes:

* grid and replication counts (taken from the data, never assumed);
* Kesten--Stigum signal-to-noise ratio (SNR) of each condition, the number of
  conditions with SNR <= 1 per K and the largest such SNR;
* ground-truth robustness: mean and fully-robust share per K, the overall
  fully-robust and >= 0.9 shares, the 95% t-interval lengths of the heatmap
  cells, the Spearman correlation with SNR and the SNR-band table;
* community tracking: per-method means by noise level and by K (all noise and
  noise = 1) with two-sided 95% t-intervals across network-seed means, the
  Phase-1 / full Leiden runtime ratio;
* accuracy relative to the detectability threshold: ARI at noise = 1 by SNR
  band, means below/above the threshold, Spearman(SNR, ARI) per method, and
  refinement gain over Mirror by SNR band and noise.

Aggregation follows the manuscript: restarts are averaged within each logical
cell (method, K, p, difficulty, network seed, noise, partition seed), every
logical cell of a condition has equal weight, and condition means are
averaged with equal weight.  This replaces the 30-seed-only
``analysis/ks_threshold.py`` with a CLI command that reads any store.

Outputs under ``--output_dir``: ``paper_stats.json`` (everything),
``*.csv`` tables, ``tables.tex`` (the two SNR tables in the manuscript's
layout) and ``summary.md`` (the quoted numbers, one line each).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GT_BANDS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf]
TRACK_BANDS = [0.0, 1.0, 2.0, 4.0, math.inf]
CONDITION_KEYS = ["number_of_communities", "p_in", "multiplier"]
CELL_KEYS = ["method", *CONDITION_KEYS, "network_seed", "noise", "partition_seed"]
TRACKING_METHODS = ["Leiden", "Hedonic", "Spectral", "OnePass", "Mirror"]
ACCURACY = ["adjusted_rand", "overlap_normalized", "nmi", "vi"]
METRICS = ["duration", "robustness", *ACCURACY]


def load_cells(data: str | Path) -> pd.DataFrame:
    """Logical-cell table from a store directory or an exported table."""
    from hedonic.experiments.disjoint import store as store_mod

    path = Path(data)
    if path.is_dir():
        return store_mod.plot_table(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def snr(frame: pd.DataFrame) -> pd.Series:
    """Kesten--Stigum SNR, p n (1-l)^2 / (K (1 + (K-1) l)), with n = K N."""
    k = frame["number_of_communities"].astype(float)
    n = k * frame["community_size"].astype(float)
    lam = frame["multiplier"].astype(float)
    return frame["p_in"] * n * (1 - lam) ** 2 / (k * (1 + (k - 1) * lam))


def _t_half_width(values: np.ndarray) -> float:
    from scipy.stats import t

    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))


def _seed_ci_table(cells: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean of network-seed means and the 95% t half-width, per group."""
    seed_means = cells.groupby([*by, "network_seed"], observed=True)[METRICS].mean()
    rows = []
    for key, sub in seed_means.groupby(level=list(range(len(by))), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["network_seeds"] = len(sub)
        for metric in METRICS:
            values = sub[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{metric}_ci95"] = _t_half_width(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _r(value, digits: int = 4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def compute_stats(cells: pd.DataFrame) -> dict[str, Any]:
    cells = cells.copy()
    for metric in ACCURACY:  # stores of schema v1 have ARI only; the others report as missing
        if metric not in cells.columns:
            cells[metric] = np.nan
    cells["snr"] = snr(cells)
    stats: dict[str, Any] = {}

    graphs = cells.drop_duplicates([*CONDITION_KEYS, "network_seed"])
    per_condition = graphs.groupby(CONDITION_KEYS).size()
    stats["grid"] = {
        "conditions": len(per_condition),
        "graphs": len(graphs),
        "network_seeds": sorted(int(s) for s in graphs["network_seed"].unique()),
        "seeds_per_condition_min": int(per_condition.min()),
        "seeds_per_condition_max": int(per_condition.max()),
        "communities": sorted(int(k) for k in cells["number_of_communities"].unique()),
        "noises": sorted(float(x) for x in cells["noise"].dropna().unique()),
        "partition_seeds": int(cells["partition_seed"].nunique()),
        "methods": sorted(cells["method"].unique()),
        "logical_cells": len(cells),
        "n_vertices": sorted(
            int(v) for v in (graphs["number_of_communities"] * graphs["community_size"]).unique()
        ),
    }
    seeds = stats["grid"]["seeds_per_condition_min"]

    # --- SNR of the grid ---------------------------------------------------
    conditions = graphs.drop_duplicates(CONDITION_KEYS)[[*CONDITION_KEYS, "community_size", "snr"]]
    below = conditions[conditions["snr"] <= 1]
    stats["snr"] = {
        "conditions_le_1_by_K": {
            int(k): int(n) for k, n in below.groupby("number_of_communities").size().items()
        },
        "conditions_by_K": {
            int(k): int(n) for k, n in conditions.groupby("number_of_communities").size().items()
        },
        "max_snr_le_1": _r(below["snr"].max()),
        "conditions_exactly_1": int((conditions["snr"] == 1).sum()),
    }

    # --- Ground-truth robustness -------------------------------------------
    gt = cells[cells["method"] == "GroundTruth"].drop_duplicates(
        [*CONDITION_KEYS, "network_seed"]
    )
    if not gt.empty:
        by_k = gt.groupby("number_of_communities")["robustness"]
        heat = gt.groupby(CONDITION_KEYS)["robustness"]
        ci_len = heat.apply(lambda s: 2 * _t_half_width(s.to_numpy(dtype=float)))
        cond_gt = heat.mean().rename("robustness").reset_index().merge(conditions, on=CONDITION_KEYS)
        cond_gt["band"] = pd.cut(cond_gt["snr"], GT_BANDS)
        band_mean = cond_gt.pivot_table(
            index="band", columns="number_of_communities", values="robustness",
            aggfunc="mean", observed=True,
        )
        band_n = cond_gt.pivot_table(
            index="band", columns="number_of_communities", values="robustness",
            aggfunc="size", observed=True,
        )
        stats["ground_truth"] = {
            "graphs": len(gt),
            "mean_by_K": {int(k): _r(v, 3) for k, v in by_k.mean().items()},
            "fully_robust_pct_by_K": {
                int(k): _r(100 * (s == 1.0).mean(), 1) for k, s in by_k
            },
            "fully_robust_pct": _r(100 * (gt["robustness"] == 1.0).mean(), 1),
            "at_least_0_9_pct": _r(100 * (gt["robustness"] >= 0.9).mean(), 1),
            "heatmap_cells": len(ci_len),
            "heatmap_ci95_length": {
                "median": _r(ci_len.median()),
                "p95": _r(ci_len.quantile(0.95)),
                "max": _r(ci_len.max()),
                "cells_le_0_01": int((ci_len <= 0.01).sum()),
                "cells_le_0_01_pct": _r(100 * (ci_len <= 0.01).mean(), 1),
            },
            "spearman_snr": _r(cond_gt[["snr", "robustness"]].corr("spearman").iloc[0, 1], 3),
            "max_abs_K_spread_above_threshold": _r(
                (band_mean.loc[[b for b in band_mean.index if b.left >= 1]].max(axis=1)
                 - band_mean.loc[[b for b in band_mean.index if b.left >= 1]].min(axis=1)).max(),
                3,
            ),
        }
        stats["_gt_band_table"] = (band_mean, band_n)

    # --- Community tracking --------------------------------------------------
    tracking = cells[cells["method"].isin(TRACKING_METHODS)]
    overall = _seed_ci_table(tracking, ["method"])
    by_noise = _seed_ci_table(tracking, ["method", "noise"])
    by_k = _seed_ci_table(tracking, ["method", "number_of_communities"])
    by_k_noise1 = _seed_ci_table(
        tracking[np.isclose(tracking["noise"], 1.0)], ["method", "number_of_communities"]
    )
    stats["tracking"] = {
        "mean_duration_s": {
            r.method: _r(r.duration, 5) for r in overall.itertuples()
        },
        "mean_robustness": {r.method: _r(r.robustness, 3) for r in overall.itertuples()},
        "mean_ari": {r.method: _r(r.adjusted_rand, 3) for r in overall.itertuples()},
        "mean_accuracy": {
            metric: {r.method: _r(getattr(r, metric), 3) for r in overall.itertuples()}
            for metric in ACCURACY
        },
    }
    durations = stats["tracking"]["mean_duration_s"]
    if durations.get("Hedonic") and durations.get("Leiden"):
        stats["tracking"]["leiden_over_phase1_duration"] = _r(
            durations["Leiden"] / durations["Hedonic"], 2
        )
    stats["_tables"] = {
        "tracking_overall": overall,
        "tracking_by_noise": by_noise,
        "tracking_by_K_all_noise": by_k,
        "tracking_by_K_noise1": by_k_noise1,
    }

    # --- Detectability threshold ---------------------------------------------
    cond = (
        tracking.groupby(["method", *CONDITION_KEYS, "noise"], observed=True)[
            [*ACCURACY, "robustness"]
        ]
        .mean()
        .reset_index()
        .merge(conditions, on=CONDITION_KEYS)
    )
    full = cond[np.isclose(cond["noise"], 1.0)]
    threshold: dict[str, Any] = {"ari_noise1": {}}
    for method in TRACKING_METHODS:
        sub = full[full["method"] == method]
        if sub.empty:
            continue
        lo, hi = sub[sub["snr"] <= 1], sub[sub["snr"] > 1]
        threshold["ari_noise1"][method] = {
            "mean_snr_le_1": _r(lo["adjusted_rand"].mean(), 3),
            "max_snr_le_1": _r(lo["adjusted_rand"].max(), 3),
            "mean_snr_gt_1": _r(hi["adjusted_rand"].mean(), 3),
            "spearman_snr": _r(sub[["snr", "adjusted_rand"]].corr("spearman").iloc[0, 1], 3),
        }
    threshold["accuracy_noise1"] = {}
    for metric in ACCURACY:
        per_method = {}
        for method in TRACKING_METHODS:
            sub = full[full["method"] == method]
            if sub.empty:
                continue
            lo, hi = sub[sub["snr"] <= 1], sub[sub["snr"] > 1]
            per_method[method] = {
                "mean_snr_le_1": _r(lo[metric].mean(), 4),
                "max_snr_le_1": _r(lo[metric].max(), 4),
                "mean_snr_gt_1": _r(hi[metric].mean(), 4),
                "spearman_snr": _r(sub[["snr", metric]].corr("spearman").iloc[0, 1], 3),
            }
        threshold["accuracy_noise1"][metric] = per_method
    threshold["below_threshold_spread"] = _below_threshold_spread(tracking, conditions)
    full = full.assign(band=pd.cut(full["snr"], TRACK_BANDS))
    band_ari = full[full["method"].isin(["Hedonic", "Leiden", "Spectral"])].pivot_table(
        index="band", columns="method", values="adjusted_rand", aggfunc="mean", observed=True
    )
    band_nc = (
        full[full["method"] == "Hedonic"].groupby("band", observed=True).size()
        if (full["method"] == "Hedonic").any()
        else pd.Series(dtype=int)
    )
    mirror = cond[cond["method"] == "Mirror"].set_index([*CONDITION_KEYS, "noise"])["adjusted_rand"]
    gains = []
    for method in ["Hedonic", "Leiden"]:
        own = cond[cond["method"] == method].set_index([*CONDITION_KEYS, "noise"])
        if own.empty or mirror.empty:
            continue
        gain = (own["adjusted_rand"] - mirror).rename("gain").to_frame().join(own["snr"]).reset_index()
        for noise in sorted(gain["noise"].unique()):
            g = gain[np.isclose(gain["noise"], noise)]
            for lo, hi in itertools.pairwise(TRACK_BANDS):
                s = g[(g["snr"] > lo) & (g["snr"] <= hi)]
                gains.append({
                    "method": method, "noise": float(noise), "band": f"({lo:g},{hi:g}]",
                    "conditions": len(s), "wins": int((s["gain"] > 0).sum()),
                    "mean_gain": _r(s["gain"].mean(), 3),
                })
            k4 = g[(g["snr"] <= 1) & (g["number_of_communities"] <= 4)]
            gains.append({
                "method": method, "noise": float(noise), "band": "K<=4, SNR<=1",
                "conditions": len(k4), "wins": int((k4["gain"] > 0).sum()),
                "mean_gain": _r(k4["gain"].mean(), 3),
            })
    threshold["refinement_vs_mirror"] = gains
    stats["threshold"] = threshold
    stats["_threshold_tables"] = (band_ari, band_nc, pd.DataFrame(gains))
    accuracy_bands = (
        full[full["method"].isin(["Hedonic", "Leiden", "Spectral"])]
        .groupby(["band", "method"], observed=True)[ACCURACY].mean()
        .reset_index()
    )
    accuracy_bands["band"] = accuracy_bands["band"].astype(str)
    stats["_tables"]["tracking_snr_band_all_metrics"] = accuracy_bands
    stats["grid"]["seeds_per_condition"] = seeds
    return stats


def _below_threshold_spread(tracking: pd.DataFrame, conditions: pd.DataFrame) -> dict:
    """Seed-level spread of accuracy from an uninformative start below SNR = 1.

    For noise = 1 and SNR <= 1, each graph (condition × network seed) gets the
    mean of its logical cells.  At finite n the chance level of a metric need
    not be 0 (maximizing over matchings biases overlap upward, NMI is positive
    for any finite pair), so the reference is empirical and paired: Mirror at
    noise 1 is a uniformly shuffled partition with K equal blocks, averaged
    over the partition seeds of the same graph.  Per method and metric this
    reports the distribution of graph values, the chance level, the paired
    excess over chance, and the share of conditions whose two-sided 95%
    t-interval of the excess (across network seeds) contains 0.
    """
    sub = tracking[np.isclose(tracking["noise"], 1.0) & (tracking["snr"] <= 1)]
    out: dict[str, Any] = {}
    if sub.empty:
        return out
    keys = [*CONDITION_KEYS, "network_seed"]
    graphs = sub.groupby(["method", *keys])[ACCURACY].mean().reset_index()
    chance = graphs[graphs["method"] == "Mirror"].set_index(keys)[ACCURACY]
    for method, gm in graphs.groupby("method"):
        out[method] = {}
        gm = gm.set_index(keys)
        for metric in ("adjusted_rand", "overlap_normalized", "nmi"):
            values = gm[metric].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            item: dict[str, Any] = {
                "graphs": len(values),
                "mean": _r(values.mean(), 4),
                "sd": _r(values.std(ddof=1) if values.size > 1 else 0.0, 4),
                "p05": _r(np.quantile(values, 0.05), 4),
                "median": _r(np.quantile(values, 0.5), 4),
                "p95": _r(np.quantile(values, 0.95), 4),
                "max": _r(values.max(), 4),
            }
            if not chance.empty:
                excess = (gm[metric] - chance[metric]).dropna()
                covers = []
                for _, cond in excess.groupby(level=list(range(len(CONDITION_KEYS)))):
                    v = cond.to_numpy(dtype=float)
                    if v.size > 1:
                        covers.append(abs(float(v.mean())) <= _t_half_width(v))
                item.update({
                    "chance_mirror_mean": _r(chance[metric].mean(), 4),
                    "excess_over_chance_mean": _r(excess.mean(), 4),
                    "excess_over_chance_sd": _r(excess.std(ddof=1), 4) if len(excess) > 1 else None,
                    "conditions": len(covers),
                    "conditions_excess_ci_contains_0_pct": (
                        _r(100 * np.mean(covers), 1) if covers else None
                    ),
                })
            out[method][metric] = item
    return out


def _tex_tables(stats: dict[str, Any]) -> str:
    seeds = stats["grid"]["seeds_per_condition_min"]
    lines = [
        "% Generated by `hedonic-exp disjoint-stats`; numbers come from the run.",
        f"% network seeds per condition: {seeds}",
    ]
    if "_gt_band_table" in stats:
        mean, count = stats["_gt_band_table"]
        ks = list(mean.columns)
        lines += [
            r"\begin{tabular}{l" + "c" * len(ks) + "}", r"\hline",
            r"$\mathrm{SNR}$ band & " + " & ".join(f"$K={k}$" for k in ks) + r" \\ \hline",
        ]
        for band in mean.index:
            label = f"$({band.left:g},{'\\infty' if math.isinf(band.right) else f'{band.right:g}'}]$"
            cells = []
            for k in ks:
                v, n = mean.loc[band, k], count.loc[band, k] if k in count.columns else np.nan
                cells.append("--" if pd.isna(v) else f"{v:.3f} ({int(n)})")
            lines.append(label + " & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", ""]
    band_ari, band_nc, gains = stats["_threshold_tables"]
    if not band_ari.empty:
        lines += [
            r"\begin{tabular}{lr|ccc|ccc}", r"\hline",
            r" & & \multicolumn{3}{c|}{ARI at $\noise=1$} & \multicolumn{3}{c}{Phase~1 $-$ Mirror (wins)} \\",
            r"$\mathrm{SNR}$ band & $n_c$ & Phase~1 & Full & Spectral & $\noise=0.1$ & $\noise=0.25$ & $\noise=0.5$ \\ \hline",
        ]
        for band in band_ari.index:
            label = f"$({band.left:g},{'\\infty' if math.isinf(band.right) else f'{band.right:g}'}]$"
            key = f"({band.left:g},{band.right:g}]"
            row = [label, str(int(band_nc.get(band, 0)))]
            for method in ["Hedonic", "Leiden", "Spectral"]:
                v = band_ari.loc[band].get(method, np.nan)
                row.append("--" if pd.isna(v) else f"{v:.3f}")
            for noise in (0.1, 0.25, 0.5):
                g = gains[(gains["method"] == "Hedonic") & np.isclose(gains["noise"], noise)
                          & (gains["band"] == key)] if not gains.empty else gains
                if g.empty or g.iloc[0]["mean_gain"] is None:
                    row.append("--")
                else:
                    row.append(f"${g.iloc[0]['mean_gain']:+.3f}$ ({int(g.iloc[0]['wins'])})")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def _summary_md(stats: dict[str, Any]) -> str:
    g = stats["grid"]
    out = [
        "# Disjoint SBM benchmark: reported statistics",
        "",
        (f"- Grid: {g['conditions']} conditions × {g['seeds_per_condition_min']}"
        f"–{g['seeds_per_condition_max']} network seeds = {g['graphs']} graphs; "
        f"noises {g['noises']}, {g['partition_seeds']} partition seeds, "
        f"{g['logical_cells']} logical cells."),
        (f"- SNR <= 1 conditions per K: {stats['snr']['conditions_le_1_by_K']} "
        f"(max SNR among them {stats['snr']['max_snr_le_1']})."),
    ]
    if "ground_truth" in stats:
        t = stats["ground_truth"]
        ci = t["heatmap_ci95_length"]
        out += [
            f"- Ground-truth robustness mean by K: {t['mean_by_K']}",
            (f"- Fully robust share by K (%): {t['fully_robust_pct_by_K']}; overall "
            f"{t['fully_robust_pct']}% fully robust, {t['at_least_0_9_pct']}% >= 0.9 "
            f"(of {t['graphs']} graphs)."),
            (f"- Heatmap 95% t-interval length over {t['heatmap_cells']} cells: median "
            f"{ci['median']}, p95 {ci['p95']}, max {ci['max']}; {ci['cells_le_0_01']} cells "
            f"({ci['cells_le_0_01_pct']}%) <= 0.01."),
            (f"- Spearman(GT robustness, SNR) = {t['spearman_snr']}; largest spread across K "
            f"within an SNR > 1 band = {t['max_abs_K_spread_above_threshold']}."),
        ]
    tr = stats["tracking"]
    out += [
        (f"- Mean runtime (s): {tr['mean_duration_s']}; full/Phase-1 ratio "
        f"{tr.get('leiden_over_phase1_duration')}."),
        f"- Mean robustness: {tr['mean_robustness']}",
        f"- Mean ARI: {tr['mean_ari']}",
    ]
    for metric, per_method in stats["threshold"]["accuracy_noise1"].items():
        if metric == "adjusted_rand":
            continue
        for method, item in per_method.items():
            if method in ("Hedonic", "Leiden", "Spectral"):
                out.append(
                    f"- {metric} at noise 1, {method}: SNR<=1 mean {item['mean_snr_le_1']} "
                    f"(max {item['max_snr_le_1']}), SNR>1 mean {item['mean_snr_gt_1']}, "
                    f"Spearman {item['spearman_snr']}."
                )
    for method, metrics in stats["threshold"]["below_threshold_spread"].items():
        for metric, item in metrics.items():
            if method in ("Hedonic", "Leiden", "Spectral"):
                out.append(
                    f"- Below SNR=1, noise 1, {method} {metric} over {item['graphs']} graphs: "
                    f"mean {item['mean']} sd {item['sd']} p95 {item['p95']} max {item['max']}; "
                    f"chance (Mirror) {item.get('chance_mirror_mean')}, excess "
                    f"{item.get('excess_over_chance_mean')}; seed CI of the excess contains 0 "
                    f"in {item.get('conditions_excess_ci_contains_0_pct')}% of "
                    f"{item.get('conditions')} conditions."
                )
    for method, item in stats["threshold"]["ari_noise1"].items():
        out.append(
            f"- ARI at noise 1, {method}: SNR<=1 mean {item['mean_snr_le_1']} (max "
            f"{item['max_snr_le_1']}), SNR>1 mean {item['mean_snr_gt_1']}, Spearman "
            f"{item['spearman_snr']}."
        )
    out += ["", "Per-noise and per-K tables: `tracking_by_noise.csv`, `tracking_by_K_*.csv`;",
            "SNR tables: `tables.tex`, `gt_snr_band.csv`, `refinement_vs_mirror.csv`.", ""]
    return "\n".join(out)


def write_stats(stats: dict[str, Any], output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in stats["_tables"].items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)
    if "_gt_band_table" in stats:
        mean, count = stats["_gt_band_table"]
        joined = mean.round(4).astype(str) + " (" + count.astype("Int64").astype(str) + ")"
        joined.to_csv(out / "gt_snr_band.csv")
        written.append(out / "gt_snr_band.csv")
    band_ari, band_nc, gains = stats["_threshold_tables"]
    band_ari.assign(conditions=band_nc).to_csv(out / "tracking_snr_band.csv")
    gains.to_csv(out / "refinement_vs_mirror.csv", index=False)
    written += [out / "tracking_snr_band.csv", out / "refinement_vs_mirror.csv"]
    (out / "tables.tex").write_text(_tex_tables(stats), encoding="utf-8")
    (out / "summary.md").write_text(_summary_md(stats), encoding="utf-8")
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    (out / "paper_stats.json").write_text(
        json.dumps(public, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    written += [out / "tables.tex", out / "summary.md", out / "paper_stats.json"]
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp disjoint-stats",
        description=(
            "Every statistic quoted in the disjoint manuscript's Empirical Analysis "
            "(grid counts, SNR threshold, ground-truth robustness, tracking means, "
            "SNR-band tables), computed from a result store."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  hedonic-exp disjoint-stats --data artifacts/disjoint/sbm-smoke\n"
            "  hedonic-exp disjoint-stats --data artifacts/disjoint/sbm100/sbm/plot_table.parquet \\\n"
            "      --output_dir artifacts/disjoint/sbm100/sbm/stats\n"
        ),
    )
    parser.add_argument(
        "--data", required=True,
        help="Result store directory or its exported logical-cell table (.parquet/.csv)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write the statistics (default: <store>/stats or <table dir>/stats)",
    )
    args = parser.parse_args(argv)
    data = Path(args.data).expanduser()
    if not data.exists():
        print(f"Not found: {data}")
        return 1
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        (data if data.is_dir() else data.parent) / "stats"
    )
    from hedonic.experiments.config import ensure_not_archived_v1020

    try:
        ensure_not_archived_v1020(output_dir.resolve(), label="stats output")
    except ValueError as exc:
        print(str(exc))
        return 1
    stats = compute_stats(load_cells(data))
    for path in write_stats(stats, output_dir):
        print(f"  {path}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
