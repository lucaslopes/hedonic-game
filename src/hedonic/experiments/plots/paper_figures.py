"""PHYSA paper figures for disjoint synthetic (V1020) experiments.

Ported from ``tmp/hedonic/scripts/plot/paper_plots/plot_figures.py``.

Archived V1020 figures (under ``.../V1020/figures/``)::

    gt_robustness.pdf   — ground-truth robustness hist + heatmaps
    noise.pdf           — metrics vs noise (bars + CI)
    n_communities.pdf   — metrics vs #communities (clean vs noise=1)
    acc_robustness.pdf  — KDE accuracy/ARI vs robustness
    acc_efficiency.pdf  — KDE accuracy/ARI vs duration

Reproduce via CLI (writes to a *new* root, e.g. V1020_CLI)::

    hedonic-exp plots \\
        --data /path/to/resultados.csv.gzip \\
        --output_dir /path/to/V1020_CLI/figures \\
        --format pdf

Or smoke (synthetic mini-dataframe, no full archive required)::

    hedonic-exp plots --smoke --output_dir /tmp/hedonic-figs
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Non-interactive backend before pyplot import (CLI / headless).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import gaussian_kde, t  # noqa: E402

from hedonic.experiments.config import SYNTHETIC_DIR

# ---------------------------------------------------------------------------
# Style constants (match archived paper figures)
# ---------------------------------------------------------------------------

METHODS_MAPPING = {
    "Leiden": "Leiden (full-fledged)",
    "Hedonic": "Leiden (phase 1)",
    "Spectral": "Spectral Clustering",
    "OnePass": "One Pass",
    "Mirror": "Mirror",
}

COLORMAP_DICT = {
    "Leiden": "Blues_r",
    "Hedonic": "Greens_r",
    "Spectral": "Oranges_r",
    "OnePass": "Reds_r",
    "Mirror": "Purples_r",
}

_TAB20B = plt.get_cmap("tab20b").colors
BAR_COLOR_DICT = {
    "Leiden": _TAB20B[2],
    "Hedonic": _TAB20B[6],
    "Spectral": _TAB20B[10],
    "OnePass": _TAB20B[14],
    "Mirror": _TAB20B[18],
}

# Filenames produced by the original V1020 paper pipeline.
FIGURE_NAMES = (
    "gt_robustness",
    "noise",
    "n_communities",
    "acc_robustness",
    "acc_efficiency",
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_data(path: str | Path) -> pd.DataFrame:
    """Load results from parquet or gzipped CSV."""
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # gzipped CSV (resultados.csv.gzip / resultados_ari.csv.gzip)
    return pd.read_csv(path, compression="gzip")


def normalize_score_column(df: pd.DataFrame, *, use_ari: bool = True) -> pd.DataFrame:
    """Ensure a single score column for plots.

    Archived ``resultados_ari`` has both ``accuracy`` and ``adjusted_rand``.
    The CLI sweep stores ARI under ``accuracy``. When ``use_ari`` is True we
    prefer ``adjusted_rand`` if present, else copy ``accuracy`` → ``adjusted_rand``.
    """
    out = df
    if use_ari:
        if "adjusted_rand" not in out.columns:
            if "accuracy" not in out.columns:
                raise ValueError(
                    "DataFrame needs 'adjusted_rand' or 'accuracy' for ARI plots"
                )
            out = out.copy()
            out["adjusted_rand"] = out["accuracy"]
    elif "accuracy" not in out.columns:
        if "adjusted_rand" in out.columns:
            out = out.copy()
            out["accuracy"] = out["adjusted_rand"]
        else:
            raise ValueError("DataFrame needs 'accuracy' or 'adjusted_rand'")
    return out


def include_spectral(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat Spectral rows (noise≈0.1) across other noise levels (paper pipeline)."""
    noises = sorted(df["noise"].unique())
    base_noise = 0.1 if 0.1 in noises or any(abs(n - 0.1) < 1e-9 for n in noises) else min(noises)
    spectral_rows = df[(df["method"] == "Spectral") & (np.isclose(df["noise"], base_noise))]
    if spectral_rows.empty:
        return df
    new_rows = []
    for noise in noises:
        if np.isclose(noise, base_noise):
            continue
        temp = spectral_rows.copy()
        temp["noise"] = noise
        new_rows.append(temp)
    if not new_rows:
        return df
    return pd.concat([df] + new_rows, ignore_index=True)


def load_or_compute(
    filepath: str | Path | None,
    compute_func: Callable[..., Any],
    *args,
    persist: bool = True,
    **kwargs,
):
    """Load pickled intermediate data if present; otherwise compute (and optionally save)."""
    if persist and filepath is not None and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded persisted data from {filepath}")
        return data
    data = compute_func(*args, **kwargs)
    if persist and filepath is not None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Computed and persisted data to {filepath}")
    return data


def set_plot_style(ax, dark_mode: bool) -> None:
    if dark_mode:
        ax.set_facecolor("black")
        ax.grid(True, color="gray", alpha=0.2)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")
    else:
        ax.set_facecolor("white")
        ax.grid(True, color="gray", alpha=0.2)
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")


def _score_key(use_ari: bool) -> str:
    return "adjusted_rand" if use_ari else "accuracy"


LOGICAL_CELL_COLUMNS = (
    "method",
    "number_of_communities",
    "p_in",
    "multiplier",
    "network_seed",
    "noise",
    "partition_seed",
)
KDE_SAMPLE_PER_METHOD = 2_000
KDE_SAMPLE_SEED = 20_260_830


def collapse_logical_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Give every experimental condition equal weight in summaries.

    Stochastic Leiden methods can emit several unique partitions for one
    condition, whereas deterministic baselines emit one. Averaging within the
    logical condition prevents the number of unique stochastic outcomes from
    changing the estimand.
    """
    if df.attrs.get("logical_cells_collapsed"):
        return df
    group_cols = [column for column in LOGICAL_CELL_COLUMNS if column in df.columns]
    value_cols = [
        column
        for column in ("duration", "robustness", "accuracy", "adjusted_rand")
        if column in df.columns
    ]
    if not group_cols or not value_cols:
        out = df.copy()
    else:
        out = (
            df.groupby(group_cols, as_index=False, sort=False, dropna=False)[value_cols]
            .mean()
        )
    out.attrs["logical_cells_collapsed"] = True
    return out


def _mean_and_seed_ci(sub: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Return the balanced mean and a two-sided 95% CI across network seeds."""
    if sub.empty or metric not in sub.columns:
        return np.nan, 0.0
    values = (
        sub.groupby("network_seed", sort=False)[metric].mean().to_numpy(dtype=float)
        if "network_seed" in sub.columns
        else sub[metric].to_numpy(dtype=float)
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0.0
    mean_val = float(values.mean())
    if values.size == 1:
        return mean_val, 0.0
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    ci = float(t.ppf(0.975, values.size - 1) * standard_error)
    return mean_val, ci


def _sample_for_kde(
    df: pd.DataFrame,
    *,
    per_method: int = KDE_SAMPLE_PER_METHOD,
    seed: int = KDE_SAMPLE_SEED,
) -> pd.DataFrame:
    """Deterministically sample equal numbers of logical cells per method."""
    sampled = []
    for offset, method in enumerate(METHODS_MAPPING):
        subset = df[df["method"] == method]
        if len(subset) > per_method:
            subset = subset.sample(n=per_method, random_state=seed + offset)
        sampled.append(subset)
    out = pd.concat(sampled, ignore_index=True) if sampled else df.iloc[0:0].copy()
    out.attrs["logical_cells_collapsed"] = True
    return out


# ---------------------------------------------------------------------------
# Figure 1 — ground-truth robustness
# ---------------------------------------------------------------------------


def compute_figure1_data(df: pd.DataFrame, cmap: str = "BuPu") -> dict:
    gt_df = df[df["method"] == "GroundTruth"]
    if gt_df.empty:
        # Fallback: empty structure so smoke still writes a figure shell.
        return {
            "communities": [],
            "global_max": 1,
            "hist_data": [],
            "heatmaps": {},
            "cmap": cmap,
        }
    graph_columns = [
        column
        for column in (
            "number_of_communities",
            "p_in",
            "multiplier",
            "network_seed",
        )
        if column in gt_df.columns
    ]
    if graph_columns:
        gt_df = (
            gt_df.groupby(graph_columns, as_index=False, sort=False)["robustness"]
            .mean()
        )
    else:
        gt_df = gt_df.drop_duplicates(subset=["robustness"])
    communities = sorted(gt_df["number_of_communities"].unique())
    global_max = 0
    hist_data = []
    for nc in communities:
        subset = gt_df[gt_df["number_of_communities"] == nc]["robustness"]
        counts, _ = np.histogram(subset, bins=100, range=(0, 1))
        percentages = 100.0 * counts / max(len(subset), 1)
        global_max = max(global_max, float(percentages.max()) if len(counts) else 0.0)
        hist_data.append((nc, subset, percentages))
    heatmaps = {}
    for nc in communities:
        sub = gt_df[gt_df["number_of_communities"] == nc]
        pivot = sub.pivot_table(
            values="robustness", index="p_in", columns="multiplier", aggfunc="mean"
        )
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        heatmaps[nc] = pivot
    return {
        "communities": communities,
        "global_max": max(global_max, 1),
        "hist_data": hist_data,
        "heatmaps": heatmaps,
        "cmap": cmap,
    }


def plot_figure1(
    data: dict,
    filename: str = "gt_robustness",
    fig_dir: str | Path | None = None,
    dark_mode: bool = False,
    file_format: str = "pdf",
) -> Path | None:
    communities = data["communities"]
    if not communities:
        print("  [figure1] no GroundTruth rows — skipping")
        return None
    global_max = data["global_max"]
    hist_data = data["hist_data"]
    heatmaps = data["heatmaps"]
    # Match the manuscript's blue-purple heatmap palette: higher robustness
    # is dark purple and lower robustness is pale blue.
    cmap = data.get("cmap", "BuPu")

    fig, axs = plt.subplots(2, len(communities), figsize=(16, 5), squeeze=False)
    fig.patch.set_facecolor("black" if dark_mode else "white")
    # Use 0.999 rather than the integer endpoint 1.0; ListedColormap treats
    # the latter as an index and wraps to its first color.
    hist_color = "white" if dark_mode else plt.get_cmap(cmap)(0.999)
    for i, (nc, subset, _counts) in enumerate(hist_data):
        ax = axs[0, i]
        weights = np.full(len(subset), 100.0 / max(len(subset), 1))
        ax.hist(subset, bins=100, range=(0, 1), weights=weights, color=hist_color)
        ax.set_title(f"{nc} Communities")
        ax.set_xlabel("Fraction of robust nodes")
        ax.set_ylim(0, global_max)
        ax.set_ylabel("Ground-truth graphs (%)" if i == 0 else "")
        set_plot_style(ax, dark_mode)
        ax.grid(False)

    im = None
    for i, nc in enumerate(communities):
        ax = axs[1, i]
        pivot = heatmaps[nc]
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_xlabel(r"Difficulty Factor ($\lambda$)")
        ax.set_ylabel(r"Intra Edge Probability ($p$)" if i == 0 else "")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], fontsize=8)
        set_plot_style(ax, dark_mode)
        ax.grid(False)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_ax.set_facecolor("black" if dark_mode else "white")
        cbar = fig.colorbar(im, cax=cbar_ax, label="Robustness")
        if dark_mode:
            cbar.ax.yaxis.label.set_color("white")
            cbar.ax.tick_params(colors="white")

    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    out = Path(fig_dir) / f"{filename}.{file_format}"
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — metrics vs noise
# ---------------------------------------------------------------------------


def compute_figure2_data(df: pd.DataFrame, use_ari: bool = True) -> dict:
    df = collapse_logical_cells(df)
    noise_levels = sorted(df["noise"].unique())
    metrics = ["duration", "robustness", _score_key(use_ari)]
    results: dict[str, dict] = {metric: {} for metric in metrics}
    for metric in metrics:
        for meth in METHODS_MAPPING:
            rows = []
            for nl in noise_levels:
                sub = df[(df["noise"] == nl) & (df["method"] == meth)]
                mean_val, ci = _mean_and_seed_ci(sub, metric)
                rows.append((nl, mean_val, ci))
            results[metric][meth] = rows
    return {
        "methods": list(METHODS_MAPPING.keys()),
        "noise_levels": noise_levels,
        "metrics": metrics,
        "results": results,
        "use_ari": use_ari,
    }


def plot_figure2(
    fig2_data: dict,
    filename: str = "noise",
    fig_dir: str | Path | None = None,
    dark_mode: bool = False,
    file_format: str = "pdf",
) -> Path:
    methods = fig2_data["methods"]
    noise_levels = fig2_data["noise_levels"]
    metrics = fig2_data["metrics"]
    results = fig2_data["results"]
    use_ari = fig2_data.get("use_ari", True)
    acc_key = _score_key(use_ari)

    fig, axs = plt.subplots(1, len(metrics), figsize=(15, 2), squeeze=False)
    fig.patch.set_facecolor("black" if dark_mode else "white")
    bar_width = 0.15
    indices = np.arange(len(noise_levels))
    handles, labels = [], []
    metric_title = {
        "duration": "Efficiency",
        "robustness": "Robustness",
        acc_key: "Accuracy",
    }
    metric_names = {
        "duration": "Time (s)",
        "robustness": "Robustness",
        acc_key: "Adjusted Rand Index" if use_ari else "Rand Index",
    }

    for col, metric in enumerate(metrics):
        ax = axs[0, col]
        for i, m in enumerate(methods):
            rows = results[metric][m]
            means = [r[1] for r in rows]
            cis = [r[2] for r in rows]
            bars = ax.bar(
                indices + i * bar_width,
                means,
                width=bar_width,
                yerr=cis,
                color=BAR_COLOR_DICT[m],
                capsize=3,
                label=METHODS_MAPPING[m] if col == 0 else "_nolegend_",
            )
            if col == 0:
                handles.append(bars[0])
                labels.append(METHODS_MAPPING[m])
        ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
        ax.set_xlabel("Noise")
        ax.set_ylabel(metric_names[metric])
        ax.set_title(metric_title[metric])
        set_plot_style(ax, dark_mode)

    fig.legend(
        handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, -0.1)
    )
    fig.subplots_adjust(bottom=0.1)
    out = Path(fig_dir) / f"{filename}.{file_format}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — metrics vs #communities
# ---------------------------------------------------------------------------


def precompute_subsets(
    df: pd.DataFrame | None = None,
    precomputed_all=None,
    precomputed_noisy=None,
    noisy_level: float = 1.0,
):
    if precomputed_all is not None:
        df_methods_all = precomputed_all
    else:
        assert df is not None
        df_methods_all = df[df["method"].isin(METHODS_MAPPING.keys())]
    if precomputed_noisy is not None:
        df_methods_noisy = precomputed_noisy
    else:
        df_methods_noisy = df_methods_all[
            np.isclose(df_methods_all["noise"].astype(float), noisy_level)
        ]
    if df_methods_all.attrs.get("logical_cells_collapsed"):
        df_methods_noisy.attrs["logical_cells_collapsed"] = True
    return df_methods_all, df_methods_noisy


def compute_figure3_data(
    df: pd.DataFrame | None = None,
    precomputed_all=None,
    precomputed_noisy=None,
    use_ari: bool = True,
) -> dict:
    df_methods_all, df_methods_noisy = precompute_subsets(
        df, precomputed_all, precomputed_noisy
    )
    df_methods_all = collapse_logical_cells(df_methods_all)
    df_methods_noisy = collapse_logical_cells(df_methods_noisy)
    communities = sorted(df_methods_all["number_of_communities"].unique())
    metrics = ["duration", "robustness", _score_key(use_ari)]
    results = {0: {m: {} for m in metrics}, 1: {m: {} for m in metrics}}
    for noise_val in (0, 1):
        subset = df_methods_noisy if noise_val == 1 else df_methods_all
        for metric in metrics:
            for meth in METHODS_MAPPING:
                rows = []
                for nc in communities:
                    sub = subset[
                        (subset["number_of_communities"] == nc) & (subset["method"] == meth)
                    ]
                    mean_val, ci = _mean_and_seed_ci(sub, metric)
                    rows.append((nc, mean_val, ci))
                results[noise_val][metric][meth] = rows
    return {
        "methods": list(METHODS_MAPPING.keys()),
        "communities": communities,
        "metrics": metrics,
        "results": results,
        "use_ari": use_ari,
    }


def plot_figure3(
    fig3_data: dict,
    filename: str = "n_communities",
    fig_dir: str | Path | None = None,
    dark_mode: bool = False,
    file_format: str = "pdf",
) -> Path:
    methods = fig3_data["methods"]
    communities = fig3_data["communities"]
    metrics = fig3_data["metrics"]
    results = fig3_data["results"]
    use_ari = fig3_data.get("use_ari", True)
    acc_key = _score_key(use_ari)

    fig, axs = plt.subplots(2, len(metrics), figsize=(15, 4), squeeze=False)
    fig.patch.set_facecolor("black" if dark_mode else "white")
    bar_width = 0.15
    indices = np.arange(len(communities))
    handles, labels = [], []
    metric_title = {
        "duration": "Efficiency",
        "robustness": "Robustness",
        acc_key: "Accuracy",
    }
    metric_names = {
        "duration": "Time (s)",
        "robustness": "Robustness",
        acc_key: "Adjusted Rand Index" if use_ari else "Rand Index",
    }

    for row, noise_val in enumerate((0, 1)):
        for col, metric in enumerate(metrics):
            ax = axs[row, col]
            for i, m in enumerate(methods):
                rows_data = results[noise_val][metric][m]
                means = [r[1] for r in rows_data]
                cis = [r[2] for r in rows_data]
                bars = ax.bar(
                    indices + i * bar_width,
                    means,
                    width=bar_width,
                    yerr=cis,
                    color=BAR_COLOR_DICT[m],
                    capsize=3,
                    label=METHODS_MAPPING[m]
                    if (row == 0 and col == 0)
                    else "_nolegend_",
                )
                if row == 0 and col == 0:
                    handles.append(bars[0])
                    labels.append(METHODS_MAPPING[m])
            ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
            ax.set_xticklabels([str(nc) for nc in communities])
            if row == 0:
                ax.set_title(metric_title[metric])
            if row == 1:
                ax.set_xlabel("Number of Communities")
            ax.set_ylabel(metric_names[metric])
            set_plot_style(ax, dark_mode)

    fig.legend(
        handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, -0.1)
    )
    fig.subplots_adjust(bottom=0)
    out = Path(fig_dir) / f"{filename}.{file_format}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — KDE contour plots
# ---------------------------------------------------------------------------


def compute_kde2d_generic(xvals, yvals, X, Y):
    if len(xvals) < 2:
        return np.zeros_like(X)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kde = gaussian_kde(np.vstack([xvals, yvals]), bw_method=0.2)
            coords = np.vstack([X.ravel(), Y.ravel()])
            return kde(coords).reshape(X.shape)
    except Exception:
        return np.zeros_like(X)


def _kde_worker(args):
    m_key, x_all, y_all, x_noisy, y_noisy, X, Y = args
    Z_clean = compute_kde2d_generic(x_all, y_all, X, Y)
    Z_noisy = compute_kde2d_generic(x_noisy, y_noisy, X, Y)
    return m_key, Z_clean, Z_noisy


def compute_figure4_data_common(
    df: pd.DataFrame | None,
    x_col: str,
    y_col: str,
    x_range: tuple,
    y_range: tuple,
    grid_size: int = 100,
    precomputed_all=None,
    precomputed_noisy=None,
    n_jobs: int = 1,
) -> tuple:
    df_methods_all, df_methods_noisy = precompute_subsets(
        df, precomputed_all, precomputed_noisy
    )
    xgrid = np.linspace(x_range[0], x_range[1], grid_size)
    ygrid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xgrid, ygrid)

    tasks = []
    for m_key in METHODS_MAPPING:
        sub_all = df_methods_all[df_methods_all["method"] == m_key]
        sub_noisy = df_methods_noisy[df_methods_noisy["method"] == m_key]
        tasks.append(
            (
                m_key,
                sub_all[x_col].values if x_col in sub_all else np.array([]),
                sub_all[y_col].values if y_col in sub_all else np.array([]),
                sub_noisy[x_col].values if x_col in sub_noisy else np.array([]),
                sub_noisy[y_col].values if y_col in sub_noisy else np.array([]),
                X,
                Y,
            )
        )

    kde_results: dict[str, dict] = {}
    if n_jobs and n_jobs != 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as ex:
                futs = [ex.submit(_kde_worker, t) for t in tasks]
                for fut in as_completed(futs):
                    m_key, Z_clean, Z_noisy = fut.result()
                    kde_results[m_key] = {"clean": Z_clean, "noisy": Z_noisy}
        except Exception as exc:
            print(f"  [kde] parallel failed ({exc}); falling back to serial")
            kde_results = {}
            for t in tasks:
                m_key, Z_clean, Z_noisy = _kde_worker(t)
                kde_results[m_key] = {"clean": Z_clean, "noisy": Z_noisy}
    else:
        for t in tasks:
            m_key, Z_clean, Z_noisy = _kde_worker(t)
            kde_results[m_key] = {"clean": Z_clean, "noisy": Z_noisy}
    return X, Y, kde_results


def compute_figure4a_robustness_data(
    df=None, precomputed_all=None, precomputed_noisy=None, use_ari: bool = True, n_jobs: int = 1
):
    y_col = _score_key(use_ari)
    X, Y, kde_results = compute_figure4_data_common(
        df,
        x_col="robustness",
        y_col=y_col,
        x_range=(-0.1, 1.2),
        y_range=(-0.1 if use_ari else 0, 1.1),
        precomputed_all=precomputed_all,
        precomputed_noisy=precomputed_noisy,
        n_jobs=n_jobs,
    )
    return {"X": X, "Y": Y, "kde_results": kde_results, "use_ari": use_ari}


def compute_figure4b_efficiency_data(
    df=None, precomputed_all=None, precomputed_noisy=None, use_ari: bool = True, n_jobs: int = 1
):
    y_col = _score_key(use_ari)
    if precomputed_all is not None:
        precomputed_all = precomputed_all.copy()
        precomputed_all["log10_duration"] = np.log10(
            precomputed_all["duration"].clip(lower=np.finfo(float).tiny)
        )
    if precomputed_noisy is not None:
        precomputed_noisy = precomputed_noisy.copy()
        precomputed_noisy["log10_duration"] = np.log10(
            precomputed_noisy["duration"].clip(lower=np.finfo(float).tiny)
        )
    if df is not None:
        df = df.copy()
        df["log10_duration"] = np.log10(
            df["duration"].clip(lower=np.finfo(float).tiny)
        )
    X, Y, kde_results = compute_figure4_data_common(
        df,
        x_col="log10_duration",
        y_col=y_col,
        x_range=(-5.2, 0.0),
        y_range=(-0.1 if use_ari else 0, 1.1),
        precomputed_all=precomputed_all,
        precomputed_noisy=precomputed_noisy,
        n_jobs=n_jobs,
    )
    return {
        "X": X,
        "Y": Y,
        "kde_results": kde_results,
        "use_ari": use_ari,
        "log_runtime": True,
    }


def plot_figure4(
    fig4_data: dict,
    xlabel: str,
    fig_dir: str | Path | None = None,
    file_format: str = "pdf",
) -> Path:
    X = fig4_data["X"]
    Y = fig4_data["Y"]
    kde_results = fig4_data["kde_results"]
    use_ari = fig4_data.get("use_ari", True)
    n_methods = len(METHODS_MAPPING)
    y_label = "Adjusted Rand Index" if use_ari else "Rand Index"

    fig, axs = plt.subplots(2, n_methods, figsize=(15, 6), squeeze=False)
    for j, (m_key, m_label) in enumerate(METHODS_MAPPING.items()):
        ax = axs[0, j]
        Z = kde_results.get(m_key, {}).get("clean")
        if Z is not None and np.any(Z):
            ax.contourf(X, Y, Z, levels=14, cmap=COLORMAP_DICT[m_key])
        ax.set_title(m_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label if j == 0 else "")
        ax2 = axs[1, j]
        Z = kde_results.get(m_key, {}).get("noisy")
        if Z is not None and np.any(Z):
            ax2.contourf(X, Y, Z, levels=14, cmap=COLORMAP_DICT[m_key])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(y_label if j == 0 else "")
        if fig4_data.get("log_runtime"):
            ticks = np.arange(-5, 1)
            tick_labels = [rf"$10^{{{tick}}}$" for tick in ticks]
            ax.set_xticks(ticks, tick_labels)
            ax2.set_xticks(ticks, tick_labels)

    fig.tight_layout()
    fname = (
        f"acc_robustness.{file_format}"
        if xlabel == "Robustness"
        else f"acc_efficiency.{file_format}"
    )
    out = Path(fig_dir) / fname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Synthetic smoke dataframe + full generation pipeline
# ---------------------------------------------------------------------------


def make_smoke_dataframe(n_per_cell: int = 8, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic results table with the same schema as V1020 CSV exports."""
    rng = np.random.default_rng(seed)
    methods = list(METHODS_MAPPING.keys()) + ["GroundTruth"]
    rows = []
    for n_comm in (2, 3):
        for noise in (0.1, 0.5, 1.0):
            for p_in in (0.05, 0.10):
                for mult in (0.3, 0.6):
                    for meth in methods:
                        for _ in range(n_per_cell):
                            acc = float(np.clip(rng.normal(0.6 if meth != "Mirror" else 0.2, 0.15), -1, 1))
                            rob = float(np.clip(rng.normal(0.5, 0.2), 0, 1))
                            dur = float(abs(rng.normal(0.05, 0.02)))
                            rows.append(
                                {
                                    "method": meth,
                                    "number_of_communities": n_comm,
                                    "community_size": 20,
                                    "p_in": p_in,
                                    "p_out": p_in * mult,
                                    "multiplier": mult,
                                    "resolution": 0.1,
                                    "duration": dur,
                                    "accuracy": acc,
                                    "adjusted_rand": acc,
                                    "robustness": rob,
                                    "noise": noise,
                                    "network_seed": 0,
                                    "partition_seed": 0,
                                }
                            )
    return pd.DataFrame(rows)


def generate_all_figures(
    df: pd.DataFrame,
    fig_dir: str | Path,
    *,
    persist_dir: str | Path | None = None,
    file_format: str = "pdf",
    dark_mode: bool = False,
    use_ari: bool = True,
    persist: bool = True,
    n_jobs: int = 1,
) -> list[Path]:
    """Compute and write all five V1020 paper figures. Returns written paths."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if persist_dir is None and persist:
        persist_dir = fig_dir.parent / "persist"
    if persist and persist_dir is not None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    df = normalize_score_column(df, use_ari=use_ari)
    written: list[Path] = []

    def _pp(name: str) -> str | None:
        if not persist or persist_dir is None:
            return None
        return str(Path(persist_dir) / name)

    print("Plotting figure 1 (gt_robustness)...")
    fig1 = load_or_compute(
        _pp("fig1_data.pkl"), compute_figure1_data, df, persist=persist
    )
    p = plot_figure1(
        fig1, "gt_robustness", fig_dir, dark_mode=dark_mode, file_format=file_format
    )
    if p is not None:
        written.append(p)
    del fig1
    gc.collect()

    df_methods_all, df_methods_noisy = precompute_subsets(
        collapse_logical_cells(include_spectral(df))
    )
    # Drop full df reference for memory on large archives.
    del df
    gc.collect()

    print("Plotting figure 2 (noise)...")
    fig2 = load_or_compute(
        _pp("fig2_data.pkl"),
        compute_figure2_data,
        df_methods_all,
        use_ari=use_ari,
        persist=persist,
    )
    written.append(
        plot_figure2(fig2, "noise", fig_dir, dark_mode=dark_mode, file_format=file_format)
    )
    del fig2
    gc.collect()

    print("Plotting figure 3 (n_communities)...")
    fig3 = load_or_compute(
        _pp("fig3_data.pkl"),
        compute_figure3_data,
        None,
        precomputed_all=df_methods_all,
        precomputed_noisy=df_methods_noisy,
        use_ari=use_ari,
        persist=persist,
    )
    written.append(
        plot_figure3(
            fig3, "n_communities", fig_dir, dark_mode=dark_mode, file_format=file_format
        )
    )
    del fig3
    gc.collect()

    kde_all = _sample_for_kde(df_methods_all)
    kde_noisy = _sample_for_kde(df_methods_noisy)
    del df_methods_all, df_methods_noisy
    gc.collect()

    print("Plotting figure 4a (acc_robustness)...")
    fig4a = load_or_compute(
        _pp("fig4a_robustness_data.pkl"),
        compute_figure4a_robustness_data,
        None,
        precomputed_all=kde_all,
        precomputed_noisy=kde_noisy,
        use_ari=use_ari,
        n_jobs=n_jobs,
        persist=persist,
    )
    written.append(
        plot_figure4(fig4a, xlabel="Robustness", fig_dir=fig_dir, file_format=file_format)
    )
    del fig4a
    gc.collect()

    print("Plotting figure 4b (acc_efficiency)...")
    fig4b = load_or_compute(
        _pp("fig4b_efficiency_data.pkl"),
        compute_figure4b_efficiency_data,
        None,
        precomputed_all=kde_all,
        precomputed_noisy=kde_noisy,
        use_ari=use_ari,
        n_jobs=n_jobs,
        persist=persist,
    )
    written.append(
        plot_figure4(
            fig4b,
            xlabel="Runtime (s; log scale)",
            fig_dir=fig_dir,
            file_format=file_format,
        )
    )
    del fig4b
    gc.collect()

    print(f"All figures written under {fig_dir}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PHYSA V1020 paper figures from disjoint experiment CSV/parquet. "
            "Produces the same five figure stems as archived V1020/figures/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Smoke (synthetic mini table, no archive required)\n"
            "  hedonic-exp plots --smoke --output_dir /tmp/hedonic-figs\n"
            "\n"
            "  # From archived V1020 results (read-only data) → new figures root\n"
            "  hedonic-exp plots \\\n"
            "      --data .../V1020/resultados_ari.csv.gzip \\\n"
            "      --output_dir .../V1020_CLI/figures --format pdf\n"
            "\n"
            "  # From CLI smoke CSV\n"
            "  hedonic-exp plots \\\n"
            "      --data .../V1020_CLI/resultados.csv.gzip \\\n"
            "      --output_dir .../V1020_CLI/figures --no-persist\n"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to resultados CSV.gzip / parquet "
            f"(default: {{SYNTHETIC_DIR}}/resultados_ari.csv.gzip → {SYNTHETIC_DIR})"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for figures (default: <data_parent>/figures or ./figures for smoke)",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=None,
        help="Directory for intermediate pickles (default: sibling 'persist')",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "svg", "png"],
        help="Output image format (default: pdf)",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Dark background / light text (figure 1 style)",
    )
    parser.add_argument(
        "--no-ari",
        action="store_true",
        help="Use accuracy column instead of adjusted_rand / ARI labeling",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not read/write intermediate pickle caches",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers for KDE (1 = serial; -1 = all CPUs)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate figures from a tiny synthetic dataframe (no data file needed)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional row cap after load (for quick checks on huge archives)",
    )
    args = parser.parse_args(argv)

    archived_figures = (
        Path("~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/figures")
        .expanduser()
        .resolve()
    )

    if args.smoke:
        print("[plots smoke] building synthetic mini results table")
        df = make_smoke_dataframe()
        fig_dir = Path(args.output_dir or "figures_smoke")
        persist = not args.no_persist
        persist_dir = args.persist_dir
    else:
        data_path = args.data
        if data_path is None:
            # Prefer ARI table if present, else plain resultados.
            candidates = [
                SYNTHETIC_DIR / "resultados_ari.csv.gzip",
                SYNTHETIC_DIR / "resultados.csv.gzip",
            ]
            data_path = next((str(p) for p in candidates if p.is_file()), str(candidates[0]))
        if not Path(data_path).is_file():
            print(f"Data file not found: {data_path}", flush=True)
            return 1
        print(f"Loading data from {data_path} ...")
        df = load_data(data_path)
        if args.max_rows is not None and len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=0)
            print(f"Sampled to {len(df)} rows (--max_rows)")
        print(f"Data loaded. Rows = {len(df)}")
        fig_dir = Path(
            args.output_dir
            if args.output_dir
            else Path(data_path).resolve().parent / "figures"
        )
        persist = not args.no_persist
        persist_dir = args.persist_dir

    fig_dir = fig_dir.resolve()
    if fig_dir == archived_figures:
        print(
            "Refusing to write figures into the archived V1020/figures folder.\n"
            "Pass a different --output_dir, e.g.\n"
            "  --output_dir ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI/figures",
            flush=True,
        )
        return 1

    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    written = generate_all_figures(
        df,
        fig_dir,
        persist_dir=persist_dir,
        file_format=args.format,
        dark_mode=args.dark_mode,
        use_ari=not args.no_ari,
        persist=persist,
        n_jobs=n_jobs,
    )
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
