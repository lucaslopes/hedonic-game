"""Small, reusable helpers for smoke reports and notebooks.

The detector adapters and their timeout policy live in :mod:`method_smoke`.
This module owns the presentation-facing composition around them: environment
configuration, source loading, a stable report object, tabular formatting, and
figures.  A QMD cell or a notebook can therefore express the whole workflow as
``report = build_smoke_report(); report.table()`` without reimplementing any
experiment policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Sequence

from hedonic.experiments.config import DBLP_DIR
from hedonic.experiments.overlapping.method_smoke import (
    METRICS,
    SmokeSource,
    default_smoke_config,
    load_smoke_source,
    run_smoke_methods,
)


TABLE_COLUMNS: tuple[str, ...] = (
    "method",
    "status",
    "wall_seconds",
    "n_communities",
    "coverage",
    *METRICS,
    "failure_or_limitation",
)


def _finite_value(row: dict[str, Any], field: str) -> float | None:
    """Read a finite numeric field without letting display sort fail."""
    try:
        value = float(row.get(field))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _quality_value(
    row: dict[str, Any],
    metrics: Sequence[str],
) -> float | None:
    """Return the mean of available quality metrics for one method row."""
    values = [
        value
        for metric in metrics
        if (value := _finite_value(row, metric)) is not None
    ]
    return sum(values) / len(values) if values else None


def order_smoke_rows(
    rows: Sequence[dict[str, Any]],
    *,
    criterion: str = "quality",
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, Any]]:
    """Order smoke results from best to worst for a chosen criterion.

    Quality scores are higher-is-better, while ``wall_seconds`` is
    lower-is-better.  The default table/metric-figure ranking uses the mean
    of the available quality metrics, so all scores share the same [0, 1]
    direction.  Missing values and non-completed rows are placed last, and
    method names provide a deterministic tie-breaker.

    ``criterion`` may be ``"quality"``, ``"wall_seconds"``, or one of the
    individual metric names.  Exposing this small policy helper keeps a
    notebook or a future report from duplicating an easy-to-get-backwards
    sorting rule.
    """
    metric_names = tuple(metrics)
    valid_criteria = {"quality", "wall_seconds", *metric_names}
    if criterion not in valid_criteria:
        choices = ", ".join(sorted(valid_criteria))
        raise ValueError(f"unknown smoke ordering criterion {criterion!r}; choose {choices}")

    decorated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        quality = _quality_value(copied, metric_names)
        wall_seconds = _finite_value(copied, "wall_seconds")
        if criterion == "quality":
            primary = quality
            lower_is_better = False
        elif criterion == "wall_seconds":
            primary = wall_seconds
            lower_is_better = True
        else:
            primary = _finite_value(copied, criterion)
            lower_is_better = False

        primary_missing = primary is None
        if primary_missing:
            primary_key: float = math.inf
        elif lower_is_better:
            primary_key = float(primary)
        else:
            primary_key = -float(primary)

        # Completed rows always precede failures/skips.  Ties in the selected
        # criterion prefer better quality, then lower runtime, then method.
        quality_key = math.inf if quality is None else -quality
        wall_key = math.inf if wall_seconds is None else wall_seconds
        key = (
            0 if copied.get("status") == "completed" else 1,
            primary_missing,
            primary_key,
            quality is None,
            quality_key,
            wall_seconds is None,
            wall_key,
            str(copied.get("method", "")),
            index,
        )
        decorated.append((key, copied))

    return [row for _, row in sorted(decorated, key=lambda item: item[0])]


@dataclass(frozen=True)
class SmokeConfig:
    """Configuration shared by a smoke report and notebook reruns."""

    dblp_root: Path
    max_vertices: int = 80
    timeout_seconds: float = 30.0
    omega_sample_size: int = 10_000
    seed: int = 7
    allow_compatibility_fallbacks: bool = True

    def __post_init__(self) -> None:
        if self.max_vertices < 2:
            raise ValueError("max_vertices must be at least 2")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if self.omega_sample_size <= 0:
            raise ValueError("omega_sample_size must be positive")

    @classmethod
    def from_environment(
        cls,
        *,
        dblp_root: str | Path | None = None,
        **overrides: Any,
    ) -> "SmokeConfig":
        """Build configuration from ``HEDONIC_*`` variables plus overrides."""
        defaults = default_smoke_config()
        configured_root = (
            dblp_root
            if dblp_root is not None
            else os.getenv("HEDONIC_DBLP_DIR", str(DBLP_DIR))
        )
        return cls(
            dblp_root=Path(configured_root).expanduser(),
            max_vertices=int(overrides.get("max_vertices", defaults["max_vertices"])),
            timeout_seconds=float(
                overrides.get("timeout_seconds", defaults["timeout_seconds"])
            ),
            omega_sample_size=int(
                overrides.get("omega_sample_size", defaults["omega_sample_size"])
            ),
            seed=int(overrides.get("seed", defaults["seed"])),
            allow_compatibility_fallbacks=bool(
                overrides.get("allow_compatibility_fallbacks", True)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly configuration metadata."""
        return {
            "dblp_root": str(self.dblp_root),
            "max_vertices": self.max_vertices,
            "timeout_seconds": self.timeout_seconds,
            "omega_sample_size": self.omega_sample_size,
            "seed": self.seed,
            "allow_compatibility_fallbacks": self.allow_compatibility_fallbacks,
        }


@dataclass(frozen=True)
class SmokeReport:
    """Loaded source, configuration, and display-ready method rows."""

    config: SmokeConfig
    source: SmokeSource
    rows: tuple[dict[str, Any], ...]

    @property
    def graph(self) -> Any | None:
        return self.source.graph

    @property
    def ground_truth(self) -> list[list[int]]:
        return self.source.ground_truth

    def summary(self) -> dict[str, Any]:
        """Return provenance and size information for a notebook display."""
        summary = dict(self.source.report)
        summary["source"] = self.source.source
        summary["configuration"] = self.config.as_dict()
        window = summary.get("window")
        if isinstance(window, dict) and "selected_vertices" in window:
            selected = list(window["selected_vertices"])
            summary["window"] = {
                key: value
                for key, value in window.items()
                if key != "selected_vertices"
            }
            summary["window"]["selected_vertex_count"] = len(selected)
            summary["window"]["selected_vertex_preview"] = selected[:8]
        if self.graph is None:
            summary["graph"] = None
        else:
            summary["graph"] = {
                "vertices": int(self.graph.vcount()),
                "edges": int(self.graph.ecount()),
                "density": float(self.graph.density()),
            }
        summary["ground_truth_communities"] = len(self.ground_truth)
        summary["method_rows"] = len(self.rows)
        summary["status_counts"] = {
            status: sum(1 for row in self.rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in self.rows})
        }
        return summary

    def table(self):
        """Return a stable pandas table, or plain rows without pandas."""
        return smoke_table(self.rows)

    def figures(self):
        """Return metric and wall-clock figures for completed rows."""
        return smoke_figures(self.rows)

    def display_figures(self, *, close: bool = True):
        """Display figures in a notebook and optionally close their handles."""
        return display_smoke_figures(self, close=close)


def build_smoke_report(config: SmokeConfig | None = None) -> SmokeReport:
    """Load the configured source and execute the registered smoke adapters."""
    selected = config or SmokeConfig.from_environment()
    source = load_smoke_source(
        selected.dblp_root,
        max_vertices=selected.max_vertices,
    )
    rows = run_smoke_methods(
        source.graph,
        source.ground_truth,
        timeout_seconds=selected.timeout_seconds,
        seed=selected.seed,
        omega_sample_size=selected.omega_sample_size,
        metrics=METRICS,
        allow_compatibility_fallbacks=selected.allow_compatibility_fallbacks,
    )
    return SmokeReport(selected, source, tuple(rows))


def smoke_table(rows: Sequence[dict[str, Any]]):
    """Normalize smoke rows into the report's stable display columns."""
    ordered = order_smoke_rows(rows)
    try:
        import pandas as pd
    except ImportError:
        return ordered

    result = pd.DataFrame(ordered)
    for column in TABLE_COLUMNS:
        if column not in result:
            result[column] = None
    return result.loc[:, TABLE_COLUMNS]


def smoke_figures(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str] = METRICS,
) -> tuple[Any, ...]:
    """Create the report's metric heatmap and wall-clock bar chart.

    Figures are returned rather than displayed, so the same helper works in a
    Jupyter notebook, a QMD, or a script that saves them to disk.
    """
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return ()

    import matplotlib.pyplot as plt
    import numpy as np

    metric_names = tuple(metrics)
    quality_ordered = order_smoke_rows(completed, metrics=metric_names)
    labels = [str(row["method"]) for row in quality_ordered]
    metric_matrix = np.asarray(
        [
            [
                _finite_value(row, metric)
                if _finite_value(row, metric) is not None
                else float("nan")
                for metric in metric_names
            ]
            for row in quality_ordered
        ]
    )

    metric_figure, metric_axis = plt.subplots(
        figsize=(max(10, len(labels) * 0.65), 5.5)
    )
    image = metric_axis.imshow(
        metric_matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    metric_axis.set_xticks(
        range(len(metric_names)),
        [metric.replace("_", " ") for metric in metric_names],
        rotation=35,
        ha="right",
    )
    metric_axis.set_yticks(range(len(labels)), labels)
    metric_axis.set_title("Métricas de alinhamento na janela/fallback do smoke")
    metric_figure.colorbar(image, ax=metric_axis, label="score")
    metric_figure.tight_layout()

    wall_ordered = order_smoke_rows(
        completed,
        criterion="wall_seconds",
        metrics=metric_names,
    )
    wall_labels = [str(row["method"]) for row in wall_ordered]
    wall_figure, wall_axis = plt.subplots(figsize=(max(10, len(labels) * 0.65), 4.5))
    wall_axis.bar(
        wall_labels,
        [
            _finite_value(row, "wall_seconds")
            if _finite_value(row, "wall_seconds") is not None
            else float("nan")
            for row in wall_ordered
        ],
        color="#4472c4",
    )
    wall_axis.set_ylabel("segundos (wall-clock)")
    wall_axis.set_title("Tempo por método concluído")
    wall_axis.tick_params(axis="x", rotation=55)
    wall_figure.tight_layout()
    return metric_figure, wall_figure


def display_smoke_figures(
    report: SmokeReport,
    *,
    close: bool = True,
) -> tuple[Any, ...]:
    """Display a report's figures without relying on notebook auto-display.

    Returning the objects keeps the helper useful for callers that want to
    save them; ``close=True`` avoids a second inline rendering in QMD cells.
    """
    from IPython.display import display

    figures = report.figures()
    for figure in figures:
        display(figure)
    if close:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)
    return figures


__all__ = [
    "METRICS",
    "TABLE_COLUMNS",
    "SmokeConfig",
    "SmokeReport",
    "build_smoke_report",
    "display_smoke_figures",
    "order_smoke_rows",
    "smoke_figures",
    "smoke_table",
]
