"""Detector orchestration for the executable overlapping-method smoke flow.

The report-facing composition lives in
``hedonic.experiments.overlapping.smoke_helpers``.  This module keeps the
source loading, normalization, scoring, timeout, and timed adapter execution
primitives reusable from a QMD, a script, a notebook, or a test.

This module is experiment-only.  It does not add optional dependencies to the
core :class:`hedonic.Game` API and it never substitutes one detector for an
unavailable optional detector.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import time
from typing import Any, Callable, Sequence

from hedonic.experiments.overlapping.metrics import (
    GAME_EVALUATION_METRICS,
    cover_to_memberships,
    score_cover,
)


# Backward-compatible name used by the original smoke tutorial and tests.
METRICS = GAME_EVALUATION_METRICS


@dataclass(frozen=True)
class SmokeSource:
    """Graph, metadata cover, and provenance for one smoke input."""

    graph: Any | None
    ground_truth: list[list[int]]
    source: str
    report: dict[str, Any]


class SmokeTimeout(TimeoutError):
    """Raised when an adapter exceeds its smoke wall-clock budget."""


def clean_cover(cover: Sequence[Sequence[int]], n_vertices: int) -> list[list[int]]:
    """Canonicalize a cover without depending on optional libraries."""
    seen: set[tuple[int, ...]] = set()
    output: list[list[int]] = []
    for community in cover:
        members = tuple(
            sorted(
                {
                    int(vertex)
                    for vertex in community
                    if 0 <= int(vertex) < int(n_vertices)
                }
            )
        )
        if len(members) < 2 or members in seen:
            continue
        seen.add(members)
        output.append(list(members))
    return output


def deterministic_dblp_window(
    graph: Any,
    cover: Sequence[Sequence[int]],
    max_vertices: int = 80,
) -> tuple[Any, list[list[int]], dict[str, Any]]:
    """Induce a deterministic, edge-bearing DBLP window.

    Metadata communities are not guaranteed to contain graph edges.  The old
    selector therefore sometimes returned a perfectly valid metadata window
    with ``ecount() == 0``; every topology-based detector then failed before
    producing a smoke result.  Keep the metadata anchor deterministic, but add
    the smallest incident graph neighbours needed to make the induced window
    executable.  The added vertices remain ordinary graph-only input and are
    never added to the ground-truth cover unless they are already members of a
    metadata community.
    """
    canonical = clean_cover(cover, graph.vcount())
    if not canonical:
        raise ValueError("DBLP ground truth has no community with at least two vertices")
    incidence = [[] for _ in range(graph.vcount())]
    for community_id, community in enumerate(canonical):
        for vertex in community:
            incidence[vertex].append(community_id)
    # Selecting the minimum union for every overlapping vertex is needlessly
    # expensive on full DBLP: each repeated set union can touch thousands of
    # metadata memberships.  The number of incident labels and vertex id give
    # a deterministic, cheap anchor while we materialize only its union.
    overlap_anchors = [
        (len(labels), vertex, labels)
        for vertex, labels in enumerate(incidence)
        if len(labels) >= 2
    ]
    if overlap_anchors:
        _label_count, anchor, labels = min(overlap_anchors, key=lambda item: (item[0], item[1]))
        selected = sorted({vertex for label in labels for vertex in canonical[label]})
    else:
        anchor = min(range(graph.vcount()), key=lambda vertex: (len(incidence[vertex]), vertex))
        selected = sorted(canonical[0])

    cap = max(2, int(max_vertices))
    metadata_seed_vertices = set(selected)
    selected = sorted(metadata_seed_vertices)[:cap]
    if anchor not in selected:
        if len(selected) < cap:
            selected.append(anchor)
        elif selected:
            selected[-1] = anchor
        else:  # pragma: no cover - canonical has at least one vertex
            selected = [anchor]
        selected = sorted(set(selected))

    def induced_edge_count(vertices: set[int]) -> int:
        return sum(
            1
            for left, right in graph.get_edgelist()
            if int(left) in vertices and int(right) in vertices
        )

    # A metadata-only union is often tiny and disconnected in the actual
    # co-authorship graph.  Add the first deterministic incident neighbour,
    # preferring an edge whose other endpoint is already selected.  If the
    # selected set is full, replace a non-anchor vertex rather than silently
    # returning an edgeless detector input.
    selected_set = set(selected)
    if induced_edge_count(selected_set) == 0 and graph.ecount() > 0:
        edge_candidates = sorted(
            (int(left), int(right))
            for left, right in graph.get_edgelist()
            if int(left) != int(right)
        )
        incident: list[tuple[int, int]] = []
        for left, right in edge_candidates:
            if left in selected_set and right not in selected_set:
                incident.append((left, right))
            elif right in selected_set and left not in selected_set:
                incident.append((right, left))
        if incident:
            _inside, outside = incident[0]
            if len(selected_set) < cap:
                selected_set.add(outside)
            elif outside != anchor:
                replace = next((vertex for vertex in sorted(selected_set) if vertex != anchor), None)
                if replace is not None:
                    selected_set.remove(replace)
                    selected_set.add(outside)
        elif len(edge_candidates) > 0:
            # The metadata window is isolated from the rest of the graph.
            # Use the first graph edge and retain the anchor when possible.
            left, right = edge_candidates[0]
            if len(selected_set) + 2 <= cap:
                selected_set.update((left, right))
            elif len(selected_set) < cap:
                selected_set.add(left)
            elif anchor not in {left, right}:
                replace = next((vertex for vertex in sorted(selected_set) if vertex != anchor), None)
                if replace is not None:
                    selected_set.remove(replace)
                    selected_set.add(left)

    selected = sorted(selected_set)[:cap]
    selected_set = set(selected)
    old_to_new = {old: new for new, old in enumerate(selected)}
    subgraph = graph.induced_subgraph(selected)
    subcover = [
        sorted(old_to_new[v] for v in community if v in selected_set)
        for community in canonical
    ]
    subcover = clean_cover(subcover, subgraph.vcount())
    if not subcover:
        raise ValueError("deterministic DBLP window has no nontrivial GT community")
    return subgraph, subcover, {
        "anchor": anchor,
        "selected_vertices": selected,
        "edge_bearing": bool(subgraph.ecount()),
        "added_graph_vertices": sorted(set(selected) - metadata_seed_vertices),
    }


def fallback_fixture() -> tuple[Any, list[list[int]]]:
    """Return the documented eight-vertex overlap fixture when DBLP is absent."""
    import igraph as ig

    graph = ig.Graph(
        n=8,
        edges=[
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
            (4, 5), (5, 3), (1, 6), (6, 7), (7, 1),
        ],
        directed=False,
    )
    return graph, [[0, 1, 2], [1, 6, 7], [3, 4, 5]]


def load_smoke_source(
    dblp_root: str | Path,
    *,
    max_vertices: int = 80,
) -> SmokeSource:
    """Load a deterministic DBLP window, or an explicit synthetic fallback."""
    root = Path(dblp_root).expanduser()
    try:
        from hedonic.experiments.overlapping.dblp_full import load_dblp

        full_graph, full_cover, _node_map = load_dblp(root)
        graph, ground_truth, window = deterministic_dblp_window(
            full_graph, full_cover, max_vertices=max_vertices
        )
        if graph.ecount() == 0:
            raise ValueError("deterministic DBLP smoke window has no graph edges")
        report = {
            "source": "DBLP subgrafo determinístico",
            "db_path": str(root),
            "full_vertices": full_graph.vcount(),
            "full_edges": full_graph.ecount(),
            "window": window,
        }
        return SmokeSource(graph, ground_truth, report["source"], report)
    except Exception as exc:
        try:
            graph, ground_truth = fallback_fixture()
            source = "fallback sintético (DBLP indisponível)"
            report = {"source": source, "db_path": str(root), "reason": f"{type(exc).__name__}: {exc}"}
            return SmokeSource(graph, ground_truth, source, report)
        except Exception as fallback_exc:
            source = "indisponível (DBLP e fallback não executáveis)"
            report = {
                "source": source,
                "db_path": str(root),
                "reason": f"{type(exc).__name__}: {exc}; fallback: {type(fallback_exc).__name__}: {fallback_exc}",
            }
            return SmokeSource(None, [], source, report)


def _short_cover(cover: Sequence[Sequence[int]], limit: int = 10) -> str:
    bodies = [list(map(int, community)) for community in cover]
    if len(bodies) <= limit:
        return str(bodies)
    return str(bodies[:limit])[:-1] + f", … ({len(bodies)} comunidades)]"


def run_with_timeout(function: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run one adapter in-process with a Unix wall-clock guard."""
    if timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        return function()

    def handler(_signum: int, _frame: Any) -> None:
        raise SmokeTimeout(f"timeout after {timeout_seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_row(
    display_name: str,
    runner: Callable[[], tuple[Sequence[Sequence[int]], dict[str, Any]]],
    *,
    graph: Any,
    ground_truth: Sequence[Sequence[int]],
    timeout_seconds: float,
    metrics: Sequence[str],
    omega_sample_size: int,
    seed: int,
    unavailable_errors: tuple[type[BaseException], ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "method": display_name,
        "status": "error",
        "wall_seconds": None,
        "coverage": None,
        "n_communities": None,
        "failure_or_limitation": None,
    }
    try:
        cover, metadata = run_with_timeout(runner, timeout_seconds)
        row.update(
            {
                "status": "completed",
                "coverage": _short_cover(cover),
                "n_communities": len(cover),
                "implementation": metadata.get("implementation", metadata.get("method", "")),
                "implementation_kind": metadata.get("implementation_kind"),
                **score_cover(
                    graph,
                    ground_truth,
                    cover,
                    metrics=metrics,
                    omega_sample_size=omega_sample_size,
                    seed=seed,
                ),
            }
        )
        if metadata.get("provider_error"):
            row["failure_or_limitation"] = (
                "compatibility smoke fallback; "
                + str(metadata["provider_error"])
            )
        elif metadata.get("failure_or_limitation"):
            row["failure_or_limitation"] = str(metadata["failure_or_limitation"])
    except unavailable_errors as exc:
        row["status"] = "unavailable"
        row["failure_or_limitation"] = str(exc)
    except SmokeTimeout as exc:
        row["status"] = "timeout"
        row["failure_or_limitation"] = str(exc)
    except Exception as exc:  # report detector failures rather than hiding them
        row["failure_or_limitation"] = f"{type(exc).__name__}: {exc}"
    row["wall_seconds"] = time.perf_counter() - started
    return row


def _compatibility_smoke_fallback(
    display_name: str,
    graph: Any,
    error: BaseException,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return an explicit graph-only fallback for the rendered smoke table.

    The historical AGMfit/MMSB/Infomap implementations do not have stable
    providers in this checkout, and Link Communities may be unavailable when
    its isolated CDlib environment is not ready.  The reusable method
    adapters correctly fail closed in that situation.  The tutorial can opt
    into this *presentation-only* fallback so its table exercises the common
    normalization/scoring path for every registered row.  The metadata keeps
    the missing provider and original exception visible; this is never used by
    the benchmark or by :func:`run_method` itself.
    """
    try:
        components = [
            sorted(map(int, component))
            for component in graph.components()
            if len(component) >= 2
        ]
    except (AttributeError, TypeError):
        components = []
    if not components and int(graph.vcount()) >= 2:
        components = [list(range(int(graph.vcount())))]
    if not components:
        raise RuntimeError(
            f"{display_name} has no valid compatibility smoke cover"
        ) from error
    method_name = display_name.removeprefix("methods.")
    return components, {
        "method": method_name,
        "family": "compatibility smoke fallback",
        "implementation": f"{method_name} compatibility smoke fallback",
        "implementation_kind": "compatibility_smoke_fallback",
        "provider_status": "unavailable",
        "provider_error": f"{type(error).__name__}: {error}",
        "metadata_free": True,
        "information_budget": "graph topology only; historical provider was not available",
    }


def run_smoke_methods(
    graph: Any,
    ground_truth: Sequence[Sequence[int]],
    *,
    timeout_seconds: float = 30.0,
    seed: int = 7,
    omega_sample_size: int = 10_000,
    metrics: Sequence[str] = METRICS,
    allow_compatibility_fallbacks: bool = False,
) -> list[dict[str, Any]]:
    """Run all registered adapters and return display-ready result rows.

    ``allow_compatibility_fallbacks`` is intentionally opt-in.  When enabled,
    only the rendered smoke report may turn a missing optional provider into a
    completed *compatibility* row; the provider error and fallback identity
    remain in the row metadata.  The default preserves fail-closed behavior.
    """
    if graph is None:
        return []
    try:
        from hedonic.experiments.overlapping.baselines import (
            BASELINES,
            MethodUnavailable as BaselineUnavailable,
            run_baseline,
        )
        from hedonic.experiments.overlapping.methods import (
            METHODS,
            MethodUnavailable,
            run_method,
        )
    except Exception as exc:
        return [{
            "method": "all adapters",
            "status": "unavailable",
            "wall_seconds": 0.0,
            "coverage": None,
            "n_communities": None,
            "failure_or_limitation": f"{type(exc).__name__}: {exc}",
        }]

    rows: list[dict[str, Any]] = []
    resolution = float(graph.density())
    cap = max(2, min(3, max((len(c) for c in ground_truth), default=2)))
    unavailable = (MethodUnavailable, BaselineUnavailable)
    for name, adapter in METHODS.items():
        effective_resolution = resolution
        multiplier = adapter.density_resolution_multiplier
        if multiplier is not None:
            effective_resolution = min(resolution * multiplier, 1.0)

        def run_method_for_smoke(
            adapter=adapter,
            effective_resolution=effective_resolution,
            display_name=f"methods.{name}",
        ):
            try:
                return run_method(
                    adapter,
                    graph,
                    max_memberships=cap,
                    resolution=effective_resolution,
                    seed=seed,
                )
            except unavailable as exc:
                if not allow_compatibility_fallbacks:
                    raise
                return _compatibility_smoke_fallback(display_name, graph, exc)
            except RuntimeError as exc:
                # Sparse windows can legitimately contain no clique or no
                # DEMON body.  Keep genuine adapter errors visible, but make
                # the rendered compatibility report total when the detector
                # only failed because its normalized cover was empty.
                if (
                    allow_compatibility_fallbacks
                    and "produced no valid communities" in str(exc)
                ):
                    return _compatibility_smoke_fallback(display_name, graph, exc)
                raise

        rows.append(
            _run_row(
                f"methods.{name}",
                run_method_for_smoke,
                graph=graph,
                ground_truth=ground_truth,
                timeout_seconds=timeout_seconds,
                metrics=metrics,
                omega_sample_size=omega_sample_size,
                seed=seed,
                unavailable_errors=unavailable,
            )
        )
    for name in BASELINES:
        rows.append(
            _run_row(
                f"baselines.{name}",
                lambda name=name: run_baseline(
                    name,
                    graph,
                    cap=cap,
                    resolution=resolution,
                    seed=seed,
                    allow_replicas=True,
                ),
                graph=graph,
                ground_truth=ground_truth,
                timeout_seconds=timeout_seconds,
                metrics=metrics,
                omega_sample_size=omega_sample_size,
                seed=seed,
                unavailable_errors=unavailable,
            )
        )
    return rows


def default_smoke_config() -> dict[str, Any]:
    """Return environment-driven defaults shared by executable tutorials."""
    return {
        "seed": 7,
        "max_vertices": int(os.getenv("HEDONIC_SMOKE_MAX_VERTICES", "80")),
        "timeout_seconds": float(os.getenv("HEDONIC_SMOKE_TIMEOUT", "30")),
        "omega_sample_size": int(os.getenv("HEDONIC_SMOKE_OMEGA_SAMPLES", "10000")),
    }


__all__ = [
    "METRICS",
    "SmokeSource",
    "SmokeTimeout",
    "clean_cover",
    "cover_to_memberships",
    "default_smoke_config",
    "deterministic_dblp_window",
    "fallback_fixture",
    "load_smoke_source",
    "run_smoke_methods",
    "run_with_timeout",
    "score_cover",
]
