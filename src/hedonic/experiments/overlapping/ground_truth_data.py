"""Explicit ground-truth data policies for the overlapping robustness study.

The existing SNAP loader is protocol-locked for the v1 paper benchmark.  This
module composes its public loader/projection functions and keeps the new
partial-cover policies isolated from that locked implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hedonic.experiments.overlapping.robustness import (
    canonicalize_cover,
    cover_to_vertex_memberships,
)
from hedonic.experiments.overlapping.snap import (
    SnapDataset,
    bounded_induced_dataset,
    common_undirected_analysis_dataset,
    content_identity,
    cover_statistics,
    load_snap_dataset,
    smoke_dataset,
)


CompletionPolicy = Literal["covered-induced", "singleton"]


@dataclass(frozen=True)
class PreparedDataset:
    """A dataset plus the explicit policy used to make its GT start valid."""

    dataset: SnapDataset
    policy: CompletionPolicy
    source_covered_vertices: tuple[int, ...]
    completion_count: int
    graph_identity: str
    ground_truth_identity: str


def covered_induced_dataset(dataset: SnapDataset) -> SnapDataset:
    """Restrict the graph to vertices present in the supplied cover.

    This is the primary exact-GT policy for partial SNAP covers: no synthetic
    memberships are added, and every retained graph vertex belongs to at least
    one supplied community.
    """
    cover = canonicalize_cover(dataset.cover, dataset.graph.vcount())
    covered = sorted({member for community in cover for member in community})
    if not covered:
        raise ValueError("ground-truth cover contains no in-graph vertices")
    old_to_new = {old: new for new, old in enumerate(covered)}
    graph = dataset.graph.induced_subgraph(covered)
    remapped = [
        [old_to_new[member] for member in community]
        for community in cover
    ]
    remapped = canonicalize_cover(remapped, graph.vcount())
    memberships = cover_to_vertex_memberships(remapped, graph.vcount())
    if any(not labels for labels in memberships):  # defensive invariant
        raise ValueError("covered-induced graph contains an uncovered vertex")
    report = dict(dataset.report)
    report.update(
        {
            "ground_truth_completion": {
                "policy": "covered-induced",
                "source_graph_vertices": dataset.graph.vcount(),
                "source_covered_vertices": len(covered),
                "retained_vertices": graph.vcount(),
                "synthetic_memberships_added": 0,
                "original_vertex_ids": covered,
            },
            "n": graph.vcount(),
            "m": graph.ecount(),
            "directed": graph.is_directed(),
            "id_mapping_strategy": report.get("id_mapping_strategy", "")
            + "+ground_truth_covered_induced",
        }
    )
    stats = cover_statistics(remapped)
    report["number_of_communities"] = stats["n_communities"]
    report["number_of_covered_nodes"] = stats["n_covered_nodes"]
    report["community_size_statistics"] = stats["community_size"]
    report["overlap_statistics"] = stats["overlap"]
    return SnapDataset(
        dataset.name,
        dataset.cover_variant,
        graph,
        remapped,
        report,
    )


def singleton_completed_dataset(dataset: SnapDataset) -> SnapDataset:
    """Complete a partial cover with one synthetic singleton per uncovered node."""
    cover = canonicalize_cover(dataset.cover, dataset.graph.vcount())
    covered = {member for community in cover for member in community}
    completed = list(cover)
    synthetic_vertices = [
        vertex for vertex in range(dataset.graph.vcount()) if vertex not in covered
    ]
    completed.extend([[vertex] for vertex in synthetic_vertices])
    completed = canonicalize_cover(completed, dataset.graph.vcount())
    memberships = cover_to_vertex_memberships(completed, dataset.graph.vcount())
    if any(not labels for labels in memberships):
        raise ValueError("singleton completion failed to cover every vertex")
    report = dict(dataset.report)
    report.update(
        {
            "ground_truth_completion": {
                "policy": "singleton",
                "source_graph_vertices": dataset.graph.vcount(),
                "source_covered_vertices": len(covered),
                "retained_vertices": dataset.graph.vcount(),
                "synthetic_memberships_added": len(synthetic_vertices),
                "synthetic_vertex_ids": synthetic_vertices,
            },
            "id_mapping_strategy": report.get("id_mapping_strategy", "")
            + "+ground_truth_uncovered_singletons",
        }
    )
    stats = cover_statistics(completed)
    report["number_of_communities"] = stats["n_communities"]
    report["number_of_covered_nodes"] = stats["n_covered_nodes"]
    report["community_size_statistics"] = stats["community_size"]
    report["overlap_statistics"] = stats["overlap"]
    return SnapDataset(
        dataset.name,
        dataset.cover_variant,
        dataset.graph,
        completed,
        report,
    )


def prepare_dataset(
    dataset: SnapDataset,
    *,
    policy: CompletionPolicy = "covered-induced",
    max_nodes: int | None = None,
) -> PreparedDataset:
    """Apply the common graph projection, optional bound, and GT policy."""
    if policy not in ("covered-induced", "singleton"):
        raise ValueError("policy must be 'covered-induced' or 'singleton'")
    # Node selection depends only on the canonical supplied cover.  Inducing a
    # selected vertex set commutes with direction collapse, loop removal, and
    # parallel-edge simplification, so bound first to avoid copying a full
    # LiveJournal graph merely to retain the reviewed 3,000-vertex subgraph.
    if max_nodes is not None and max_nodes > 0:
        projected = common_undirected_analysis_dataset(
            bounded_induced_dataset(dataset, max_nodes)
        )
    else:
        projected = common_undirected_analysis_dataset(dataset)
    source_covered = tuple(
        sorted({member for community in projected.cover for member in community})
    )
    prepared = (
        covered_induced_dataset(projected)
        if policy == "covered-induced"
        else singleton_completed_dataset(projected)
    )
    if prepared.graph.vcount() == 0:
        raise ValueError("prepared graph is empty")
    identity = content_identity(prepared.graph, prepared.cover)
    return PreparedDataset(
        dataset=prepared,
        policy=policy,
        source_covered_vertices=source_covered,
        completion_count=int(
            prepared.report.get("ground_truth_completion", {}).get(
                "synthetic_memberships_added", 0
            )
        ),
        graph_identity=str(identity["graph_sha256"]),
        ground_truth_identity=str(identity["ground_truth_cover_sha256"]),
    )


def load_prepared_dataset(
    name: str,
    *,
    cover_variant: str = "top5000",
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    policy: CompletionPolicy = "covered-induced",
    max_nodes: int | None = None,
    smoke: bool = False,
) -> PreparedDataset:
    """Load one real or built-in dataset using an explicit GT policy."""
    raw = (
        smoke_dataset(name, cover_variant=cover_variant)
        if smoke
        else load_snap_dataset(
            name,
            cover_variant=cover_variant,
            data_root=data_root,
            cache_dir=cache_dir,
        )
    )
    return prepare_dataset(raw, policy=policy, max_nodes=max_nodes)


__all__ = [
    "CompletionPolicy",
    "PreparedDataset",
    "covered_induced_dataset",
    "load_prepared_dataset",
    "prepare_dataset",
    "singleton_completed_dataset",
]
