"""Explicit ground-truth data policies for the overlapping robustness study.

The existing SNAP loader is protocol-locked for the v1 paper benchmark.  This
module composes its public loader/projection functions and keeps the new
partial-cover policies isolated from that locked implementation.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import igraph as ig
import numpy as np

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
from hedonic.experiments.overlapping import snap as _snap


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


# ------------------------------------------------------------------ covered-induced graph read as a stream
STREAM_COVERED_BYTES = 2 * 1024**3  # a raw edge file at least this large (compressed) is streamed, not materialised
STREAM_COVERED_CHUNK = 20_000_000  # edge rows parsed at a time
STREAM_COVERED_SCHEMA = 1


def stream_covered_wanted(name: str, data_root: str | Path | None = None, *, cover_variant: str = "top5000") -> bool:
    """Should this pair's covered-induced graph be read as a stream? Only a graph too large to materialise first
    (the raw edge file is at least ``STREAM_COVERED_BYTES`` compressed, e.g. Friendster) and without a trusted pickle:
    every network with a pickle or a smaller archive keeps the loader it always had, so no earlier record changes.
    ``HEDONIC_SNAP_STREAM_COVERED=1`` forces the stream (``0`` forbids it); a test uses it to compare the two loaders."""
    forced = os.environ.get("HEDONIC_SNAP_STREAM_COVERED", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    spec = _snap.SPECS.get(name.strip().lower())
    if spec is None:
        return False
    root = Path(data_root).expanduser() if data_root is not None else _snap.DEFAULT_NETWORKS_DIR
    base = _snap._dataset_dir(root, spec)
    if _snap._first_existing(base, spec.graph_files) is not None:
        return False
    edges = _snap._first_existing(base, spec.raw_edge_files)
    return edges is not None and edges.stat().st_size >= STREAM_COVERED_BYTES


def _covered_stream_cache_path(cache_root: Path, spec, cover_variant: str, edge_path: Path, cover_path: Path) -> Path:
    key = hashlib.sha256(_covered_stream_key(spec.name, cover_variant, edge_path, cover_path).encode()).hexdigest()[:16]
    return cache_root / f"{spec.name}-{cover_variant}-covered-stream-v{STREAM_COVERED_SCHEMA}-{key}.pkl"


def _covered_stream_key(name: str, cover_variant: str, edge_path: Path, cover_path: Path) -> str:
    parts = [name, cover_variant]
    for path in (edge_path, cover_path):
        info = path.stat()
        parts += [str(path.resolve()), str(info.st_size), str(int(info.st_mtime))]
    return "|".join(parts)


def load_covered_induced_streaming(name: str, *, cover_variant: str, data_root: str | Path | None = None,
                                   cache_dir: str | Path | None = None) -> SnapDataset:
    """The covered-induced dataset (``covered_induced_dataset`` of the full load) without building the full graph.

    The raw edge file is read in chunks and only the edges with both endpoints in the cover are kept, so a graph of 1.8
    billion edges costs the memory of its covered subgraph. The result is the same labelled graph the full load followed by
    ``covered_induced_dataset`` gives: vertices are the cover's members that are endpoints of some edge, numbered in the
    order of their original IDs (the raw loader's own order, ``raw_sorted_original_id_to_contiguous_index``), every induced
    edge is kept with its multiplicity, and the cover goes through the same remapping and canonicalisation. Only the order
    of the edge list can differ (``tests`` compare the sorted edge lists); neighbour order in igraph does not depend on it.
    The result is cached under the SNAP cache, keyed by the two source files' path, size and modification time.
    """
    import pandas as pd

    key = name.strip().lower()
    cover_variant = _snap.canonical_cover_variant(cover_variant)
    spec = _snap.SPECS[key]
    root = Path(data_root).expanduser() if data_root is not None else _snap.DEFAULT_NETWORKS_DIR
    base = _snap._dataset_dir(root, spec)
    edge_path = _snap._first_existing(base, spec.raw_edge_files)
    cover_path = _snap._first_existing(base, spec.raw_cover_files.get(cover_variant, ()))
    if edge_path is None or cover_path is None:
        raise _snap.SnapLoadError(f"no raw edge and cover files for {key}/{cover_variant} under {base}")
    cache_root = Path(cache_dir).expanduser() if cache_dir else _snap.default_cache_dir()
    cache_path = _covered_stream_cache_path(cache_root, spec, cover_variant, edge_path, cover_path)
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return SnapDataset(spec.name, cover_variant, payload["graph"], payload["cover"], payload["report"])

    raw_cover = _snap._read_cover_text(cover_path, wikipedia_categories=spec.name == "wikipedia")
    wanted = np.unique(np.fromiter((int(v) for community in raw_cover for v in community), dtype=np.int64))
    present = np.zeros(len(wanted), dtype=bool)
    kept: list[np.ndarray] = []
    rows = 0
    reader = pd.read_csv(edge_path, sep=r"\s+", comment="#", header=None, usecols=[0, 1], dtype=np.int64,
                         chunksize=STREAM_COVERED_CHUNK)
    for chunk in reader:
        source, target = chunk[0].to_numpy(), chunk[1].to_numpy()
        rows += len(source)
        hits = []
        for ends in (source, target):
            position = np.searchsorted(wanted, ends)
            position[position == len(wanted)] = 0
            hit = wanted[position] == ends
            present[position[hit]] = True
            hits.append((position, hit))
        both = hits[0][1] & hits[1][1]
        if both.any():
            kept.append(np.stack([source[both], target[both]], axis=1))
    present_ids = wanted[present]
    edges = np.concatenate(kept) if kept else np.zeros((0, 2), dtype=np.int64)
    edges = np.searchsorted(present_ids, edges)
    graph = ig.Graph(n=len(present_ids), edges=[(int(a), int(b)) for a, b in edges], directed=spec.directed)
    id_to_index = {int(old): index for index, old in enumerate(present_ids)}
    cover, validation = _snap._remap_cover(raw_cover, id_to_index)
    validation["cover"] = cover
    report = _snap._make_report(
        spec=spec, cover_variant=cover_variant, graph=graph, graph_path=str(edge_path.relative_to(base)),
        cover_path=str(cover_path.relative_to(base)),
        mapping_strategy="raw_sorted_original_id_to_contiguous_index+streamed_covered_subgraph",
        remap=validation, source_kind="streamed_raw_gzip_covered_only")
    report["stream_edge_rows_read"] = rows
    dataset = covered_induced_dataset(SnapDataset(spec.name, cover_variant, graph, cover, report))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"graph": dataset.graph, "cover": dataset.cover, "report": dataset.report}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache_path)
    return dataset


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
