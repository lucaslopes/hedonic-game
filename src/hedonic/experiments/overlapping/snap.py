"""Portable, ID-safe loaders for the overlapping SNAP benchmark datasets.

The saved SNAP archives use original (often sparse) numeric identifiers while
igraph works with contiguous vertex indices.  This module is deliberately the
only place that knows the archive layout: experiments consume
:class:`SnapDataset`, whose cover is already normalized to igraph indices.

Input archives are read-only.  A validated normalized cache is written below
the repository artifact root by default, never next to the archived SNAP
files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import random
import statistics
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import igraph as ig

from hedonic.experiments.config import NETWORKS_DIR, SNAP_CACHE_DIR, expand_path

# Alias retained as part of this loader's public surface; configuration owns
# the portable default and environment/TOML precedence.
DEFAULT_NETWORKS_DIR = NETWORKS_DIR
CACHE_SCHEMA_VERSION = 5
ANALYSIS_GRAPH_POLICY = "common_undirected_simple_v1"
CONTENT_IDENTITY_SCHEMA_VERSION = 2

# ``mapequation-networks`` is an optional, lazy data catalogue.  It is not
# imported at module import time and is never installed as a core dependency.
# The catalogue uses the official SNAP names below and caches downloads outside
# the hedonic wheel.  Local archives still take precedence over this fallback.
MAPEQUATION_SNAP_NAMES: dict[str, str] = {
    "amazon": "com-Amazon",
    "youtube": "com-Youtube",
    "dblp": "com-DBLP",
    "livejournal": "com-LiveJournal",
    "orkut": "com-Orkut",
    "friendster": "com-Friendster",
    "wikipedia": "wiki-topcats",
}

# Official SNAP file endpoints.  The optional ``mapequation-networks`` package
# can provide the same catalogue, but the direct HTTP fallback keeps the
# reproduction extra self-contained and available even when that package is
# not published on the active Python index.
SNAP_DOWNLOAD_URLS: dict[str, dict[str, str]] = {
    "amazon": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.dedup.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz",
    },
    "youtube": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz",
    },
    "dblp": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz",
    },
    "livejournal": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-lj.all.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-lj.top5000.cmty.txt.gz",
    },
    "orkut": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.all.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.top5000.cmty.txt.gz",
    },
    "friendster": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/com-friendster.all.cmty.txt.gz",
        "top5000": "https://snap.stanford.edu/data/bigdata/communities/com-friendster.top5000.cmty.txt.gz",
    },
    "wikipedia": {
        "graph": "https://snap.stanford.edu/data/bigdata/communities/wiki-topcats.txt.gz",
        "all": "https://snap.stanford.edu/data/bigdata/communities/wiki-topcats-categories.txt.gz",
    },
}


class SnapLoadError(RuntimeError):
    """The requested SNAP dataset could not be loaded safely."""


class UnsupportedCoverVariant(SnapLoadError):
    """A dataset has no supplied cover for the requested variant."""


@dataclass(frozen=True)
class SnapDatasetSpec:
    """Static archive information for one supported overlapping benchmark."""

    name: str
    directory: str
    directed: bool
    ground_truth_type: str
    graph_files: tuple[str, ...]
    raw_edge_files: tuple[str, ...]
    cover_files: dict[str, tuple[str, ...]]
    raw_cover_files: dict[str, tuple[str, ...]]


SPECS: dict[str, SnapDatasetSpec] = {
    "amazon": SnapDatasetSpec(
        name="amazon",
        directory="Amazon",
        directed=False,
        ground_truth_type="overlapping_product_community_cover",
        graph_files=("com-amazon.ungraph.pkl",),
        raw_edge_files=("com-amazon.ungraph.txt.gz",),
        cover_files={
            "all": ("com-amazon.all.dedup.cmty.pkl", "com-amazon.all.cmty.pkl"),
            "top5000": ("com-amazon.top5000.cmty.pkl", "top5000.cmty.pkl"),
        },
        raw_cover_files={
            "all": (
                "com-amazon.all.dedup.cmty.txt.gz",
                "com-amazon.all.cmty.txt.gz",
            ),
            "top5000": ("com-amazon.top5000.cmty.txt.gz", "top5000.cmty.txt.gz"),
        },
    ),
    "dblp": SnapDatasetSpec(
        name="dblp",
        directory="DBLP",
        directed=False,
        ground_truth_type="overlapping_coauthorship_community_cover",
        graph_files=("pkl/com-dblp.ungraph.pkl", "com-dblp.ungraph.pkl"),
        raw_edge_files=("raw/com-dblp.ungraph.txt.gz", "com-dblp.ungraph.txt.gz"),
        cover_files={
            "all": ("pkl/com-dblp.all.cmty.pkl", "com-dblp.all.cmty.pkl"),
            "top5000": (
                "pkl/com-dblp.top5000.cmty.pkl",
                "pkl/top5000.cmty.pkl",
                "com-dblp.top5000.cmty.pkl",
                "top5000.cmty.pkl",
            ),
        },
        raw_cover_files={
            "all": (
                "raw/com-dblp.all.cmty.txt.gz",
                "com-dblp.all.cmty.txt.gz",
            ),
            "top5000": (
                "raw/com-dblp.top5000.cmty.txt.gz",
                "raw/top5000.cmty.txt.gz",
                "com-dblp.top5000.cmty.txt.gz",
                "top5000.cmty.txt.gz",
            ),
        },
    ),
    "livejournal": SnapDatasetSpec(
        name="livejournal",
        directory="LiveJournal",
        directed=False,
        ground_truth_type="overlapping_social_community_cover",
        graph_files=("com-lj.ungraph.pkl",),
        raw_edge_files=("com-lj.ungraph.txt.gz",),
        cover_files={
            "all": ("com-lj.all.cmty.pkl",),
            "top5000": ("com-lj.top5000.cmty.pkl", "top5000.cmty.pkl"),
        },
        raw_cover_files={
            "all": ("com-lj.all.cmty.txt.gz",),
            "top5000": ("com-lj.top5000.cmty.txt.gz", "top5000.cmty.txt.gz"),
        },
    ),
    "youtube": SnapDatasetSpec(
        name="youtube",
        directory="Youtube",
        directed=False,
        ground_truth_type="overlapping_channel_community_cover",
        graph_files=("com-youtube.ungraph.pkl",),
        raw_edge_files=("com-youtube.ungraph.txt.gz",),
        cover_files={
            "all": ("com-youtube.all.cmty.pkl",),
            "top5000": ("com-youtube.top5000.cmty.pkl", "top5000.cmty.pkl"),
        },
        raw_cover_files={
            "all": ("com-youtube.all.cmty.txt.gz",),
            "top5000": ("com-youtube.top5000.cmty.txt.gz", "top5000.cmty.txt.gz"),
        },
    ),
    "friendster": SnapDatasetSpec(
        name="friendster",
        directory="Friendster",
        directed=False,
        ground_truth_type="overlapping_social_community_cover",
        graph_files=("com-friendster.ungraph.pkl",),
        raw_edge_files=("com-friendster.ungraph.txt.gz",),
        cover_files={
            "all": ("com-friendster.all.cmty.pkl",),
            "top5000": ("com-friendster.top5000.cmty.pkl", "top5000.cmty.pkl"),
        },
        raw_cover_files={
            "all": ("com-friendster.all.cmty.txt.gz",),
            "top5000": ("com-friendster.top5000.cmty.txt.gz", "top5000.cmty.txt.gz"),
        },
    ),
    "orkut": SnapDatasetSpec(
        name="orkut",
        directory="Orkut",
        directed=False,
        ground_truth_type="overlapping_social_community_cover",
        graph_files=("com-orkut.ungraph.pkl",),
        raw_edge_files=("com-orkut.ungraph.txt.gz",),
        cover_files={
            "all": ("com-orkut.all.cmty.pkl",),
            "top5000": ("com-orkut.top5000.cmty.pkl", "top5000.cmty.pkl"),
        },
        raw_cover_files={
            "all": ("com-orkut.all.cmty.txt.gz",),
            "top5000": ("com-orkut.top5000.cmty.txt.gz", "top5000.cmty.txt.gz"),
        },
    ),
    "wikipedia": SnapDatasetSpec(
        name="wikipedia",
        directory="Wikipedia",
        directed=True,
        ground_truth_type="overlapping_wikipedia_category_cover",
        graph_files=("wiki-topcats.pkl",),
        raw_edge_files=("wiki-topcats.txt.gz",),
        cover_files={"all": ("wiki-topcats-categories.pkl",)},
        raw_cover_files={"all": ("wiki-topcats-categories.txt.gz",)},
    ),
}


@dataclass
class SnapDataset:
    """A graph and cover whose members are contiguous igraph indices."""

    name: str
    cover_variant: str
    graph: ig.Graph
    cover: list[list[int]]
    report: dict[str, Any]


def network_names() -> tuple[str, ...]:
    """Names in the historical five-dataset benchmark, stable display order."""
    return ("amazon", "dblp", "livejournal", "youtube", "wikipedia")


def default_cache_dir() -> Path:
    """Return the repository-local normalized-data cache location."""
    configured = os.getenv("HEDONIC_SNAP_CACHE_DIR")
    return expand_path(configured) if configured else SNAP_CACHE_DIR


def _dataset_dir(root: Path, spec: SnapDatasetSpec) -> Path:
    """Accept the documented directory plus common archive spelling variants."""
    candidates = [root / spec.directory]
    if spec.name == "youtube":
        candidates.extend((root / "YouTube", root / "youtube"))
    if spec.name == "livejournal":
        candidates.extend((root / "LiveJournal", root / "livejournal"))
    if spec.name == "wikipedia":
        candidates.extend((root / "Wikipedia", root / "wiki-topcats"))
    return next((p for p in candidates if p.exists()), candidates[0])


def _first_existing(base: Path, names: Iterable[str]) -> Path | None:
    return next((base / name for name in names if (base / name).is_file()), None)


class _LegacyGraphUnpickler(pickle.Unpickler):
    """Load trusted local caches serialized before ``Game.py`` was renamed."""

    def find_class(self, module: str, name: str):  # noqa: D401 - pickle protocol
        if module == "hedonic.game" and name in {"HedonicGame", "Game"}:
            return ig.Graph
        return super().find_class(module, name)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return _LegacyGraphUnpickler(f).load()


def _read_edge_list(path: Path) -> tuple[list[tuple[int, int]], set[int]]:
    """Stream a SNAP gzip edge list, retaining only graph construction data."""
    edges: list[tuple[int, int]] = []
    node_ids: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise SnapLoadError(f"Malformed edge line {line_number} in {path}")
            try:
                source, target = int(fields[0]), int(fields[1])
            except ValueError as exc:
                raise SnapLoadError(
                    f"Non-integer edge line {line_number} in {path}"
                ) from exc
            edges.append((source, target))
            node_ids.add(source)
            node_ids.add(target)
    return edges, node_ids


def _read_cover_text(path: Path, *, wikipedia_categories: bool) -> list[list[int]]:
    """Stream a SNAP community/category gzip file into original-ID members."""
    cover: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(";", 1)[1].split() if wikipedia_categories else line.split()
            try:
                cover.append([int(member) for member in fields])
            except ValueError as exc:
                raise SnapLoadError(
                    f"Non-integer community member on line {line_number} in {path}"
                ) from exc
    return cover


def _coerce_cover(value: Any, path: Path) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SnapLoadError(f"Cover pickle is not a sequence: {path}")
    result: list[list[int]] = []
    for index, community in enumerate(value):
        if not isinstance(community, Sequence) or isinstance(community, (str, bytes)):
            raise SnapLoadError(f"Community {index} is not a member sequence: {path}")
        try:
            result.append([int(member) for member in community])
        except (TypeError, ValueError) as exc:
            raise SnapLoadError(f"Invalid member in community {index}: {path}") from exc
    return result


def _mapping_for_cached_graph(graph: ig.Graph) -> tuple[dict[int, int], str]:
    labels = graph.vs["label"] if "label" in graph.vertex_attributes() else None
    if labels is not None and len(labels) == graph.vcount():
        try:
            mapping = {int(label): index for index, label in enumerate(labels)}
        except (TypeError, ValueError) as exc:
            raise SnapLoadError("Cached graph has non-integer vertex labels") from exc
        if len(mapping) != graph.vcount():
            raise SnapLoadError("Cached graph vertex labels are not unique")
        return mapping, "cached_vertex_label_attribute"
    return {index: index for index in range(graph.vcount())}, "cached_identity_vertex_index"


def _remap_cover(
    raw_cover: Sequence[Sequence[int]],
    id_to_index: dict[int, int],
    *,
    canonicalize_communities: bool = False,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Map IDs while retaining labelled-cover provenance until bounding.

    Duplicate labels in the supplied archive are counted and retained here so
    their provenance is not lost.  The bounded selector canonicalizes this
    labelled multicover before ranking vertices, then canonicalizes once more
    after projection so both source duplicates and projection collisions are
    removed from the registered set-cover semantics.
    """
    remapped: list[list[int]] = []
    missing_member_count = 0
    missing_ids: set[int] = set()
    duplicate_member_count = 0
    dropped_communities = 0
    for community in raw_cover:
        members: list[int] = []
        seen: set[int] = set()
        for old_id in community:
            new_id = id_to_index.get(int(old_id))
            if new_id is None:
                missing_member_count += 1
                if len(missing_ids) < 20:
                    missing_ids.add(int(old_id))
                continue
            if new_id in seen:
                duplicate_member_count += 1
                continue
            seen.add(new_id)
            members.append(new_id)
        if len(members) >= 2:
            remapped.append(members)
        else:
            dropped_communities += 1
    canonical, canonicalization = canonicalize_cover(
        remapped, n_vertices=len(id_to_index), minimum_size=2
    )
    normalized = canonical if canonicalize_communities else remapped
    return normalized, {
        "raw_community_count": len(raw_cover),
        "dropped_communities_lt_2_members": dropped_communities,
        "missing_member_count": missing_member_count,
        "missing_id_examples": sorted(missing_ids),
        "duplicate_members_removed": duplicate_member_count,
        "duplicate_communities_observed": canonicalization[
            "duplicate_communities_removed"
        ],
        "duplicate_communities_removed": (
            canonicalization["duplicate_communities_removed"]
            if canonicalize_communities
            else 0
        ),
        "labeled_community_count": len(remapped),
        "canonical_community_count": len(canonical),
        "community_semantics": (
            "canonical_set_cover"
            if canonicalize_communities
            else "labeled_multicover_provenance"
        ),
    }


def canonicalize_cover(
    cover: Sequence[Sequence[int]],
    *,
    n_vertices: int | None = None,
    minimum_size: int = 1,
) -> tuple[list[list[int]], dict[str, int]]:
    """Return a label-invariant cover made of unique sorted vertex sets.

    This is the experiment-wide serialization boundary for supplied and
    predicted covers.  Community labels and detector iteration order are not
    scientific content, while duplicate members or duplicate vertex sets
    would otherwise change metric denominators.
    """
    if minimum_size < 1:
        raise ValueError("minimum_size must be >= 1")
    communities: set[tuple[int, ...]] = set()
    duplicate_members = 0
    duplicate_communities = 0
    dropped_small = 0
    for community in cover:
        members: list[int] = []
        seen: set[int] = set()
        for member in community:
            vertex = int(member)
            if n_vertices is not None and not 0 <= vertex < n_vertices:
                raise ValueError("cover contains a vertex outside the graph")
            if vertex in seen:
                duplicate_members += 1
                continue
            seen.add(vertex)
            members.append(vertex)
        key = tuple(sorted(members))
        if len(key) < minimum_size:
            dropped_small += 1
            continue
        if key in communities:
            duplicate_communities += 1
            continue
        communities.add(key)
    canonical = [list(members) for members in sorted(communities)]
    return canonical, {
        "input_community_count": len(cover),
        "canonical_community_count": len(canonical),
        "duplicate_members_removed": duplicate_members,
        "duplicate_communities_removed": duplicate_communities,
        "communities_below_minimum_size_dropped": dropped_small,
    }


def cover_sha256(cover: Sequence[Sequence[int]]) -> str:
    """Hash an already canonicalized cover using a portable JSON encoding."""
    payload = json.dumps(cover, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_sha256(graph: ig.Graph) -> str:
    """Hash graph topology in a deterministic vertex/edge representation."""
    directed = bool(graph.is_directed())
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "directed": directed,
                "n": int(graph.vcount()),
                "m": int(graph.ecount()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    # Source-major sorted adjacency is a canonical edge stream and avoids
    # materializing/sorting tens of millions of edge tuples for LiveJournal.
    mode = "out" if directed else "all"
    for source in range(graph.vcount()):
        for target in sorted(int(value) for value in graph.neighbors(source, mode=mode)):
            if directed or source <= target:
                digest.update(f"{source},{target};".encode())
    return digest.hexdigest()


def _ordered_cover_sha256(cover: Sequence[Sequence[int]]) -> str:
    payload = [[int(member) for member in community] for community in cover]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _cache_content_identity(
    graph: ig.Graph, cover: Sequence[Sequence[int]]
) -> dict[str, Any]:
    return {
        "graph_sha256": graph_sha256(graph),
        "labeled_cover_sha256": _ordered_cover_sha256(cover),
        "n": graph.vcount(),
        "m": graph.ecount(),
        "directed": graph.is_directed(),
        "labeled_community_count": len(cover),
    }


def content_identity(graph: ig.Graph, cover: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Return exact topology and canonical-ground-truth identities."""
    canonical, _ = canonicalize_cover(
        cover, n_vertices=graph.vcount(), minimum_size=1
    )
    memberships = Counter(
        vertex for community in canonical for vertex in community
    )
    return {
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "graph_sha256": graph_sha256(graph),
        "ground_truth_cover_sha256": cover_sha256(canonical),
        "n": int(graph.vcount()),
        "m": int(graph.ecount()),
        "directed": bool(graph.is_directed()),
        "ground_truth_community_count": len(canonical),
        "ground_truth_max_memberships_per_node": max(
            memberships.values(), default=0
        ),
    }


def cover_statistics(cover: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Return bounded-memory cover and overlap diagnostics for reports."""
    sizes = [len(community) for community in cover]
    memberships = Counter(member for community in cover for member in community)
    sorted_sizes = sorted(sizes)
    p95_index = min(len(sorted_sizes) - 1, int(0.95 * (len(sorted_sizes) - 1))) if sorted_sizes else 0
    n_covered = len(memberships)
    n_overlapping = sum(count > 1 for count in memberships.values())
    total_memberships = sum(sizes)
    return {
        "n_communities": len(sizes),
        "n_covered_nodes": n_covered,
        "community_size": {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": statistics.fmean(sizes) if sizes else 0.0,
            "median": statistics.median(sizes) if sizes else 0.0,
            "p95": sorted_sizes[p95_index] if sorted_sizes else 0,
            "singleton_count": sum(size == 1 for size in sizes),
        },
        "overlap": {
            "n_overlapping_nodes": n_overlapping,
            "overlapping_node_fraction_of_covered": (
                n_overlapping / n_covered if n_covered else 0.0
            ),
            "memberships": total_memberships,
            "mean_memberships_per_covered_node": (
                total_memberships / n_covered if n_covered else 0.0
            ),
            "max_memberships_per_node": max(memberships.values(), default=0),
        },
    }


def _make_report(
    *,
    spec: SnapDatasetSpec,
    cover_variant: str,
    graph: ig.Graph,
    graph_path: str,
    cover_path: str,
    mapping_strategy: str,
    remap: dict[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    stats = cover_statistics(remap.pop("cover"))
    return {
        "dataset": spec.name,
        "graph_path": graph_path,
        "ground_truth_path": cover_path,
        "ground_truth_type": spec.ground_truth_type,
        "cover_variant": cover_variant,
        "n": graph.vcount(),
        "m": graph.ecount(),
        "directed": graph.is_directed(),
        "number_of_communities": stats["n_communities"],
        "number_of_covered_nodes": stats["n_covered_nodes"],
        "community_size_statistics": stats["community_size"],
        "overlap_statistics": stats["overlap"],
        "id_mapping_strategy": mapping_strategy,
        "source_kind": source_kind,
        "validation": remap,
    }


def _source_root_sha256(root: Path) -> str:
    return hashlib.sha256(str(root.expanduser().resolve()).encode()).hexdigest()


def _source_artifact_identity(
    spec: SnapDatasetSpec, base: Path, cover_variant: str
) -> tuple[str, list[dict[str, Any]]]:
    """Fingerprint the exact source pair selected by loader precedence."""
    graph_path = _first_existing(base, spec.graph_files)
    cover_path = _first_existing(base, spec.cover_files.get(cover_variant, ()))
    source_kind = "trusted_archive_pickle"
    if graph_path is None or cover_path is None:
        graph_path = _first_existing(base, spec.raw_edge_files)
        cover_path = _first_existing(base, spec.raw_cover_files.get(cover_variant, ()))
        source_kind = "streamed_raw_gzip"
    if graph_path is None or cover_path is None:
        return source_kind, []
    artifacts = []
    for role, path in (("graph", graph_path), ("ground_truth", cover_path)):
        artifacts.append(
            {
                "role": role,
                "relative_path": path.relative_to(base).as_posix(),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return source_kind, artifacts


def _source_fingerprint(source_kind: str, artifacts: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"source_kind": source_kind, "artifacts": artifacts},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _normalized_cache_path(
    cache_dir: Path,
    name: str,
    cover_variant: str,
    source_root_sha256: str,
    source_fingerprint: str,
) -> Path:
    # Namespace caches by the requested archive root.  A shared cache directory
    # can therefore never return another root's graph merely because dataset
    # and cover names happen to match.
    return cache_dir / (
        f"{name}-{cover_variant}-{source_root_sha256[:12]}-"
        f"{source_fingerprint[:16]}-"
        f"normalized-v{CACHE_SCHEMA_VERSION}.pkl"
    )


def _load_normalized_cache(
    path: Path,
    *,
    expected_dataset: str,
    expected_cover_variant: str,
    expected_source_root_sha256: str,
    expected_source_kind: str,
    expected_source_artifacts: list[dict[str, Any]],
) -> SnapDataset | None:
    try:
        payload = _read_pickle(path)
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA_VERSION:
            return None
        if payload.get("source_root_sha256") != expected_source_root_sha256:
            return None
        if payload.get("source_kind") != expected_source_kind:
            return None
        if payload.get("source_artifacts") != expected_source_artifacts:
            return None
        graph, cover, report = payload["graph"], payload["cover"], payload["report"]
        if not isinstance(graph, ig.Graph) or not isinstance(report, dict):
            return None
        if (
            report.get("dataset") != expected_dataset
            or report.get("cover_variant") != expected_cover_variant
            or report.get("source_kind") != expected_source_kind
            or report.get("source_artifacts") != expected_source_artifacts
            or report.get("normalized_cache_path") != path.name
            or report.get("normalized_cache_namespace")
            != expected_source_root_sha256[:16]
            or report.get("source_artifact_fingerprint_sha256")
            != _source_fingerprint(expected_source_kind, expected_source_artifacts)
        ):
            return None
        source_paths = {
            artifact["role"]: artifact["relative_path"]
            for artifact in expected_source_artifacts
        }
        if (
            report.get("graph_path") != source_paths.get("graph")
            or report.get("ground_truth_path") != source_paths.get("ground_truth")
        ):
            return None
        cover = _coerce_cover(cover, path)
        if any(member < 0 or member >= graph.vcount() for c in cover for member in c):
            return None
        if any(
            len(community) < 2
            or len(community) != len(set(community))
            for community in cover
        ):
            return None
        if payload.get("cache_content_identity") != _cache_content_identity(
            graph, cover
        ):
            return None
        stats = cover_statistics(cover)
        if any(
            report.get(key) != expected
            for key, expected in {
                "n": graph.vcount(),
                "m": graph.ecount(),
                "directed": graph.is_directed(),
                "number_of_communities": stats["n_communities"],
                "number_of_covered_nodes": stats["n_covered_nodes"],
                "community_size_statistics": stats["community_size"],
                "overlap_statistics": stats["overlap"],
            }.items()
        ):
            return None
        report = dict(report)
        canonical, _ = canonicalize_cover(
            cover, n_vertices=graph.vcount(), minimum_size=2
        )
        if (
            isinstance(report.get("bounded_content_identity"), dict)
            and report["bounded_content_identity"]
            != content_identity(graph, canonical)
        ):
            return None
        if (
            isinstance(report.get("content_identity"), dict)
            and report["content_identity"] != content_identity(graph, canonical)
        ):
            return None
        report["source_kind"] = "validated_normalized_cache"
        report["normalized_cache_path"] = path.name
        return SnapDataset(str(report["dataset"]), str(report["cover_variant"]), graph, cover, report)
    except (
        OSError,
        EOFError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pickle.PickleError,
        SnapLoadError,
    ):
        return None


def _write_normalized_cache(
    path: Path,
    dataset: SnapDataset,
    *,
    source_root_sha256: str,
    source_kind: str,
    source_artifacts: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "source_root_sha256": source_root_sha256,
        "source_kind": source_kind,
        "source_artifacts": source_artifacts,
        "cache_content_identity": _cache_content_identity(
            dataset.graph, dataset.cover
        ),
        "graph": dataset.graph,
        "cover": dataset.cover,
        "report": dataset.report,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _load_from_pickles(
    spec: SnapDatasetSpec, base: Path, cover_variant: str
) -> SnapDataset | None:
    graph_path = _first_existing(base, spec.graph_files)
    cover_path = _first_existing(base, spec.cover_files.get(cover_variant, ()))
    if graph_path is None or cover_path is None:
        return None
    graph = _read_pickle(graph_path)
    if not isinstance(graph, ig.Graph):
        raise SnapLoadError(f"Graph pickle is not an igraph.Graph: {graph_path}")
    raw_cover = _coerce_cover(_read_pickle(cover_path), cover_path)
    id_to_index, strategy = _mapping_for_cached_graph(graph)
    cover, validation = _remap_cover(raw_cover, id_to_index)
    validation["cover"] = cover
    report = _make_report(
        spec=spec,
        cover_variant=cover_variant,
        graph=graph,
        graph_path=str(graph_path.relative_to(base)),
        cover_path=str(cover_path.relative_to(base)),
        mapping_strategy=strategy,
        remap=validation,
        source_kind="trusted_archive_pickle",
    )
    return SnapDataset(spec.name, cover_variant, graph, cover, report)


def _load_from_raw(spec: SnapDatasetSpec, base: Path, cover_variant: str) -> SnapDataset:
    edge_path = _first_existing(base, spec.raw_edge_files)
    cover_path = _first_existing(base, spec.raw_cover_files.get(cover_variant, ()))
    if edge_path is None or cover_path is None:
        if cover_variant not in spec.raw_cover_files:
            raise UnsupportedCoverVariant(
                f"{spec.name} has no supplied {cover_variant!r} cover; "
                f"available variants: {', '.join(spec.raw_cover_files)}"
            )
        raise SnapLoadError(
            f"Could not find both graph and cover files for {spec.name} under {base}. "
            f"Expected graph candidates {spec.raw_edge_files} and cover candidates "
            f"{spec.raw_cover_files[cover_variant]}"
        )
    edges, old_ids = _read_edge_list(edge_path)
    ordered_ids = sorted(old_ids)
    id_to_index = {old: index for index, old in enumerate(ordered_ids)}
    graph = ig.Graph(
        n=len(ordered_ids),
        edges=[(id_to_index[src], id_to_index[tgt]) for src, tgt in edges],
        directed=spec.directed,
    )
    raw_cover = _read_cover_text(
        cover_path, wikipedia_categories=spec.name == "wikipedia"
    )
    cover, validation = _remap_cover(raw_cover, id_to_index)
    validation["cover"] = cover
    report = _make_report(
        spec=spec,
        cover_variant=cover_variant,
        graph=graph,
        graph_path=str(edge_path.relative_to(base)),
        cover_path=str(cover_path.relative_to(base)),
        mapping_strategy="raw_sorted_original_id_to_contiguous_index",
        remap=validation,
        source_kind="streamed_raw_gzip",
    )
    return SnapDataset(spec.name, cover_variant, graph, cover, report)


def _mapequation_artifact_identity(
    name: str, cover_variant: str
) -> tuple[str, list[dict[str, Any]]]:
    """Return a stable cache identity for a catalogue-backed dataset.

    The catalogue owns the actual archive paths and may change its local cache
    layout between releases.  Cache identity therefore uses the official
    dataset name and requested cover variant rather than guessing paths inside
    a third-party package.
    """
    catalog_name = MAPEQUATION_SNAP_NAMES[name]
    return "snap_catalog", [
        {
            "role": "graph",
            "relative_path": SPECS[name].raw_edge_files[0],
            "url": SNAP_DOWNLOAD_URLS[name]["graph"],
            "sha256": None,
            "size_bytes": None,
        },
        {
            "role": "ground_truth",
            "relative_path": SPECS[name].raw_cover_files[cover_variant][0],
            "url": SNAP_DOWNLOAD_URLS[name][cover_variant],
            "sha256": None,
            "size_bytes": None,
        },
    ]


def _catalog_source_identity(name: str, cover_variant: str) -> tuple[str, list[dict[str, Any]]]:
    """Return one cache identity for either catalogue backend."""
    return _mapequation_artifact_identity(name, cover_variant)


def _download_catalog_file(
    url: str,
    destination: Path,
    *,
    max_download_bytes: int | None = None,
) -> None:
    """Atomically download one official SNAP file with an optional size cap."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if (
                max_download_bytes is not None
                and content_length
                and int(content_length) > max_download_bytes
            ):
                raise SnapLoadError(
                    f"SNAP download {url} is {content_length} bytes, above "
                    f"the configured --max-download-bytes={max_download_bytes}"
                )
            written = 0
            with temporary.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if max_download_bytes is not None and written > max_download_bytes:
                        raise SnapLoadError(
                            f"SNAP download {url} exceeded --max-download-bytes="
                            f"{max_download_bytes}"
                        )
                    stream.write(chunk)
        temporary.replace(destination)
    except SnapLoadError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    except (OSError, urllib.error.URLError, ValueError) as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise SnapLoadError(f"could not download SNAP file {url}: {exc}") from exc


def _load_from_official_snap(
    spec: SnapDatasetSpec,
    cover_variant: str,
    *,
    cache_dir: Path,
    max_download_bytes: int | None = None,
) -> SnapDataset:
    """Download raw official SNAP files into the user cache and parse them."""
    urls = SNAP_DOWNLOAD_URLS[spec.name]
    base = cache_dir / "raw" / spec.name
    graph_path = base / spec.raw_edge_files[0]
    cover_path = base / spec.raw_cover_files[cover_variant][0]
    _download_catalog_file(
        urls["graph"], graph_path, max_download_bytes=max_download_bytes
    )
    _download_catalog_file(
        urls[cover_variant], cover_path, max_download_bytes=max_download_bytes
    )
    dataset = _load_from_raw(spec, base, cover_variant)
    dataset.report["source_kind"] = "snap_catalog"
    dataset.report["catalog_backend"] = "official_snap_http"
    dataset.report["catalog_urls"] = {
        "graph": urls["graph"],
        "ground_truth": urls[cover_variant],
    }
    dataset.report["catalog_cache_root"] = str(base)
    return dataset


def prepare_snap_dataset(
    name: str,
    *,
    cover_variant: str = "top5000",
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_catalog: bool = True,
    max_download_bytes: int | None = None,
) -> dict[str, Any]:
    """Ensure graph/cover source files exist without parsing a huge graph.

    This is the setup/doctor boundary.  Unlike :func:`load_snap_dataset`, it
    never constructs an igraph object, so preparing Friendster or LiveJournal
    does not allocate their full adjacency structure just to verify a URL.
    """
    key = name.strip().lower()
    if key not in SPECS:
        raise SnapLoadError(f"Unknown SNAP overlap dataset {name!r}")
    spec = SPECS[key]
    if cover_variant not in spec.raw_cover_files:
        raise UnsupportedCoverVariant(
            f"{spec.name} has no supplied {cover_variant!r} cover"
        )
    root = Path(data_root).expanduser() if data_root is not None else DEFAULT_NETWORKS_DIR
    base = _dataset_dir(root, spec)
    source_kind, artifacts = _source_artifact_identity(spec, base, cover_variant)
    if artifacts:
        return {
            "status": "ready",
            "dataset": key,
            "cover_variant": cover_variant,
            "source_kind": source_kind,
            "source_artifacts": artifacts,
            "root": str(base),
        }
    if not allow_catalog:
        return {
            "status": "unavailable",
            "dataset": key,
            "cover_variant": cover_variant,
            "reason": f"local graph/cover files are absent under {base}",
        }
    cache_root = expand_path(cache_dir) if cache_dir else default_cache_dir()
    try:
        from networks import snap as catalog  # type: ignore[import-not-found]

        catalog_name = MAPEQUATION_SNAP_NAMES[key]
        if max_download_bytes is None:
            network = catalog.load(catalog_name)
        else:
            try:
                network = catalog.load(catalog_name, max_bytes=int(max_download_bytes))
            except TypeError:
                network = catalog.load(catalog_name)
        truth = catalog.load_ground_truth(catalog_name, variant=cover_variant)
        return {
            "status": "ready",
            "dataset": key,
            "cover_variant": cover_variant,
            "source_kind": "snap_catalog",
            "catalog_backend": "mapequation_networks",
            "catalog_dataset": catalog_name,
            "network_locator": _catalog_object_path(network),
            "ground_truth_locator": _catalog_object_path(truth),
        }
    except Exception as catalog_error:
        # The direct downloader below is the supported fallback when the
        # catalogue package is absent or unavailable on the active index.
        try:
            urls = SNAP_DOWNLOAD_URLS[key]
            raw_base = cache_root / "raw" / key
            graph_path = raw_base / spec.raw_edge_files[0]
            cover_path = raw_base / spec.raw_cover_files[cover_variant][0]
            _download_catalog_file(
                urls["graph"], graph_path, max_download_bytes=max_download_bytes
            )
            _download_catalog_file(
                urls[cover_variant], cover_path, max_download_bytes=max_download_bytes
            )
        except SnapLoadError as download_error:
            raise SnapLoadError(
                f"catalogue setup failed for {key}: {catalog_error}; "
                f"official SNAP download failed: {download_error}"
            ) from download_error
        return {
            "status": "ready",
            "dataset": key,
            "cover_variant": cover_variant,
            "source_kind": "snap_catalog",
            "catalog_backend": "official_snap_http",
            "catalog_urls": {
                "graph": urls["graph"],
                "ground_truth": urls[cover_variant],
            },
            "cache_root": str(raw_base),
        }


def _catalog_object_path(value: Any) -> str | None:
    """Best-effort path extraction for catalogue provenance reports."""
    for attribute in ("path", "filename", "file", "source", "url"):
        candidate = getattr(value, attribute, None)
        if candidate is not None and not callable(candidate):
            return str(candidate)
    return None


def _catalog_edges(network: Any) -> tuple[list[tuple[int, int]], set[int]]:
    """Normalize the small API variants exposed by networks releases."""
    edge_reader = getattr(network, "edges", None)
    if callable(edge_reader):
        try:
            values = edge_reader(cast=int)
        except TypeError:
            values = edge_reader()
    else:
        values = edge_reader or ()
    if isinstance(values, (int, float, str, bytes)):
        values = ()
    edges: list[tuple[int, int]] = []
    node_ids: set[int] = set()
    for edge in values:
        if len(edge) < 2:
            continue
        source, target = int(edge[0]), int(edge[1])
        edges.append((source, target))
        node_ids.update((source, target))
    node_reader = getattr(network, "nodes", None)
    if callable(node_reader):
        try:
            node_values = node_reader()
        except TypeError:
            node_values = node_reader
    else:
        node_values = node_reader or ()
    if isinstance(node_values, (int, float, str, bytes)):
        node_values = ()
    for node in node_values:
        try:
            node_ids.add(int(node))
        except (TypeError, ValueError):
            continue
    return edges, node_ids


def _catalog_communities(ground_truth: Any) -> list[list[int]]:
    reader = getattr(ground_truth, "communities", None)
    values = reader() if callable(reader) else ground_truth
    communities: list[list[int]] = []
    for community in values or ():
        members: list[int] = []
        for member in community:
            try:
                members.append(int(member))
            except (TypeError, ValueError):
                continue
        communities.append(members)
    return communities


def _load_from_mapequation(
    spec: SnapDatasetSpec,
    cover_variant: str,
    *,
    max_download_bytes: int | None = None,
) -> SnapDataset:
    """Load one dataset through the optional MapEquation SNAP catalogue.

    This function is deliberately called only after local archive precedence
    has failed.  Importing ``networks`` here keeps ``pip install hedonic``
    lightweight while ``pip install 'hedonic[reproduce]'`` enables a fresh
    machine to fetch the official graph and supplied cover on first use.
    """
    try:
        from networks import snap as catalog  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SnapLoadError(
            "no local SNAP archive was found and the optional MapEquation "
            "catalogue is not installed; install `hedonic[reproduce]` or "
            "set HEDONIC_NETWORKS_DIR to the downloaded archives"
        ) from exc
    catalog_name = MAPEQUATION_SNAP_NAMES[spec.name]
    try:
        if max_download_bytes is None:
            network = catalog.load(catalog_name)
        else:
            network = catalog.load(catalog_name, max_bytes=int(max_download_bytes))
    except TypeError:
        # Older catalogue releases do not expose ``max_bytes``.  Retrying
        # without it keeps the adapter compatible while the setup command
        # records the requested bound in its provenance manifest.
        network = catalog.load(catalog_name)
    except Exception as exc:
        raise SnapLoadError(
            f"MapEquation catalogue could not load {catalog_name}: {exc}"
        ) from exc
    try:
        ground_truth = catalog.load_ground_truth(catalog_name, variant=cover_variant)
    except TypeError:
        ground_truth = catalog.load_ground_truth(catalog_name, cover_variant)
    except Exception as exc:
        raise UnsupportedCoverVariant(
            f"MapEquation catalogue has no {cover_variant!r} cover for {catalog_name}: {exc}"
        ) from exc

    edges, node_ids = _catalog_edges(network)
    raw_cover = _catalog_communities(ground_truth)
    node_ids.update(member for community in raw_cover for member in community)
    ordered_ids = sorted(node_ids)
    id_to_index = {old: index for index, old in enumerate(ordered_ids)}
    graph = ig.Graph(
        n=len(ordered_ids),
        edges=[(id_to_index[src], id_to_index[tgt]) for src, tgt in edges],
        directed=spec.directed,
    )
    cover, validation = _remap_cover(raw_cover, id_to_index)
    validation["cover"] = cover
    source_graph_locator = _catalog_object_path(network)
    source_cover_locator = _catalog_object_path(ground_truth)
    # The cache validator expects the report paths to match the stable
    # artifact identity.  Keep volatile package-local locators as separate
    # provenance fields instead of embedding them in the cache key.
    source_graph = spec.raw_edge_files[0]
    source_cover = spec.raw_cover_files[cover_variant][0]
    report = _make_report(
        spec=spec,
        cover_variant=cover_variant,
        graph=graph,
        graph_path=source_graph,
        cover_path=source_cover,
        mapping_strategy="catalog_original_id_to_contiguous_index",
        remap=validation,
        source_kind="snap_catalog",
    )
    report["catalog_dataset"] = catalog_name
    report["catalog_package"] = "mapequation-networks"
    report["catalog_backend"] = "mapequation_networks"
    report["catalog_graph_locator"] = source_graph_locator
    report["catalog_ground_truth_locator"] = source_cover_locator
    return SnapDataset(spec.name, cover_variant, graph, cover, report)


def load_snap_dataset(
    name: str,
    *,
    cover_variant: str = "top5000",
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    use_normalized_cache: bool = True,
    allow_catalog: bool = True,
    max_download_bytes: int | None = None,
) -> SnapDataset:
    """Load, remap, validate and report one supported overlapping SNAP dataset.

    Validated normalized caches are preferred, then trusted local archive
    pickles, then streamed compressed text files.  The returned cover always
    consists of lists of valid ``0 <= v < graph.vcount()`` indices.
    """
    key = name.strip().lower()
    if key not in SPECS:
        raise SnapLoadError(
            f"Unknown SNAP overlap dataset {name!r}; choose from {', '.join(network_names())}"
        )
    spec = SPECS[key]
    if cover_variant not in {"all", "top5000"}:
        raise UnsupportedCoverVariant("cover_variant must be 'all' or 'top5000'")
    if cover_variant not in spec.raw_cover_files:
        raise UnsupportedCoverVariant(
            f"{spec.name} has no supplied {cover_variant!r} cover; "
            f"use 'all' (Wikipedia categories) instead"
        )
    root = Path(data_root).expanduser() if data_root is not None else DEFAULT_NETWORKS_DIR
    root_digest = _source_root_sha256(root)
    base = _dataset_dir(root, spec)
    source_kind, source_artifacts = _source_artifact_identity(
        spec, base, cover_variant
    )
    local_source_available = bool(source_artifacts)
    if not local_source_available and allow_catalog:
        source_kind, source_artifacts = _catalog_source_identity(key, cover_variant)
    source_fingerprint = _source_fingerprint(source_kind, source_artifacts)
    cache_root = expand_path(cache_dir) if cache_dir else default_cache_dir()
    normalized_path = _normalized_cache_path(
        cache_root, key, cover_variant, root_digest, source_fingerprint
    )
    if use_normalized_cache and normalized_path.is_file():
        cached = _load_normalized_cache(
            normalized_path,
            expected_dataset=key,
            expected_cover_variant=cover_variant,
            expected_source_root_sha256=root_digest,
            expected_source_kind=source_kind,
            expected_source_artifacts=source_artifacts,
        )
        if cached is not None:
            return cached

    dataset = _load_from_pickles(spec, base, cover_variant)
    if dataset is None and local_source_available:
        dataset = _load_from_raw(spec, base, cover_variant)
    if dataset is None:
        if not allow_catalog:
            dataset = _load_from_raw(spec, base, cover_variant)
        else:
            try:
                dataset = _load_from_mapequation(
                    spec,
                    cover_variant,
                    max_download_bytes=max_download_bytes,
                )
            except SnapLoadError as catalog_error:
                try:
                    dataset = _load_from_official_snap(
                        spec,
                        cover_variant,
                        cache_dir=cache_root,
                        max_download_bytes=max_download_bytes,
                    )
                except SnapLoadError as download_error:
                    raise SnapLoadError(
                        f"no local SNAP archive and catalogue backends failed for "
                        f"{key}: optional package: {catalog_error}; official "
                        f"SNAP download: {download_error}"
                    ) from download_error
    dataset.report["normalized_cache_path"] = normalized_path.name
    dataset.report["normalized_cache_namespace"] = root_digest[:16]
    dataset.report["source_artifacts"] = source_artifacts
    dataset.report["source_artifact_fingerprint_sha256"] = source_fingerprint
    _write_normalized_cache(
        normalized_path,
        dataset,
        source_root_sha256=root_digest,
        source_kind=source_kind,
        source_artifacts=source_artifacts,
    )
    return dataset


def bounded_induced_dataset(dataset: SnapDataset, max_nodes: int | None) -> SnapDataset:
    """Return a deterministic GT-informed induced subgraph when bounded.

    The standard benchmark profile uses this to keep social-network runs
    practical.  It first retains two high-membership nodes per ground-truth
    community, then fills remaining capacity by membership frequency.  Covers
    are remapped again and communities with fewer than two retained members are
    reported explicitly.
    """
    if max_nodes is None or max_nodes <= 0 or dataset.graph.vcount() <= max_nodes:
        canonical, validation = canonicalize_cover(
            dataset.cover, n_vertices=dataset.graph.vcount(), minimum_size=2
        )
        report = dict(dataset.report)
        stats = cover_statistics(canonical)
        report.update(
            {
                "n": dataset.graph.vcount(),
                "m": dataset.graph.ecount(),
                "directed": dataset.graph.is_directed(),
                "number_of_communities": stats["n_communities"],
                "number_of_covered_nodes": stats["n_covered_nodes"],
                "community_size_statistics": stats["community_size"],
                "overlap_statistics": stats["overlap"],
            }
        )
        report["bounded_subgraph"] = {
            "applied": False,
            "max_nodes": max_nodes,
            "selected_nodes": dataset.graph.vcount(),
            "strategy": "not_required",
            "cover_canonicalization": validation,
        }
        report["bounded_content_identity"] = content_identity(
            dataset.graph, canonical
        )
        report["content_identity"] = report["bounded_content_identity"]
        return SnapDataset(
            dataset.name, dataset.cover_variant, dataset.graph, canonical, report
        )
    selection_cover, source_canonicalization = canonicalize_cover(
        dataset.cover, n_vertices=dataset.graph.vcount(), minimum_size=2
    )
    frequency = Counter(member for community in selection_cover for member in community)
    ranked = lambda members: sorted(members, key=lambda v: (-frequency[v], v))
    selected: set[int] = set()
    for community in sorted(selection_cover, key=lambda c: (-len(c), tuple(c))):
        for member in ranked(community)[:2]:
            if len(selected) >= max_nodes:
                break
            selected.add(member)
        if len(selected) >= max_nodes:
            break
    for member, _count in sorted(frequency.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= max_nodes:
            break
        selected.add(member)
    selected_indices = sorted(selected)
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    graph = dataset.graph.induced_subgraph(selected_indices)
    cover, validation = _remap_cover(
        selection_cover, old_to_new, canonicalize_communities=True
    )
    report = dict(dataset.report)
    report.update(
        {
            "n": graph.vcount(),
            "m": graph.ecount(),
            "bounded_subgraph": {
                "applied": True,
                "max_nodes": max_nodes,
                "selected_nodes": len(selected_indices),
                "strategy": "gt_membership_frequency_then_induced_subgraph",
                "source_cover_canonicalization_before_selection": source_canonicalization,
                "dropped_communities_lt_2_members": validation[
                    "dropped_communities_lt_2_members"
                ],
                "projected_cover_canonicalization": validation,
                # Compatibility alias; the explicit source/projected fields
                # above distinguish archive duplicates from induction
                # collisions.
                "cover_canonicalization": validation,
            },
            "id_mapping_strategy": report.get("id_mapping_strategy", "identity")
            + "+bounded_induced_subgraph_remap",
        }
    )
    stats = cover_statistics(cover)
    report["number_of_communities"] = stats["n_communities"]
    report["number_of_covered_nodes"] = stats["n_covered_nodes"]
    report["community_size_statistics"] = stats["community_size"]
    report["overlap_statistics"] = stats["overlap"]
    report["bounded_content_identity"] = content_identity(graph, cover)
    report["content_identity"] = report["bounded_content_identity"]
    return SnapDataset(dataset.name, dataset.cover_variant, graph, cover, report)


def _agmfit_induced_dataset_for_anchor(
    dataset: SnapDataset,
    *,
    anchor: int,
    seed: int,
    canonical: list[list[int]],
    incident: list[list[int]],
    source_canonicalization: dict[str, int],
) -> SnapDataset:
    """Build one AGMfit window after its anchor has been selected."""
    # Yang & Leskovec, “Community-Affiliation Graph Model for Overlapping
    # Network Community Detection,” ICDM 2012, §VI “Experimental setup,”
    # PDF p. 7, Fig. 8: choose a random node in at least two communities and
    # induce the subnetwork on the union of its incident communities.
    selected = {
        vertex
        for community_id in incident[anchor]
        for vertex in canonical[community_id]
    }
    selected_indices = sorted(selected)
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    graph = dataset.graph.induced_subgraph(selected_indices)
    projected = [
        [old_to_new[vertex] for vertex in community if vertex in old_to_new]
        for community in canonical
    ]
    cover, projection_canonicalization = canonicalize_cover(
        projected, n_vertices=graph.vcount(), minimum_size=2
    )
    report = dict(dataset.report)
    report.update(
        {
            "n": graph.vcount(),
            "m": graph.ecount(),
            "agmfit_subgraph": {
                "applied": True,
                "seed": int(seed),
                "anchor_vertex_original_index": int(anchor),
                "anchor_membership_count": len(incident[anchor]),
                "selected_nodes": len(selected_indices),
                "incident_community_count": len(incident[anchor]),
                "strategy": "random_overlap_anchor_union_induced_subgraph",
                "source_cover_canonicalization": source_canonicalization,
                "projected_cover_canonicalization": projection_canonicalization,
            },
            "id_mapping_strategy": report.get("id_mapping_strategy", "identity")
            + "+agmfit_induced_subgraph_remap",
        }
    )
    stats = cover_statistics(cover)
    report["number_of_communities"] = stats["n_communities"]
    report["number_of_covered_nodes"] = stats["n_covered_nodes"]
    report["community_size_statistics"] = stats["community_size"]
    report["overlap_statistics"] = stats["overlap"]
    report["bounded_content_identity"] = content_identity(graph, cover)
    report["content_identity"] = report["bounded_content_identity"]
    return SnapDataset(dataset.name, dataset.cover_variant, graph, cover, report)


def agmfit_induced_dataset(dataset: SnapDataset, *, seed: int = 0) -> SnapDataset:
    """Construct one AGMfit-style overlap-centered induced subgraph.

    The selected anchor is sampled from vertices belonging to at least two
    ground-truth communities.  The vertex set is the union of those incident
    communities, and the returned graph is the induced subgraph on that set.
    This is intentionally independent of the bounded deterministic selector
    above, which remains available for protocols with a fixed node budget.
    """
    canonical, source_canonicalization = canonicalize_cover(
        dataset.cover, n_vertices=dataset.graph.vcount(), minimum_size=2
    )
    incident: list[list[int]] = [[] for _ in range(dataset.graph.vcount())]
    for community_id, community in enumerate(canonical):
        for vertex in community:
            incident[vertex].append(community_id)
    eligible = [vertex for vertex, labels in enumerate(incident) if len(labels) >= 2]
    if not eligible:
        raise ValueError(
            "AGMfit sampling requires at least one vertex in two ground-truth communities"
        )
    anchor = random.Random(int(seed)).choice(eligible)
    return _agmfit_induced_dataset_for_anchor(
        dataset,
        anchor=anchor,
        seed=seed,
        canonical=canonical,
        incident=incident,
        source_canonicalization=source_canonicalization,
    )


def sample_agmfit_subgraphs(
    dataset: SnapDataset,
    *,
    n_subgraphs: int = 500,
    seed: int = 0,
) -> list[SnapDataset]:
    """Return reproducible AGMfit-style overlap-centered subgraphs.

    Each replicate uses a deterministic child seed.  Anchors are sampled
    without replacement while possible, so the common case yields distinct
    windows just as the AGMfit evaluation describes; if fewer eligible
    vertices exist than requested replicates, deterministic re-use is allowed.
    """
    if n_subgraphs <= 0:
        raise ValueError("n_subgraphs must be positive")
    canonical, source_canonicalization = canonicalize_cover(
        dataset.cover, n_vertices=dataset.graph.vcount(), minimum_size=2
    )
    incident: list[list[int]] = [[] for _ in range(dataset.graph.vcount())]
    for community_id, community in enumerate(canonical):
        for vertex in community:
            incident[vertex].append(community_id)
    eligible = [vertex for vertex, labels in enumerate(incident) if len(labels) >= 2]
    if not eligible:
        raise ValueError(
            "AGMfit sampling requires at least one vertex in two ground-truth communities"
        )
    rng = random.Random(int(seed))
    anchors = rng.sample(eligible, k=min(n_subgraphs, len(eligible)))
    while len(anchors) < n_subgraphs:
        anchors.append(eligible[rng.randrange(len(eligible))])
    # Child seeds preserve deterministic sampling while keeping each returned
    # report independently replayable.
    return [
        _agmfit_induced_dataset_for_anchor(
            dataset,
            anchor=anchor,
            seed=seed,
            canonical=canonical,
            incident=incident,
            source_canonicalization=source_canonicalization,
        )
        for anchor, seed in zip(anchors, (rng.randrange(2**63) for _ in anchors))
    ]


# Descriptive alias for callers that prefer the noun used in the paper.
agmfit_induced_datasets = sample_agmfit_subgraphs


def common_undirected_analysis_dataset(dataset: SnapDataset) -> SnapDataset:
    """Return the single undirected simple graph used by every detector.

    SNAP's Wikipedia archive is directed, whereas the formal hedonic model and
    the two external baselines in this benchmark are undirected.  Keeping the
    raw directionality in the loader is useful provenance, but comparing a
    directed hedonic run with undirected baselines is not a valid method
    comparison.  This explicit boundary therefore projects *every* loaded
    graph to the same undirected, loop-free, simple representation before any
    resolution, detection, quality, or metric calculation.
    """
    source = dataset.graph
    graph = source.copy()
    if graph.is_directed():
        graph.to_undirected(mode="collapse")
    before_simplify_edges = graph.ecount()
    graph.simplify(multiple=True, loops=True, combine_edges=None)
    canonical_cover, cover_validation = canonicalize_cover(
        dataset.cover, n_vertices=graph.vcount(), minimum_size=2
    )
    stats = cover_statistics(canonical_cover)
    report = dict(dataset.report)
    source_graph = {
        "n": source.vcount(),
        "m": source.ecount(),
        "directed": source.is_directed(),
    }
    analysis_graph = {
        "policy": ANALYSIS_GRAPH_POLICY,
        "n": graph.vcount(),
        "m": graph.ecount(),
        "directed": graph.is_directed(),
        "source_directed": source.is_directed(),
        "edges_after_direction_collapse_before_simplify": before_simplify_edges,
        "edges_removed_by_simplification": before_simplify_edges - graph.ecount(),
        "applied_before_all_methods_and_metrics": True,
    }
    report.update(
        {
            "source_graph": source_graph,
            "analysis_graph": analysis_graph,
            "n": graph.vcount(),
            "m": graph.ecount(),
            "directed": graph.is_directed(),
            "number_of_communities": stats["n_communities"],
            "number_of_covered_nodes": stats["n_covered_nodes"],
            "community_size_statistics": stats["community_size"],
            "overlap_statistics": stats["overlap"],
            "analysis_cover_canonicalization": cover_validation,
            "content_identity": content_identity(graph, canonical_cover),
        }
    )
    return SnapDataset(
        dataset.name, dataset.cover_variant, graph, canonical_cover, report
    )


def smoke_dataset(name: str, *, cover_variant: str = "top5000") -> SnapDataset:
    """A tiny built-in overlapping graph used by the CLI smoke profile."""
    if name not in SPECS:
        raise SnapLoadError(f"Unknown smoke dataset {name!r}")
    spec = SPECS[name]
    if name == "wikipedia" and cover_variant == "top5000":
        raise UnsupportedCoverVariant("Wikipedia has no supplied top5000 category cover")
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 0)]
    graph = ig.Graph(n=6, edges=edges, directed=spec.directed)
    cover = [[0, 1, 2], [2, 3, 4], [0, 4, 5]]
    validation = {
        "raw_community_count": len(cover),
        "dropped_communities_lt_2_members": 0,
        "missing_member_count": 0,
        "missing_id_examples": [],
        "duplicate_members_removed": 0,
        "cover": cover,
    }
    report = _make_report(
        spec=spec,
        cover_variant=cover_variant,
        graph=graph,
        graph_path="<built-in-smoke-graph>",
        cover_path="<built-in-smoke-cover>",
        mapping_strategy="built_in_contiguous_indices",
        remap=validation,
        source_kind="built_in_smoke_fixture",
    )
    return SnapDataset(name, cover_variant, graph, cover, report)


def synthetic_agmfit_dataset(
    *,
    n_nodes: int = 1_000,
    seed: int = 0,
    n_communities: int = 24,
    overlap_probability: float = 0.18,
    second_overlap_probability: float = 0.04,
    intra_community_probability: float = 0.20,
) -> SnapDataset:
    """Build a deterministic, AGMfit-like overlapping smoke fixture.

    This is intentionally not presented as a SNAP result.  It is an archive-
    free affiliation graph used to validate the complete benchmark plumbing
    (method dispatch, cover scoring, resource accounting, resumable ledgers)
    on roughly the scale of a small AGMfit-centered induced window.  Every
    vertex has a ground-truth membership and a controlled fraction has two or
    three memberships.  Community edges are sampled from shared affiliations,
    with a ring backbone so the fixture remains connected for every seed.
    """
    if n_nodes < 8:
        raise ValueError("n_nodes must be at least 8")
    if n_communities < 2:
        raise ValueError("n_communities must be at least 2")
    if not 0.0 <= overlap_probability <= 1.0:
        raise ValueError("overlap_probability must be in [0, 1]")
    if not 0.0 <= second_overlap_probability <= 1.0:
        raise ValueError("second_overlap_probability must be in [0, 1]")
    if not 0.0 <= intra_community_probability <= 1.0:
        raise ValueError("intra_community_probability must be in [0, 1]")

    rng = random.Random(int(seed))
    n_communities = min(int(n_communities), n_nodes)
    memberships: list[list[int]] = [[] for _ in range(n_nodes)]
    for vertex in range(n_nodes):
        primary = vertex % n_communities
        memberships[vertex].append(primary)
        if rng.random() < overlap_probability:
            offset = 1 + rng.randrange(n_communities - 1)
            memberships[vertex].append((primary + offset) % n_communities)
        if rng.random() < second_overlap_probability:
            offset = 1 + rng.randrange(n_communities - 1)
            candidate = (primary + offset) % n_communities
            if candidate not in memberships[vertex]:
                memberships[vertex].append(candidate)

    communities = [
        sorted(vertex for vertex, labels in enumerate(memberships) if community in labels)
        for community in range(n_communities)
    ]
    communities = [community for community in communities if len(community) >= 2]
    if len(communities) < 2:
        raise RuntimeError("synthetic AGMfit fixture generated too few communities")

    edge_set: set[tuple[int, int]] = set()
    for community in communities:
        for left_index, left in enumerate(community):
            for right in community[left_index + 1 :]:
                if rng.random() < intra_community_probability:
                    edge_set.add((left, right))
    # The backbone is deterministic and keeps isolated affiliation samples
    # from making method failures look like dependency failures.
    edge_set.update(
        (vertex, (vertex + 1) % n_nodes) for vertex in range(n_nodes)
    )
    graph = ig.Graph(n=n_nodes, edges=sorted(edge_set), directed=False)
    stats = cover_statistics(communities)
    report = {
        "dataset": "synthetic_agmfit",
        "graph_path": "<generated:synthetic-agmfit-like>",
        "ground_truth_path": "<generated:affiliation-cover>",
        "ground_truth_type": "synthetic_agmfit_like_overlapping_affiliation_cover",
        "cover_variant": "all",
        "n": graph.vcount(),
        "m": graph.ecount(),
        "directed": False,
        "number_of_communities": stats["n_communities"],
        "number_of_covered_nodes": stats["n_covered_nodes"],
        "community_size_statistics": stats["community_size"],
        "overlap_statistics": stats["overlap"],
        "id_mapping_strategy": "generated_contiguous_indices",
        "source_kind": "synthetic_agmfit_smoke_fixture",
        "validation": {
            "raw_community_count": len(communities),
            "dropped_communities_lt_2_members": 0,
            "missing_member_count": 0,
            "duplicate_members_removed": 0,
        },
        "generator": {
            "seed": int(seed),
            "n_nodes": int(n_nodes),
            "n_communities_requested": int(n_communities),
            "overlap_probability": float(overlap_probability),
            "second_overlap_probability": float(second_overlap_probability),
            "intra_community_probability": float(intra_community_probability),
            "backbone": "cycle",
        },
    }
    report["content_identity"] = content_identity(graph, communities)
    return SnapDataset("synthetic_agmfit", "all", graph, communities, report)


def print_dataset_report(report: dict[str, Any]) -> None:
    """Print the required compact load report in a human-readable form."""
    print(
        "[dataset] "
        f"{report['dataset']} | cover={report['cover_variant']} | "
        f"n={report['n']:,} m={report['m']:,} directed={report['directed']}"
    )
    print(f"  graph path       : {report['graph_path']}")
    print(f"  ground truth path: {report['ground_truth_path']}")
    print(f"  ground truth type: {report['ground_truth_type']}")
    print(
        "  cover             : "
        f"{report['number_of_communities']:,} communities, "
        f"{report['number_of_covered_nodes']:,} covered nodes"
    )
    print(f"  community sizes   : {json.dumps(report['community_size_statistics'], sort_keys=True)}")
    print(f"  overlap           : {json.dumps(report['overlap_statistics'], sort_keys=True)}")
    print(f"  ID mapping        : {report['id_mapping_strategy']}")
    validation = report.get("validation", {})
    if validation.get("missing_member_count") or validation.get("dropped_communities_lt_2_members"):
        print(
            "  validation        : "
            f"missing members={validation.get('missing_member_count', 0):,}, "
            f"dropped communities={validation.get('dropped_communities_lt_2_members', 0):,}"
        )
