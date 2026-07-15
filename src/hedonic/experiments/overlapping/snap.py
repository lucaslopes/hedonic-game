"""Portable, ID-safe loaders for the overlapping SNAP benchmark datasets.

The saved SNAP archives use original (often sparse) numeric identifiers while
igraph works with contiguous vertex indices.  This module is deliberately the
only place that knows the archive layout: experiments consume
:class:`SnapDataset`, whose cover is already normalized to igraph indices.

Input archives are read-only.  A validated normalized cache is written to a
user cache directory (``~/.cache/hedonic/snap`` by default), never next to the
archived SNAP files.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import igraph as ig

from hedonic.experiments.config import NETWORKS_DIR

# Alias retained as part of this loader's public surface; configuration owns
# the portable default and environment/TOML precedence.
DEFAULT_NETWORKS_DIR = NETWORKS_DIR
CACHE_SCHEMA_VERSION = 1


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
    """Names accepted by the benchmark CLI, in stable display order."""
    return tuple(SPECS)


def default_cache_dir() -> Path:
    """Return the external normalized-data cache location."""
    return Path(
        os.path.expanduser(
            os.getenv("HEDONIC_SNAP_CACHE_DIR", "~/.cache/hedonic/snap")
        )
    )


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
    raw_cover: Sequence[Sequence[int]], id_to_index: dict[int, int]
) -> tuple[list[list[int]], dict[str, Any]]:
    """Map every original ID and make dropped members/communities explicit."""
    normalized: list[list[int]] = []
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
            normalized.append(members)
        else:
            dropped_communities += 1
    return normalized, {
        "raw_community_count": len(raw_cover),
        "dropped_communities_lt_2_members": dropped_communities,
        "missing_member_count": missing_member_count,
        "missing_id_examples": sorted(missing_ids),
        "duplicate_members_removed": duplicate_member_count,
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


def _normalized_cache_path(cache_dir: Path, name: str, cover_variant: str) -> Path:
    return cache_dir / f"{name}-{cover_variant}-normalized-v{CACHE_SCHEMA_VERSION}.pkl"


def _load_normalized_cache(path: Path) -> SnapDataset | None:
    try:
        payload = _read_pickle(path)
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA_VERSION:
            return None
        graph, cover, report = payload["graph"], payload["cover"], payload["report"]
        if not isinstance(graph, ig.Graph) or not isinstance(report, dict):
            return None
        cover = _coerce_cover(cover, path)
        if any(member < 0 or member >= graph.vcount() for c in cover for member in c):
            return None
        report = dict(report)
        report["source_kind"] = "validated_normalized_cache"
        report["normalized_cache_path"] = str(path)
        return SnapDataset(str(report["dataset"]), str(report["cover_variant"]), graph, cover, report)
    except (OSError, KeyError, pickle.PickleError, SnapLoadError):
        return None


def _write_normalized_cache(path: Path, dataset: SnapDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
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
        graph_path=str(graph_path),
        cover_path=str(cover_path),
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
        graph_path=str(edge_path),
        cover_path=str(cover_path),
        mapping_strategy="raw_sorted_original_id_to_contiguous_index",
        remap=validation,
        source_kind="streamed_raw_gzip",
    )
    return SnapDataset(spec.name, cover_variant, graph, cover, report)


def load_snap_dataset(
    name: str,
    *,
    cover_variant: str = "top5000",
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    use_normalized_cache: bool = True,
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
    cache_root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    normalized_path = _normalized_cache_path(cache_root, key, cover_variant)
    if use_normalized_cache and normalized_path.is_file():
        cached = _load_normalized_cache(normalized_path)
        if cached is not None:
            return cached

    root = Path(data_root).expanduser() if data_root is not None else DEFAULT_NETWORKS_DIR
    base = _dataset_dir(root, spec)
    dataset = _load_from_pickles(spec, base, cover_variant)
    if dataset is None:
        dataset = _load_from_raw(spec, base, cover_variant)
    dataset.report["normalized_cache_path"] = str(normalized_path)
    _write_normalized_cache(normalized_path, dataset)
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
        return dataset
    frequency = Counter(member for community in dataset.cover for member in community)
    ranked = lambda members: sorted(members, key=lambda v: (-frequency[v], v))
    selected: set[int] = set()
    for community in sorted(dataset.cover, key=lambda c: (-len(c), tuple(c))):
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
    cover, validation = _remap_cover(dataset.cover, old_to_new)
    report = dict(dataset.report)
    report.update(
        {
            "n": graph.vcount(),
            "m": graph.ecount(),
            "bounded_subgraph": {
                "max_nodes": max_nodes,
                "selected_nodes": len(selected_indices),
                "strategy": "gt_membership_frequency_then_induced_subgraph",
                "dropped_communities_lt_2_members": validation[
                    "dropped_communities_lt_2_members"
                ],
            },
            "id_mapping_strategy": report["id_mapping_strategy"]
            + "+bounded_induced_subgraph_remap",
        }
    )
    stats = cover_statistics(cover)
    report["number_of_communities"] = stats["n_communities"]
    report["number_of_covered_nodes"] = stats["n_covered_nodes"]
    report["community_size_statistics"] = stats["community_size"]
    report["overlap_statistics"] = stats["overlap"]
    return SnapDataset(dataset.name, dataset.cover_variant, graph, cover, report)


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
