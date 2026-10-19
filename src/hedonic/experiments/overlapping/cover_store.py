"""What a benchmark unit keeps of its answer, so it can be audited and rescored later without running its detector.

A unit run with ``--save-covers`` writes, next to its record ``runs/<net>/<runner>.json``, under ``covers/<net>/``:

``<runner>.json.gz``         the scoring cover: a JSON list of communities, each a sorted list of vertex ids of the
                             analysis graph (the graph the detector ran on).
``<runner>.labels.json.gz``  the exact labelled memberships a Hedonic detector returned, one row of labels per vertex
                             (``detector.final_memberships_path``). The cover merges labels whose members are the same,
                             so it is not the profile the detector reached; the ``labelled`` audit needs these rows.
                             Records written before this file existed keep them inline in ``detector.final_memberships``.
``vertex_ids.json.gz``       the analysis graph's vertices as SNAP ids, one per vertex id, with the graph's SHA-256.
                             One file per seed folder, shared by its methods (they all ran on the same graph).

The record's ``dataset_report.graph_sha256`` (``graph_sha256``) names the analysis graph exactly: an audit or a rescoring
checks the graph it loaded against it, not only the vertex and edge counts. Every file here is a gzip with no time stamp
in its header (neither time nor name), so the same answer is the same bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
LABELS_SUFFIX = ".labels.json.gz"
VERTEX_IDS_NAME = "vertex_ids.json.gz"
LABELS_FIELD = "final_memberships"
LABELS_PATH_FIELD = "final_memberships_path"


def graph_sha256(graph) -> str:
    """SHA-256 of the graph's sorted undirected edge list (int64 pairs, smaller end first) and its vertex count.

    The same bytes as ``cover_audit.graph_identity`` has always hashed, so the audits already recorded agree with it;
    numpy sorts the edges, so a graph of millions of edges takes about a second.
    """
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    edges.sort(axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return hashlib.sha256(np.ascontiguousarray(edges).tobytes() + str(graph.vcount()).encode()).hexdigest()


def write_json_gz(path: Path, value: Any) -> None:
    """Write ``value`` as gzipped JSON, atomically, with no time stamp in the gzip header (same value, same bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(json.dumps(value, separators=(",", ":")).encode("utf-8"))
    os.replace(tmp, path)


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- vertex ids
def snap_vertex_ids(dataset, network_root: str | Path | None = None) -> tuple[list[int] | None, str]:
    """The SNAP id of each vertex of the analysis graph, and how it was found; (None, why) when the loader lost them.

    The loaders number vertices three ways (``dataset_report.id_mapping_strategy``): a cached graph with a ``label``
    attribute keeps the SNAP id there (DBLP); a cached graph without one uses the SNAP id as the vertex index (Amazon,
    YouTube, LiveJournal), and the covered subgraph records the indices it kept; the streamed loader (Friendster)
    numbers the cover's vertices that appear in an edge in increasing SNAP id, so when every vertex of the raw cover
    file is in the graph, vertex i is the i-th smallest id of that file.
    """
    graph = dataset.graph
    report = dataset.report or {}
    strategy = str(report.get("id_mapping_strategy") or "")
    completion = report.get("ground_truth_completion") or {}
    kept = completion.get("original_vertex_ids") if completion.get("policy") == "covered-induced" else None
    if strategy.startswith("cached_vertex_label_attribute") and "label" in graph.vertex_attributes():
        return [int(label) for label in graph.vs["label"]], "vertex label attribute"
    if strategy.startswith("cached_identity_vertex_index"):
        ids = [int(v) for v in kept] if kept is not None else list(range(graph.vcount()))
        return ids, "vertex index (identity)" + (" of the covered subgraph's kept vertices" if kept is not None else "")
    if "streamed_covered_subgraph" in strategy and network_root is not None:
        ids = _streamed_ids(dataset, Path(network_root))
        if ids is not None:
            return ids, "sorted SNAP ids of the raw cover file (every one is in the graph)"
        return None, "streamed loader: some cover vertices have no edge, so the numbering is not recoverable here"
    return None, f"the loader ({strategy or 'unknown'}) keeps no SNAP ids"


def _streamed_ids(dataset, root: Path) -> list[int] | None:
    from hedonic.experiments.overlapping import snap

    try:
        spec = snap.SPECS[dataset.name]
        path = snap._first_existing(snap._dataset_dir(root, spec), spec.raw_cover_files.get(dataset.cover_variant, ()))
        if path is None:
            return None
        raw = snap._read_cover_text(path, wikipedia_categories=spec.name == "wikipedia")
    except Exception:  # noqa: BLE001 - without the raw file the ids are simply not recorded
        return None
    wanted = sorted({int(v) for community in raw for v in community})
    return wanted if len(wanted) == dataset.graph.vcount() else None


def write_vertex_ids(folder: Path, dataset, graph_sha: str, network_root: str | Path | None = None) -> Path:
    """``covers/<net>/vertex_ids.json.gz`` of a seed folder, written once (again only for another graph). Returns it."""
    path = folder / "covers" / dataset.name / VERTEX_IDS_NAME
    if path.is_file():
        try:
            if read_json_gz(path).get("graph_sha256") == graph_sha:
                return path
        except (OSError, EOFError, ValueError, AttributeError):
            pass
    ids, basis = snap_vertex_ids(dataset, network_root)
    report = dataset.report or {}
    completion = report.get("ground_truth_completion") or {}
    write_json_gz(path, {
        "schema": SCHEMA, "network": dataset.name, "cover_variant": dataset.cover_variant,
        "n": dataset.graph.vcount(), "m": dataset.graph.ecount(), "graph_sha256": graph_sha,
        "id_mapping_strategy": report.get("id_mapping_strategy"), "snap_ids_basis": basis, "snap_ids": ids,
        "source_graph_vertices": completion.get("source_graph_vertices"),
        "source_vertex_ids": completion.get("original_vertex_ids"),
    })
    return path


def record_report(report: dict, graph_sha: str, vertex_ids: str | None) -> dict:
    """The ``dataset_report`` a record keeps: the loader's report with the graph's SHA-256. With a saved id map, the
    covered subgraph's list of kept vertices lives in that file (``vertex_ids``), not in every record of the folder."""
    out = dict(report)
    out["graph_sha256"] = graph_sha
    completion = out.get("ground_truth_completion")
    if vertex_ids and isinstance(completion, dict) and "original_vertex_ids" in completion:
        completion = dict(completion)
        kept = completion.pop("original_vertex_ids")
        completion["original_vertex_ids_count"] = len(kept)
        completion["original_vertex_ids_in"] = vertex_ids
        out["ground_truth_completion"] = completion
    out["vertex_ids_path"] = vertex_ids
    return out


# --------------------------------------------------------------------------- the answer
def save_answer(folder: Path, network: str, runner: str, cover, detector: dict) -> str:
    """Write a unit's cover (and a Hedonic detector's labelled rows, taken out of ``detector``). Returns the cover path,
    relative to the seed folder, as the record names it."""
    cover_path = folder / "covers" / network / f"{runner}.json.gz"
    write_json_gz(cover_path, [sorted(int(v) for v in community) for community in cover])
    rows = detector.get(LABELS_FIELD)
    if rows:
        labels = folder / "covers" / network / f"{runner}{LABELS_SUFFIX}"
        write_json_gz(labels, [[int(label) for label in row] for row in rows])
        detector.pop(LABELS_FIELD)
        detector[LABELS_PATH_FIELD] = str(labels.relative_to(folder))
        detector["final_memberships_rows"] = len(rows)
    return str(cover_path.relative_to(folder))


def labels_path(record: dict) -> str | None:
    """The labelled rows' file a record names (relative to its seed folder), or None."""
    path = (record.get("detector") or {}).get(LABELS_PATH_FIELD)
    return str(path) if path else None


def has_labels(record: dict) -> bool:
    detector = record.get("detector") or {}
    return bool(detector.get(LABELS_FIELD) or detector.get(LABELS_PATH_FIELD))


def read_labels(record: dict, folder: Path) -> list[list[int]] | None:
    """The exact labelled memberships of a Hedonic unit: inline in an older record, else from its file."""
    detector = record.get("detector") or {}
    if detector.get(LABELS_FIELD):
        return detector[LABELS_FIELD]
    path = labels_path(record)
    return read_json_gz(folder / path) if path else None


def read_vertex_ids(folder: Path, network: str) -> dict | None:
    path = folder / "covers" / network / VERTEX_IDS_NAME
    try:
        return read_json_gz(path)
    except (OSError, EOFError, ValueError):
        return None
