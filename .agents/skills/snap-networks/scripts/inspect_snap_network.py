#!/usr/bin/env python3
"""Bounded, read-only inspection of the SNAP network archive.

By default this reads filenames, the SNAP ``# Nodes: / Edges:`` headers and a
few data rows of each raw file; it never loads a graph. ``--pairs`` adds, for
every network:cover pair hedonic knows, the two files its loader would read
(``snap.pair_files``: nothing is parsed or downloaded). ``--pickles`` loads
trusted ``.pkl`` caches of this archive only: unpickling runs code, and the
LiveJournal or Wikipedia pickles take GBs and minutes (run under guard.sh).

    python inspect_snap_network.py [ROOT] [--dataset DBLP] [--pairs] [--pickles] [--format json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Any

HEADER_RE = re.compile(r"\b(Nodes|Edges):\s*([\d,]+)", re.I)
RAW_SUFFIXES = (".txt.gz", ".tab", ".txt", ".cites", ".content")
# Folders of old experiment outputs inside the archive: never graph input.
OUTPUT_PARTS = {"DBLP_CLI", "results", "resolution_spectra", "resolution_f1", "resolution_f1_log"}


def _is_output(path: Path) -> bool:
    return any(part in OUTPUT_PARTS or part.endswith("_CLI") for part in path.parts)


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _raw_summary(path: Path, preview_limit: int = 3, header_lines: int = 50) -> dict[str, Any]:
    headers: dict[str, str] = {}
    preview: list[str] = []
    with _open_text(path) as handle:
        for i, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            for match in HEADER_RE.finditer(stripped):
                headers[match.group(1).lower()] = match.group(2)
            if not stripped.startswith("#"):
                preview.append(stripped[:240])
            if len(preview) >= preview_limit or i >= header_lines:
                break
    return {"path": str(path), "bytes": path.stat().st_size, "headers": headers, "preview": preview}


class _LegacyUnpickler(pickle.Unpickler):
    """Old caches name ``hedonic.game.HedonicGame``; read them as igraph graphs, as the loader does."""

    def find_class(self, module: str, name: str):
        if module == "hedonic.game" and name in {"HedonicGame", "Game"}:
            import igraph

            return igraph.Graph
        return super().find_class(module, name)


def _pickle_summary(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    try:
        with path.open("rb") as handle:
            obj = _LegacyUnpickler(handle).load()
        record["type"] = type(obj).__name__
        if hasattr(obj, "vcount") and hasattr(obj, "ecount"):
            record.update(n=int(obj.vcount()), m=int(obj.ecount()), directed=bool(obj.is_directed()),
                          graph_attributes=list(obj.attributes()), vertex_attributes=list(obj.vs.attributes()))
            if "label" in obj.vs.attributes() and obj.vcount():
                labels = [int(x) for x in obj.vs["label"]]
                record.update(vertex_label_min=min(labels), vertex_label_max=max(labels))
        elif isinstance(obj, (list, tuple)):
            sizes = [len(item) for item in obj if hasattr(item, "__len__")]
            record["length"] = len(obj)
            if sizes:
                record.update(community_size_min=min(sizes), community_size_max=max(sizes),
                              memberships=sum(sizes))
        elif isinstance(obj, (dict, set)):
            record["length"] = len(obj)
    except Exception as exc:  # a failed legacy load says nothing about the raw file
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _pairs(root: Path) -> list[dict[str, Any]]:
    """The files each known pair resolves to (hedonic's own resolution; reads nothing)."""
    try:
        from hedonic.experiments.overlapping import snap
    except ImportError as exc:
        return [{"error": f"hedonic is not importable: {exc}"}]
    rows = []
    for network, spec in snap.SPECS.items():
        for variant in spec.raw_cover_files:
            found = snap.pair_files(network, variant, root)
            row: dict[str, Any] = {"pair": f"{network}:{snap.cover_variant_label(variant)}"}
            if found is None:
                row["status"] = "missing"
            else:
                row.update(status="present", graph=str(found.graph), cover=str(found.cover_file),
                           bytes=found.graph.stat().st_size + found.cover_file.stat().st_size)
            rows.append(row)
    return rows


def inspect(root: Path, dataset: str | None, read_pickles: bool, pairs: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    dirs = [root / dataset] if dataset else sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    output: dict[str, Any] = {"root": str(root), "datasets": []}
    for directory in dirs:
        if not directory.is_dir() or _is_output(directory.relative_to(root)):
            continue
        files = sorted(p for p in directory.rglob("*") if p.is_file() and not _is_output(p.relative_to(root)))
        item: dict[str, Any] = {"name": directory.name, "file_count": len(files)}
        raw = [p for p in files if p.name.endswith(RAW_SUFFIXES)]
        item["raw_preview"] = [_raw_summary(p) for p in raw[:12]]
        if read_pickles:
            item["pickles"] = [_pickle_summary(p) for p in files if p.suffix == ".pkl"][:20]
        output["datasets"].append(item)
    if pairs:
        output["pairs"] = _pairs(root)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default="~/Databases/Hedonic/Networks", type=Path)
    parser.add_argument("--dataset", help="inspect one folder of the root, e.g. DBLP")
    parser.add_argument("--pairs", action="store_true", help="the files hedonic's loader reads for each pair")
    parser.add_argument("--pickles", action="store_true", help="load the trusted .pkl caches (slow for big networks)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    report = inspect(args.root, args.dataset, args.pickles, args.pairs)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"root: {report['root']}")
    for item in report["datasets"]:
        print(f"\n[{item['name']}] files={item['file_count']}")
        for raw in item["raw_preview"]:
            print(f"  raw: {raw['path']}  ({raw['bytes']:,} bytes)")
            if raw["headers"]:
                print(f"    headers: {raw['headers']}")
            for row in raw["preview"]:
                print(f"    sample: {row}")
        for cache in item.get("pickles", []):
            print(f"  pickle: {cache}")
    if "pairs" in report:
        print("\npairs (as hedonic's loader resolves them):")
        for row in report["pairs"]:
            if "error" in row:
                print(f"  {row['error']}")
            elif row["status"] == "missing":
                print(f"  {row['pair']:<22} missing")
            else:
                print(f"  {row['pair']:<22} {Path(row['graph']).name} + {Path(row['cover']).name}  ({row['bytes'] / 2**20:,.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
