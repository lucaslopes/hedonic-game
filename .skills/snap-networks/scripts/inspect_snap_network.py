#!/usr/bin/env python3
"""Bounded, read-only inspection of a SNAP-style network archive.

By default this reads filenames, compressed-file headers, and a few data rows.
Pass --pickles only for trusted local pickle caches; pickle loading can execute
arbitrary code and large graph caches may take time and memory to deserialize.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any


HEADER_RE = re.compile(r"\b(Nodes|Edges):\s*([\d,]+)", re.I)


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _raw_summary(path: Path, preview_limit: int = 3) -> dict[str, Any]:
    headers: dict[str, str] = {}
    preview: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            for match in HEADER_RE.finditer(stripped):
                headers[match.group(1).lower()] = match.group(2)
            if stripped.startswith("#"):
                continue
            preview.append(stripped[:240])
            if len(preview) >= preview_limit:
                break
    return {"path": str(path), "headers": headers, "preview": preview}


def _install_legacy_pickle_alias() -> None:
    """Allow inspection of old hedonic graph pickles when repo code is importable."""
    try:
        module = importlib.import_module("hedonic.Game")
    except ImportError:
        return
    sys.modules.setdefault("hedonic.game", module)
    if hasattr(module, "Game") and not hasattr(module, "HedonicGame"):
        module.HedonicGame = module.Game


def _pickle_summary(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    try:
        _install_legacy_pickle_alias()
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        record["type"] = type(obj).__name__
        if hasattr(obj, "vcount") and hasattr(obj, "ecount"):
            record.update(
                n=int(obj.vcount()),
                m=int(obj.ecount()),
                directed=bool(obj.is_directed()),
                graph_attributes=list(obj.attributes()),
                vertex_attributes=list(obj.vs.attributes()),
            )
            if "label" in obj.vs.attributes() and obj.vcount():
                labels = [int(x) for x in obj.vs["label"]]
                record["vertex_label_min"] = min(labels)
                record["vertex_label_max"] = max(labels)
        elif isinstance(obj, (list, tuple, dict, set)):
            record["length"] = len(obj)
            if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
                sizes = [len(item) for item in obj if hasattr(item, "__len__")]
                if sizes:
                    record.update(
                        community_size_min=min(sizes),
                        community_size_max=max(sizes),
                        communities_with_members=len(sizes),
                    )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def inspect(root: Path, dataset: str | None, read_pickles: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    dirs = [root / dataset] if dataset else sorted(p for p in root.iterdir() if p.is_dir())
    output: dict[str, Any] = {"root": str(root), "datasets": []}
    for directory in dirs:
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        item: dict[str, Any] = {"name": directory.name, "file_count": len(files)}
        raw_files = [
            p for p in files
            if p.name.endswith((".txt.gz", ".tab", ".txt", ".cites", ".content"))
            and "DBLP_CLI" not in p.parts
        ]
        item["raw_preview"] = [_raw_summary(p) for p in raw_files[:12]]
        if read_pickles:
            pickle_files = [
                p for p in files
                if p.suffix == ".pkl" and "DBLP_CLI" not in p.parts
            ]
            item["pickles"] = [_pickle_summary(p) for p in pickle_files[:20]]
        output["datasets"].append(item)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", default="~/Databases/Hedonic/Networks", type=Path
    )
    parser.add_argument(
        "--dataset", help="Inspect one immediate child directory, e.g. DBLP"
    )
    parser.add_argument("--pickles", action="store_true", help="Load trusted .pkl caches")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    report = inspect(args.root, args.dataset, args.pickles)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"root: {report['root']}")
    for item in report["datasets"]:
        print(f"\n[{item['name']}] files={item['file_count']}")
        for raw in item["raw_preview"]:
            print(f"  raw: {raw['path']}")
            if raw["headers"]:
                print(f"    headers: {raw['headers']}")
            for row in raw["preview"]:
                print(f"    sample: {row}")
        for cache in item.get("pickles", []):
            print(f"  pickle: {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
