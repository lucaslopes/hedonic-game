#!/usr/bin/env python3
"""Run the CoDeSEG authors' NcGame script as a cover-producing worker.

The upstream ``NCG.py`` is an experiment script rather than a command-line
program: it imports local evaluation modules, hard-codes the dataset loop,
and writes its predicted cover through ``draw.coms_to_csv``.  This adapter
keeps that detector implementation intact while supplying the small runtime
boundary needed by ``hedonic-exp overlapping-codeseg``.

Usage::

    python ncgame_worker.py \
        --upstream-root /path/to/CoDeSEG \
        --input graph.txt --ground-truth ground_truth.txt \
        --output communities.txt --dataset dblp

The input graph is the four-header-line SNAP edge-list emitted by the runner.
The output is one integer community per line.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Any


def _read_cover(path: Path) -> list[list[int]]:
    cover: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [int(token) for token in line.replace(",", " ").split()]
        if values:
            cover.append(values)
    return cover


def _install_runtime_stubs(ground_truth_path: Path, capture: dict[str, Any]) -> None:
    """Provide only the evaluation hooks imported by upstream ``NCG.py``."""

    ground_truth = _read_cover(ground_truth_path)
    keep_nodes = {node for community in ground_truth for node in community}

    comm_utils = types.ModuleType("comm_utils")

    def read_ture_cluster(*_args: Any, **_kwargs: Any):
        return ground_truth, keep_nodes, None

    def filter_nodes(communities: list[list[int]], nodes: set[int]):
        return [
            sorted({int(node) for node in community if int(node) in nodes})
            for community in communities
            if any(int(node) in nodes for node in community)
        ]

    comm_utils.read_ture_cluster = read_ture_cluster  # type: ignore[attr-defined]
    comm_utils.filter_nodes = filter_nodes  # type: ignore[attr-defined]
    comm_utils.DATASETS = {}  # type: ignore[attr-defined]

    draw = types.ModuleType("draw")

    def coms_to_csv(communities: list[list[int]], *_args: Any, **_kwargs: Any):
        capture["communities"] = communities

    draw.coms_to_csv = coms_to_csv  # type: ignore[attr-defined]

    onmi = types.ModuleType("onmi")
    onmi.overlapping_normalized_mutual_information = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]
    xmeasures = types.ModuleType("xmeasures")
    xmeasures.f1 = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]

    sys.modules.update(
        {
            "comm_utils": comm_utils,
            "draw": draw,
            "onmi": onmi,
            "xmeasures": xmeasures,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--stop-criterion", type=float, default=0.01)
    parser.add_argument("--similarity-type", default="HP")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    upstream_root = args.upstream_root.expanduser().resolve()
    code_dir = upstream_root / "code_py"
    ncg_path = code_dir / "NCG.py"
    if not ncg_path.is_file():
        raise SystemExit(f"upstream NcGame script is missing: {ncg_path}")

    args.input = args.input.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # NCG.py writes its own diagnostic files one level above the current
    # directory.  The runner gives workers an isolated temporary cwd.
    (Path.cwd().parent / "eva_txt" / "ncg").mkdir(parents=True, exist_ok=True)

    capture: dict[str, Any] = {}
    _install_runtime_stubs(args.ground_truth, capture)
    sys.path.insert(0, str(code_dir))
    ncg = importlib.import_module("NCG")
    ncg.DETECT_COMMUNITIES(
        str(args.input),
        str(args.ground_truth),
        str(args.dataset),
        stopCRITERION=float(args.stop_criterion),
        similarity_type=str(args.similarity_type),
    )
    communities = capture.get("communities")
    if not isinstance(communities, list) or not communities:
        raise SystemExit("upstream NcGame did not emit a non-empty cover")
    with args.output.open("w", encoding="utf-8") as stream:
        for community in communities:
            values = [str(int(node)) for node in community]
            if values:
                stream.write(" ".join(values) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
