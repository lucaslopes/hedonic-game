"""Durable exhaustive integrity grid for the overlapping Leiden binding.

The ``standard`` profile enumerates every nonempty labelled simple graph on
two through five vertices, every wrapper-valid two-label state at cap two,
and every binary disjoint control state.  Only violations are stored in full;
atomic shards bind every attempted case through deterministic digests.

Raw ``igraph.Graph.community_leiden`` calls are intentional here because this
is a binding-integrity experiment.  Ordinary experiments should continue to
use :meth:`hedonic.Game.community_hedonic`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hedonic.experiments.overlapping._integrity_cases import (
    DEFAULT_GAMMAS,
    PROTOCOL_NAME,
    expected_calls_per_graph,
    expected_total_calls,
    graph_edges,
    graph_mask_count,
    graph_pairs,
    iter_disjoint_states,
    iter_q2_states,
    valid_disjoint_state_count,
    valid_q2_state_count,
)
from hedonic.experiments.overlapping._integrity_durable import run


DEFAULT_OUTPUT = Path("artifacts/overlapping/integrity_grid")

__all__ = [
    "DEFAULT_GAMMAS",
    "PROTOCOL_NAME",
    "build_parser",
    "expected_calls_per_graph",
    "expected_total_calls",
    "graph_edges",
    "graph_mask_count",
    "graph_pairs",
    "iter_disjoint_states",
    "iter_q2_states",
    "main",
    "run",
    "valid_disjoint_state_count",
    "valid_q2_state_count",
]


def _parse_gammas(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("gammas must be finite comma-separated values")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpointed exhaustive tiny-graph integrity grid for raw "
            "overlapping Leiden versus the independent unit-l2 oracle."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--gammas", type=_parse_gammas, default=DEFAULT_GAMMAS)
    parser.add_argument("--batch-graphs", type=int, default=8)
    parser.add_argument(
        "--positive-budget",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include one-iteration multilevel quality-guard cases (standard default: on)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-shards", type=int, default=None)
    parser.add_argument(
        "--require-igraph-version",
        default="auto",
        help=(
            "required lucas-igraph distribution version (exact pin); auto "
            "requires 1.0.0.4 or later whenever positive-budget checks are "
            "enabled"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                key: manifest.get(key)
                for key in (
                    "protocol_name",
                    "run_identity",
                    "status",
                    "expected_shards",
                    "completed_shards",
                    "expected_calls",
                    "observed_calls",
                    "failure_count",
                    "maxima",
                )
            },
            indent=2,
        )
    )
    if manifest["status"] in {"complete", "preflight_complete"}:
        return 0
    if manifest["status"] == "incomplete":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
