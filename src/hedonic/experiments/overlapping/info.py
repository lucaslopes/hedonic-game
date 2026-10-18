"""``hedonic show``: what the disjoint benchmark can run.

    hedonic show disjoint             the disjoint presets, grids and estimates

Add ``--json`` for machine-readable output. This reads no data and downloads nothing. The overlap topics of earlier
releases (methods, networks, metrics, covers, table, figure) belong to hedonic 2.
"""

from __future__ import annotations

import argparse
import json

TOPICS = ("disjoint",)
HEDONIC_2_TOPICS = ("methods", "networks", "metrics", "covers", "table", "figure")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    removed = next((arg for arg in argv if arg in HEDONIC_2_TOPICS), None)
    if removed is not None:
        raise SystemExit(f"hedonic show {removed}: the overlapping benchmark and its SNAP data are hedonic 2; "
                         "hedonic 1 shows the disjoint benchmark only: hedonic show disjoint")
    p = argparse.ArgumentParser(prog="hedonic show", description="Show the disjoint SBM benchmark's presets.")
    p.add_argument("what", choices=TOPICS, nargs="?", default="disjoint")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from hedonic.experiments.disjoint import front

    if a.json:
        print(json.dumps({name: {**front.resolve(front.build_parser().parse_args(["--preset", name])),
                                 **front.plan(front.resolve(front.build_parser().parse_args(["--preset", name])))}
                          for name in front.PRESETS}, indent=2))
    else:
        print(front.render_presets())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
