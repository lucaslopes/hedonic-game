"""Illustration of the participation budget: a vertex in k communities gives each of them intensity 1/sqrt(k).

One panel per multiplicity k = 1..K. Panel k stacks k equal circles (one per label) into a pyramid standing on a ground
line shared by every panel, with its label below, all filled with the colour of the per-label intensity f = 1/sqrt(k), so the squared intensities always sum to one while each
label receives less as k grows. The figure is model-only: it reads no data.

    hedonic-exp participation-budget --output artifacts/overlapping/participation_budget/participation_budget.pdf
    hedonic-exp participation-budget --k-max 6 --width 3.3 --output fig.pdf fig.png
    hedonic-exp participation-budget --bars --k-max 124 --output bars.pdf   # one bar per k, single column

``--bars`` draws the same intensities as one bar chart instead: bar k has height 1/sqrt(k) and the colour of that
intensity in the same colour map.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from hedonic.experiments.config import OUTPUT_DIR, expand_path  # noqa: E402

DEFAULT_OUTPUT = Path(OUTPUT_DIR) / "overlapping" / "participation_budget" / "participation_budget.pdf"


def intensity(k: int) -> float:
    """The unit-l2 participation 1/sqrt(k) a vertex gives each of its k communities."""
    if k < 1:
        raise ValueError(f"a vertex is in at least one community, got k={k}")
    return 1.0 / math.sqrt(k)


def pyramid_positions(n: int, spacing: float = 1.0) -> list[tuple[float, float]]:
    """Centres of n circles stacked into a pyramid on the ground, bottom row first, bottom centres at y = 0.

    The bottom row holds the smallest b circles with b(b+1)/2 >= n; each row above holds one fewer and rests in the
    grooves of the row below (rows spacing*sqrt(3)/2 apart, adjacent centres `spacing` apart), and only the top row
    may be partial, filled from its centre (left of centre on a tie). Every row is centred on x = 0.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    base = 0
    while base * (base + 1) // 2 < n:
        base += 1
    positions: list[tuple[float, float]] = []
    row = 0
    while len(positions) < n:
        size = base - row
        count = min(size, n - len(positions))
        start = (size - count) // 2
        y = row * spacing * math.sqrt(3) / 2
        positions.extend(((col - (size - 1) / 2) * spacing, y) for col in range(start, start + count))
        row += 1
    return positions


def figure(k_max: int = 10, *, cmap: str = "YlGn", width: float = 7.0, height: float = 1.0,
           radius: float = 0.42, colorbar: bool = True) -> Figure:
    """One row of k_max panels; panel k shows k communities coloured by their intensity 1/sqrt(k).

    The colour scale is the intensity itself, from 0 to 1.
    """
    if k_max < 1:
        raise ValueError(f"k_max must be at least 1, got {k_max}")
    stacks = [pyramid_positions(k) for k in range(1, k_max + 1)]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)  # the colour is the intensity itself
    colours = plt.get_cmap(cmap)
    pad = radius + 0.12
    half = max(abs(x) for x, _ in stacks[-1]) + pad
    top = max(y for stack in stacks for _, y in stack) + pad
    fig, axes = plt.subplots(1, k_max, figsize=(width, height), squeeze=False, layout="constrained")
    for k, (ax, stack) in enumerate(zip(axes[0], stacks), start=1):
        f = intensity(k)
        for x, y in stack:
            ax.add_patch(Circle((x, y), radius, facecolor=colours(norm(f)), edgecolor="#444444", linewidth=0.5))
        ax.set_xlim(-half, half)
        ax.set_ylim(-radius, top)  # the ground: every bottom row touches it
        ax.set_aspect("equal", anchor="S")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set(color="#777777", linewidth=0.6)
        ax.set_xlabel(f"$k={k}$\n$1/\\sqrt{{{k}}}={f:.2f}$", fontsize=6.5, labelpad=2)
    if colorbar:
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=colours), ax=axes[0].tolist(), shrink=0.8, aspect=15,
                           pad=0.01)
        bar.set_label(r"$f_{vc}=1/\sqrt{k_v}$", fontsize=6.5)
        bar.ax.tick_params(labelsize=5.5, width=0.4, length=2)
        bar.outline.set_linewidth(0.4)
    return fig


def bar_figure(k_max: int = 124, *, cmap: str = "YlGn", width: float = 3.33, height: float = 1.6,
               colorbar: bool = False) -> Figure:
    """One bar per multiplicity k = 1..k_max, of height 1/sqrt(k), coloured by that intensity (1 darkest)."""
    if k_max < 1:
        raise ValueError(f"k_max must be at least 1, got {k_max}")
    ks = list(range(1, k_max + 1))
    values = [intensity(k) for k in ks]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)  # the colour is the intensity itself
    colours = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.bar(ks, values, width=1.0, color=[colours(norm(v)) for v in values], edgecolor="#444444", linewidth=0.15)
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of communities $k_v$", fontsize=7)
    ax.set_ylabel(r"$f_{vc}=1/\sqrt{k_v}$", fontsize=7)
    ax.tick_params(labelsize=6, width=0.4, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if colorbar:
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=colours), ax=ax, aspect=20, pad=0.01)
        bar.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hedonic-exp participation-budget", description=__doc__.split("\n\n")[0])
    parser.add_argument("--k-max", type=int, default=10, help="largest number of communities drawn (default 10)")
    parser.add_argument("--output", nargs="+", default=[str(DEFAULT_OUTPUT)], metavar="FILE",
                        help="figure file(s); the suffix picks the format (default %(default)s)")
    parser.add_argument("--cmap", default="YlGn", help="matplotlib colour map, dark = high intensity (default YlGn)")
    parser.add_argument("--width", type=float, default=7.0, help="figure width in inches (default 7.0)")
    parser.add_argument("--height", type=float, default=1.0, help="figure height in inches (default 1.0)")
    parser.add_argument("--no-colorbar", action="store_true", help="omit the intensity colour bar")
    parser.add_argument("--bars", action="store_true",
                        help="one bar chart (bar k = 1/sqrt(k)) instead of pyramids; no colour bar unless --colorbar")
    parser.add_argument("--colorbar", action="store_true", help="with --bars: add the intensity colour bar")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bars:
        fig = bar_figure(args.k_max, cmap=args.cmap, width=args.width, height=args.height, colorbar=args.colorbar)
    else:
        fig = figure(args.k_max, cmap=args.cmap, width=args.width, height=args.height, colorbar=not args.no_colorbar)
    for name in args.output:
        path = Path(expand_path(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"  wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
