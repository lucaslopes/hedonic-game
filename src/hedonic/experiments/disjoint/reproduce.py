"""End-to-end disjoint synthetic reproduction: sweep → CSV → paper figures.

Chains the CLI pieces needed to reproduce the PHYSA V1020 pipeline without
touching the archived ``.../V1020`` tree (writes under a caller-provided root
such as ``V1020_CLI``).

::

    hedonic-exp reproduce-disjoint --preset v1020-smoke \\
        --output_root .../V1020_CLI

    # Full grid (very large) + figures when CSV is ready:
    hedonic-exp reproduce-disjoint --preset v1020 \\
        --output_root .../V1020_CLI

    # Figures only from an existing archive CSV (read-only data):
    hedonic-exp reproduce-disjoint --plots-only \\
        --data .../V1020/resultados_ari.csv.gzip \\
        --output_root .../V1020_CLI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hedonic.experiments.config import SYNTHETIC_DIR
from hedonic.experiments.disjoint import data_loader, sbm_sweep
from hedonic.experiments.plots import paper_figures

ARCHIVED_V1020 = Path(
    "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
).resolve()


def _ok(result) -> bool:
    if result is None or result is True:
        return True
    if result is False:
        return False
    if isinstance(result, int):
        return result == 0
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Complete disjoint synthetic reproduction: "
            "SBM sweep → combined CSV → PHYSA paper figures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "pipeline:\n"
            "  1. hedonic-exp disjoint --preset … --output_root ROOT\n"
            "  2. hedonic-exp disjoint-load --results_folder ROOT/resultados --simple\n"
            "  3. hedonic-exp plots --data ROOT/resultados.csv.gzip --output_dir ROOT/figures\n"
            "\n"
            "examples:\n"
            "  hedonic-exp reproduce-disjoint --preset v1020-smoke \\\n"
            "      --output_root .../V1020_CLI\n"
            "  hedonic-exp reproduce-disjoint --plots-only \\\n"
            "      --data .../V1020/resultados_ari.csv.gzip \\\n"
            "      --output_root .../V1020_CLI --max_rows 50000\n"
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(sbm_sweep.PRESETS.keys()),
        default="v1020-smoke",
        help="Sweep preset (default: v1020-smoke)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Root for resultados/, figures/, persist/ "
            "(must NOT be the archived V1020 folder)"
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip sweep + load; only generate figures from --data",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="CSV/parquet for plots-only (or override CSV after sweep)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "svg", "png"],
        help="Figure format (default: pdf)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not cache figure intermediates",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Cap rows used for plotting (quick checks on full archive CSV)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="KDE workers for plots (-1 = all CPUs)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run sweep + CSV only (no figures)",
    )
    args = parser.parse_args(argv)

    root = Path(args.output_root).resolve()
    if root == ARCHIVED_V1020:
        print(
            "Refusing to use the archived V1020 folder as --output_root.\n"
            "Use e.g. .../Synthetic_Networks/V1020_CLI",
            file=sys.stderr,
        )
        return 1

    root.mkdir(parents=True, exist_ok=True)
    resultados_dir = root / "resultados"
    csv_path = root / "resultados.csv.gzip"
    figures_dir = root / "figures"

    if not args.plots_only:
        print("=" * 60)
        print(f"  [1/3] disjoint sweep  preset={args.preset}")
        print("=" * 60)
        ok = _ok(
            sbm_sweep.main(
                [
                    "--preset",
                    args.preset,
                    "--output_root",
                    str(root),
                ]
            )
        )
        if not ok:
            print("Sweep failed.", file=sys.stderr)
            return 1

        print("=" * 60)
        print("  [2/3] combine JSON → CSV")
        print("=" * 60)
        ok = _ok(
            data_loader.main(
                [
                    "--results_folder",
                    str(resultados_dir),
                    "--output",
                    str(csv_path),
                    "--simple",
                ]
            )
        )
        if not ok:
            print("Load/combine failed.", file=sys.stderr)
            return 1
        data_for_plots = str(csv_path)
    else:
        data_for_plots = args.data
        if not data_for_plots:
            # Default: archived ARI table (read-only) if present.
            for cand in (
                ARCHIVED_V1020 / "resultados_ari.csv.gzip",
                ARCHIVED_V1020 / "resultados.csv.gzip",
                SYNTHETIC_DIR / "resultados_ari.csv.gzip",
            ):
                if cand.is_file():
                    data_for_plots = str(cand)
                    break
        if not data_for_plots or not Path(data_for_plots).is_file():
            print(
                "plots-only requires --data pointing to an existing CSV/parquet.",
                file=sys.stderr,
            )
            return 1

    if args.data and args.plots_only is False and args.data:
        # Explicit override of plot input after sweep.
        data_for_plots = args.data

    if args.skip_plots:
        print("Skipping plots (--skip-plots).")
        print(f"CSV: {csv_path if csv_path.is_file() else data_for_plots}")
        return 0

    print("=" * 60)
    print("  [3/3] paper figures")
    print("=" * 60)
    plot_argv = [
        "--data",
        data_for_plots,
        "--output_dir",
        str(figures_dir),
        "--format",
        args.format,
        "--n_jobs",
        str(args.n_jobs),
    ]
    if args.no_persist:
        plot_argv.append("--no-persist")
    else:
        plot_argv.extend(["--persist_dir", str(root / "persist")])
    if args.max_rows is not None:
        plot_argv.extend(["--max_rows", str(args.max_rows)])

    ok = _ok(paper_figures.main(plot_argv))
    if not ok:
        print("Plots failed.", file=sys.stderr)
        return 1

    print()
    print("Reproduction complete.")
    print(f"  root:    {root}")
    if not args.plots_only:
        print(f"  results: {resultados_dir}")
        print(f"  csv:     {csv_path}")
    print(f"  figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
