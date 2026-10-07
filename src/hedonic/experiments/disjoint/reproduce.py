"""End-to-end disjoint synthetic reproduction: sweep → CSV → paper figures.

Chains the CLI pieces needed to reproduce the PHYSA V1020 pipeline without
touching the archived synthetic input tree (writes below the repository
artifact root by default).

::

    hedonic-exp reproduce-disjoint --preset v1020-smoke \\
        --output_root artifacts/disjoint/v1020

    # Full grid (very large) + figures when CSV is ready:
    hedonic-exp reproduce-disjoint --preset v1020 \\
        --output_root artifacts/disjoint/v1020

    # Figures only from an existing archive CSV (read-only data):
    hedonic-exp reproduce-disjoint --plots-only \\
        --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \\
        --output_root artifacts/disjoint/v1020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hedonic.experiments.config import (
    ARCHIVED_V1020_DIR,
    DISJOINT_ARTIFACTS_DIR,
    SYNTHETIC_DIR,
    ensure_not_archived_v1020,
    expand_path,
)
from hedonic.experiments.disjoint import data_loader, sbm_sweep
from hedonic.experiments.plots import paper_figures

# Backwards-compatible module alias used by callers/tests and help examples.
# The shared guard also rejects descendants of this path.
ARCHIVED_V1020 = ARCHIVED_V1020_DIR


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
            "      --output_root artifacts/disjoint/v1020\n"
            "  # No-write check before a long run\n"
            "  hedonic-exp reproduce-disjoint --preset v1020 \\\n"
            "      --output_root artifacts/disjoint/v1020 --preflight\n"
            "  hedonic-exp reproduce-disjoint --plots-only \\\n"
            "      --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \\\n"
            "      --output_root artifacts/disjoint/v1020 --max_rows 50000\n"
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
        default=str(DISJOINT_ARTIFACTS_DIR / "v1020"),
        help=(
            "Root for resultados/, figures/, persist/ "
            "(default: repository artifacts/disjoint/v1020; "
            "must NOT be the archived V1020 folder)"
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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume schema-valid sweep cells when possible (default: true; "
            "use --no-resume to force recomputation)."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the selected sweep and generated graph sizes, then exit "
            "without detectors, aggregation, or figures."
        ),
    )
    args = parser.parse_args(argv)

    root = expand_path(args.output_root).resolve()
    try:
        ensure_not_archived_v1020(root, label="output root")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.preflight:
        preflight_argv = [
            "--preset",
            args.preset,
            "--output_root",
            str(root),
            "--preflight",
        ]
        return 0 if _ok(sbm_sweep.main(preflight_argv)) else 1

    root.mkdir(parents=True, exist_ok=True)
    resultados_dir = root / "resultados"
    csv_path = root / "resultados.csv.gzip"
    figures_dir = root / "figures"

    if not args.plots_only:
        print("=" * 60)
        print(f"  [1/3] disjoint sweep  preset={args.preset}")
        print("=" * 60)
        sweep_argv = [
            "--preset",
            args.preset,
            "--output_root",
            str(root),
        ]
        if args.resume:
            sweep_argv.append("--resume")
        ok = _ok(sbm_sweep.main(sweep_argv))
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
        data_for_plots = str(expand_path(args.data)) if args.data else None
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
