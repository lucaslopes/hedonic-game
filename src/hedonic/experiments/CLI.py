"""CLI entrypoint for hedonic experiments.

Install with experiments extra, then::

    hedonic-exp --help
    hedonic-exp list
    hedonic-exp smoke
    hedonic-exp disjoint --smoke --output_root artifacts/disjoint/smoke
    hedonic-exp disjoint --preset v1020 --preflight --output_root artifacts/disjoint/v1020
    hedonic-exp overlapping-small
    hedonic-exp overlapping-dnn --output artifacts/overlapping/dnn_certificate/dnn_certificate.json
    hedonic-exp overlapping-dnn-rational --exact-only
    hedonic-exp overlapping-oracle --list-examples
    hedonic-exp overlapping-oracle --proof-check
    hedonic-exp overlapping-oracle --fee-check
    hedonic-exp overlapping-oracle --optimization-check
    hedonic-exp overlapping-oracle --random-cases 300 --token-cases 100
    hedonic-exp overlapping-oracle --native-differential
    hedonic-exp overlapping-integrity --profile smoke
    hedonic-exp overlapping-certificate-reconcile
    hedonic-exp overlapping-controlled --smoke --output artifacts/overlapping/controlled_overlap/controlled_overlap.json
    hedonic-exp overlapping-lfr --profile smoke --output-dir artifacts/overlapping/overlap_lfr/smoke
    hedonic-exp overlapping-lfr --profile pilot --dry-run --output-dir artifacts/overlapping/overlap_lfr/pilot
    hedonic-exp overlapping-tracking --profile smoke --output-dir artifacts/overlapping/tracking_triangle/smoke
    hedonic-exp overlapping-tracking --profile standard --preflight --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json --output-dir artifacts/overlapping/tracking_triangle/standard
    hedonic-exp overlapping-baselines --profile smoke --output artifacts/overlapping/baselines/smoke.json
    hedonic-exp overlapping-resource-envelope --profile smoke --output-dir artifacts/overlapping/resource_envelope/smoke
    hedonic-exp overlapping-subgraph --levels 1 --n_communities 5
    hedonic-exp overlapping-full --n_iterations -1
    hedonic-exp overlapping-scale --smoke --output_dir artifacts/overlapping/complexity_scale/smoke
    hedonic-exp overlapping-resolution --smoke --singleton-mode both --output_dir artifacts/overlapping/resolution_f1/smoke
    hedonic-exp overlapping-full-snap --preflight --output-dir artifacts/overlapping/full-snap-v1/preflight
    hedonic-exp overlapping-benchmark --profile smoke --output_dir artifacts/overlapping/snap_benchmark/smoke
    hedonic-exp overlapping-benchmark --list-methods
    hedonic-exp overlapping-codeseg --datasets dblp --max-nodes 5000 --output-dir artifacts/overlapping/codeseg/dblp-local
    hedonic-exp overlapping-reproduce --profile smoke
    hedonic-exp overlapping-reproduce --profile full --datasets dblp
    hedonic-exp codeseg-setup --all --build-native
    hedonic-exp codeseg-doctor --require-all
    hedonic-exp overlapping-gt-robustness --smoke --output-dir artifacts/overlapping/ground_truth_robustness_v3/smoke
    hedonic-exp disjoint-load --results_folder /path/to/jsons
    hedonic-exp plots --smoke --output_dir artifacts/disjoint/figures_smoke
    hedonic-exp reproduce-disjoint --preset v1020-smoke --output_root artifacts/disjoint/v1020

**Agents:** when you add, remove, rename, or change the CLI of any experiment
module under ``hedonic.experiments``, you **must** update this file (registry +
help text) and the CLI section in ``AGENTS.md`` in the same change.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Command:
    """Registered experiment subcommand."""

    name: str
    module: str
    attr: str
    summary: str
    needs_data: str | None = None  # "dblp" | "synthetic" | None
    has_argparse_help: bool = True


# ---------------------------------------------------------------------------
# Registry — single source of truth for subcommands.
# Add new experiments here AND document them in AGENTS.md.
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Command] = {
    "smoke": Command(
        name="smoke",
        module="",  # handled specially
        attr="",
        summary="Isolated local check (no DBLP): small graphs + tiny SBM sweep",
        needs_data=None,
        has_argparse_help=False,
    ),
    "disjoint": Command(
        name="disjoint",
        module="hedonic.experiments.disjoint.sbm_sweep",
        attr="main",
        summary=(
            "SBM parameter sweep for disjoint community detection "
            "(presets: v1020, v1020-smoke)"
        ),
        needs_data="synthetic",
    ),
    "disjoint-load": Command(
        name="disjoint-load",
        module="hedonic.experiments.disjoint.data_loader",
        attr="main",
        summary="Combine disjoint JSON results into a gzipped CSV",
        needs_data="synthetic",
    ),
    "overlapping-small": Command(
        name="overlapping-small",
        module="hedonic.experiments.overlapping.small_graphs",
        attr="main",
        summary="Quick overlapping smoke tests on small graphs (no data dirs)",
        needs_data=None,
        has_argparse_help=False,
    ),
    "overlapping-controlled": Command(
        name="overlapping-controlled",
        module="hedonic.experiments.overlapping.controlled_overlap",
        attr="main",
        summary=(
            "LFR-derived controlled overlap: cap, initialization, phase, "
            "and resolution ablations (not canonical overlapping LFR)"
        ),
        needs_data=None,
    ),
    "overlapping-lfr": Command(
        name="overlapping-lfr",
        module="hedonic.experiments.overlapping.overlap_lfr",
        attr="main",
        summary=(
            "Planted-overlap study: smoke compatibility fixture; official "
            "LFR binary required for pilot/standard"
        ),
        needs_data=None,
    ),
    "overlapping-tracking": Command(
        name="overlapping-tracking",
        module="hedonic.experiments.overlapping.tracking",
        attr="main",
        summary=(
            "Mirror/one-sweep/local/multiphase noisy-cover tracking "
            "and graph-level robustness/recovery triangle"
        ),
        needs_data=None,
    ),
    "overlapping-baselines": Command(
        name="overlapping-baselines",
        module="hedonic.experiments.overlapping.baselines",
        attr="main",
        summary=(
            "Bounded SLPA/DEMON/KCP adapters, scoped Chen replica, and "
            "trivial controls (external SLPA/DEMON required for standard)"
        ),
        needs_data=None,
    ),
    "overlapping-resource-envelope": Command(
        name="overlapping-resource-envelope",
        module="hedonic.experiments.overlapping.resource_envelope",
        attr="main",
        summary=(
            "Original/token/audit resource envelope with RSS, timings, "
            "projection and capacity outcomes"
        ),
        needs_data=None,
    ),
    "overlapping-dnn": Command(
        name="overlapping-dnn",
        module="hedonic.experiments.overlapping.dnn_certificate",
        attr="main",
        summary=(
            "Locked tiny-graph exact-cover enumeration + DNN SDP certificate diagnostic"
        ),
        needs_data=None,
    ),
    "overlapping-dnn-rational": Command(
        name="overlapping-dnn-rational",
        module="hedonic.experiments.overlapping.dnn_rational_certificate",
        attr="main",
        summary=(
            "Versioned exact rational diagonally-dominant DNN certificates "
            "+ optional numerical calibration; candidate trace audits label-reducing ties"
        ),
        needs_data=None,
    ),
    "overlapping-oracle": Command(
        name="overlapping-oracle",
        module="hedonic.experiments.overlapping.unit_l2_oracle",
        attr="main",
        summary=(
            "Independent unit-ℓ₂ prefix oracle, CE1–CE7, CE5 token "
            "normalization, collision lists, --proof-check for numbered "
            "propositions, --fee-check for the separate fee-v1 objective, and "
            "--optimization-check for cached/reference equivalence; optional "
            "native differential"
        ),
        needs_data=None,
    ),
    "overlapping-integrity": Command(
        name="overlapping-integrity",
        module="hedonic.experiments.overlapping.integrity_grid",
        attr="main",
        summary=(
            "Checkpointed exhaustive tiny-graph native/oracle integrity grid "
            "(--profile smoke|standard; resumable shards, native-library fingerprints "
            "and versioned diagnostic trace)"
        ),
        needs_data=None,
    ),
    "overlapping-subgraph": Command(
        name="overlapping-subgraph",
        module="hedonic.experiments.overlapping.dblp_subgraph",
        attr="main",
        summary="DBLP L-hop subgraph overlapping experiment",
        needs_data="dblp",
    ),
    "overlapping-full": Command(
        name="overlapping-full",
        module="hedonic.experiments.overlapping.dblp_full",
        attr="main",
        summary="Full DBLP overlapping experiment (+ optional resolution sweep)",
        needs_data="dblp",
    ),
    "overlapping-scale": Command(
        name="overlapping-scale",
        module="hedonic.experiments.overlapping.complexity_scale",
        attr="main",
        summary=(
            "Wallclock scaling of overlapping hedonic as subnetworks grow "
            "(local_move_only T/F vs size; timeout stop + plot)"
        ),
        needs_data="dblp",
    ),
    "overlapping-resolution": Command(
        name="overlapping-resolution",
        module="hedonic.experiments.overlapping.resolution_f1",
        attr="main",
        summary=(
            "Full-DBLP overlap metrics vs resolution for multi-phase hedonic; "
            "cache-only rescoring, singleton modes, optional sampled Omega"
        ),
        needs_data="dblp",
    ),
    "overlapping-full-snap": Command(
        name="overlapping-full-snap",
        module="hedonic.experiments.overlapping.full_snap",
        attr="main",
        summary="Separate full-snap-v1: complete raw graphs, unsupervised variants, resource/provenance records",
        needs_data="snap-networks",
    ),
    "overlapping-benchmark": Command(
        name="overlapping-benchmark",
        module="hedonic.experiments.overlapping.benchmark",
        attr="main",
        summary=(
            "Resumable Amazon/DBLP/LiveJournal/YouTube/Wikipedia overlap "
            "benchmark (hedonic + CPM/DEMON; optional AGMfit/Link/MMSB/Infomap)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-codeseg": Command(
        name="overlapping-codeseg",
        module="hedonic.experiments.overlapping.codeseg_reproduction",
        attr="main",
        summary=(
            "CoDeSEG WWW'25 nine-method SNAP overlap reproduction plus "
            "community_hedonic comparator (shared cover metrics + LFK-ONMI)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-reproduce": Command(
        name="overlapping-reproduce",
        module="hedonic.experiments.overlapping.turnkey",
        attr="main",
        summary=(
            "One-command SNAP setup plus the complete registered method, "
            "accuracy, runtime, and resource benchmark"
        ),
        needs_data="snap-networks",
    ),
    "codeseg-setup": Command(
        name="codeseg-setup",
        module="hedonic.experiments.overlapping.codeseg_setup",
        attr="setup_main",
        summary=(
            "Lazy-download/cache SNAP data and prepare the nine CoDeSEG "
            "method runtimes"
        ),
        needs_data="snap-networks",
    ),
    "codeseg-doctor": Command(
        name="codeseg-doctor",
        module="hedonic.experiments.overlapping.codeseg_setup",
        attr="doctor_main",
        summary="Read-only readiness audit for CoDeSEG data and method runtimes",
        needs_data="snap-networks",
    ),
    "overlapping-gt-robustness": Command(
        name="overlapping-gt-robustness",
        module="hedonic.experiments.overlapping.ground_truth_robustness",
        attr="main",
        summary=(
            "Ground-truth overlap robustness plus GT-seeded local/multi-phase "
            "equilibrium search"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-gt-spectrum": Command(
        name="overlapping-gt-spectrum",
        module="hedonic.experiments.overlapping.gt_spectrum",
        attr="main",
        summary=(
            "SNAP ground-truth robustness spectrum: stable-vertex fraction vs resolution for "
            "every locally available overlapping cover, plus seeded GT-started local moving"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-audit": Command(
        name="overlapping-audit",
        module="hedonic.experiments.overlapping.protocol",
        attr="main",
        summary="Read-only reconciliation of the 125 locked paper conditions",
        needs_data=None,
    ),
    "overlapping-certificate-reconcile": Command(
        name="overlapping-certificate-reconcile",
        module="hedonic.experiments.overlapping.certificate_reconcile",
        attr="main",
        summary=(
            "Detector-free 3840/864/125/tiny ledger classification "
            "(status_coverage.csv, certificate_manifest.json, displayed TeX "
            "fragments; optional --replay-dnn / --replay-gt / --replay-paper; "
            "--gate-h-check verifies frozen locks, numerical/rational dual "
            "witnesses, proofs, wrapper/metric contracts, and TeX/PDF hashes)"
        ),
        needs_data=None,
    ),
    "reproduce-overlapping-paper": Command(
        name="reproduce-overlapping-paper",
        module="hedonic.experiments.overlapping.reproduce_paper",
        attr="main",
        summary=(
            "TOML + tmux full SNAP paper reproduction: safe parallel shards "
            "→ merged caches/plots/tables → manuscript PDF"
        ),
        needs_data="snap-networks",
    ),
    "plots": Command(
        name="plots",
        module="hedonic.experiments.plots.paper_figures",
        attr="main",
        summary=(
            "PHYSA V1020 paper figures from disjoint CSV "
            "(gt_robustness, noise, n_communities, acc_*)"
        ),
        needs_data="synthetic",
    ),
    "reproduce-disjoint": Command(
        name="reproduce-disjoint",
        module="hedonic.experiments.disjoint.reproduce",
        attr="main",
        summary=(
            "End-to-end disjoint reproduction: sweep → CSV → paper figures "
            "(presets: v1020, v1020-smoke)"
        ),
        needs_data="synthetic",
    ),
    "info": Command(
        name="info",
        module="hedonic.experiments.overlapping.info",
        attr="main",
        summary=(
            "Show the screened methods (ready / installable / missing tools), the SNAP "
            "networks (size, on-disk copy, download size), and the accuracy metrics "
            "(also `hedonic show`)"
        ),
        needs_data=None,
    ),
    "quickstart": Command(
        name="quickstart",
        module="hedonic.experiments.overlapping.quickstart",
        attr="main",
        summary=(
            "Tiny SNAP com-DBLP benchmark (one seed): HOC-local vs CoDeSEG, all "
            "accuracy metrics; fetches data and builds CoDeSEG on first use "
            "(also `hedonic run exp`)"
        ),
        needs_data="snap-networks",
    ),
}

SUBCOMMANDS: tuple[str, ...] = tuple(COMMANDS.keys())


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return 0

    if argv[0] in ("-V", "--version"):
        print(_package_version())
        return 0

    if argv[0] == "list":
        _print_command_list()
        return 0

    command = argv[0]
    rest = argv[1:]

    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `hedonic-exp list` or `hedonic-exp --help`.", file=sys.stderr)
        return 2

    if rest and rest[0] in ("-h", "--help"):
        return _dispatch_help(command)

    return _dispatch(command, rest)


# ``hedonic`` console script: a short, task-oriented front door.
#   hedonic run exp [flags]                      start a benchmark (wizard without flags)
#   hedonic run list|status|attach|logs|stop|resume [NAME]   manage durable runs
RUN_ALIASES: dict[str, str] = {"exp": "quickstart", "paper": "quickstart", "spectrum": "overlapping-gt-spectrum"}
RUN_VERBS = ("list", "status", "attach", "logs", "stop", "resume")


def hedonic_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[:2] in (["run", "-h"], ["run", "--help"]):
        print(
            "usage: hedonic run exp [options]      start a benchmark (interactive wizard without options)\n"
            "       hedonic run paper              reproduce the paper experiment (= run exp --config paper)\n"
            "       hedonic run spectrum           SNAP ground-truth robustness spectrum of the local covers (wizard without options)\n"
            "       hedonic run list               runs and their progress\n"
            "       hedonic run status [NAME]      snapshot of a run (default: latest)\n"
            "       hedonic run attach [NAME]      live view of a running benchmark\n"
            "       hedonic run logs [NAME]        raw output (tmux pane or log file)\n"
            "       hedonic run stop NAME          stop a run (runs survive Ctrl-C and closed terminals)\n"
            "       hedonic run resume [NAME]      continue an interrupted run; finished runs are reused\n"
            "       hedonic run                    your favourite config (hedonic config set run NAME), else the wizard\n"
            "       hedonic guide [TOPIC]          a short tour of the CLI and suggested next steps\n"
            "       hedonic update [--yes]         compare with the latest release on PyPI; upgrade only when asked\n"
            "       hedonic config [set KEY VALUE]  per-user settings (~/.config/hedonic/config.toml)\n"
            "       hedonic show methods           methods: ready / installed on first use / missing tools\n"
            "       hedonic show networks          SNAP networks: vertices, edges, on disk, --remote sizes\n"
            "       hedonic show covers            graph/cover pairs found in the local SNAP cache (present / missing / unsupported)\n"
            "       hedonic show metrics           what each accuracy metric measures\n\n"
            "`hedonic run exp --help` lists every option. Every experiment is also available through\n"
            "`hedonic-exp` (`hedonic-exp list`)."
        )
        return 0
    if argv[0] in ("-V", "--version"):
        print(_package_version())
        return 0
    if argv[0] == "guide":
        from hedonic.experiments.overlapping import guide

        return guide.main(argv[1:])
    if argv[0] == "update":
        from hedonic.experiments.overlapping import selfupdate

        return selfupdate.main(argv[1:])
    if argv[0] == "config":
        from hedonic.experiments.overlapping import userconfig

        return userconfig.main(argv[1:])
    if argv == ["run"] or (argv[0] == "run" and len(argv) > 1 and argv[1].startswith("-")):
        # bare `hedonic run`: the favourite config from ~/.config/hedonic/config.toml, else the wizard
        from hedonic.experiments.overlapping import userconfig

        favourite = userconfig.favourite()
        extra = argv[1:]
        if favourite:
            return _dispatch("quickstart", ["--config", favourite, "--confirm", *extra])
        return _dispatch("quickstart", extra)
    if argv[0] == "show":
        from hedonic.experiments.overlapping import info

        return info.main(argv[1:])
    if argv[0] != "run" or len(argv) < 2 or (argv[1] not in RUN_ALIASES and argv[1] not in RUN_VERBS):
        print(f"Unknown command: {' '.join(argv)}. Try `hedonic --help`.", file=sys.stderr)
        return 2
    if argv[1] in RUN_VERBS:
        from hedonic.experiments.overlapping import runmanager

        return runmanager.dispatch(argv[1:])
    command, rest = RUN_ALIASES[argv[1]], argv[2:]
    if argv[1] == "spectrum":  # SNAP ground-truth robustness spectrum through the durable front door
        from hedonic.experiments.overlapping import gt_spectrum

        return gt_spectrum.front_main(rest)
    if argv[1] == "paper":  # the paper's reproducible experiment: `hedonic run exp --config paper`
        rest = ["--config", "paper", *rest]
    if rest and rest[0] in ("-h", "--help"):
        return _dispatch_help(command)
    return _dispatch(command, rest)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return f"hedonic {version('hedonic')}"
    except Exception:
        return "hedonic (version unknown)"


def _print_root_help() -> None:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp",
        description=(
            "Reproduce hedonic experiments: disjoint SBM sweeps, "
            "overlapping DBLP/SNAP runs, and isolated smoke checks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples_epilog(),
    )
    parser.add_argument(
        "command",
        choices=(*SUBCOMMANDS, "list"),
        help="Experiment or meta-command to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected experiment",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=_package_version(),
    )
    parser.print_help()
    print()
    _print_command_list()


def _examples_epilog() -> str:
    return """\
examples:
  # Isolated local check (no databases required)
  hedonic-exp smoke
  hedonic-exp overlapping-small
  hedonic-exp overlapping-dnn --output artifacts/overlapping/dnn_certificate/dnn_certificate.json
  hedonic-exp overlapping-dnn --list-instances
  hedonic-exp overlapping-dnn-rational \
      --instances collision_path3,duplicate_cap4 --exact-only
  hedonic-exp overlapping-dnn-rational --exact-only --require-debug-trace \
      --output artifacts/overlapping/dnn_rational_certificate/candidate-1.0.0.5.json
  hedonic-exp overlapping-oracle --list-examples
  hedonic-exp overlapping-oracle --proof-check
  hedonic-exp overlapping-oracle --fee-check
  hedonic-exp overlapping-oracle --random-cases 300 --token-cases 100
  hedonic-exp overlapping-oracle --native-differential
  hedonic-exp overlapping-integrity --profile smoke \\\
      --output-dir artifacts/overlapping/integrity_grid/smoke
  hedonic-exp overlapping-integrity --profile standard --resume \\\
      --require-igraph-version 1.0.0.4 \\\
      --output-dir artifacts/overlapping/integrity_grid/v1
  hedonic-exp overlapping-certificate-reconcile \\
      --output-dir artifacts/evidence/overlapping_communities
  hedonic-exp overlapping-certificate-reconcile \\
      --output-dir artifacts/evidence/overlapping_communities \\
      --replay-dnn --replay-gt --replay-paper
  hedonic-exp overlapping-certificate-reconcile \\
      --output-dir artifacts/evidence/overlapping_communities \\
      --gate-h-check
  hedonic-exp overlapping-controlled --smoke \
      --output artifacts/overlapping/controlled_overlap/controlled_overlap.json
  # Planted-overlap benchmark (official binary required for pilot/standard;
  # smoke is an explicitly labelled compatibility fixture)
  hedonic-exp overlapping-lfr --profile smoke \
      --output-dir artifacts/overlapping/overlap_lfr/smoke
  hedonic-exp overlapping-lfr --profile standard --dry-run \
      --output-dir artifacts/overlapping/overlap_lfr/standard
  # Tracking triangle: Mirror, one-sweep controls, local and multiphase
  hedonic-exp overlapping-tracking --profile smoke \
      --output-dir artifacts/overlapping/tracking_triangle/smoke
  hedonic-exp overlapping-tracking --profile standard --preflight \
      --graph-ledger artifacts/overlapping/overlap_lfr/standard/graphs.json \
      --output-dir artifacts/overlapping/tracking_triangle/standard
  # Fair baseline adapter smoke (SLPA, DEMON, KCP, Chen-style, controls)
  hedonic-exp overlapping-baselines --profile smoke \
      --output artifacts/overlapping/baselines/smoke.json
  # Original/token/audit resource envelope; standard is prospective
  hedonic-exp overlapping-resource-envelope --profile smoke \
      --output-dir artifacts/overlapping/resource_envelope/smoke
  hedonic-exp disjoint --smoke --output_root artifacts/disjoint/smoke

  # Tiny V1020-compatible structural smoke (safe output root)
  hedonic-exp disjoint --preset v1020-smoke \\
      --output_root artifacts/disjoint/v1020

  # Full PHYSA V1020 grid (very large — do not write into archived V1020)
  hedonic-exp disjoint --preset v1020 \\
      --output_root artifacts/disjoint/v1020 --resume

  # No-write readiness check for a targeted recovery task
  hedonic-exp disjoint --preset v1020 \\
      --max_n_nodes 1020 --n_communities 5 --seeds 3 \\
      --p_in 0.01 --difficulty 0.30 \\
      --noises 0.10 0.25 0.50 0.75 1.00 --partition_seeds 10 \\
      --output_root artifacts/disjoint/v1020-recovery --preflight

  # Ad-hoc disjoint SBM sweep
  hedonic-exp disjoint --folder_name exp --max_n_nodes 60 \\
      --n_communities 2 --seeds 42 --p_in 0.1 --difficulty 0.5

  # Load disjoint JSON results → CSV
  hedonic-exp disjoint-load --results_folder /path/to/resultados --simple
  hedonic-exp disjoint-load \\
      --results_folder artifacts/disjoint/v1020/resultados \\
      --output artifacts/disjoint/v1020/resultados.csv.gzip --simple

  # Paper figures (same stems as archived V1020/figures/)
  hedonic-exp plots --smoke --output_dir artifacts/disjoint/figures_smoke
  hedonic-exp plots --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \\
      --output_dir artifacts/disjoint/v1020/figures --format pdf

  # Complete disjoint pipeline (sweep → CSV → figures)
  hedonic-exp reproduce-disjoint --preset v1020-smoke \\
      --output_root artifacts/disjoint/v1020
  hedonic-exp reproduce-disjoint --plots-only \\
      --data ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados_ari.csv.gzip \\
      --output_root artifacts/disjoint/v1020 --max_rows 50000

  # Overlapping on DBLP (set HEDONIC_DBLP_DIR if needed)
  # Defaults: n_iterations=-1 (native no-change stop; not a certificate),
  # max_memberships=max per-node GT memberships
  hedonic-exp overlapping-subgraph --levels 1 --n_communities 5 \\
      --methods leiden,hedonic_v1 --output artifacts/overlapping/dblp_subgraph/smoke.json
  hedonic-exp overlapping-full --resolution 1e-4 --output artifacts/overlapping/dblp_full/results.json

  # Reproducible five-network SNAP benchmark (three multi-phase hedonic
  # density variants plus CPM/DEMON; no database required for smoke)
  hedonic-exp overlapping-benchmark --profile smoke \\
      --output_dir artifacts/overlapping/snap_benchmark/smoke
  # Standard profile uses deterministic 3,000-node induced subgraphs; top5000
  # is unavailable for Wikipedia and is recorded as an intentional skip.
  hedonic-exp overlapping-benchmark --datasets amazon,dblp,livejournal,youtube,wikipedia \\
      --cover top5000 --profile standard \\
      --output_dir artifacts/overlapping/snap_benchmark
  # Inspect every registered detector and its pinned dependency availability.
  hedonic-exp overlapping-benchmark --list-methods

  # CoDeSEG WWW'25 nine-method protocol plus the local community_hedonic
  # comparator. This bounded DBLP invocation is a local validation run;
  # remove --max-nodes for the full archive.
  hedonic-exp overlapping-codeseg --datasets dblp --max-nodes 5000 \\
      --output-dir artifacts/overlapping/codeseg/dblp-local
  hedonic-exp overlapping-codeseg --preflight --datasets amazon,youtube,dblp,livejournal,orkut,friendster,wikipedia
  # Turnkey archive-free validation (~1,000-node AGMfit-like fixture; all methods)
  hedonic-exp overlapping-reproduce --profile smoke
  # Full DBLP SNAP run; setup/download/build is automatic and resumable
  hedonic-exp overlapping-reproduce --profile full --resume
  # Prepare the optional catalogue cache, isolated CDlib runtime, upstream
  # checkouts, and a machine-readable setup manifest.
  hedonic-exp codeseg-setup --all --build-native
  hedonic-exp codeseg-doctor --require-all

  # Robustness of supplied metadata covers and equilibria seeded from their
  # complete memberships (separate from the locked paper benchmark).
  hedonic-exp overlapping-gt-robustness --smoke \\
      --output-dir artifacts/overlapping/ground_truth_robustness_v3/smoke
  # Canonical v3: four eligible top5000 covers, target labeled-incidence
  # distances 0/.5/2/5%, and 3,840 distinct cells.
  hedonic-exp overlapping-gt-robustness \\
      --config configs/overlapping-ground-truth.toml

  # SNAP ground-truth robustness spectrum (separately versioned; v3 untouched):
  # discover the local cache, dry-run, then run every available graph/cover pair.
  hedonic-exp overlapping-gt-spectrum --discover
  hedonic-exp overlapping-gt-spectrum --dry-run --inspect
  hedonic-exp overlapping-gt-spectrum --profile smoke \\
      --output-dir artifacts/overlapping/gt_spectrum_smoke
  hedonic-exp overlapping-gt-spectrum --config configs/overlapping-gt-spectrum.toml --resume
  hedonic run spectrum                     # the same study through the durable `hedonic` front door

  # Read-only reconciliation of the 125 locked paper conditions.
  hedonic-exp overlapping-audit --output artifacts/evidence/overlapping_communities/protocol_audit.json

  # Full overlapping-paper protocol: preflights graph/cover/method memory,
  # serializes LiveJournal, and writes a plan without launching detectors.
  uv run hedonic-exp reproduce-overlapping-paper --dry-run
  # main.tex compiles only after every expected full-protocol record completes.
  uv run hedonic-exp reproduce-overlapping-paper
  tmux attach -t hedonic-overlapping-paper

  # Complexity scale: size vs wallclock (local-moving vs full multi-phase)
  # Stops each line when --timeout is hit; does not require full DBLP finish
  hedonic-exp overlapping-scale --smoke --output_dir artifacts/overlapping/complexity_scale
  hedonic-exp overlapping-scale --timeout 30 --max-levels 6 \\
      --output_dir artifacts/overlapping/complexity_scale-dblp
  # Full multi-phase only, 10 min budget per size point
  hedonic-exp overlapping-scale --variant full --timeout 600 --max-levels 6 \\
      --community_idx 1004 --output_dir artifacts/overlapping/complexity_scale-dblp-full

  # Full-DBLP overlap metrics vs resolution; --smoke skips DBLP
  # Caches covers under <output_dir>/runs/ (resume + detector-free re-score)
  # Default TOML: configs/hedonic.toml (paths use ~/…)
  hedonic-exp overlapping-resolution --smoke --output_dir artifacts/overlapping/resolution_f1/smoke
  hedonic-exp overlapping-resolution --config configs/hedonic.toml \\
      --resolutions 0:1:11 --seeds 0-4
  hedonic-exp overlapping-resolution --rescore-only --singleton-mode both \\
      --output_dir artifacts/overlapping/resolution_f1
  # Optional sampled Omega (memory-safe for full DBLP)
  hedonic-exp overlapping-resolution --rescore-only --omega \\
      --omega-sample-size 100000 \\
      --output_dir artifacts/overlapping/resolution_f1
"""


def _print_command_list() -> None:
    print("Subcommands:")
    width = max(len(name) for name in COMMANDS)
    for name, cmd in COMMANDS.items():
        data = f"  [{cmd.needs_data}]" if cmd.needs_data else ""
        print(f"  {name:<{width}}  {cmd.summary}{data}")
    print()
    print("Meta:")
    print("  list       List subcommands")
    print("  -h/--help  This help")
    print("  -V/--version  Package version")
    print()
    print("Data paths (env overrides / TOML — see configs/hedonic.toml):")
    print("  HEDONIC_DBLP_DIR       DBLP network directory")
    print("  HEDONIC_SYNTHETIC_DIR  Synthetic/SBM results root")
    print("  HEDONIC_OUTPUT_DIR     Default experiment artifacts root")
    print("  HEDONIC_NETWORKS_DIR   Root containing saved SNAP network archives")
    print("  HEDONIC_CONFIG         Path to TOML (default: configs/hedonic.toml)")


def _load_main(command: str) -> Callable:
    cmd = COMMANDS[command]
    if not cmd.module:
        raise ValueError(f"Command {command!r} has no module (special handler)")
    mod = importlib.import_module(cmd.module)
    return getattr(mod, cmd.attr)


def _dispatch_help(command: str) -> int:
    cmd = COMMANDS[command]
    if command == "smoke":
        print(
            "usage: hedonic-exp smoke [--keep-output]\n\n"
            "Isolated local check that needs no DBLP or large data dirs:\n"
            "  1. overlapping-small (Petersen + tiny SBM metrics)\n"
            "  2. disjoint --smoke into a temp directory\n\n"
            "Options:\n"
            "  --keep-output  Print and keep the temp output root "
            "(default: delete after run)\n"
        )
        return 0
    if command == "overlapping-small" or not cmd.has_argparse_help:
        print(
            f"usage: hedonic-exp {command}\n\n"
            f"{cmd.summary}.\n"
            "No additional flags.\n"
        )
        return 0
    try:
        _load_main(command)(["--help"])
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


def _call_experiment(fn: Callable, rest: list[str]) -> int:
    """Invoke experiment main(); normalize return / SystemExit to exit code."""
    try:
        result = fn(rest)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        # argparse / explicit SystemExit("message") — surface the text
        print(exc.code, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if result is None or result is True:
        return 0
    if result is False:
        return 1
    if isinstance(result, int):
        return result
    return 0


def _run_smoke(rest: list[str]) -> int:
    """Isolated reproduction check without external databases."""
    keep = False
    unknown: list[str] = []
    for arg in rest:
        if arg == "--keep-output":
            keep = True
        elif arg in ("-h", "--help"):
            return _dispatch_help("smoke")
        else:
            unknown.append(arg)
    if unknown:
        print(f"Unknown smoke args: {unknown}", file=sys.stderr)
        return 2

    print("=" * 55)
    print("  hedonic-exp smoke  (isolated, no DBLP required)")
    print("=" * 55)

    from hedonic.experiments.overlapping.small_graphs import main as small_main

    code = _call_experiment(small_main, [])
    if code != 0:
        return code

    from hedonic.experiments.disjoint.sbm_sweep import main as sweep_main

    if keep:
        tmp = tempfile.mkdtemp(prefix="hedonic-smoke-")
        print(f"\n[smoke] disjoint SBM → {tmp} (kept)")
        code = _call_experiment(
            sweep_main,
            ["--smoke", "--folder_name", "smoke", "--output_root", tmp],
        )
    else:
        with tempfile.TemporaryDirectory(prefix="hedonic-smoke-") as tmp:
            print(f"\n[smoke] disjoint SBM → {tmp}")
            code = _call_experiment(
                sweep_main,
                ["--smoke", "--folder_name", "smoke", "--output_root", tmp],
            )

    if code != 0:
        return code
    print("\n[smoke] All isolated checks passed.")
    return 0


def _dispatch(command: str, rest: list[str]) -> int:
    if command == "smoke":
        return _run_smoke(rest)
    return _call_experiment(_load_main(command), rest)


if __name__ == "__main__":
    raise SystemExit(main())
