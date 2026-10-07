"""One-command setup and execution for the SNAP overlap benchmark.

``overlapping-codeseg`` remains the lower-level, protocol-oriented runner.
This module is the reproducibility entry point for a fresh machine: the full
profile prepares the requested SNAP archives and external runtimes, then runs
the complete registered method catalogue; the smoke profile uses the
archive-free 1,000-node AGMfit-like fixture and still records every method as
completed, unavailable, or failed.  No unavailable method is relabelled as a
different detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hedonic.experiments.config import NETWORKS_DIR, OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping import codeseg_reproduction, codeseg_setup


TURNKEY_SCHEMA_VERSION = 1


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-reproduce",
        description=(
            "Prepare SNAP/method runtimes and run the complete overlapping "
            "community benchmark in one resumable command."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke is archive-free (~1,000 nodes); full downloads/uses SNAP archives",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="comma-separated SNAP datasets, or all (full defaults to DBLP; smoke defaults to synthetic_agmfit)",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated registered methods, or all (unavailable implementations are recorded explicitly)",
    )
    parser.add_argument("--cover", choices=("all", "top5000"), default="all")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR))
    parser.add_argument("--cache-dir", default=str(codeseg_setup.DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="result directory (default: artifacts/overlapping/turnkey/<profile>)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-nodes", type=int, default=1_000)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="per-detector timeout; smoke defaults to 30 seconds")
    parser.add_argument("--omega-sample-size", type=int, default=100_000)
    parser.add_argument("--no-omega", dest="compute_omega", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="prepare/load data and dependencies, write the manifest, but do not run detectors",
    )
    parser.add_argument("--offline", action="store_true", help="use only local data/checkouts and do not download")
    parser.add_argument("--max-download-bytes", type=int, default=None)
    parser.add_argument("--require-all", action="store_true", help="fail if any selected method is unavailable or fails")
    parser.add_argument(
        "--bootstrap",
        dest="bootstrap",
        action="store_true",
        help="prepare data and runtimes before smoke (full bootstraps by default)",
    )
    parser.add_argument(
        "--no-bootstrap",
        dest="bootstrap",
        action="store_false",
        help="skip setup and use already available resources",
    )
    parser.add_argument(
        "--build-native",
        dest="build_native",
        action="store_true",
        help="build configured C/C++ runtimes during bootstrap",
    )
    parser.add_argument(
        "--no-build-native",
        dest="build_native",
        action="store_false",
        help="do not build C/C++ runtimes during bootstrap",
    )
    parser.set_defaults(bootstrap=None, build_native=None, compute_omega=True)
    return parser


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets is None:
        return ["dblp"] if args.profile == "full" else [codeseg_reproduction.SMOKE_DATASET]
    if args.datasets.strip().lower() == "all":
        return list(codeseg_reproduction.DATASETS) if args.profile == "full" else [codeseg_reproduction.SMOKE_DATASET]
    selected = _csv(args.datasets)
    allowed = set(codeseg_reproduction.DATASETS)
    if args.profile == "smoke":
        allowed.add(codeseg_reproduction.SMOKE_DATASET)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(
            f"unknown --datasets: {', '.join(unknown)}; choose from "
            f"{', '.join(sorted(allowed))}"
        )
    if not selected:
        raise ValueError("--datasets must contain at least one value")
    return list(dict.fromkeys(selected))


def _resolve_methods(args: argparse.Namespace) -> str:
    value = str(args.methods).strip().lower()
    if value in {"", "all", "literature"}:
        return "all"
    selected = [
        codeseg_reproduction.canonical_method_name(item)
        for item in _csv(value)
    ]
    allowed = set(codeseg_reproduction.EXTENDED_METHODS)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(
            f"unknown --methods: {', '.join(unknown)}; choose from "
            f"{', '.join(codeseg_reproduction.EXTENDED_METHODS)}"
        )
    if not selected:
        raise ValueError("--methods must contain at least one value")
    return ",".join(dict.fromkeys(selected))


def _setup_args(
    *,
    datasets: list[str],
    methods: str,
    args: argparse.Namespace,
    manifest: Path,
) -> list[str]:
    real_datasets = [name for name in datasets if name != codeseg_reproduction.SMOKE_DATASET]
    command = [
        "--datasets",
        ",".join(real_datasets) if real_datasets else "dblp",
        "--methods",
        methods,
        "--cover",
        args.cover,
        "--network-root",
        str(expand_path(args.network_root)),
        "--cache-dir",
        str(expand_path(args.cache_dir)),
        "--manifest",
        str(manifest),
    ]
    if not real_datasets:
        command.append("--skip-data")
    if args.offline:
        command.append("--offline")
    if args.build_native:
        command.append("--build-native")
    if args.max_download_bytes is not None:
        command.extend(["--max-download-bytes", str(args.max_download_bytes)])
    return command


def _runner_args(
    *,
    datasets: list[str],
    methods: str,
    args: argparse.Namespace,
    output_dir: Path,
    setup_manifest: Path | None,
) -> list[str]:
    command = [
        "--datasets",
        ",".join(datasets),
        "--methods",
        methods,
        "--cover",
        args.cover,
        "--network-root",
        str(expand_path(args.network_root)),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--smoke-nodes",
        str(args.smoke_nodes),
        "--omega-sample-size",
        str(args.omega_sample_size),
    ]
    if args.profile == "smoke":
        command.append("--smoke")
        # This also tells the QOCE adapter that the run is a bounded validation
        # when the synthetic graph is exactly the requested smoke size.
        command.extend(["--max-nodes", str(args.max_nodes or args.smoke_nodes)])
    elif args.max_nodes is not None:
        command.extend(["--max-nodes", str(args.max_nodes)])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.profile == "full" and not args.offline:
        command.append("--auto-download")
    elif args.profile == "full":
        command.append("--no-auto-download")
    if setup_manifest is not None:
        command.extend(["--setup-manifest", str(setup_manifest)])
    if args.resume:
        command.append("--resume")
    if args.preflight:
        command.append("--preflight")
    if args.require_all:
        command.append("--require-all")
    if not args.compute_omega:
        command.append("--no-omega")
    if args.max_download_bytes is not None:
        command.extend(["--max-download-bytes", str(args.max_download_bytes)])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        datasets = _resolve_datasets(args)
        methods = _resolve_methods(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke_nodes < 6:
        parser.error("--smoke-nodes must be at least 6")
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.omega_sample_size <= 0:
        parser.error("--omega-sample-size must be positive")
    if args.profile == "full" and codeseg_reproduction.SMOKE_DATASET in datasets:
        parser.error("synthetic_agmfit is only a smoke-profile dataset")

    output_dir = expand_path(args.output_dir) if args.output_dir else (
        OVERLAPPING_ARTIFACTS_DIR / "turnkey" / args.profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = args.bootstrap if args.bootstrap is not None else args.profile == "full"
    build_native = args.build_native if args.build_native is not None else args.profile == "full"
    timeout = args.timeout if args.timeout is not None else (30.0 if args.profile == "smoke" else 1_800.0)
    args.timeout = timeout
    args.bootstrap = bootstrap
    args.build_native = build_native
    setup_manifest_path = output_dir / "setup_manifest.json"
    setup_manifest: Path | None = setup_manifest_path

    setup_code: int | None = None
    setup_error: str | None = None
    if bootstrap:
        try:
            setup_code = codeseg_setup.setup_main(
                _setup_args(
                    datasets=datasets,
                    methods=methods,
                    args=args,
                    manifest=setup_manifest_path,
                )
            )
        except Exception as exc:  # setup is best-effort; the runner records unavailable methods
            setup_error = f"{type(exc).__name__}: {exc}"
            print(f"[setup-error] {setup_error}")
    else:
        setup_manifest = setup_manifest_path if setup_manifest_path.is_file() else None

    runner_code = codeseg_reproduction.main(
        _runner_args(
            datasets=datasets,
            methods=methods,
            args=args,
            output_dir=output_dir,
            setup_manifest=setup_manifest,
        )
    )
    runner_manifest_path = output_dir / ("preflight.json" if args.preflight else "manifest.json")
    runner_status_counts: dict[str, Any] = {}
    if not args.preflight and runner_manifest_path.is_file():
        try:
            loaded_manifest = json.loads(runner_manifest_path.read_text(encoding="utf-8"))
            value = loaded_manifest.get("status_counts")
            if isinstance(value, dict):
                runner_status_counts = dict(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            runner_status_counts = {}
    has_method_issues = any(
        int(runner_status_counts.get(status, 0) or 0) > 0
        for status in ("unavailable", "failed")
    )
    payload = {
        "schema": TURNKEY_SCHEMA_VERSION,
        "profile": args.profile,
        "datasets": datasets,
        "methods": methods,
        "cover": args.cover,
        "output_dir": str(output_dir),
        "bootstrap": bool(bootstrap),
        "build_native": bool(build_native),
        "setup": {
            "manifest": str(setup_manifest_path),
            "returncode": setup_code,
            "error": setup_error,
        },
        "runner": {
            "returncode": int(runner_code),
            "manifest": str(runner_manifest_path),
            "metrics": None if args.preflight else str(output_dir / "metrics.csv"),
            "status_counts": runner_status_counts,
        },
        "status": (
            "failed"
            if runner_code != 0
            else "partial"
            if setup_error or (setup_code not in (None, 0)) or has_method_issues
            else "completed"
        ),
    }
    _write_json(output_dir / "turnkey_manifest.json", payload)
    print(f"[turnkey] profile={args.profile} runner={runner_code} output={output_dir}")
    return int(runner_code)


__all__ = ["TURNKEY_SCHEMA_VERSION", "main"]
