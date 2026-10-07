"""Identity, checkpointing, and orchestration for the integrity grid."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import igraph as ig
import numpy as np

from hedonic.experiments.overlapping._integrity_cases import (
    ABS_TOLERANCE,
    PROTOCOL_NAME,
    REL_TOLERANCE,
    SCHEMA_VERSION,
    Mode,
    _adjacency,
    _canonical_json,
    _run_case,
    _sha256_bytes,
    expected_total_calls,
)
from hedonic.experiments.overlapping._integrity_fixtures import _contract_fixtures
from hedonic.experiments.overlapping._integrity_runner import (
    _failure_rank,
    _run_shard,
    _shard_id,
    _shard_specs,
    _utc_now,
    _valid_shard,
)


_SOURCE_FILENAMES = (
    "integrity_grid.py",
    "_integrity_cases.py",
    "_integrity_fixtures.py",
    "_integrity_trace.py",
    "_integrity_runner.py",
    "_integrity_durable.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = b"".join(_canonical_json(row) + b"\n" for row in rows)
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _debug_trace_records(fixtures: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = (fixtures.get("cases") or {}).get("debug_trace") or {}
    trace = fixture.get("trace") or {}
    context = fixture.get("context")
    records = []
    for event, key in (("accepted_move", "moves"), ("projection", "projections")):
        for index, row in enumerate(trace.get(key) or []):
            records.append(
                {
                    "schema_version": 1,
                    "protocol_name": PROTOCOL_NAME,
                    "fixture": fixture.get("name"),
                    "context": context,
                    "event": event,
                    "event_index": index,
                    "values": row,
                }
            )
    return records


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Positive-budget and diagnostic checks need the 1.0.0.4 native fixes; later
# releases keep them. ``auto`` therefore records a lower bound, while an
# explicit ``--require-igraph-version`` value remains an exact pin.
MINIMUM_POSITIVE_BUDGET_IGRAPH = "1.0.0.4"


def _version_key(version: str | None) -> tuple[int, ...] | None:
    if version is None:
        return None
    parts = version.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_at_least(version: str | None, minimum: str) -> bool:
    found, bound = _version_key(version), _version_key(minimum)
    return found is not None and bound is not None and found >= bound


def _version_satisfies(version: str | None, requirement: str) -> bool:
    if requirement.startswith(">="):
        return _version_at_least(version, requirement[2:])
    return version == requirement


def _requires_domain_rejections(config: dict[str, Any]) -> bool:
    return bool(config.get("require_igraph_version")) and _version_at_least(
        _distribution_version("lucas-igraph"), MINIMUM_POSITIVE_BUDGET_IGRAPH
    )


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def _symbol_provider_path(function) -> Path:
    """Find the image providing a resolved native function in this process."""
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError(f"native provider identity requires dladdr on {system}")
    loader = ctypes.CDLL(None)
    try:
        dladdr = loader.dladdr
    except AttributeError:
        library = ctypes.util.find_library("dl") if system == "Linux" else None
        if library is None:
            raise RuntimeError("native provider identity cannot resolve dladdr") from None
        loader = ctypes.CDLL(library)
        dladdr = loader.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if not dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) or not info.filename:
        raise RuntimeError("dladdr could not identify the loaded igraph provider")
    return Path(os.fsdecode(info.filename)).resolve(strict=True)


def _loaded_image_paths() -> list[str]:
    """Paths of every shared image loaded into this process."""
    system = platform.system()
    if system == "Darwin":
        loader = ctypes.CDLL(None)
        count = loader._dyld_image_count
        count.restype = ctypes.c_uint32
        name = loader._dyld_get_image_name
        name.argtypes = [ctypes.c_uint32]
        name.restype = ctypes.c_char_p
        return [os.fsdecode(name(i)) for i in range(count()) if name(i)]
    if system == "Linux":
        paths = []
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) >= 6 and fields[5].startswith("/"):
                    paths.append(fields[5])
        return sorted(set(paths))
    raise RuntimeError(f"native provider identity cannot list loaded images on {system}")


def _is_igraph_library(path: str) -> bool:
    return Path(path).name.startswith("libigraph.")


def _embedded_core_identity(extension: Path) -> dict[str, Any]:
    """The core of a wheel build: compiled into the extension with hidden
    symbols, so dladdr cannot name it. It is identified by evidence instead of
    assumption: no igraph library image is loaded in this process, so every
    igraph function the extension calls is its own code."""
    images = _loaded_image_paths()
    libraries = [path for path in images if _is_igraph_library(path)]
    if libraries:
        raise RuntimeError(
            "igraph symbols are hidden but an igraph library is loaded: "
            + ", ".join(sorted(libraries))
        )
    version = getattr(ig._igraph, "__igraph_version__", None)
    if not version:
        raise RuntimeError("the binding reports no igraph core version")
    return {
        "identification": "embedded_without_igraph_library_image",
        "runtime_version": str(version),
        "symbols": {},
        "providers": [
            {
                "path": str(extension), "sha256": _sha256_file(extension),
                "bytes": extension.stat().st_size,
                "embedded_in_extension": True,
            }
        ],
    }


def _native_core_identity(extension: Path) -> dict[str, Any]:
    """Fingerprint the loaded core, including dynamically linked developer builds.

    With exported symbols, dladdr names the image that provides them (the
    extension itself for an embedded core). A wheel embeds the core with
    hidden symbols; it is accepted only when no igraph library image is
    loaded in the process (_embedded_core_identity). Anything else fails
    closed. Libraries must not be replaced while a process is running: these
    hashes describe the provider files on disk.
    """
    try:
        binding = ctypes.CDLL(str(extension))
        if getattr(binding, "igraph_version", None) is None:
            return _embedded_core_identity(extension)
        symbols = {
            name: getattr(binding, name)
            for name in ("igraph_version", "igraph_community_leiden")
        }
        for name in (
            "igraph_community_leiden_with_constraints", "igraph_community_leiden_with_diagnostics"
        ):
            function = getattr(binding, name, None)
            if function is not None:
                symbols[name] = function
        providers = {
            name: _symbol_provider_path(function)
            for name, function in symbols.items()
        }
        version = ctypes.c_char_p()
        version_function = symbols["igraph_version"]
        version_function.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        version_function.restype = None
        version_function(ctypes.byref(version), None, None, None)
        if not version.value:
            raise RuntimeError("igraph_version returned no runtime version")
        return {
            "identification": "dladdr",
            "runtime_version": version.value.decode("utf-8"),
            "symbols": {name: str(path) for name, path in providers.items()},
            "providers": [
                {
                    "path": str(path), "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                    "embedded_in_extension": path == extension,
                }
                for path in sorted(set(providers.values()))
            ],
        }
    except (OSError, AttributeError, UnicodeError, RuntimeError) as exc:
        raise RuntimeError(
            "cannot fingerprint the loaded igraph core; durable integrity runs "
            "require resolvable native provider symbols and readable provider files: "
            f"{exc}"
        ) from exc


def _extension_identity() -> dict[str, Any]:
    extension = Path(ig._igraph.__file__).resolve(strict=True)
    return {
        "path": str(extension),
        "sha256": _sha256_file(extension),
        "bytes": extension.stat().st_size,
        "native_core": _native_core_identity(extension),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _source_identity() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    root = _repo_root()
    files = []
    for filename in _SOURCE_FILENAMES:
        source = directory / filename
        files.append(
            {
                "path": str(source.relative_to(root)),
                "sha256": _sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    return {
        "files": files,
        "sha256": _sha256_bytes(_canonical_json(files)),
        "git_head": _git_head(root),
    }


def _progress_payload(
    *,
    run_identity: str,
    status: str,
    phase: str,
    active_shard: str | None,
    completed_shards: int,
    total_shards: int,
    completed_calls: int,
    total_calls: int,
    started: float,
) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started)
    rate = completed_calls / elapsed if elapsed and completed_calls else 0.0
    remaining = max(0, total_calls - completed_calls)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "run_identity": run_identity,
        "status": status,
        "phase": phase,
        "active_shard": active_shard,
        "completed_shards": completed_shards,
        "total_shards": total_shards,
        "completed_calls": completed_calls,
        "total_calls": total_calls,
        "fraction": completed_calls / total_calls if total_calls else 1.0,
        "elapsed_seconds": elapsed,
        "calls_per_second": rate,
        "eta_seconds": remaining / rate if rate else None,
        "pid": os.getpid(),
        "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "updated_utc": _utc_now(),
    }


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    default_max = 3 if args.profile == "smoke" else 5
    max_n = default_max if args.max_n is None else args.max_n
    positive = (
        args.profile == "standard"
        if args.positive_budget is None
        else args.positive_budget
    )
    required_version = args.require_igraph_version
    if required_version == "auto":
        required_version = (
            f">={MINIMUM_POSITIVE_BUDGET_IGRAPH}" if positive else None
        )
    if args.min_n < 2 or max_n < args.min_n or max_n > 5:
        raise ValueError("integrity grid requires 2 <= min_n <= max_n <= 5")
    if args.batch_graphs < 1:
        raise ValueError("batch_graphs must be positive")
    return {
        "profile": args.profile,
        "min_n": args.min_n,
        "max_n": max_n,
        "gammas": list(args.gammas),
        "include_positive_budget": bool(positive),
        "require_debug_trace": bool(positive),
        "batch_graphs": args.batch_graphs,
        "tolerance": {"atol": ABS_TOLERANCE, "rtol": REL_TOLERANCE},
        "require_igraph_version": required_version,
    }


def _preflight(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = _source_identity()
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "igraph_python": getattr(ig, "__version__", None),
        "igraph_c_reported_by_binding": getattr(ig._igraph, "__igraph_version__", None),
        "lucas_igraph_distribution": _distribution_version("lucas-igraph"),
        "numpy": np.__version__,
        "extension": _extension_identity(),
    }
    required = config.get("require_igraph_version")
    if required and not _version_satisfies(
        environment["lucas_igraph_distribution"], required
    ):
        raise RuntimeError(
            f"required lucas-igraph {required}, found "
            f"{environment['lucas_igraph_distribution']}"
        )
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "config": config,
        "source": source,
        "environment": environment,
    }
    run_identity = _sha256_bytes(_canonical_json(identity_payload))
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / f".write-probe-{os.getpid()}.json"
    _atomic_json(probe, {"run_identity": run_identity})
    reloaded = json.loads(probe.read_text(encoding="utf-8"))
    probe.unlink()
    if reloaded.get("run_identity") != run_identity:
        raise RuntimeError("output atomic-write/reload probe failed")
    smoke_graph = ig.Graph(n=2, edges=[(0, 1)], directed=False)
    smoke_adjacency = _adjacency(2, [(0, 1)])
    smoke_mode = Mode(
        "preflight_negative_multilevel_open",
        -1,
        False,
        True,
        True,
    )
    smoke_outcome = _run_case(
        graph=smoke_graph,
        adjacency=smoke_adjacency,
        graph_mask=1,
        initial_rows=((0,), (1,)),
        cap=2,
        gamma=0.5,
        mode=smoke_mode,
    )
    if smoke_outcome["failure"] is not None:
        raise RuntimeError(
            f"native/oracle preflight failed: {smoke_outcome['failure']}"
        )
    contract_preflight = _contract_fixtures(
        config["include_positive_budget"],
        require_domain_rejections=_requires_domain_rejections(config),
        require_debug_trace=config["require_debug_trace"],
    )
    if not contract_preflight["ok"]:
        raise RuntimeError(f"contract fixture preflight failed: {contract_preflight}")
    disk = shutil.disk_usage(output_dir)
    return {
        **identity_payload,
        "run_identity": run_identity,
        "output_dir": str(output_dir),
        "disk_free_bytes": disk.free,
        "preflight_utc": _utc_now(),
        "atomic_reload_ok": True,
        "native_oracle_smoke": smoke_outcome["stable"],
        "contract_fixtures": contract_preflight,
        "positive_guard_smoke": contract_preflight["cases"].get("positive_guard"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    config = _run_config(args)
    preflight = _preflight(output_dir, config)
    run_identity = preflight["run_identity"]
    _atomic_json(output_dir / "preflight.json", preflight)

    specs = _shard_specs(config["min_n"], config["max_n"], config["batch_graphs"])
    total_calls = expected_total_calls(
        config["min_n"],
        config["max_n"],
        config["gammas"],
        include_positive_budget=config["include_positive_budget"],
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "run_identity": run_identity,
        "config": config,
        "expected_shards": len(specs),
        "expected_calls": total_calls,
        "shards": [{**spec, "shard_id": _shard_id(spec)} for spec in specs],
    }
    _atomic_json(output_dir / "plan.json", plan)
    if args.preflight_only:
        return {**plan, "status": "preflight_complete"}

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed_paths: list[Path] = []
    completed_calls = 0
    executed_this_run = 0

    for spec in specs:
        shard_id = _shard_id(spec)
        path = shard_dir / f"{shard_id}.json"
        if args.resume and _valid_shard(path, run_identity, spec):
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed_paths.append(path)
            completed_calls += int(payload["observed_calls"])
            continue
        _atomic_json(
            output_dir / "progress.json",
            _progress_payload(
                run_identity=run_identity,
                status="running",
                phase="exhaustive_shards",
                active_shard=shard_id,
                completed_shards=len(completed_paths),
                total_shards=len(specs),
                completed_calls=completed_calls,
                total_calls=total_calls,
                started=started,
            ),
        )
        payload = _run_shard(
            spec,
            run_identity=run_identity,
            gammas=config["gammas"],
            include_positive_budget=config["include_positive_budget"],
        )
        _atomic_json(path, payload)
        if not _valid_shard(path, run_identity, spec):
            raise RuntimeError(f"shard failed reload validation: {path}")
        completed_paths.append(path)
        completed_calls += int(payload["observed_calls"])
        executed_this_run += 1
        if args.stop_after_shards and executed_this_run >= args.stop_after_shards:
            break

    all_complete = len(completed_paths) == len(specs)
    fixtures = None
    debug_trace_artifact = None
    if all_complete:
        fixtures = _contract_fixtures(
            config["include_positive_budget"],
            require_domain_rejections=_requires_domain_rejections(config),
            require_debug_trace=config["require_debug_trace"],
        )
        _atomic_json(output_dir / "fixtures.json", fixtures)
        debug_trace_records = _debug_trace_records(fixtures)
        if debug_trace_records:
            debug_trace_path = output_dir / "projection_trace.jsonl"
            _atomic_jsonl(debug_trace_path, debug_trace_records)
            debug_trace_artifact = {
                "path": str(debug_trace_path.relative_to(output_dir)),
                "sha256": _sha256_file(debug_trace_path),
                "bytes": debug_trace_path.stat().st_size,
                "records": len(debug_trace_records),
                "accepted_move_records": sum(
                    row["event"] == "accepted_move" for row in debug_trace_records
                ),
                "projection_records": sum(
                    row["event"] == "projection" for row in debug_trace_records
                ),
            }

    shard_rows = []
    failure_count = 0
    minimal_by_violation: dict[str, dict[str, Any]] = {}
    maxima = {"quality_error": 0.0, "initial_decrease": 0.0, "terminal_regret": 0.0}
    for path in completed_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        failure_count += int(payload["failure_count"])
        for failure in payload.get("failures") or []:
            for violation in failure.get("violations") or ["unknown"]:
                previous = minimal_by_violation.get(str(violation))
                if previous is None or _failure_rank(failure) < _failure_rank(previous):
                    minimal_by_violation[str(violation)] = failure
        for key in maxima:
            maxima[key] = max(maxima[key], float(payload["maxima"][key]))
        shard_rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "observed_calls": payload["observed_calls"],
                "failure_count": payload["failure_count"],
                "case_result_sha256": payload["case_result_sha256"],
            }
        )

    status = (
        "complete"
        if all_complete and failure_count == 0 and fixtures and fixtures["ok"]
        else ("failed" if all_complete else "incomplete")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "run_identity": run_identity,
        "status": status,
        "config": config,
        "preflight": preflight,
        "expected_shards": len(specs),
        "completed_shards": len(completed_paths),
        "expected_calls": total_calls,
        "observed_calls": sum(row["observed_calls"] for row in shard_rows),
        "failure_count": failure_count,
        "minimal_counterexamples": [
            {"violation": violation, "case": minimal_by_violation[violation]}
            for violation in sorted(minimal_by_violation)
        ],
        "maxima": maxima,
        "fixtures": fixtures,
        "debug_trace_artifact": debug_trace_artifact,
        "shards": shard_rows,
        "completed_utc": _utc_now() if all_complete else None,
    }
    manifest_path = output_dir / "integrity_manifest.json"
    if _extension_identity() != preflight["environment"]["extension"]:
        raise RuntimeError(
            "igraph extension or native provider changed during the integrity run; "
            "refusing to write a terminal manifest"
        )
    _atomic_json(manifest_path, manifest)
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reloaded.get("run_identity") != run_identity or reloaded.get("status") != status:
        raise RuntimeError("terminal manifest reload validation failed")
    _atomic_json(
        output_dir / "progress.json",
        _progress_payload(
            run_identity=run_identity,
            status=status,
            phase="terminal_verification" if all_complete else "checkpointed",
            active_shard=None,
            completed_shards=len(completed_paths),
            total_shards=len(specs),
            completed_calls=manifest["observed_calls"],
            total_calls=total_calls,
            started=started,
        ),
    )
    return manifest
