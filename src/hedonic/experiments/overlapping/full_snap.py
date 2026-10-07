"""Versioned complete SNAP track: no graph cap, no metadata-informed detection.

Run ``hedonic-exp overlapping-full-snap --help``. Raw gzip archives stay read-only;
all generated artifacts live in a new full-snap-v1 namespace. A preflight-only
receipt is an explicit not-run record, never a completed benchmark result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

from hedonic.experiments.config import NETWORKS_DIR
from .full_snap_execution import atomic_json, run_unit
from .snap import (SPECS, _dataset_dir, _first_existing, _load_from_raw,
                   common_undirected_analysis_dataset, smoke_dataset)

PROTOCOL = "full-snap-v1"
METHODS = ("hedonic_local_fixed", "hedonic_full_fixed", "hedonic_local_unlimited",
           "hedonic_full_unlimited", "leiden_disjoint", "cpm", "demon")
METRIC_CONVENTIONS = {
    "recovery_implementation": "hedonic.experiments.overlapping.metrics.evaluate_cover",
    "prediction_and_metadata": "canonical distinct bodies, singleton_mode=all",
    "matching": "maximum-weight optional one-to-one F1; exact sparse assignment for large components",
    "best_match": "symmetric best community matches; size-weighted F1 separately",
    "node_membership": "micro/macro multilabel F1 after one-to-one label alignment",
    "omega": "raw and chance-adjusted, uniform pairs with replacement on all graph vertices",
    "onmi": "unavailable: no independently qualified implementation in this protocol",
    "objective": "independent full_snap_audit unit-l2 scorer, labelled supports retained",
    "audit": "numerical prefix best response; all labels for declared sampled vertices",
    "metadata": "supplied overlapping metadata, not universal ground truth; absent nodes remain in graph",
    "structural_overlap": "canonical distinct bodies; labelled overlap also reported separately",
    "variability": "per-seed values and sample standard deviation for completed runs only",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path: Path):
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": hasher.hexdigest()}


def source_identity(path):
    root = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    head = git("rev-parse", "HEAD").decode().strip()
    diff = git("diff", "HEAD", "--binary")
    # Record untracked implementation files as well as the tracked dirty diff.
    files = {}
    for item in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if item:
            rel = item.decode()
            if Path(rel).suffix in {".py", ".c", ".h", ".toml", ".json"}:
                files[rel] = file_identity(root / rel)["sha256"]
    return {"root": str(root), "head": head, "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_code_sha256": files}


def loaded_native_libraries():
    """Identity of the actual loaded libigraph image, not a guessed build path."""
    paths = set()
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        process._dyld_image_count.restype = ctypes.c_uint32
        process._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        process._dyld_get_image_name.restype = ctypes.c_char_p
        for index in range(process._dyld_image_count()):
            name = process._dyld_get_image_name(index)
            if name and "libigraph" in Path(name.decode()).name:
                paths.add(Path(name.decode()).resolve())
    elif Path("/proc/self/maps").exists():
        for line in Path("/proc/self/maps").read_text().splitlines():
            candidate = line.split()[-1]
            if candidate.startswith("/") and "libigraph" in Path(candidate).name:
                paths.add(Path(candidate).resolve())
    return [file_identity(path) for path in sorted(paths)]


def runtime_code_files():
    from hedonic import Game
    from . import metrics
    return [Path(p) for p in (__file__, inspect.getfile(Game), inspect.getfile(metrics),
        str(Path(__file__).with_name("full_snap_audit.py")),
        str(Path(__file__).with_name("full_snap_execution.py")),
        str(Path(__file__).with_name("snap.py")))]


def environment(native_source=None, python_source=None, build_receipt=None):
    import igraph as ig
    import igraph._igraph as extension
    import scipy
    from hedonic import Game
    from .methods import method_dependency_identity
    import hedonic.experiments.overlapping.metrics as metrics
    root = Path(__file__).resolve().parents[4]
    result = {
        "interpreter": str(Path(sys.executable).resolve()), "python": sys.version,
        "igraph_version": ig.__version__, "native_version": getattr(ig, "__igraph_version__", None),
        "extension": file_identity(Path(extension.__file__)), "native_libraries": loaded_native_libraries(), "scipy": scipy.__version__,
        "hedonic": source_identity(root),
        "native": source_identity(native_source) if native_source else None,
        "python_binding": source_identity(python_source) if python_source else None,
        "module_hashes": {p.name: file_identity(p)["sha256"] for p in runtime_code_files()},
        "baselines": {name: method_dependency_identity(name) for name in ("cpm", "demon")},
        "other_baselines": {name: "unavailable: no pinned adapter in full-snap-v1"
                            for name in ("bigclam", "slpa", "oslom", "link_communities", "mmsb", "codeseg")},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "build_binding": "unverified",
    }
    if shutil.which("ps") is None:
        raise RuntimeError("preflight requires ps for the RSS watchdog")
    if build_receipt:
        expected = json.loads(Path(build_receipt).read_text())
        actual = build_binding(result)
        if any(expected.get(key) != value for key, value in actual.items()):
            raise ValueError("build receipt does not match source revisions/diffs and loaded native extension")
        result["build_binding"] = "verified_against_explicit_build_receipt"
        result["build_receipt"] = file_identity(Path(build_receipt))
    return result


def build_binding(env):
    if env["native"] is None or env["python_binding"] is None:
        raise ValueError("native and Python source paths are required for build verification")
    return {"native_git_head": env["native"]["head"],
            "native_diff_sha256": env["native"]["diff_sha256"],
            "python_git_head": env["python_binding"]["head"],
            "python_diff_sha256": env["python_binding"]["diff_sha256"],
            "extension_sha256": env["extension"]["sha256"],
            "native_library_identities": env["native_libraries"]}


def inspect_archive(name, root, *, scan_counts=False):
    """Hash compressed bytes; read a bounded header before optional full ID scan."""
    spec = SPECS[name]
    base = _dataset_dir(Path(root).expanduser(), spec)
    graph = _first_existing(base, spec.raw_edge_files)
    cover = _first_existing(base, spec.raw_cover_files["all"])
    if graph is None or cover is None:
        return {"dataset": name, "status": "unavailable_data",
                "reason": "complete raw gzip graph and all supplied cover are required"}
    header, n, m = [], None, None
    with gzip.open(graph, "rt") as stream:
        for _, line in zip(range(32), stream):
            if not line.startswith("#"):
                break
            header.append(line.strip())
            match = re.search(r"Nodes:\s*(\d+)\s+Edges:\s*(\d+)", line)
            if match:
                n, m = map(int, match.groups())
    identity = {"dataset": name, "status": "available", "graph": file_identity(graph),
                "cover": file_identity(cover), "cover_variant": "all", "header": header,
                "header_nodes": n, "header_edges": m,
                "source_directed": spec.directed, "ground_truth_type": spec.ground_truth_type,
                "count_scope": "source header only; observed counts checked in worker",
                "analysis_policy": "all edge endpoints; sorted original IDs; undirected simple projection"}
    if scan_counts:
        ids, edges = set(), 0
        with gzip.open(graph, "rt") as stream:
            for line in stream:
                if not line.strip() or line.startswith("#"):
                    continue
                a, b = map(int, line.split()[:2])
                ids.update((a, b))
                edges += 1
        identity.update(observed_source_nodes=len(ids), observed_source_edges=edges,
                        count_scope="complete streaming endpoint-ID and edge-line census")
    # Also identify alternate supplied covers without substituting them in scoring.
    identity["other_supplied_covers"] = [file_identity(p) for kind in spec.raw_cover_files
        if kind != "all" and (p := _first_existing(base, spec.raw_cover_files[kind])) is not None]
    return identity


def resource_boundary(archive, memory_limit_bytes, method):
    if archive["status"] != "available":
        return {"status": archive["status"], "reason": archive["reason"],
                "stage": archive.get("stage", "input_availability")}
    n = archive.get("observed_source_nodes", archive.get("header_nodes"))
    m = archive.get("observed_source_edges", archive.get("header_edges"))
    if n is None or m is None:
        return {"status": "not_completed_resource_boundary", "stage": "preflight_counts",
                "reason": "raw file has no count header; memory estimate unavailable",
                "next_step": "run --scan-counts preflight under an explicit census resource budget"}
    # Conservative loader + native/NetworkX estimate, NOT a proved upper bound.
    # It rejects resource use, never truncates the graph or changes detector parameters.
    estimate = 320 * m + 384 * n + 256 * 1024**2
    if method in {"cpm", "demon"}:
        estimate += 384 * m + 256 * n
    if estimate > memory_limit_bytes:
        return {"status": "not_completed_resource_boundary", "stage": "load_and_adapter",
                "reason": "conservative estimated working set exceeds requested RSS budget",
                "estimated_working_set_bytes": estimate, "memory_limit_bytes": memory_limit_bytes,
                "estimate_is_proved_bound": False,
                "next_step": "qualify a streaming/compact loader or deliberately raise the RSS budget"}
    return {"status": "eligible", "estimated_working_set_bytes": estimate,
            "estimate_is_proved_bound": False}


def _phase(directory, phase, **details):
    event = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
             "monotonic_seconds": time.monotonic(), "pid": os.getpid(), **details}
    history_path = directory / "stage_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(event)
    atomic_json(history_path, history)
    atomic_json(directory / "stage.json", event)


def _detector(graph, method, cap, seed, gamma):
    import igraph as ig
    from hedonic import Game
    from .metrics import partition_to_cover_lists
    ig.set_random_number_generator(random.Random(seed))
    if method.startswith("hedonic_") or method == "leiden_disjoint":
        actual_cap = 1 if method == "leiden_disjoint" else -1 if method.endswith("unlimited") else cap
        result = Game(graph).community_hedonic(resolution=gamma, max_memberships=actual_cap,
            local_move_only="local" in method, n_iterations=-1, allow_isolation=True,
            initial_membership=None, seed=seed)
        return partition_to_cover_lists(result), actual_cap
    # Full-network clique percolation intentionally has no output-count truncation.
    import networkx as nx
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.vcount()))
    nx_graph.add_edges_from((e.source, e.target) for e in graph.es)
    if method == "cpm":
        result = nx.algorithms.community.k_clique_communities(nx_graph, 3)
        return [sorted(c) for c in result], None
    if method == "demon":
        from demon.alg.Demon import Demon
        random.seed(seed)
        result = Demon(graph=nx_graph, epsilon=0.25, min_community_size=2).execute()
        return [sorted(map(int, c)) for c in result], None
    raise ValueError(f"unknown method {method}")


def execute_unit(config, directory):
    """Run complete pipeline inside the watched process, including final reload."""
    if "environment" in config:
        current_modules = {p.name: file_identity(p)["sha256"] for p in runtime_code_files()}
        if current_modules != config["environment"]["module_hashes"]:
            raise ValueError("source modules changed after preflight")
        if loaded_native_libraries() != config["environment"]["native_libraries"]:
            raise ValueError("loaded native library changed after preflight")
        import igraph._igraph as extension
        if file_identity(Path(extension.__file__)) != config["environment"]["extension"]:
            raise ValueError("loaded Python extension changed after preflight")
        if config.get("method") in {"cpm", "demon"}:
            from .methods import method_dependency_identity
            method = config["method"]
            if method_dependency_identity(method) != config["environment"]["baselines"][method]:
                raise ValueError("baseline implementation changed after preflight")
    if config.get("census_only"):
        _phase(directory, "streaming_endpoint_census")
        archive = inspect_archive(config["dataset"], config["data_root"], scan_counts=True)
        _phase(directory, "complete")
        return {"status": "completed", "archive": archive}
    from .metrics import evaluate_cover
    from .full_snap_audit import memberships, omega_pair_agreement, unit_l2_audit
    from .snap import canonicalize_cover
    _phase(directory, "load_graph_and_metadata")
    if config["smoke"]:
        dataset = smoke_dataset(config["dataset"], cover_variant="all")
    else:
        for key in ("graph", "cover"):
            if file_identity(Path(config["archive"][key]["path"])) != config["archive"][key]:
                raise ValueError("input identity drift since preflight")
        spec = SPECS[config["dataset"]]
        dataset = _load_from_raw(spec, _dataset_dir(Path(config["data_root"]), spec), "all")
        expected = config["archive"]
        if dataset.graph.vcount() != expected.get("observed_source_nodes", expected.get("header_nodes")):
            raise ValueError("observed complete endpoint graph count differs from preflight census/header")
        if dataset.graph.ecount() != expected.get("observed_source_edges", expected.get("header_edges")):
            raise ValueError("observed raw edge count differs from preflight census/header")
    dataset = common_undirected_analysis_dataset(dataset)
    graph, truth = dataset.graph, dataset.cover
    metadata_vertices = {v for c in truth for v in c}
    dataset.report["vertices_without_metadata"] = graph.vcount() - len(metadata_vertices)
    atomic_json(directory / "graph.json", dataset.report)
    gamma = graph.density(loops=False)
    _phase(directory, "detect", graph_n=graph.vcount(), graph_m=graph.ecount())
    started = time.monotonic()
    predicted, cap = _detector(graph, config["method"], config["fixed_cap"], config["seed"], gamma)
    detection_seconds = time.monotonic() - started
    # Strict validation; never silently discard invalid detector memberships.
    memberships(predicted, graph.vcount())
    _phase(directory, "serialize_cover")
    path = directory / "cover.json.gz"
    with gzip.open(path.with_suffix(".gz.tmp"), "wt") as stream:
        json.dump({"protocol": PROTOCOL, "run_identity": config["run_identity"],
                   "cover": predicted}, stream, separators=(",", ":"), allow_nan=False)
    path.with_suffix(".gz.tmp").replace(path)
    # The committed cover is reloaded before scoring; a later timeout retains it.
    with gzip.open(path, "rt") as stream:
        restored = json.load(stream)
    if restored["cover"] != predicted or restored["run_identity"] != config["run_identity"]:
        raise RuntimeError("serialized cover reload mismatch")
    _phase(directory, "score_recovery")
    recovery = evaluate_cover(predicted, truth, graph.vcount(), compute_omega=False,
                              singleton_mode="all", matching_weight="f1")
    canonical, cleanup = canonicalize_cover(predicted, n_vertices=graph.vcount(), minimum_size=1)
    canonical_truth, _ = canonicalize_cover(truth, n_vertices=graph.vcount(), minimum_size=1)
    omega = omega_pair_agreement(canonical, canonical_truth, graph.vcount(),
                                sample_size=config["omega_pairs"], seed=config["seed"])
    rows = memberships(predicted, graph.vcount())
    canonical_rows = memberships(canonical, graph.vcount())
    structural = {"vertex_coverage": sum(bool(r) for r in rows) / graph.vcount() if rows else 1,
                  "overlap_fraction_all_vertices": sum(len(r) > 1 for r in canonical_rows) / len(canonical_rows) if canonical_rows else 0,
                  "labelled_overlap_fraction_all_vertices": sum(len(r) > 1 for r in rows) / len(rows) if rows else 0,
                  "labelled_community_count": len(predicted), "canonical_community_count": len(canonical),
                  "metadata_community_count": len(canonical_truth),
                  "community_count_error": len(canonical) - len(canonical_truth),
                  "duplicate_body_rate": cleanup["duplicate_communities_removed"] / len(predicted) if predicted else 0}
    atomic_json(directory / "scores.json", {"recovery": recovery, "omega": omega, "structural": structural})
    _phase(directory, "independent_unit_l2_audit")
    audit = unit_l2_audit(graph, predicted, gamma, cap if cap is not None else max(1, graph.vcount()),
                         max_vertices=config["audit_vertices"], max_label_visits=config["audit_label_visits"],
                         seed=config["seed"])
    if cap is None:
        audit["baseline_warning"] = "diagnostic in Hedonic objective only; not the baseline's optimization claim"
    atomic_json(directory / "audit.json", audit)
    _phase(directory, "terminal_reload")
    for filename in ("graph.json", "scores.json", "audit.json"):
        json.loads((directory / filename).read_text())
    _phase(directory, "complete")
    return {"status": "completed", "graph": dataset.report, "recovery": recovery,
            "structural": structural, "omega": omega, "onmi": {"status": "unavailable"},
            "audit": audit, "detector_seconds": detection_seconds, "resolution": gamma,
            "parameters": {"max_memberships": cap, "initialization": "singleton/native default",
                "global_upper_bound": None, "exact_community_count": None, "n_iterations": -1,
                "allow_isolation": True, "resolution_rule": "complete analysis graph density",
                "clique_size": 3 if config["method"] == "cpm" else None,
                "epsilon": 0.25 if config["method"] == "demon" else None,
                "min_community_size": 2 if config["method"] == "demon" else None}}


def _verify_artifacts(record):
    for artifact in record.get("artifacts", []):
        if file_identity(Path(artifact["path"])) != artifact:
            raise ValueError("cached artifact identity mismatch; use a new output namespace")


def _summary(records):
    groups = {}
    for record in records:
        key = f"{record['dataset']}/{record['method']}"
        group = groups.setdefault(key, {"outcomes": Counter(), "completed_seed_values": []})
        group["outcomes"][record["status"]] += 1
        if record["status"] == "completed":
            group["completed_seed_values"].append({"seed": record["seed"],
                "f1": record["recovery"].get("f1"),
                "detector_seconds": record["detector_seconds"],
                "peak_rss_bytes": max(record.get("sampled_peak_rss_bytes", 0), record.get("worker_peak_rss_bytes", 0)),
                "one_to_one_f1": record["recovery"].get("matching_f1")})
    for group in groups.values():
        group["outcomes"] = dict(group["outcomes"])
        scores = [row["f1"] for row in group["completed_seed_values"] if row["f1"] is not None]
        group["best_match_f1_mean"] = statistics.mean(scores) if scores else None
        group["best_match_f1_sample_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        group["variability_status"] = "measured" if len(scores) > 1 else "insufficient_completed_seeds"
    return groups


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(NETWORKS_DIR))
    p.add_argument("--output-dir", default=f"artifacts/overlapping/{PROTOCOL}")
    p.add_argument("--datasets", default=",".join(SPECS))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--fixed-cap", type=int, default=4)
    p.add_argument("--timeout-seconds", type=float, default=90)
    p.add_argument("--memory-mib", type=int, default=4096)
    p.add_argument("--omega-pairs", type=int, default=10000)
    p.add_argument("--audit-vertices", type=int, default=32)
    p.add_argument("--audit-label-visits", type=int, default=2_000_000)
    p.add_argument("--preflight", action="store_true", help="header/hash inventory and explicit not-run records")
    p.add_argument("--scan-counts", action="store_true", help="optional complete endpoint census in a watched subprocess under the same RAM/time budgets")
    p.add_argument("--smoke", action="store_true", help="complete disposable tiny pipeline; never a full-network measurement")
    p.add_argument("--resume", action="store_true", help="verify hashes and reuse all terminal outcomes, including failures")
    p.add_argument("--native-source")
    p.add_argument("--python-source")
    p.add_argument("--build-receipt", help="explicit source-diff/revision/extension hash binding from the sibling build")
    return p


def main(argv=None):
    task_started = time.monotonic()
    args = parser().parse_args(argv)
    datasets, methods = args.datasets.split(","), args.methods.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]
    if not datasets or any(d not in SPECS for d in datasets) or any(m not in METHODS for m in methods):
        raise ValueError("unknown/empty dataset or method selection")
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("unique nonnegative integer seeds required")
    if len(set(datasets)) != len(datasets) or len(set(methods)) != len(methods):
        raise ValueError("duplicate dataset/method selection")
    if min(args.fixed_cap, args.memory_mib, args.omega_pairs, args.audit_vertices, args.audit_label_visits) < 1 or args.timeout_seconds <= 0:
        raise ValueError("positive cap, resource, and audit parameters required")
    output, data_root = Path(args.output_dir).expanduser().resolve(), Path(args.data_root).expanduser().resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("output must be outside read-only SNAP archives")
    output.mkdir(parents=True, exist_ok=True)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    env = environment(args.native_source, args.python_source, args.build_receipt)
    archives = {}
    for dataset in datasets:
        archive = {"dataset": dataset, "status": "synthetic_fixture"} if args.smoke else inspect_archive(dataset, data_root)
        if args.scan_counts and not args.smoke and archive["status"] == "available":
            census = run_unit({"dataset": dataset, "data_root": str(data_root), "census_only": True,
                "timeout_seconds": args.timeout_seconds, "memory_limit_bytes": args.memory_mib * 1024**2},
                output / "census" / dataset,
                heartbeat=lambda elapsed, rss, pid: atomic_json(output / "progress.json",
                    {"status": "running", "phase": "streaming_census", "active_item": dataset,
                     "elapsed_seconds": elapsed, "worker_rss_bytes": rss, "worker_pid": pid}))
            if census["status"] == "completed":
                archive = census["archive"]
            else:
                archive.update(status="not_completed_resource_boundary", stage="streaming_census",
                    reason="complete endpoint census did not finish within the resource envelope")
            # Keep measured attempt provenance outside the deterministic cache key.
            atomic_json(output / "census" / f"{dataset}-receipt.json", census)
        archives[dataset] = archive
    config = {"protocol": PROTOCOL, "data_root": str(data_root), "smoke": args.smoke,
              "datasets": datasets, "methods": methods, "seeds": seeds, "fixed_cap": args.fixed_cap,
              "timeout_seconds": args.timeout_seconds, "memory_limit_bytes": args.memory_mib * 1024**2,
              "omega_pairs": args.omega_pairs, "audit_vertices": args.audit_vertices,
              "audit_label_visits": args.audit_label_visits, "supervision": "unsupervised_discovery",
              "archives": archives, "environment": env, "metric_conventions": METRIC_CONVENTIONS}
    run_identity = digest(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume or previous["run_identity"] != run_identity:
            raise ValueError("existing output/config/source identity mismatch; use --resume unchanged or a new directory")
    else:
        atomic_json(manifest_path, {"run_identity": run_identity, "config": config,
                                   "created_at": datetime.now(timezone.utc).isoformat()})
    available = {m: not(m in {"cpm", "demon"}) or bool(env["baselines"][m]["implementation_sha256"]) for m in methods}
    total = len(datasets) * len(methods) * len(seeds)
    records, started = [], task_started
    def progress(phase, active=None, **details):
        elapsed = time.monotonic() - started
        atomic_json(output / "progress.json", {"run_identity": run_identity, "status": "running",
            "phase": phase, "active_item": active, "completed_units": len(records), "total_units": total,
            "fraction": len(records) / total, "elapsed_seconds": elapsed,
            "rate_units_per_second": len(records) / elapsed if elapsed else None,
            "eta_seconds": None, "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), **details})
    progress("qualification")
    qualification = {}
    if not args.preflight and not args.smoke:
        if env["build_binding"] == "unverified":
            raise ValueError("real full runs require --native-source, --python-source and --build-receipt")
        if shutil.disk_usage(output).free < 1024**3:
            raise ValueError("full run requires at least 1 GiB free output space")
        # Every available selected adapter exercises load→detect→score→audit→reload
        # on a disposable complete fixture before the first real graph is loaded.
        for method in methods:
            if not available[method]:
                continue
            fixture = dict(config, dataset="amazon", method=method, seed=0, smoke=True,
                           run_identity=run_identity)
            qualification[method] = run_unit(fixture, output / "qualification" / method)
        atomic_json(output / "qualification.json", qualification)
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                key = f"{dataset}-{method}-seed{seed}"
                path = output / "runs" / f"{key}.json"
                progress("run", key)
                if path.exists() and args.resume:
                    old = json.loads(path.read_text())
                    if old["run_identity"] != run_identity or old["record_sha256"] != digest({k: v for k, v in old.items() if k != "record_sha256"}):
                        raise ValueError("cached run identity/hash mismatch")
                    if old["status"] != "not_run_preflight":
                        _verify_artifacts(old)
                        records.append(old)
                        continue
                record = {"run_identity": run_identity, "dataset": dataset, "method": method,
                          "seed": seed, "supervision": "unsupervised_discovery", "measurement_scope":
                          "synthetic_fixture" if args.smoke else "complete_raw_graph"}
                boundary = {"status": "eligible"} if args.smoke else resource_boundary(archives[dataset], config["memory_limit_bytes"], method)
                if not available[method]:
                    record.update(status="unavailable_dependency", reason="pinned implementation import/source identity unavailable")
                elif boundary["status"] != "eligible":
                    record.update(boundary)
                elif args.preflight:
                    record.update(status="not_run_preflight", resource_preflight=boundary)
                elif qualification and qualification.get(method, {}).get("status") != "completed":
                    record.update(status="not_completed_preflight_failure", qualification=qualification.get(method))
                else:
                    unit = dict(config, dataset=dataset, method=method, seed=seed,
                                run_identity=run_identity, archive=archives[dataset])
                    record.update(run_unit(unit, output / "units" / key,
                        heartbeat=lambda elapsed, rss, pid: progress("run", key, worker_elapsed_seconds=elapsed,
                                                                   worker_rss_bytes=rss, worker_pid=pid)))
                record["record_sha256"] = digest(record)
                atomic_json(path, record)
                records.append(record)
    progress("aggregate_and_reload")
    atomic_json(output / "summary.json", {"protocol": PROTOCOL, "run_identity": run_identity,
        "scope": "synthetic_fixture" if args.smoke else "complete_network_track",
        "metric_conventions": METRIC_CONVENTIONS, "comparisons": _summary(records)})
    base_fields = ["dataset", "method", "seed", "status", "measurement_scope", "supervision", "detector_seconds", "total_wall_seconds", "sampled_peak_rss_bytes", "worker_peak_rss_bytes"]
    table_rows = []
    for record in records:
        row = {key: record.get(key) for key in base_fields}
        for section in ("recovery", "structural", "omega", "audit"):
            for key, value in record.get(section, {}).items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    row[f"{section}__{key}"] = value
        for key, value in (record.get("audit", {}).get("objective") or {}).items():
            row[f"objective__{key}"] = value
        row["onmi__status"] = record.get("onmi", {}).get("status", "not_scored")
        row["failure__stage"] = record.get("stage", (record.get("last_committed_stage") or {}).get("phase"))
        row["failure__reason"] = record.get("reason", record.get("error"))
        table_rows.append(row)
    fields = base_fields + sorted(set().union(*(r.keys() for r in table_rows)) - set(base_fields))
    with (output / "summary.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)
    (output / "summary.csv.tmp").replace(output / "summary.csv")
    terminal = {"run_identity": run_identity, "status": "recorded", "completed_units": len(records),
                "total_units": total, "outcomes": dict(Counter(r["status"] for r in records)),
                "manifest": file_identity(manifest_path), "summary": file_identity(output / "summary.json"),
                "elapsed_seconds": time.monotonic() - started,
                "meaning": "all outcomes recorded; this is not a claim all detections completed"}
    atomic_json(output / "terminal.json", terminal)
    json.loads((output / "terminal.json").read_text())
    atomic_json(output / "progress.json", dict(terminal, phase="terminal_verified", fraction=1,
                active_item=None, eta_seconds=0, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(terminal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
