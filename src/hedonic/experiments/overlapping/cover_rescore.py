"""Score the covers a benchmark saved again, with other metric settings, without running any detector.

A benchmark run with ``--save-covers`` keeps every unit's cover (``cover_store``); this command loads each pair's graph
and reference cover once, as the runner does, checks it is the graph the detector saw (vertex and edge counts, and the
SHA-256 newer records carry), and scores the saved cover with ``codeseg_reproduction.score_cover``, the runner's own
scoring. What can change:

``--no-paper-filter``      score the whole cover against the whole reference (the paper filter restricts both to the
                           reference's communities of at least three vertices and their vertices)
``--omega/--no-omega``     the sampled Omega index, ``--omega-sample-size`` its vertex-pair sample
``--singleton-mode``,      the options of ``hedonic.metrics.evaluate_cover``
``--matching-weight``
``--nf1``                  also Rossetti et al.'s NF1 (``hedonic.metrics.normalized_f1_rossetti``)
``--as-recorded``          each unit's own settings (its run's paper filter, Omega and sample size): the record's metrics
``--check``                ``--as-recorded``, and compare every metric with the record (the morning check of scoring)

    hedonic-exp overlapping-rescore --source RUN_DIR --networks dblp:top5000,amazon:all --dry-run
    hedonic-exp overlapping-rescore --source RUN_DIR --networks dblp:top5000 --no-paper-filter --nf1 --name whole
    hedonic-exp overlapping-rescore --source RUN_DIR --networks dblp:top5000 --check

The output folder (``<first source>/rescore/<name>`` unless ``--output-dir``) holds ``rescore.jsonl`` (one JSON line
per unit and scoring setting, appended and fsynced: a rerun skips what is recorded), ``rescore.csv`` and
``summary.json``. Records are keyed by the setting's digest, so several settings can share a folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hedonic.experiments.config import NETWORKS_DIR, expand_path
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import cover_store
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import replication
from hedonic.experiments.overlapping.start_quality import append_record, repair_tail

PROTOCOL = "cover-rescore-v1"
RESULTS_NAME = "rescore.jsonl"
TABLE_NAME = "rescore.csv"
SUMMARY_NAME = "summary.json"
SINGLETON_MODES = ("all", "size_ge_2")
MATCHING_WEIGHTS = ("f1", "jaccard")
SUMMARY_METRICS = ("f1", "onmi", "omega", "nf1", "jaccard", "matching_f1")
CHECK_ABS = 1e-12  # a rescored metric this close to the record's is the same number (the scoring is deterministic)


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Score the covers a benchmark saved again (other metric settings), with no detector run.")
    parser.add_argument("--source", nargs="+", required=True, metavar="RUN_DIR",
                        help="benchmark run folders holding the units' runs/ and covers/ (the first with a unit wins)")
    parser.add_argument("--networks", required=True, help="comma list of NETWORK[:COVER] targets, e.g. dblp:top5000")
    parser.add_argument("--methods", help="comma list of method keys (default: those found in the sources)")
    parser.add_argument("--seeds", default=ca.DEFAULT_SEEDS, help=f"seeds (default {ca.DEFAULT_SEEDS})")
    parser.add_argument("--replication", choices=replication.MODES, default=replication.DEFAULT_MODE,
                        help="auto (default): a deterministic method's cover at the lowest seed only, as the benchmark "
                             "runs it; all: at every seed")
    parser.add_argument("--induce", choices=("full", "covered"), default=None,
                        help="the graph the source runs detected on (default: whichever folder the unit has)")
    parser.add_argument("--network-root", help="folder of the SNAP networks (default HEDONIC_NETWORKS_DIR)")
    parser.add_argument("--snap-cache-dir", help="cache of the streamed covered subgraphs (default <cache_dir>/snap)")
    parser.add_argument("--paper-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="restrict cover and reference to the reference's scored communities (default on)")
    parser.add_argument("--omega", action=argparse.BooleanOptionalAction, default=True, help="sampled Omega index (default on)")
    parser.add_argument("--omega-sample-size", type=int, default=100_000, help="vertex pairs Omega samples (default 100000)")
    parser.add_argument("--singleton-mode", choices=SINGLETON_MODES, default="all",
                        help="which communities the node-membership metrics count (default all)")
    parser.add_argument("--matching-weight", choices=MATCHING_WEIGHTS, default="f1",
                        help="weight of the one-to-one community matching (default f1)")
    parser.add_argument("--nf1", action="store_true", help="also NF1 (Rossetti, Pappalardo and Rinzivillo 2016)")
    parser.add_argument("--as-recorded", action="store_true",
                        help="each unit's own scoring settings (its run's paper filter and Omega), as its record has them")
    parser.add_argument("--check", action="store_true",
                        help="--as-recorded, and compare every metric with the record's; exit 1 on any difference")
    parser.add_argument("--name", help="name of this rescoring (default rescore, or check under --check)")
    parser.add_argument("--output-dir", help="output folder (default <first source>/rescore/NAME)")
    parser.add_argument("--overwrite", action="store_true", help="delete this folder's records before starting")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load nothing and write nothing")
    return parser


def resolve(argv: list[str], prog: str = "hedonic-exp overlapping-rescore") -> dict[str, Any]:
    """The plan of a rescoring. Reads folder names only: no graph, no record, nothing written."""
    args = build_parser(prog).parse_args(argv)
    networks = [net for net in (part.strip() for part in args.networks.split(",")) if net]
    unknown = [net for net in networks if qs.split_target(net)[0] not in qs.NETWORKS]
    if unknown:
        raise ValueError(f"unknown network(s) {', '.join(unknown)}; choose from {', '.join(qs.NETWORKS)}")
    seeds = qs.parse_seeds(args.seeds)
    sources = [expand_path(source).resolve() for source in args.source]
    methods = (qs.parse_choice(args.methods, [m.key for m in qs.METHODS], qs.ALIASES, "method", qs.CONTROL_KEYS)
               if args.methods else ca.discover_methods(sources, networks, seeds, args.induce))
    recorded = bool(args.as_recorded or args.check)
    spec = None if recorded else {
        "paper_filter": bool(args.paper_filter), "compute_omega": bool(args.omega),
        "omega_sample_size": int(args.omega_sample_size), "singleton_mode": args.singleton_mode,
        "matching_weight": args.matching_weight, "nf1": bool(args.nf1)}
    name = qs.parse_name(args.name or ("check" if args.check else "rescore"))
    output = (expand_path(args.output_dir).resolve() if args.output_dir
              else (sources[0] / "rescore" / name).resolve())
    from hedonic.experiments.overlapping import userconfig

    cache = (expand_path(args.snap_cache_dir) if args.snap_cache_dir
             else expand_path(userconfig.defaults().get("cache_dir") or "~/.cache/hedonic") / "snap")
    return {"sources": sources, "networks": networks, "methods": methods, "seeds": seeds, "induce": args.induce,
            "replication": args.replication, "spec": spec, "nf1": bool(args.nf1), "check": bool(args.check),
            "name": name, "output": output, "overwrite": bool(args.overwrite), "dry_run": bool(args.dry_run),
            "network_root": expand_path(args.network_root or NETWORKS_DIR).resolve(), "snap_cache_dir": cache}


def spec_digest(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def units(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (pair, seed, method) of the plan, pair outermost (a pair's graph is loaded once)."""
    out = []
    for network, method, seed in itertools.product(plan["networks"], plan["methods"], plan["seeds"]):
        if seed in replication.planned_seeds(method, plan["seeds"], plan["replication"]):
            out.append({"network": network, "seed": seed, "method": method})
    return out


def recorded_spec(record: dict[str, Any], unit_dir: Path, nf1: bool) -> dict[str, Any]:
    """The scoring settings a unit's record was scored with (its run's preflight config)."""
    config = {}
    for name in ("preflight.json", "manifest.json"):
        try:
            config = json.loads((unit_dir / name).read_text()).get("config") or {}
        except (OSError, ValueError, AttributeError):
            continue
        if config:
            break
    metrics = record.get("metrics") or {}
    return {"paper_filter": bool(record.get("paper_filter_enabled", config.get("paper_filter", True))),
            "compute_omega": bool(config.get("compute_omega", "omega" in metrics)),
            "omega_sample_size": int(config.get("omega_sample_size", 100_000)),
            "singleton_mode": "all", "matching_weight": "f1", "nf1": nf1}


def differences(rescored: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Metrics of the record that the rescoring does not reproduce (missing, or another value)."""
    out = []
    for key, value in recorded.items():
        new = rescored.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if new != value:
                out.append(key)
        elif not isinstance(new, (int, float)) or abs(float(new) - float(value)) > CHECK_ABS:
            out.append(key)
    return out


def load_pair(plan: dict[str, Any], network: str, variant: str, induce: str):
    from hedonic.experiments.overlapping import codeseg_reproduction as cr

    return cr._load_dataset(qs.split_target(network)[0], cover_variant=variant, network_root=plan["network_root"],
                            max_nodes=None, smoke=False, snap_cache_dir=plan["snap_cache_dir"], allow_catalog=False,
                            induce=induce)


def rescore_unit(unit: dict[str, Any], plan: dict[str, Any], loaded: dict) -> dict[str, Any]:
    """One unit's record of this rescoring (a unit whose run ended without a cover is a finished record too)."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr
    from hedonic.metrics import normalized_f1_rossetti

    network, seed, method = unit["network"], unit["seed"], unit["method"]
    source, path = ca.find_record(plan, network, seed, method)
    if path is None:
        raise FileNotFoundError(f"no benchmark record for {network} seed {seed} {method} in the sources")
    record = json.loads(path.read_text())
    unit_dir = path.parents[2]
    spec = plan["spec"] or recorded_spec(record, unit_dir, plan["nf1"])
    digest = spec_digest(spec)
    base = {"key": key_of(network, seed, method, digest), "protocol": PROTOCOL, "network": network, "seed": seed,
            "method": method, "spec": spec, "spec_digest": digest, "source": str(source),
            "record_file": str(path.relative_to(source)), "config_digest": record.get("config_digest")}
    if record.get("status") != "completed":
        return {**base, "status": "no_cover", "reason": f"the detector run ended {record.get('status')!r}"}
    if not record.get("cover_path"):
        return {**base, "status": "no_cover", "reason": "the run saved no cover (it needs --save-covers)"}
    report = record.get("dataset_report") or {}
    variant = str(report.get("cover_variant") or "all")
    induce = "covered" if unit_dir.parent.name.endswith("-covered") or plan.get("induce") == "covered" else "full"
    graph_key = (network, variant, induce)
    if graph_key not in loaded:
        loaded.clear()  # one pair at a time: a covered LiveJournal is GBs
        print(f"  loading {network} ({variant} cover, {induce} graph)", flush=True)
        dataset = load_pair(plan, network, variant, induce)
        ground_truth, gt_nodes, _report = cr._cover_for_paper_metrics(dataset)
        loaded[graph_key] = (dataset, ground_truth, gt_nodes, cover_store.graph_sha256(dataset.graph))
    dataset, ground_truth, gt_nodes, graph_sha = loaded[graph_key]
    if (report.get("n"), report.get("m")) != (dataset.graph.vcount(), dataset.graph.ecount()):
        raise ValueError(f"{path}: the record was produced on a graph of n={report.get('n')}, m={report.get('m')}, "
                         f"but the loaded one has n={dataset.graph.vcount()}, m={dataset.graph.ecount()}")
    if report.get("graph_sha256") and report["graph_sha256"] != graph_sha:
        raise ValueError(f"{path}: the record names graph {report['graph_sha256'][:12]}..., the loaded graph is "
                         f"{graph_sha[:12]}... (same n and m, other edges)")
    cover_file = unit_dir / record["cover_path"]
    cover = cover_store.read_json_gz(cover_file)
    started = time.perf_counter()
    options = {"singleton_mode": spec["singleton_mode"], "matching_weight": spec["matching_weight"]}
    metrics, prediction, reference = cr.score_cover(
        cover, dataset, ground_truth, gt_nodes, paper_filter=spec["paper_filter"], compute_omega=spec["compute_omega"],
        omega_sample_size=spec["omega_sample_size"], seed=int(record.get("seed", seed)), **options)
    if spec["nf1"]:
        metrics["nf1"] = float(normalized_f1_rossetti(prediction, reference))
    out = {**base, "status": "completed", "metrics": metrics, "graph_sha256": graph_sha,
           "cover_file_sha256": hashlib.sha256(cover_file.read_bytes()).hexdigest(),
           "scored_prediction_communities": len(prediction), "scored_reference_communities": len(reference),
           "seconds": round(time.perf_counter() - started, 3)}
    if plan["check"]:
        out["differs"] = differences(metrics, record.get("metrics") or {})
    return out


def key_of(network: str, seed: int, method: str, digest: str) -> str:
    return f"{network}/seed{seed}/{method}/{digest[:12]}"


# --------------------------------------------------------------------------- outputs
def table_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records.values():
        row = {key: record.get(key) for key in ("network", "seed", "method", "status", "spec_digest")}
        for key, value in sorted((record.get("metrics") or {}).items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
        if "differs" in record:
            row["differs"] = ";".join(record["differs"])
        rows.append(row)
    return rows


def write_outputs(plan: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    out = plan["output"]
    rows = table_rows(records)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out / TABLE_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for record in records.values():
        if record.get("status") != "completed":
            continue
        cell = cells.setdefault((record["network"], record["method"], record["spec_digest"]), {})
        for metric in SUMMARY_METRICS:
            value = (record.get("metrics") or {}).get(metric)
            if isinstance(value, (int, float)):
                cell.setdefault(metric, []).append(float(value))
    summary = {"protocol": PROTOCOL, "sources": [str(s) for s in plan["sources"]], "cells": [
        {"network": net, "method": method, "spec_digest": digest,
         **{metric: {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
            for metric, v in values.items()}}
        for (net, method, digest), values in cells.items()]}
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plan_lines(plan: dict[str, Any]) -> list[str]:
    todo = units(plan)
    missing = [u for u in todo if ca.find_record(plan, u["network"], u["seed"], u["method"])[1] is None]
    spec = plan["spec"]
    scoring = ("each unit's own settings" + (" (compared with its record)" if plan["check"] else "")) if spec is None else (
        f"paper filter {'on' if spec['paper_filter'] else 'off'} · Omega "
        + (f"on ({spec['omega_sample_size']:,} pairs)" if spec["compute_omega"] else "off")
        + f" · singletons {spec['singleton_mode']} · matching {spec['matching_weight']}"
        + (" · + NF1" if spec["nf1"] else ""))
    lines = [f"hedonic-exp overlapping-rescore · {len(plan['networks'])} pairs × {len(plan['methods'])} methods · "
             f"{len(todo):,} units ({PROTOCOL})",
             "  sources  " + ", ".join(map(str, plan["sources"])),
             "  pairs    " + ", ".join(plan["networks"]),
             "  methods  " + (", ".join(plan["methods"]) or "none found in the sources"),
             "  scoring  " + scoring,
             f"  records  {plan['output'] / RESULTS_NAME}"]
    if missing:
        lines.append(f"  waiting  {len(missing):,} units have no benchmark record yet, e.g. "
                     f"{missing[0]['network']} seed {missing[0]['seed']} {missing[0]['method']}")
    return lines


def run(plan: dict[str, Any], emit: Callable[[str], Any] = print) -> int:
    out = plan["output"]
    out.mkdir(parents=True, exist_ok=True)
    results = out / RESULTS_NAME
    if plan["overwrite"]:
        for name in (RESULTS_NAME, TABLE_NAME, SUMMARY_NAME):
            (out / name).unlink(missing_ok=True)
    repair_tail(results)
    records = ca.read_records(results)
    todo = units(plan)
    loaded: dict = {}
    waiting, differ, done = [], [], 0
    emit(f"rescore: {len(todo)} units; records in {results}")
    for unit in todo:
        source, path = ca.find_record(plan, unit["network"], unit["seed"], unit["method"])
        if path is None:
            waiting.append(unit)
            continue
        if plan["spec"] is not None:  # a fixed setting: the key is known before the record is read
            key = key_of(unit["network"], unit["seed"], unit["method"], spec_digest(plan["spec"]))
            if key in records:
                done += 1
                continue
        record = rescore_unit(unit, plan, loaded)
        if record["key"] in records:
            done += 1
            continue
        append_record(results, record)
        records[record["key"]] = record
        done += 1
        if record.get("differs"):
            differ.append(record["key"])
        metrics = record.get("metrics") or {}
        tail = (f"F1 {metrics['f1']:.4f} · ONMI {metrics['onmi']:.4f}" if "f1" in metrics else record.get("reason", ""))
        if plan["check"] and record["status"] == "completed":
            tail += " · " + ("same as its record" if not record["differs"] else f"DIFFERS: {', '.join(record['differs'])}")
        emit(f"  [{done}/{len(todo)}] {record['key']} · {record['status']} · {tail}")
    write_outputs(plan, records)
    if waiting:
        emit(f"rescore: {len(waiting)} units have no benchmark record yet (first: {waiting[0]['network']} seed "
             f"{waiting[0]['seed']} {waiting[0]['method']})")
    emit(f"rescore: wrote {results}, {out / TABLE_NAME} and {out / SUMMARY_NAME}")
    if plan["check"]:
        emit(f"check: {len(differ)} units differ from their records" if differ else "check: every unit matches its record")
        return 1 if differ else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        plan = resolve(argv)
        ca.check_sources(plan)
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic-exp overlapping-rescore: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        print("  dry run: no graph loaded, nothing written")
        return 0
    try:
        return run(plan)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"hedonic-exp overlapping-rescore: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
