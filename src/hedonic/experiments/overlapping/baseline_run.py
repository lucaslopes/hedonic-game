"""The baselines' calibration as a durable, queueable run (``hedonic run calibrate-baselines``).

``baseline_calibration`` holds one calibration function per baseline method and the label-free criterion they share. This
module runs them for **every method on every network/cover pair** of a list, the way ``hedonic run calibrate`` does for
Hedonic, so a queue can have it as a step before the benchmark:

    hedonic run calibrate-baselines --networks dblp:top5k,amazon:all --name baseline-calibration
    hedonic run exp --config CONFIG --tuned hoc_tuned.json,baseline-calibration/baseline_tuned.json

For each pair the graph ``G[S]`` (the subgraph induced by the vertices the cover contains) is loaded once, and each method is
calibrated on it by its own function (searched parameters: CliquePerc ``k``, ANGEL ``phi``, BigCLAM and NEO-K-Means ``K``;
the others take the authors' published values). Nothing reads the ground-truth communities: the context a calibration
receives has no cover (``baseline_calibration.Context``). A method whose runs under the calibration's settings are replicas
(``replication.calibration_kind``; none of the registered methods now) is run ``--replicates`` times per setting and scored
by the mean.

The run folder holds ``baseline_tuned.json`` (``complete`` once every method of every pair has a record; ``settings`` is
``{method: {network:cover: method_params}}``, exactly the parameters the benchmark runner reads, which ``--tuned`` lays over
``method_params``; ``searches`` has each method's rule, candidates, stop reason, bracketing and notes), ``ledger/`` (every
evaluation, memoised: a resume or a larger budget reuses them), ``progress.json`` and ``worker.log``. A method that could not
be calibrated (runtime unavailable, every probe failed) is recorded with its reason and has no entry in ``settings``, so the
benchmark keeps its own default for it and says unavailable where that is the truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hedonic.experiments.config import RUNS_ROOT, expand_path
from hedonic.experiments.overlapping import baseline_calibration as bc
from hedonic.experiments.overlapping import replication
from hedonic.experiments.overlapping import parallel
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.runtime import runtime_versions

PROTOCOL = bc.PROTOCOL
TUNED_NAME = "baseline_tuned.json"
DEFAULT_NAME = "baseline-calibration"
DEFAULT_NETWORKS = "dblp:top5k,amazon:top5k"

Progress = Callable[[dict[str, Any]], None]


def _silent(event: dict[str, Any]) -> None:
    return None


# --------------------------------------------------------------------------- plan
def build_parser(prog: str = "hedonic run calibrate-baselines") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Calibrate the baseline methods on the graph induced by each pair's cover, without ground truth "
                     "(held-out pair likelihood, structural rules, or the authors' published values); writes "
                     "baseline_tuned.json for the benchmark's --tuned."))
    parser.add_argument("--networks", default=DEFAULT_NETWORKS,
                        help="comma list; network:cover (dblp:top5k) is the graph induced by that cover's vertices, a name "
                             f"alone is its `all` cover (default {DEFAULT_NETWORKS})")
    parser.add_argument("--covers", metavar="LIST", help="covers to cross with every network, e.g. all,top5k")
    parser.add_argument("--methods", default="all",
                        help=f"comma list or `all` of: {', '.join(bc.CALIBRATORS)} (the order is the order they run)")
    parser.add_argument("--network-root", help="folder of the SNAP networks (default HEDONIC_NETWORKS_DIR)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the held-out split and of the detector runs (default 0)")
    parser.add_argument("--max-evals", type=int, default=8, metavar="N",
                        help="settings tried per method and pair, at most (0 = no cap; default 8)")
    parser.add_argument("--max-minutes", type=float, default=0.0, metavar="M",
                        help="single-core minutes per method and pair (0 = no cap)")
    parser.add_argument("--eval-timeout-minutes", type=float, default=60.0, metavar="M",
                        help="one detector run is killed after M minutes and recorded as a failed probe (default 60)")
    parser.add_argument("--replicates", type=int, default=bc.DEFAULT_REPLICATES, metavar="R",
                        help="independent runs averaged per setting for the methods whose runs are replicas "
                             f"(BigCLAM, NEO-K-Means, FOX; default {bc.DEFAULT_REPLICATES}; 1 = one run)")
    parser.add_argument("--method-budget", default="", metavar="M=N[:MIN],...",
                        help="a method's own budget instead of --max-evals/--max-minutes, e.g. angel=3:60 (3 settings, 60 "
                             "single-core minutes per pair)")
    parser.add_argument("--transfer", default="", metavar="PAIRS",
                        help="pairs whose searched methods are not searched: each takes the setting of the nearest pair "
                             "calibrated in this run (the same network's other cover first, else the nearest density), K "
                             "scaled by the vertex count for BigCLAM and NEO-K-Means, recorded as transferred")
    parser.add_argument("--transfer-missing", action="store_true",
                        help="a search that ended without a setting (every probe failed or timed out) also takes a "
                             "transferred setting; a method whose runtime is unavailable never does")
    parser.add_argument("--adopt-from", default="", metavar="DIR,...",
                        help="earlier calibration folders: the searches of this plan they finished are copied in (with "
                             "their ledgers, so a search they left unfinished resumes from its evaluations); copied, never "
                             "moved; each copied search names its source and build")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="pairs of one method calibrated side by side (each forked child loads its own pair); default "
                             "the performance cores, HEDONIC_WORKERS overrides; a new pair starts only while free memory "
                             "is well above the floor, and the floor stops the run instead of swapping. 1 = one at a time")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also score the neighbours of the published settings (FOX 0.01, NcGame 0.005, CoDeSEG 0.2/0.4)")
    parser.add_argument("--runtime-manifest", help="the `hedonic setup` manifest (default ~/.cache/hedonic/codeseg/setup_manifest.json, HEDONIC_CODESEG_CACHE)")
    parser.add_argument("--config", metavar="FILE.toml",
                        help="a TOML file whose [calibrate_baselines] table gives the options below (keys are the flag names "
                             "with underscores, lists allowed for networks/covers/methods); explicit flags override it")
    parser.add_argument("--name", help=f"run name (default {DEFAULT_NAME})")
    parser.add_argument("--output-dir", help="run folder (default: the user config's output folder / NAME)")
    parser.add_argument("--overwrite", action="store_true", help="delete this calibration's results and ledger first")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load no graph, run nothing")
    return parser


def config_defaults(parser: argparse.ArgumentParser, path: str) -> dict[str, Any]:
    """The ``[calibrate_baselines]`` table of a TOML file as parser defaults (an unknown key is an error, naming the file)."""
    import tomllib

    from hedonic.experiments.resources import config_path

    file = config_path(path)  # a path, or the name of a shipped config (baseline-calibration-paper)
    try:
        data = tomllib.loads(file.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read --config {file}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{file}: {exc}") from exc
    table = data.get("calibrate_baselines", data)
    known = {action.dest for action in parser._actions if action.dest not in ("help", "config", "dry_run", "overwrite")}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"{file}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(known))}")
    out: dict[str, Any] = {}
    for key, value in table.items():
        out[key] = ",".join(str(v) for v in value) if isinstance(value, list) else value
    return out


def resolve(argv: list[str], prog: str = "hedonic run calibrate-baselines") -> dict[str, Any]:
    """The plan of ``hedonic run calibrate-baselines ARGV``. Reads nothing but the arguments; writes nothing."""
    from hedonic.experiments.overlapping import userconfig

    parser = build_parser(prog)
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        parser.set_defaults(**config_defaults(parser, pre.config))
    args = parser.parse_args(argv)
    networks = qs.parse_targets(args.networks)
    if args.covers:
        networks = qs.expand_covers(networks, args.covers)
    methods = list(bc.CALIBRATORS) if args.methods.strip() == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in bc.CALIBRATORS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}; choose from {', '.join(bc.CALIBRATORS)}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods names a method twice")
    if not networks or not methods:
        raise ValueError("nothing to calibrate")
    if args.max_evals < 0 or args.max_minutes < 0 or args.eval_timeout_minutes <= 0:
        raise ValueError("--max-evals and --max-minutes cannot be negative and --eval-timeout-minutes must be positive")
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    method_budgets = parse_method_budgets(args.method_budget, methods)
    transfer = qs.parse_targets(args.transfer) if args.transfer.strip() else []
    stray = [t for t in transfer if t not in networks]
    if stray:
        raise ValueError(f"--transfer names pair(s) not in --networks: {', '.join(stray)}")
    if transfer and len(transfer) == len(networks):
        raise ValueError("--transfer names every pair: there is no calibrated pair to transfer from")
    adopt = [expand_path(d.strip()).resolve() for d in args.adopt_from.split(",") if d.strip()]
    name = qs.parse_name(args.name or DEFAULT_NAME)
    root = Path(userconfig.defaults().get("output_dir") or RUNS_ROOT)
    output = expand_path(args.output_dir).resolve() if args.output_dir else expand_path(root / name).resolve()
    return {"networks": networks, "methods": methods, "network_root": args.network_root, "seed": int(args.seed),
            "max_evals": int(args.max_evals), "max_minutes": float(args.max_minutes),
            "eval_timeout_minutes": float(args.eval_timeout_minutes), "replicates": int(args.replicates), "workers": args.workers,
            "sensitivity": bool(args.sensitivity), "runtime_manifest": args.runtime_manifest,
            "method_budgets": method_budgets, "transfer": transfer, "transfer_missing": bool(args.transfer_missing),
            "adopt_from": adopt, "name": name, "output": output, "overwrite": bool(args.overwrite), "dry_run": bool(args.dry_run)}


def parse_method_budgets(text: str, methods: list[str]) -> dict[str, tuple[int, float]]:
    """``angel=3:60,clique_percolation=4`` -> {method: (settings, single-core minutes; 0 = no cap)}."""
    out: dict[str, tuple[int, float]] = {}
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        method, sep, value = part.partition("=")
        method = method.strip()
        if not sep or method not in bc.CALIBRATORS:
            raise ValueError(f"--method-budget {part!r}: expected METHOD=N[:MINUTES] with METHOD one of {', '.join(bc.CALIBRATORS)}")
        if method not in methods:
            raise ValueError(f"--method-budget names {method}, which is not in --methods")
        evals, _, minutes = value.partition(":")
        try:
            budget = (int(evals), float(minutes or 0))
        except ValueError:
            raise ValueError(f"--method-budget {part!r}: N must be an integer and MINUTES a number") from None
        if budget[0] < 0 or budget[1] < 0:
            raise ValueError(f"--method-budget {part!r} cannot be negative")
        out[method] = budget
    return out


def budget_of(plan: dict[str, Any], method: str | None = None) -> bc.Budget:
    evals, minutes = (plan.get("method_budgets") or {}).get(method, (plan["max_evals"], plan["max_minutes"]))
    return bc.Budget(max_evals=evals or None, max_seconds=minutes * 60 or None,
                     eval_timeout=plan["eval_timeout_minutes"] * 60)


def searches(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Method by method, in the plan's order; each method on every pair, in the plan's order."""
    return [{"key": f"{method}/{target}", "method": method, "network": target}
            for method in plan["methods"] for target in plan["networks"]]


def describe(plan: dict[str, Any]) -> str:
    def n(count, word):
        return f"{count} {word}{'' if count == 1 else 's'}"
    return (f"baseline calibration ({bc.CRITERION}) · {n(len(searches(plan)), 'calibration')} = "
            f"{n(len(plan['methods']), 'method')} x {n(len(plan['networks']), 'pair')}")


def replica_line(plan: dict[str, Any]) -> list[str]:
    """The replica policy, only when a method of the plan has runs that are replicas of one another."""
    replicated = [m for m in plan["methods"] if replication.calibration_kind(m) == "replicas"]
    if plan["replicates"] > 1 and replicated:
        return [f"  replicas {plan['replicates']} per setting for {', '.join(replicated)} (their seed does not reach the program)"]
    return []


def plan_lines(plan: dict[str, Any]) -> list[str]:
    done = recorded(plan)
    budget = budget_of(plan)
    lines = [f"hedonic run calibrate-baselines · {describe(plan)} ({PROTOCOL})",
             f"  build    {qs.environment_line()}",
             f"  folder   {plan['output']} ({TUNED_NAME}, ledger/, progress.json, worker.log)",
             "  inputs   G[S] only: the graph induced by each cover's vertices; the cover's communities are never read",
             f"  budget   at most {budget.max_evals or 'unlimited'} settings per method and pair"
             + (f", {plan['max_minutes']:g} single-core minutes" if plan["max_minutes"] else "")
             + f"; one detector run is killed after {plan['eval_timeout_minutes']:g} min",
             *replica_line(plan),
             f"  seed     {plan['seed']}" + ("   sensitivity: neighbours of the published settings are scored too"
                                              if plan["sensitivity"] else "")]
    for method, (evals, minutes) in (plan.get("method_budgets") or {}).items():
        lines.append(f"  budget   {method}: at most {evals or 'unlimited'} settings"
                     + (f", {minutes:g} single-core minutes" if minutes else "") + " per pair")
    if plan.get("transfer"):
        lines.append(f"  transfer {', '.join(plan['transfer'])}: not searched; each searched method takes the setting of "
                     "the nearest calibrated pair (same network first, else nearest density; K scaled by n), recorded "
                     "as transferred")
    if plan.get("transfer_missing"):
        lines.append("  transfer also every search that ends without a setting (not an unavailable runtime)")
    for source, keys in adoptable(plan).items():
        lines.append(f"  adopt    {len(keys)} finished searches from {source} (copied with its ledger at the start)")
    for method in plan["methods"]:
        info = bc.REPLICATION[method]
        fixed = method in bc.FIXED_METHODS
        lines.append(f"  {method:<19} {'published setting' if fixed else 'searched (' + bc.CRITERION + ')' if method != 'clique_percolation' else 'searched (percolation transition)':<42}"
                     f"benchmark: {info['runs']} run{'s' if info['runs'] != 1 else ''} per pair ({info['kind']})")
    lines.append(f"  present  {len(done & {s['key'] for s in searches(plan)})} of {len(searches(plan))} calibrations already recorded "
                 + ("(--overwrite: deleted first)" if plan["overwrite"] else "(a run resumes the rest)"))
    return lines


# --------------------------------------------------------------------------- results
def read_tuned(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads((plan["output"] / TUNED_NAME).read_text())
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def recorded(plan: dict[str, Any]) -> set[str]:
    return set((read_tuned(plan).get("searches") or {}))


def state_of(plan: dict[str, Any]) -> str:
    """complete | partial | missing, from the result file alone: complete once every search of *this* plan is recorded.

    The file's own ``complete`` flag describes the plan that wrote it, which may have had other pairs (a calibration
    copied from an earlier run, with its ledger, so its finished searches are reused); it is not consulted here."""
    done = recorded(plan)
    wanted = {s["key"] for s in searches(plan)}
    if wanted <= done:
        return "complete"
    return "partial" if done & wanted else "missing"


def mark_complete(plan: dict[str, Any]) -> dict[str, Any]:
    """Set the file's ``complete`` flag for this plan (atomically) and return the file's content."""
    data = read_tuned(plan)
    flag = {s["key"] for s in searches(plan)} <= set(data.get("searches") or {})
    if data and data.get("complete") != flag:
        data["complete"] = flag
        path = plan["output"] / TUNED_NAME
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)
    return data


def record_search(plan: dict[str, Any], search: dict[str, Any], result: bc.CalibrationResult, graph: dict[str, Any],
                  **extra: Any) -> dict[str, Any]:
    """Add one finished calibration to baseline_tuned.json (atomically) and return the file's content."""
    entry = {**result.to_dict(), "network": search["network"], "graph": graph,
             "runner_params": bc.runner_params(result.method, result.params), "environment": runtime_versions(), **extra}
    data = read_tuned(plan) or {}
    data.update({"protocol": PROTOCOL, "criterion": bc.CRITERION, "uses_ground_truth": False, "environment": runtime_versions(),
                 "plan": {k: plan[k] for k in ("seed", "max_evals", "max_minutes", "eval_timeout_minutes", "replicates",
                                               "sensitivity")}})
    data.setdefault("searches", {})[search["key"]] = entry
    settings = data.setdefault("settings", {}).setdefault(search["method"], {})
    if result.params is not None:
        settings[search["network"]] = entry["runner_params"]
    else:
        settings.pop(search["network"], None)
    data["complete"] = {s["key"] for s in searches(plan)} <= set(data["searches"])
    write_tuned(plan, data)
    return data


def write_tuned(plan: dict[str, Any], data: dict[str, Any]) -> None:
    path = plan["output"] / TUNED_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


ADOPTABLE = ("calibrated", "fixed")  # a failed or unavailable search is not adopted: this run tries it again


def adoptable(plan: dict[str, Any]) -> dict[str, list[str]]:
    """{source folder: keys of this plan it finished and this run has not recorded}, read from the files only."""
    mine = recorded(plan)
    wanted = [s["key"] for s in searches(plan)]
    out: dict[str, list[str]] = {}
    for source in plan.get("adopt_from") or []:
        found = read_tuned({"output": Path(source)}).get("searches") or {}
        keys = [k for k in wanted if k not in mine and (found.get(k) or {}).get("status") in ADOPTABLE]
        mine |= set(keys)
        out[str(source)] = keys
    return out


def adopt(plan: dict[str, Any], emit: Callable[[str], Any] = print) -> int:
    """Copy into this run the searches of its plan that earlier calibrations finished, and those calibrations' ledgers.

    A source must be the same protocol and criterion and must not use the ground truth; anything else is refused. The
    copied search keeps its own record and gains ``adopted_from`` (source folder and the build that ran it). Ledger files
    (``evals-METHOD.jsonl``, memoised by method, parameters, seed and graph fingerprint, not by build) are copied when this
    run has none for that pair and method, so a search a source left unfinished resumes from its evaluations. Nothing in a
    source is changed."""
    import shutil

    count = 0
    for source, keys in adoptable(plan).items():
        src = read_tuned({"output": Path(source)})
        if not src:
            emit(f"  adopt: {source} has no {TUNED_NAME}; nothing copied")
            continue
        if src.get("protocol") != PROTOCOL or src.get("criterion") != bc.CRITERION or src.get("uses_ground_truth"):
            raise ValueError(f"--adopt-from {source}: protocol {src.get('protocol')} / criterion {src.get('criterion')} "
                             f"is not this calibration's ({PROTOCOL} / {bc.CRITERION})")
        data = read_tuned(plan) or {"protocol": PROTOCOL, "criterion": bc.CRITERION, "uses_ground_truth": False}
        for key in keys:
            entry = dict(src["searches"][key])
            entry["adopted_from"] = {"run": source, "environment": entry.get("environment") or src.get("environment"),
                                     "plan": src.get("plan")}
            method, target = key.split("/", 1)
            data.setdefault("searches", {})[key] = entry
            if entry.get("runner_params") is not None:
                data.setdefault("settings", {}).setdefault(method, {})[target] = entry["runner_params"]
            count += 1
        data["complete"] = {s["key"] for s in searches(plan)} <= set(data.get("searches") or {})
        write_tuned(plan, data)
        copied = 0
        for target in plan["networks"]:
            folder = Path(source) / "ledger" / target.replace(":", "_")
            for file in sorted(folder.glob("evals-*.jsonl")) if folder.is_dir() else []:
                method = file.name[len("evals-"):-len(".jsonl")]
                dest = plan["output"] / "ledger" / folder.name / file.name
                if method in plan["methods"] and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
                    copied += 1
        emit(f"  adopted {len(keys)} finished searches and {copied} ledger files from {source}")
    return count


K_KEYS = {"bigclam": "communities", "neo_kmeans": "clusters"}  # a count of communities: scaled by the vertex count


def _density(graph: dict[str, Any] | None) -> float | None:
    if not graph or not graph.get("n") or graph["n"] < 2:
        return None
    return 2.0 * graph["m"] / (graph["n"] * (graph["n"] - 1))


def transfer_source(method: str, target: str, results: dict[str, Any], graph: dict[str, Any] | None) -> str | None:
    """The calibrated pair a transferred setting comes from: the same network's other cover if it was calibrated, else
    the pair whose G[S] density is nearest (log scale). Only searched, not transferred, settings are sources."""
    import math

    pool = {k.split("/", 1)[1]: v for k, v in results.items()
            if k.split("/", 1)[0] == method and v.get("status") == "calibrated" and v.get("params") is not None
            and k.split("/", 1)[1] != target}
    if not pool:
        return None
    network = target.split(":", 1)[0]
    same = sorted(t for t in pool if t.split(":", 1)[0] == network)
    if same:
        return same[0]
    want = _density(graph)

    def gap(t: str) -> float:
        have = _density(pool[t].get("graph"))
        return abs(math.log(have) - math.log(want)) if want and have else math.inf
    return min(sorted(pool), key=gap)


def transferred_result(method: str, target: str, results: dict[str, Any], graph: dict[str, Any] | None,
                       why: str) -> tuple[bc.CalibrationResult, str | None]:
    """The setting of the nearest calibrated pair, as a record that says so (status ``transferred``)."""
    source = transfer_source(method, target, results, graph)
    if source is None:
        return (bc.CalibrationResult(method, "failed", None, f"{why}; no calibrated pair of {method} to transfer from",
                                     None, stop_reason="no transfer source", replication=dict(bc.REPLICATION[method])),
                None)
    record = results[f"{method}/{source}"]
    params = dict(record["params"])
    notes = [f"transferred from {source}: not searched on {target} ({why})"]
    key = K_KEYS.get(method)
    if key in params:
        n_from, n_to = (record.get("graph") or {}).get("n"), (graph or {}).get("n")
        if n_from and n_to:
            scaled = max(2, int(round(params[key] * n_to / n_from)))
            notes.append(f"{key} {params[key]} x n {n_to} / {n_from} = {scaled} (the same fraction K/n as {source})")
            params[key] = scaled
        else:
            notes.append(f"{key} not scaled: a vertex count is not known")
    rule = f"transferred from {source} ({why}); no search on this pair, no ground truth"
    return (bc.CalibrationResult(method, "transferred", params, rule, None, criterion=None, stop_reason="transferred",
                                 notes=notes, replication=dict(bc.REPLICATION[method])), source)


def reset(plan: dict[str, Any]) -> None:
    (plan["output"] / TUNED_NAME).unlink(missing_ok=True)
    ledger = plan["output"] / "ledger"
    for path in ledger.rglob("*") if ledger.is_dir() else []:
        if path.is_file():
            path.unlink()


# --------------------------------------------------------------------------- run
START_STAGGER_SECONDS = 20.0  # between two pairs of a method starting: one's memory is visible before the next begins
ADMIT_MARGIN = 15.0           # a pair starts only while the machine's free memory is this many points above the floor


def _memory_allows(running: int) -> bool:
    """May another pair start now? Yes unless free memory is within ``ADMIT_MARGIN`` points of the floor."""
    free = parallel.free_memory_percent()
    return free is None or free >= parallel.memory_floor() + ADMIT_MARGIN


def run_calibration(plan: dict[str, Any], *, progress: Progress = _silent, emit: Callable[[str], Any] = print) -> int:
    out = plan["output"]
    out.mkdir(parents=True, exist_ok=True)
    if plan["overwrite"]:
        reset(plan)
    todo = searches(plan)
    wanted = {s["key"] for s in todo}
    if plan.get("adopt_from"):
        adopt(plan, emit)
    done = recorded(plan) & wanted
    emit(f"baseline calibration: {len(todo)} calibrations; {len(done)} already recorded")
    finished = len(done)
    from hedonic.experiments.overlapping import extra_methods

    extra_methods.install()  # opt-in runners (Ego-Splitting) outside the 25-method registry
    runtime = bc.runtime_parameters(plan["runtime_manifest"])
    workers = parallel.default_workers(plan.get("workers"))
    sizes = {s_["network"]: s_.get("graph") for s_ in (read_tuned(plan).get("searches") or {}).values()}

    def calibrate_one(search):
        """One method on one pair, in a forked child that loads the pair's graph (an unloaded graph costs nothing to hold).
        A fixed-protocol method without a sensitivity check runs nothing, so its graph is not loaded: the record is the
        authors' setting, the same on every pair."""
        target = search["network"]
        if search["method"] in bc.FIXED_METHODS and not plan["sensitivity"]:
            return bc.calibrate(search["method"], None), sizes.get(target) or {}
        ctx = bc.load_context(target, network_root=plan["network_root"], induce="covered", seed=plan["seed"],
                              budget=budget_of(plan, search["method"]), ledger_dir=out / "ledger" / target.replace(":", "_"), runtime=runtime)
        bc.with_replicates(ctx, plan["replicates"])
        ctx.replica_workers = max(1, min(int(plan["replicates"] or 1), 3))  # a setting's replicas run side by side
        graph = {"n": ctx.graph.vcount(), "m": ctx.graph.ecount(), "sha256": ctx.fingerprint()}
        try:
            result = bc.calibrate_method(ctx, search["method"], sensitivity=plan["sensitivity"])
        except Exception as exc:  # noqa: BLE001 - one method's failure is a record, never the end of the others
            result = bc.CalibrationResult(search["method"], "failed", None, "calibration raised an exception", None,
                                          notes=[f"{type(exc).__name__}: {exc}"[:500]],
                                          replication=dict(bc.REPLICATION[search["method"]]))
        return result, graph

    # Method by method (the plan's order: the cheapest first), every pair of a method side by side. A pair's work is
    # independent of the others (its own ledger folder, its own detector children); calibration time is not a result.
    def size_of(search):
        """G[S]'s size for a pair no search has loaded yet (a transfer scales K by it), in a forked child."""
        ctx = bc.load_context(search["network"], network_root=plan["network_root"], induce="covered", seed=plan["seed"],
                              budget=budget_of(plan), ledger_dir=None, runtime=runtime)
        return {"n": ctx.graph.vcount(), "m": ctx.graph.ecount(), "sha256": ctx.fingerprint()}

    transfer = set(plan.get("transfer") or [])
    for method in plan["methods"]:
        pending = [s_ for s_ in todo if s_["method"] == method and s_["key"] not in done]
        moved = [s_ for s_ in pending if s_["network"] in transfer and method not in bc.FIXED_METHODS]
        pending = [s_ for s_ in pending if s_ not in moved]
        finished += _run_method(plan, method, pending, sizes, workers, calibrate_one, finished, len(todo), progress, emit)
        if plan.get("transfer_missing"):
            results = read_tuned(plan).get("searches") or {}
            moved += [s_ for s_ in todo if s_["method"] == method and s_ not in moved
                      and (results.get(s_["key"]) or {}).get("status") == "failed"
                      and (results.get(s_["key"]) or {}).get("stop_reason") != "no transfer source"]
        unknown = [s_ for s_ in moved if not sizes.get(s_["network"])]
        for index, graph in parallel.run_forked(size_of, unknown, workers, admit=_memory_allows) if unknown else []:
            sizes[unknown[index]["network"]] = graph
        for search in moved:
            results = read_tuned(plan).get("searches") or {}
            earlier = results.get(search["key"])
            why = ("its search ended without a setting" if earlier else "--transfer: this pair is not searched")
            result, source = transferred_result(method, search["network"], results, sizes.get(search["network"]), why)
            extra = {"transferred_from": source}
            if earlier:
                extra["failed_search"] = {k: earlier.get(k) for k in ("status", "rule", "candidates", "stop_reason",
                                                                       "evaluations", "seconds", "notes")}
            record_search(plan, search, result, sizes.get(search["network"]) or {}, **extra)
            if not earlier:
                finished += 1
            emit(f"  {qs.snap.public_target(search['network']):<22} {result.status:<11} "
                 f"{json.dumps(bc.runner_params(method, result.params))}  [from {source or 'nowhere'}]")
            progress({"type": "unit", "done": finished, "total": len(todo), "key": search["key"]})
    final = mark_complete(plan)
    problems = [k for k, v in (final.get("searches") or {}).items() if v.get("status") in ("failed", "unavailable")]
    if problems:
        emit("baseline calibration: no setting for " + ", ".join(sorted(problems)) + " (the benchmark keeps its default)")
    moved_all = sorted(k for k, v in (final.get("searches") or {}).items() if v.get("status") == "transferred" and k in wanted)
    if moved_all:
        emit("baseline calibration: transferred (not searched) " + ", ".join(
            f"{k} <- {final['searches'][k].get('transferred_from')}" for k in moved_all))
    emit(f"baseline calibration: {'complete' if final.get('complete') else 'NOT complete'}; wrote {out / TUNED_NAME}")
    return 0 if final.get("complete") else 1


def _run_method(plan, method, pending, sizes, workers, calibrate_one, finished, total, progress, emit) -> int:
    """Every pending pair of one method, side by side; returns how many were recorded."""
    if not pending:
        return 0
    # largest graph first, so the long jobs start early; a pair whose size is not recorded yet is assumed large
    pending.sort(key=lambda s_: -(sizes.get(s_["network"]) or {"m": float("inf")})["m"])
    emit(f"== {method}: {len(pending)} pairs, {min(workers, len(pending))} at a time")
    progress({"type": "stage", "text": f"{method}: calibrating {len(pending)} pairs, {workers} at once"})
    count = 0
    for index, (result, graph) in parallel.run_forked(calibrate_one, pending, workers, admit=_memory_allows,
                                                     stagger=START_STAGGER_SECONDS):
        search = pending[index]
        sizes[search["network"]] = graph
        record_search(plan, search, result, graph)
        count += 1
        bracket = "" if result.bracketed is None else ", bracketed" if result.bracketed else ", NOT bracketed"
        emit(f"  {qs.snap.public_target(search['network']):<22} {result.status:<11} {json.dumps(bc.runner_params(method, result.params))}"
             f"  [{result.evaluations} settings, {result.seconds / 60:.1f} min, stop: {result.stop_reason}{bracket}]")
        progress({"type": "unit", "done": finished + count, "total": total, "key": search["key"]})
    return count


# --------------------------------------------------------------------------- CLI entry points
def front_study(argv: list[str]) -> dict[str, Any]:
    """The plan of ``hedonic run calibrate-baselines ARGV``. Loads nothing and writes nothing."""
    return resolve([arg for arg in argv if arg != "--detach"])


def launch_argv(plan: dict[str, Any]) -> list[str]:
    """Every setting spelled out, so a resume rebuilds the same plan."""
    argv = ["--networks", ",".join(plan["networks"]), "--methods", ",".join(plan["methods"]), "--seed", str(plan["seed"]),
            "--max-evals", str(plan["max_evals"]), "--max-minutes", f"{plan['max_minutes']:g}",
            "--eval-timeout-minutes", f"{plan['eval_timeout_minutes']:g}", "--replicates", str(plan["replicates"]),
            *(["--workers", str(plan["workers"])] if plan.get("workers") else []),
            *(["--method-budget", ",".join(f"{m}={e}:{mi:g}" for m, (e, mi) in plan["method_budgets"].items())]
              if plan.get("method_budgets") else []),
            *(["--transfer", ",".join(plan["transfer"])] if plan.get("transfer") else []),
            *(["--transfer-missing"] if plan.get("transfer_missing") else []),
            *(["--adopt-from", ",".join(str(d) for d in plan["adopt_from"])] if plan.get("adopt_from") else []),
            "--name", plan["name"], "--output-dir", str(plan["output"])]
    if plan["network_root"]:
        argv += ["--network-root", str(plan["network_root"])]
    if plan["runtime_manifest"]:
        argv += ["--runtime-manifest", str(plan["runtime_manifest"])]
    return argv + (["--sensitivity"] if plan["sensitivity"] else [])


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m``; the console verb is ``hedonic run calibrate-baselines``."""
    return front_main(list(sys.argv[1:] if argv is None else argv))


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run calibrate-baselines``: a durable run; ``--dry-run`` only prints the plan."""
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    detach = "--detach" in argv
    try:
        plan = resolve([arg for arg in argv if arg != "--detach"])
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic run calibrate-baselines: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        print(style("  dry run: nothing started, nothing written", "dim"))
        return 0
    if plan["overwrite"]:
        try:
            runmanager._refuse_if_running(plan["name"])  # never delete results under a live worker
        except SystemExit as exc:
            print(f"hedonic run calibrate-baselines: {exc.code}", file=sys.stderr)
            return 2
        reset(plan)
    entry = runmanager.launch_baseline_calibration(launch_argv(plan), plan["name"], str(plan["output"]))
    where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
             else f"background process (log {entry['log']})")
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


# --------------------------------------------------------------------------- durable run
def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body of a baseline calibration: the calibrations, with their progress mirrored into progress.json."""
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    plan = resolve(entry["argv"], prog="hedonic run calibrate-baselines")
    todo = searches(plan)
    already = len(recorded(plan) & {s["key"] for s in todo})
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "baseline_calibrate", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": describe(plan), "command": entry["command"],
                 "environment": qs.environment(), "done": already, "total": len(todo), "resumed_from": already,
                 "eta_seconds": None, "stage": "starting the calibration", "output": str(plan["output"] / TUNED_NAME),
                 "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(plan))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    started = time.time()

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "stage":
            prog.update(status="running", stage=event["text"], updated=time.time())
        elif event["type"] == "unit":
            fresh = event["done"] - prog["resumed_from"]
            eta = (time.time() - started) / fresh * (event["total"] - event["done"]) if fresh else None
            prog.update(status="running", done=event["done"], total=event["total"], last=event["key"], eta_seconds=eta,
                        updated=time.time())
        rm.write_json(ppath, prog)

    prog.update(status="running", updated=time.time())
    rm.write_json(ppath, prog)
    code = run_calibration(plan, progress=on_event)
    if code != 0:
        raise RuntimeError("the baseline calibration did not complete")
    prog.update(status="completed", done=len(todo), eta_seconds=0, stage=None, finished=time.time(), updated=time.time())
    rm.write_json(ppath, prog)
    print(style(f"  wrote {plan['output'] / TUNED_NAME}", "green"), flush=True)
    return 0


def snapshot(entry: dict[str, Any], prog: dict[str, Any], frame: str) -> str:
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    status = rm.effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "interrupted": "red",
              "stopped": "yellow", "failed": "red"}.get(status, "dim")
    done, total = int(prog.get("done", 0)), int(prog.get("total", 0))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting") else prog.get("updated", time.time()))
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "baseline calibration"), "dim")]
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = f"  {done}/{total} calibrations · elapsed {qs.fmt_time(end - started)}"
        if status in ("running", "setup") and prog.get("eta_seconds") is not None:
            tail += f" · ETA ~{qs.fmt_time(prog['eta_seconds'])} (rough: methods and pairs differ in cost)"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    if status in ("running", "setup") and prog.get("stage"):
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    if status in ("running", "setup") and prog.get("last"):
        lines.append(style(f"  last finished: {prog['last']}", "dim"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); restart with: hedonic run resume {entry['name']}", "red"))
    elif status == "stopped":
        lines.append(style(f"\n  stopped by you; resume keeps the finished calibrations and the ledger: "
                           f"hedonic run resume {entry['name']}", "yellow"))
    if status == "completed":
        lines.append(style(f"\n  settings  {prog.get('output', entry['output_dir'])}", "dim"))
    return "\n".join(lines)


def tree(entry: dict[str, Any], prog: dict[str, Any]):
    """The run as the interactive tree of ``queuetui``: pair > method > every setting tried, with scores and ETAs."""
    from hedonic.experiments.overlapping import queuebuild as qb
    from hedonic.experiments.overlapping import queuemodel as qm
    from hedonic.experiments.overlapping import runmanager as rm

    plan = resolve(entry["argv"], prog="hedonic run calibrate-baselines")
    todo = searches(plan)
    results = read_tuned(plan).get("searches") or {}
    status = rm.effective_status(entry, prog)
    model = qb.baselines(plan, todo, results)
    done = len(set(results) & {s["key"] for s in todo})
    node_status = {"completed": "done", "setup": "running", "interrupted": "stopped"}.get(status, status)
    if status == "stopped" and done and done < len(todo):
        node_status = "stopped"
    node = qm.run_node(1, entry["name"], describe(plan), node_status, model, (done, len(todo), "calibrations"),
                       "Every baseline's parameters are chosen on G[S] alone (no ground truth): held-out pair likelihood, "
                       "a structural rule, or the authors' published setting.",
                       took=rm.elapsed_seconds(entry, prog, status) if status == "completed" else None)
    qb.decorate_baselines(node, plan, results)
    return qm.queue_node(entry["name"], [node], status, str(plan["output"]))


if __name__ == "__main__":
    raise SystemExit(main())
