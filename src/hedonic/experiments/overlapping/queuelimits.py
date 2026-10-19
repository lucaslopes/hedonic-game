"""A queue's time limits (``limits``) applied to its items: what one run is for each kind, how ``timeout`` reaches the
runner, and how ``max_eta`` drops the pairs whose single run is estimated longer than the limit.

``timeout`` is passed as the runner's ``--limit-seconds``, which lowers its own per-run timeout and never enters a record's
identity. ``max_eta`` is judged by the worker when an item (or a pair of a by-pair item) comes up: the estimate is that of
**one run** on the pair (a benchmark unit's detector run, a baseline calibration's evaluation, a start-quality condition,
a spectrum detector run), from the item's own records first, then the same runs on other pairs scaled by edges, then the
method's planning constant; it is never capped below what the run's own timeout allows, and never the total of the item.
A pair without an estimate is never skipped. The Hedonic calibration is never skipped on an estimate: its searches decide
their next probe from the last one, so only ``timeout`` bounds them, probe by probe.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from hedonic.experiments.overlapping import limits
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import threadio

BENCHMARK_VERBS = ("exp", "paper", "paper-hoc")
SPECTRUM_VERBS = ("spectrum", "spectrum-full")
LIMITED_VERBS = (*BENCHMARK_VERBS, "calibrate", "calibrate-baselines", "startcurve")  # take --limit-seconds
SKIPPABLE_VERBS = (*BENCHMARK_VERBS, "calibrate-baselines", "startcurve", *SPECTRUM_VERBS)  # max_eta drops their pairs

ONE_RUN = {
    **{verb: "one detector run of a unit (network, seed, method)" for verb in BENCHMARK_VERBS},
    "calibrate": "one detector run of a search (each probe on its own, never the search's total)",
    "calibrate-baselines": "one evaluation of a search (one detector run on a pair, never the search's total)",
    "startcurve": "one condition's detector run (pair, corruption level, seed)",
    **{verb: "one detector run (pair, policy, resolution, seed)" for verb in SPECTRUM_VERBS},
}
NOT_APPLIED = {
    **{verb: "its per-run timeout is part of every condition's identity (--timeout-per-run, default 1h), so a queue "
             "limit does not lower it; max_eta still skips pairs" for verb in SPECTRUM_VERBS},
    "controlled": "its 5 s per-cell timeout is a protocol parameter; a queue limit does not change it",
    "disjoint": "its runs are seconds each and have no per-run timeout",
    "posthoc": "an exact analysis with no per-run timeout",
    "audit": "an exact audit with no per-run timeout",
}


def _option(run: list[str], flag: str) -> str | None:
    return run[run.index(flag) + 1] if flag in run and run.index(flag) + 1 < len(run) else None


def _with_option(run: list[str], flag: str, value: str) -> list[str]:
    run = list(run)
    if flag in run and run.index(flag) + 1 < len(run):
        run[run.index(flag) + 1] = value
        return run
    return [*run, flag, value]


def _without_option(run: list[str], flag: str) -> list[str]:
    run = list(run)
    if flag in run:
        del run[run.index(flag):run.index(flag) + 2]
    return run


def apply(run: list[str], lim: dict) -> list[str]:
    """The item's arguments with its ``timeout`` passed on: ``--limit-seconds`` for the runners that take it (the smaller
    of the two when the item already sets one); a nested queue gets ``--timeout``/``--max-eta`` for its own items."""
    if run[0] == "queue":
        for key, flag in (("timeout", "--timeout"), ("max_eta", "--max-eta")):
            if lim.get(key) is not None and _option(run, flag) is None:
                run = _with_option(run, flag, f"{lim[key]:g}")
        return run
    timeout = lim.get("timeout")
    if timeout is None or run[0] not in LIMITED_VERBS:
        return run
    given = _option(run, "--limit-seconds")
    return _with_option(run, "--limit-seconds", f"{limits.effective(float(given) if given else None, timeout):g}")


def describe(run: list[str], lim: dict) -> list[str]:
    """How each limit reads for this item: what one run is, or why a limit does not apply."""
    verb, lines = run[0], []
    if lim.get("timeout") is not None:
        if verb in LIMITED_VERBS:
            lines.append(f"timeout {limits.fmt(lim['timeout'])} per {ONE_RUN[verb]}")
        elif verb != "queue":
            lines.append(f"timeout {limits.fmt(lim['timeout'])} not applied: {NOT_APPLIED.get(verb, 'no per-run timeout')}")
    if lim.get("max_eta") is not None:
        if verb in SKIPPABLE_VERBS:
            lines.append(f"max_eta {limits.fmt(lim['max_eta'])}: a pair whose {ONE_RUN[verb]} is estimated longer is "
                         "skipped")
        elif verb == "calibrate":
            lines.append(f"max_eta {limits.fmt(lim['max_eta'])} not applied: a search is never skipped on an estimate "
                         "(timeout bounds each probe)")
        elif verb != "queue":
            lines.append(f"max_eta {limits.fmt(lim['max_eta'])} not applied: {NOT_APPLIED.get(verb, 'no pairs to skip')}")
    if verb == "queue" and lim:
        lines.append(f"{limits.describe(lim)}, passed to every item of the queue it runs")
    return lines


# --------------------------------------------------------------------------- one-run estimates
def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _edges() -> dict[str, int]:
    from hedonic.experiments.overlapping import runqueue

    return runqueue._covered_edges()


def _scaled(target: str, measured: dict[str, list[float]], edges: dict[str, int]) -> tuple[float, str] | None:
    """The target priced from the same runs on other pairs, by seconds per edge (the median over those pairs)."""
    if target not in edges:
        return None
    rates = [_median(seconds) / edges[other] for other, seconds in measured.items()
             if other != target and seconds and edges.get(other)]
    if not rates:
        return None
    return statistics.median(rates) * edges[target], "the same runs on other pairs, scaled by edges"


def _pick(target: str, measured: dict[str, list[float]], edges: dict[str, int],
          planning: float | None, cap: float | None) -> tuple[float, str] | None:
    """Measured on the pair, else scaled from other pairs, else the planning constant; at most the run's timeout."""
    if measured.get(target):
        found = (_median(measured[target]), "measured on this pair")
    else:
        found = _scaled(target, measured, edges) or (
            (planning, "the method's planning constant") if planning is not None else None)
    if found is None:
        return None
    seconds, basis = found
    if cap is not None and seconds > cap:  # a run cannot last longer than its timeout
        return cap, f"{basis}, bounded by the run's timeout"
    return seconds, basis


def _benchmark(run: list[str]) -> dict[str, tuple[float, str]]:
    rest = run[1:] if run[0] == "exp" else ["--config", run[0], *run[1:]]
    with threadio.redirected():
        args = qs.build_parser().parse_args(rest)
        cfg = qs.config_from_args(args)
    if args.name:
        cfg.output_dir = str(qs.run_folder(cfg, args.name))
    edges = dict(_edges()) if cfg.induce == "covered" else {}
    cap = qs.run_timeout(cfg)
    out: dict[str, tuple[float, str]] = {}
    methods = [m for m in qs.METHODS + list(getattr(qs, "CONTROLS", [])) if m.key in cfg.methods]
    per_method: dict[str, dict[str, list[float]]] = {}
    for m in methods:
        measured = per_method.setdefault(m.key, {})
        for net in cfg.networks:
            for seed in cfg.seeds:
                _argv, folder = qs.unit_argv(cfg, net, seed, m, None)
                try:
                    record = json.loads((folder / "runs" / qs.split_target(net)[0] / f"{m.runner}.json").read_text())
                except (OSError, ValueError):
                    continue
                seconds = record.get("detection_seconds")
                if record.get("status") == "completed" and isinstance(seconds, (int, float)):
                    measured.setdefault(net, []).append(float(seconds))
                    size = (record.get("dataset_report") or {}).get("m")
                    if isinstance(size, int) and size > 0:
                        edges.setdefault(net, size)
    for net in cfg.networks:
        best: tuple[float, str] | None = None
        for m in methods:
            planning = (m.seconds * edges[net] / qs.DBLP_EDGES) if net in edges else qs.expected_seconds(m, cfg, net) - 2.0
            found = _pick(net, per_method[m.key], edges, planning, cap)
            if found is not None and (best is None or found[0] < best[0]):
                best = (found[0], f"{m.label}: {found[1]}")
        if best is not None:
            out[net] = best
    return out


def _baselines(run: list[str]) -> dict[str, tuple[float, str]]:
    from hedonic.experiments.overlapping import baseline_run as br

    with threadio.redirected():
        plan = br.resolve(run[1:])
    cap = limits.effective(plan["eval_timeout_minutes"] * 60, plan.get("limit"))
    folders = [plan["output"], *plan.get("adopt_from", [])]
    edges = dict(_edges())
    keys = {m.key: m for m in qs.METHODS}
    out: dict[str, tuple[float, str]] = {}
    for method in plan["methods"]:
        measured: dict[str, list[float]] = {}
        for target in plan["networks"]:
            for folder in folders:
                path = Path(folder) / "ledger" / target.replace(":", "_") / f"evals-{method}.jsonl"
                try:
                    lines = path.read_text().splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("status") == "completed" and isinstance(rec.get("detect_seconds"), (int, float)):
                        measured.setdefault(target, []).append(float(rec["detect_seconds"]))
        for target in plan["networks"]:
            m = keys.get(method)
            planning = m.seconds * edges[target] / qs.DBLP_EDGES if m is not None and target in edges else None
            found = _pick(target, measured, edges, planning, cap)
            if found is not None and (target not in out or found[0] < out[target][0]):
                out[target] = (found[0], f"{method}: {found[1]}")
    return out


def _startcurve(run: list[str]) -> dict[str, tuple[float, str]]:
    from hedonic.experiments.overlapping import queuebuild as qb
    from hedonic.experiments.overlapping import start_quality as sq

    plan = sq.front_study(run[1:])
    measured: dict[str, list[float]] = {}
    edges = dict(_edges())
    for record in sq.read_records(Path(plan["output"]) / sq.RESULTS_NAME).values():
        if sq.completed(record) and isinstance(record.get("seconds"), (int, float)):
            target = qb.record_target(record)
            measured.setdefault(target, []).append(float(record["seconds"]))
            if isinstance(record.get("n_edges"), int):
                edges.setdefault(target, record["n_edges"])
    out = {}
    for target in plan["networks"]:
        found = _pick(target, measured, edges, None, plan.get("limit"))
        if found is not None:
            out[target] = found
    return out


def _spectrum(run: list[str]) -> dict[str, tuple[float, str]]:
    from hedonic.experiments.overlapping import runqueue

    options = runqueue._spectrum_options(run[0], run[1:])
    folder = _option(run, "--output-dir")
    measured: dict[str, list[float]] = {}
    if folder:
        try:
            lines = (qs.expand_path(folder) / "results.jsonl").read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seconds = row.get("detector_runtime_seconds")
            if isinstance(seconds, (int, float)) and row.get("dataset"):
                measured.setdefault(f"{row['dataset']}:{row.get('cover', 'all')}", []).append(float(seconds))
    edges = dict(_edges())
    out = {}
    for dataset, cover in options.get("pairs") or []:
        target = f"{dataset}:{cover}"
        found = _pick(target, measured, edges, None, options.get("timeout_seconds"))
        if found is not None:
            out[f"{dataset}/{cover}"] = found
    return out


def estimates(run: list[str]) -> dict[str, tuple[float, str]]:
    """Seconds one run of each of the item's pairs should take (keyed as the item spells its pairs), with where the
    figure comes from. Empty when nothing can be estimated (a pair with no estimate is never skipped)."""
    verb = run[0]
    build = {**{v: _benchmark for v in BENCHMARK_VERBS}, "calibrate-baselines": _baselines, "startcurve": _startcurve,
             **{v: _spectrum for v in SPECTRUM_VERBS}}.get(verb)
    if build is None:
        return {}
    # a calibration file an earlier item writes may not exist yet; it changes settings, not how long one run takes
    for attempt in (run, _without_option(run, "--tuned")):
        try:
            return build(attempt)
        except (SystemExit, Exception):  # noqa: BLE001 - without an estimate nothing is skipped
            continue
    return {}


def over(run: list[str], max_eta: float | None) -> dict[str, tuple[float, str]]:
    """The item's pairs whose single run is estimated longer than ``max_eta``."""
    if max_eta is None or run[0] not in SKIPPABLE_VERBS:
        return {}
    return {target: found for target, found in estimates(run).items() if found[0] > max_eta}


def without(run: list[str], targets: list[str]) -> list[str] | None:
    """The item's arguments with ``targets`` left out (None when nothing would be left to run). A baseline calibration
    keeps them in its plan as transferred pairs (the benchmark still gets a setting for them); the others drop them."""
    from hedonic.experiments.overlapping import runqueue

    if not targets:
        return run
    verb = run[0]
    if verb == "calibrate-baselines":
        from hedonic.experiments.overlapping import baseline_run as br

        plan = br.resolve(run[1:])
        moved = [t for t in plan["networks"] if t in targets or t in plan["transfer"]]
        if len(moved) >= len(plan["networks"]):
            return None
        return _with_option(run, "--transfer", ",".join(moved))
    kept = [t for t in runqueue.item_targets(run) if t not in targets]
    if not kept:
        return None
    if verb in SPECTRUM_VERBS:
        return _with_option(run, "--pairs", ",".join(kept))
    return _with_option(_without_option(run, "--covers"), "--networks", ",".join(kept))
