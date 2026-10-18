"""A budgeted ("lazy") run of the Hedonic calibration search: ``hedonic run calibrate --budget N --cap-minutes M``.

The search (``calibration_search``) is sequential and deterministic, so a search stopped after E evaluations is exactly
the first E evaluations of the full one. This runner lets the search's ``main`` run unchanged and stops the search before
it asks for a new selection evaluation when either

* ``--budget`` distinct selection configurations have been asked for (repeats of a cached configuration are free), or
* the selection evaluations already run cost ``--cap-minutes`` single-core minutes (detection + scoring + audit).

The setting it reports is the best-observed (gamma, M) by selection-seed F1, the first found winning a tie. There are no
finishing steps (cap delta rule, rational snap, confirmation, branch check): they save labels, not quality, and cost about
a third of the full search. The record appended to the search ledger has the shape ``calibrate.final_of`` reads, plus a
``lazy`` block with the budget, what stopped the search and the configurations asked, so a result is never mistaken for a
full calibration. A search that ends by its own rules inside the budget is a full search and is written by the search.

The search is the packaged ``hedonic.experiments.overlapping.calibration_search``; ``--script FILE`` runs a search script
with the same interface instead (the 1.x ``bisect_tuning.py``).

    python -m hedonic.experiments.overlapping.lazy_search --budget 28 --cap-minutes 60 \\
        tune --instance livejournal:all --phase local --tag livejournal_all-local --out RUN/ledger
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SELECTION_SEEDS = [0]  # the seeds of a selection probe in the search (``tune(seeds=[0])``)


class Stop(Exception):
    """The next selection evaluation would exceed the budget or the cost cap."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason  # "budget" | "cap"
        self.detail = detail


def load_script(path: Path):
    """Import a search script as the module ``bisect_tuning`` (it is also what a script run would execute)."""
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("bisect_tuning", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the search script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bisect_tuning"] = module
    spec.loader.exec_module(module)
    return module


def packaged_search():
    """The search shipped with the package."""
    from hedonic.experiments.overlapping import calibration_search

    return calibration_search


def install(bt, *, budget: int, cap_seconds: float | None) -> list[str]:
    """Count the selection configurations in the order the search asks for them and stop it at the budget or the cap.

    ``Probe.prefetch`` only warms the cache (the scan points are independent), so it is cut to the budget that is left and
    not counted. Returns the live list of keys asked, in order. :func:`uninstall` puts the search's own methods back."""
    asked: list[str] = []
    seen: set[str] = set()
    call, prefetch = bt.Probe.__call__, bt.Probe.prefetch
    bt.Probe._unbudgeted = (call, prefetch)

    def spent() -> float:
        total = 0.0
        for key in asked:
            record = bt.CACHE.get(key)
            if record and record.get("status") == "completed":
                total += record["detection_seconds"] + record["scoring_seconds"] + record["audit_seconds"]
        return total

    def keys_of(probe, over: dict) -> list[str]:
        return [bt.cfg_key(probe.spec, bt._normal({**probe.base, **over, "seed": s})) for s in probe.seeds]

    def counted_call(self, **over):
        if self.seeds == SELECTION_SEEDS:
            new = [k for k in dict.fromkeys(keys_of(self, over)) if k not in seen]
            if new and len(asked) + len(new) > budget:
                raise Stop("budget", f"{len(asked)} evaluations")
            if new and cap_seconds is not None and spent() >= cap_seconds:
                raise Stop("cap", f"{len(asked)} evaluations, {spent() / 60:.0f} probe minutes")
            for key in new:
                seen.add(key)
                asked.append(key)
        return call(self, **over)

    def counted_prefetch(self, overs):
        if self.seeds == SELECTION_SEEDS:
            room = budget - len(asked)
            kept = []
            for over in overs:
                if any(k not in seen for k in keys_of(self, over)):
                    if room <= 0:
                        continue
                    room -= 1
                kept.append(over)
            overs = kept
        return prefetch(self, overs)

    bt.Probe.__call__ = counted_call
    bt.Probe.prefetch = counted_prefetch
    return asked


def uninstall(bt) -> None:
    """Undo :func:`install` (the packaged search is a shared module; a later search in this process runs unbudgeted)."""
    original = bt.Probe.__dict__.get("_unbudgeted")
    if original is not None:
        bt.Probe.__call__, bt.Probe.prefetch = original
        del bt.Probe._unbudgeted


def summarize(bt, asked: list[str], stop: Stop, *, budget: int, cap_minutes: float | None, started: float,
              instance: str, phase: str) -> dict[str, Any]:
    """The ledger record of a stopped search: the best-observed selection evaluation as ``final``."""
    done = [bt.CACHE[k] for k in asked if k in bt.CACHE and bt.CACHE[k].get("status") == "completed"]
    if not done:
        raise RuntimeError(f"the lazy search for {instance} stopped ({stop.detail}) without a completed evaluation")
    best = done[0]
    for record in done[1:]:
        if record["metrics"]["f1"] > best["metrics"]["f1"]:  # first found wins a tie
            best = record
    inst = bt.INST[instance]
    gamma, cap = float(best["config"]["gamma"]), int(best["config"]["M"])
    cost = sum(r["detection_seconds"] + r["scoring_seconds"] + r["audit_seconds"] for r in done)
    metrics = best["metrics"]
    return {
        "kind": "tune", "instance": bt.instance_summary(inst), "phase": phase, "objective": "f1", "seeds_per_probe": SELECTION_SEEDS,
        "snap": None, "branch_check": None,
        "final": {"gamma": gamma, "M": cap, "c": gamma / inst["rho"], "t": 1 - math.log(gamma) / math.log(inst["rho"]),
                  "gamma_over_rho_in": gamma / inst["rho_in"], "confirm_seeds": SELECTION_SEEDS, "completed": 1,
                  "f1_median": metrics["f1"], "onmi_median": metrics["onmi"], "f1_sd": None,
                  "detection_median": best["detection_seconds"], "K_median": metrics.get("predicted_community_count")},
        "detector_evaluations": len(done), "wall_hours": (time.time() - started) / 3600.0,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lazy": {"budget": budget, "cap_minutes": cap_minutes, "stopped_by": stop.reason, "stopped_after": stop.detail,
                 "evaluations": len(done), "probe_seconds": cost,
                 "detection_seconds": sum(r["detection_seconds"] for r in done),
                 "selection": "best observed by selection-seed F1 (no snap, no confirmation)",
                 "asked": [{"gamma": r["config"]["gamma"], "M": r["config"]["M"], "f1": r["metrics"]["f1"],
                            "onmi": r["metrics"]["onmi"]} for r in done]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hedonic.experiments.overlapping.lazy_search", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", help="a search script with the search's interface (default: the packaged "
                                         "calibration_search)")
    parser.add_argument("--budget", type=int, required=True, help="selection evaluations before the search is stopped")
    parser.add_argument("--cap-minutes", type=float, help="single-core probe minutes before the search is stopped")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.budget < 1:
        parser.error("--budget must be at least 1")
    if args.cap_minutes is not None and not args.cap_minutes > 0:
        parser.error("--cap-minutes must be positive")
    bt = load_script(Path(args.script)) if args.script else packaged_search()
    asked = install(bt, budget=args.budget, cap_seconds=None if args.cap_minutes is None else args.cap_minutes * 60.0)
    flag = lambda name, default=None: rest[rest.index(name) + 1] if name in rest else default  # noqa: E731
    instance, phase = flag("--instance", "dblp"), flag("--phase", "local")
    started = time.time()
    try:
        if args.script:  # a script's main reads sys.argv
            sys.argv = [args.script, *rest]
            code = bt.main()
        else:
            code = bt.main(rest)
    except Stop as stop:
        summary = summarize(bt, asked, stop, budget=args.budget, cap_minutes=args.cap_minutes, started=started,
                            instance=instance, phase=phase)
        bt.append_line(bt.SEARCHES, json.dumps(summary, default=str))
        final = summary["final"]
        print(f"== {instance} {phase}: lazy search stopped ({stop.reason}) after {stop.detail}: gamma={final['gamma']:.6g} "
              f"M={final['M']} F1={final['f1_median']:.4f} ONMI={final['onmi_median']:.4f}", flush=True)
        return 0
    finally:
        uninstall(bt)
    print(f"== {instance} {phase}: the search ended by its own rules inside the budget ({len(asked)} evaluations)", flush=True)
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
