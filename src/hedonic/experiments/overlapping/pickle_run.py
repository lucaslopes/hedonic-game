"""``hedonic pickle run``: Hedonic on the pickled ``Game`` of each network:cover pair, one run per pair, with the
time and memory each run costs.

Each pair runs in a worker process of its own (so its peak memory is its own and a hung run can be killed at
``--timeout``): unpickle the ``Game`` (``hedonic pickle`` made it), call ``Game.community_hedonic`` with the pair's
calibrated (gamma, cap) exactly as the calibration's evaluations call it (local moving to an equilibrium,
``allow_isolation=True``, ``seed``), and score the cover against the reference cover stored in the pickle (F1
and LFK-ONMI of the shared metric code). A row records the load time, the detection wall-clock and CPU time, the scoring
time, the peak resident memory after the load and at the end, and the cover's shape.

Settings come from ``--tuned FILE`` (``hedonic_tuned.json`` of ``hedonic run calibrate``; repeat the option to lay several
files over one another, the last wins). A pair is **calibrated** when a file has the ``hedonic`` entry for its own
``network:cover`` key. Any other pair borrows the setting of the calibrated pair of nearest graph density
(2m / n(n-1), compared on a log scale) among the pairs of the same invocation, and the row says so (``setting_source``).

    hedonic pickle run --all --tuned hedonic_tuned.json --tuned friendster-tuned.json
    hedonic pickle run dblp:top5k wikipedia:all --tuned hedonic_tuned.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

METHOD = "hedonic"


def density(n: int, m: int) -> float:
    return 2.0 * m / (n * (n - 1)) if n > 1 else 0.0


def own_setting(tuned: dict, network: str, cover: str) -> dict | None:
    return (tuned.get(METHOD) or {}).get(f"{network}:{cover}")


def borrow(pair: tuple[str, str], rho: float, calibrated: dict[tuple[str, str], dict]) -> tuple[tuple[str, str], dict]:
    """The calibrated pair of nearest density (log scale) and its setting."""
    if not calibrated:
        raise ValueError("no calibrated pair to borrow a setting from")
    best = min(calibrated, key=lambda other: abs(math.log(calibrated[other]["density"]) - math.log(rho)))
    return best, calibrated[best]["setting"]


def load_tuned(files: list[Path]) -> dict:
    from hedonic.experiments.overlapping import quickstart as qs

    merged: dict = {}
    for file in files:
        for method, nets in qs.load_tuned_file(file).items():
            merged.setdefault(method, {}).update(nets)
    return merged


# --------------------------------------------------------------------------- the worker (one pair, own process)
def _rss(which: int = resource.RUSAGE_SELF) -> int:
    peak = resource.getrusage(which).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def run_pair(network: str, cover: str, pickle_file: Path, gamma: float, cap: int, seed: int) -> dict[str, Any]:
    from hedonic.experiments.overlapping import codeseg_reproduction as cr, snap_pickle
    from hedonic.experiments.overlapping.metrics import evaluate_cover, overlapping_normalized_mutual_information_lfk
    from hedonic.experiments.runtime import runtime_versions

    snap_pickle._start_guard(f"{network}:{cover}", None)
    t0 = time.perf_counter()
    game = snap_pickle.load_game(pickle_file)
    load_seconds = time.perf_counter() - t0
    rss_loaded = _rss()
    gt = [list(c) for c in game["cover"]]
    n, m = game.vcount(), game.ecount()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t1 = time.perf_counter()
    game.community_hedonic(None, max_memberships=int(cap), resolution=float(gamma), allow_isolation=True,
                           seed=int(seed))
    detection_seconds = time.perf_counter() - t1
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    rss_detected = _rss()
    rows = [[int(label) for label in labels] for labels in (game.memberships or [])]
    t2 = time.perf_counter()
    communities: dict[int, list[int]] = {}
    for vertex, labels in enumerate(rows):
        for label in labels:
            communities.setdefault(label, []).append(vertex)
    gt_nodes = {v for community in gt for v in community}
    pred = cr._filter_prediction(list(communities.values()), gt_nodes=gt_nodes, n_vertices=n)
    metrics = evaluate_cover(pred, gt, n, compute_omega=False)
    onmi = overlapping_normalized_mutual_information_lfk(pred, gt)
    scoring_seconds = time.perf_counter() - t2
    sizes = sorted(len(c) for c in communities.values())
    return {
        "pair": f"{network}:{cover}", "status": "completed", "n": n, "m": m, "density": density(n, m),
        "gamma": gamma, "cap": int(cap), "seed": seed,
        "load_seconds": round(load_seconds, 3), "detection_seconds": round(detection_seconds, 3),
        "detection_cpu_seconds": round((cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime), 3),
        "scoring_seconds": round(scoring_seconds, 3),
        "rss_after_load_bytes": rss_loaded, "rss_peak_bytes": max(rss_detected, _rss()),
        "detection_added_bytes": max(rss_detected - rss_loaded, 0),
        "f1": float(metrics["f1"]), "onmi": float(onmi), "reference_communities": len(gt),
        "predicted_communities": len(communities), "median_size": sizes[len(sizes) // 2] if sizes else 0,
        "max_size": sizes[-1] if sizes else 0,
        "mean_memberships": sum(len(r) for r in rows) / max(len(rows), 1),
        "runtime": runtime_versions(),
    }


# --------------------------------------------------------------------------- the driver
def _measure(pickle_file: Path) -> tuple[int, int]:
    """(vertices, edges) of a pickled Game, read in a process of its own so the memory is given back."""
    run = subprocess.run([sys.executable, "-m", "hedonic.experiments.overlapping.pickle_run", "--measure", str(pickle_file)],
                         stdout=subprocess.PIPE, text=True, check=True)
    n, m = json.loads(run.stdout.strip().splitlines()[-1])
    return int(n), int(m)


def _worker(pair: tuple[str, str], pickle_file: Path, setting: dict, seed: int, timeout: float | None) -> dict[str, Any]:
    argv = [sys.executable, "-m", "hedonic.experiments.overlapping.pickle_run", "--run-one", f"{pair[0]}:{pair[1]}",
            "--pickle", str(pickle_file), "--gamma", repr(setting["absolute_resolution"]),
            "--cap", str(setting["max_memberships"]), "--seed", str(seed)]
    started = time.perf_counter()
    try:
        run = subprocess.run(argv, stdout=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"pair": f"{pair[0]}:{pair[1]}", "status": "timeout", "reason": f"no result within {timeout:.0f} s",
                "wall_seconds": round(time.perf_counter() - started, 1)}
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    if lines:
        row = json.loads(lines[-1])
    else:
        row = {"pair": f"{pair[0]}:{pair[1]}", "status": "failed", "reason": f"worker exited with code {run.returncode}"}
    row["wall_seconds"] = round(time.perf_counter() - started, 1)
    return row


def _gib(value: float | None) -> str:
    return "-" if value is None else f"{value / 1024**3:.2f}"


def format_rows(rows: list[dict[str, Any]]) -> str:
    head = (f"{'pair':<21}{'n':>10}{'m':>12}{'gamma':>11}{'cap':>4}{'load s':>8}{'detect s':>10}{'cpu s':>9}"
            f"{'RSS load':>10}{'RSS peak':>10}{'F1':>7}{'ONMI':>7}{'K':>8}  setting")
    out = [head]
    for r in rows:
        if r.get("status") != "completed":
            out.append(f"{r['pair']:<21}{r.get('status')}: {r.get('reason', '')}")
            continue
        out.append(f"{r['pair']:<21}{r['n']:>10,}{r['m']:>12,}{r['gamma']:>11.3g}{r['cap']:>4}{r['load_seconds']:>8.1f}"
                   f"{r['detection_seconds']:>10.1f}{r['detection_cpu_seconds']:>9.1f}{_gib(r['rss_after_load_bytes']):>10}"
                   f"{_gib(r['rss_peak_bytes']):>10}{r['f1']:>7.3f}{r['onmi']:>7.3f}{r['predicted_communities']:>8,}  "
                   f"{r.get('setting_source', '')}")
    return "\n".join(out) + "\n(RSS in GiB; detect s is wall-clock of community_hedonic; cpu s is user+system of the process)"


def main(argv: list[str] | None = None) -> int:
    from hedonic.experiments.overlapping import snap_pickle as sp

    parser = argparse.ArgumentParser(prog="hedonic pickle run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="*", type=sp.parse_pair, metavar="NETWORK:COVER")
    parser.add_argument("--all", action="store_true", help="every pair of Table 1 that has a pickle")
    parser.add_argument("--tuned", action="append", type=Path, default=[], metavar="FILE",
                        help="calibrated settings (hedonic_tuned.json); repeat to lay several files, the last wins")
    parser.add_argument("--out-dir", type=Path, help="folder of the pickles (as `hedonic pickle`)")
    parser.add_argument("--output-dir", type=Path, help="where results.json/results.jsonl go (default <out-dir>/hedonic-runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per pair (default 3600; 0 = none)")
    parser.add_argument("--dry-run", action="store_true", help="print the settings each pair would use; run nothing")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--run-one", metavar="NETWORK:COVER", type=sp.parse_pair, help=argparse.SUPPRESS)
    parser.add_argument("--measure", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pickle", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--cap", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.measure:
        from hedonic.experiments.overlapping import snap_pickle as _sp

        game = _sp.load_game(args.measure)
        print(json.dumps([game.vcount(), game.ecount()]), flush=True)
        return 0
    if args.run_one:
        try:
            row = run_pair(*args.run_one, args.pickle, args.gamma, args.cap, args.seed)
        except Exception as exc:  # noqa: BLE001 - the row records why a pair failed; the other pairs still run
            row = {"pair": f"{args.run_one[0]}:{args.run_one[1]}", "status": "failed",
                   "reason": f"{type(exc).__name__}: {exc}", "peak_rss_bytes": _rss()}
        print(json.dumps(row), flush=True)
        return 0
    out_dir = (args.out_dir or sp.default_out_dir()).expanduser()
    pairs = list(sp.TABLE_PAIRS) if args.all else []
    pairs += [p for p in args.pairs if p not in pairs]
    if not pairs:
        parser.error("name at least one NETWORK:COVER, or use --all")
    if not args.tuned:
        parser.error("give the calibration with --tuned FILE (hedonic_tuned.json of `hedonic run calibrate`)")
    tuned = load_tuned(args.tuned)
    present = [p for p in pairs if sp.pickle_path(out_dir, *p).is_file()]
    for p in pairs:
        if p not in present:
            print(f"skipped {p[0]}:{p[1]}: no pickle at {sp.pickle_path(out_dir, *p)} (hedonic pickle {p[0]}:{p[1]})",
                  file=sys.stderr)
    calibrated = [p for p in present if own_setting(tuned, *p)]
    others = [p for p in present if p not in calibrated]
    if others and not calibrated:
        parser.error("none of the pairs has its own calibration to borrow from; add calibrated pairs or --tuned files")
    timeout = args.timeout or None
    output_dir = (args.output_dir or out_dir / "hedonic-runs").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sizes = {pair: _measure(sp.pickle_path(out_dir, *pair)) for pair in calibrated + others}
    known = {pair: {"density": density(*sizes[pair]), "setting": own_setting(tuned, *pair)} for pair in calibrated}
    for pair in pairs:
        if pair not in present:
            continue
        path = sp.pickle_path(out_dir, *pair)
        if pair in calibrated:
            setting, source = own_setting(tuned, *pair), "calibrated"
        else:
            rho = density(*sizes[pair])
            donor, setting = borrow(pair, rho, known)
            source = f"borrowed from {donor[0]}:{donor[1]} (density {known[donor]['density']:.2e} vs {rho:.2e})"
        if args.dry_run:
            rows.append({"pair": f"{pair[0]}:{pair[1]}", "status": "planned", "gamma": setting["absolute_resolution"],
                         "cap": setting["max_memberships"], "setting_source": source})
            continue
        print(f"== {pair[0]}:{pair[1]}  gamma={setting['absolute_resolution']:.6g} cap={setting['max_memberships']}  "
              f"({source})", file=sys.stderr, flush=True)
        row = _worker(pair, path, setting, args.seed, timeout)
        row["setting_source"] = source
        rows.append(row)
        with (output_dir / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    if not args.dry_run:
        (output_dir / "results.json").write_text(json.dumps(rows, indent=1))
    if args.dry_run:
        print("\n".join(f"{r['pair']:<21} gamma={r['gamma']:.6g} cap={r['cap']:<3} {r['setting_source']}" for r in rows))
    else:
        print(json.dumps(rows, indent=1) if args.json else format_rows(rows))
    return 1 if any(r.get("status") in ("failed", "timeout") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
