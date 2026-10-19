"""``hedonic pickle``: write the :class:`hedonic.Game` of a SNAP ``network:cover`` pair to a pickle, once, so later runs
open it instead of parsing the raw edge list again.

The graph is the one a covered-induced benchmark run hands to every detector (``--induce covered``): loaded by
``codeseg_reproduction._load_dataset`` (a raw edge file of at least 2 GiB is streamed, so Friendster's covered subgraph is
built without its 1.8 billion edges ever being in memory), made undirected and simple when the network is directed, and
wrapped as ``Game``. The reference cover rides on the graph as graph attributes::

    game = snap_pickle.load_game("dblp-top5000.game.pkl")
    game["cover"]            # list of communities (lists of vertex indices), the paper-metrics cover
    game["network"], game["cover_variant"], game["report"]

Each pair is written by its own worker process, so its peak memory is its own and is released when it ends; a second
worker then reopens the file to measure the unpickle time and peak memory. A pair whose estimated memory does not fit the
free memory is reported as ``skipped`` with the reason (``--force`` tries anyway), and every worker watches the machine's
free memory and stops itself below the floor (``HEDONIC_MEMORY_FLOOR``, default 10%).

    hedonic pickle dblp:top5000 amazon:all --out-dir <data root>/Pickles
    hedonic pickle --all --dry-run
    hedonic pickle run --all --tuned hoc_tuned.json     # HOC-local on each pickle: time and memory (see pickle_run)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = 1

# The pairs of the paper's Table 1 (the six SNAP com-* networks; Orkut's top5000 is not part of it): (network, cover), and for each the sizes used to decide, before anything is read,
# whether it fits: edges of the raw network and edges of the covered subgraph (Table 1; Friendster top5000 is unknown
# until it has been streamed, and is the figure the calibration of that pair measured when known).
TABLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("dblp", "top5000"), ("dblp", "all"),
    ("amazon", "top5000"), ("amazon", "all"),
    ("youtube", "top5000"), ("youtube", "all"),
    ("livejournal", "top5000"), ("livejournal", "all"),
    ("orkut", "all"),
    ("friendster", "top5000"), ("friendster", "all"),
)
SIZES: dict[tuple[str, str], tuple[int, int | None]] = {  # (raw edges, covered-subgraph edges)
    ("dblp", "top5000"): (1_049_866, 335_520), ("dblp", "all"): (1_049_866, 950_059),
    ("amazon", "top5000"): (925_872, 48_739), ("amazon", "all"): (925_872, 872_935),
    ("youtube", "top5000"): (2_987_624, 224_235), ("youtube", "all"): (2_987_624, 318_432),
    ("wikipedia", "all"): (28_511_807, 28_511_807),
    ("livejournal", "top5000"): (34_681_189, 1_521_988), ("livejournal", "all"): (34_681_189, 16_899_734),
    ("orkut", "top5000"): (117_185_083, 21_992_510), ("orkut", "all"): (117_185_083, 94_765_983),
    ("friendster", "top5000"): (1_806_067_135, None), ("friendster", "all"): (1_806_067_135, 447_219_610),
}
# Peak bytes per edge measured on this code: a whole-graph load plus the covered copy and the Game copy (Orkut top5000:
# 26.9 GB for 117 M raw edges), and the streamed load plus the Game copy per kept edge (Friendster top5000 is checked
# by the worker's own guard). Deliberately generous: the estimate only decides what is attempted without ``--force``.
BYTES_PER_RAW_EDGE = 230  # the whole graph in memory while the covered subgraph is cut from it
BYTES_PER_COPY_EDGE = 300  # the covered subgraph, its edge list and the Game copy (whole-graph load)
BYTES_PER_KEPT_EDGE = 420  # streamed load: kept edges only
GUARD_SECONDS = 0.5


def pair_label(network: str, cover: str) -> str:
    return f"{network}:{cover}"


def parse_pair(text: str) -> tuple[str, str]:
    network, _, cover = text.strip().lower().partition(":")
    if not network or not cover:
        raise argparse.ArgumentTypeError(f"{text!r} is not NETWORK:COVER (for example dblp:top5000)")
    return network, cover


def default_out_dir() -> Path:
    from hedonic.experiments.overlapping import userconfig

    from hedonic.experiments.config import DATA_ROOT

    configured = userconfig.defaults().get("pickle_dir")
    return Path(configured).expanduser() if configured else Path(DATA_ROOT) / "Pickles"


def pickle_path(out_dir: Path, network: str, cover: str) -> Path:
    return Path(out_dir) / f"{network}-{cover}.game.pkl"


def estimate_bytes(network: str, cover: str, streamed: bool) -> int | None:
    sizes = SIZES.get((network, cover))
    if sizes is None:
        return None
    raw, kept = sizes
    if streamed:
        return None if kept is None else int(kept * BYTES_PER_KEPT_EDGE)
    return int(raw * BYTES_PER_RAW_EDGE + (kept or raw) * BYTES_PER_COPY_EDGE)


def plan_pair(network: str, cover: str, network_root: Path, out_dir: Path, *, force: bool = False,
              overwrite: bool = False) -> dict[str, Any]:
    """What would happen to one pair: ``write``, ``exists`` (kept) or ``skipped`` (with the reason), before any load."""
    from hedonic.experiments.overlapping import capacity, ground_truth_data as gtd, snap as _snap

    row: dict[str, Any] = {"pair": pair_label(network, cover), "network": network, "cover": cover,
                           "path": str(pickle_path(out_dir, network, cover))}
    spec = _snap.SPECS.get(network)
    if spec is None:
        return {**row, "action": "skipped", "reason": f"unknown network {network!r}", "bad_input": True}
    try:
        variant = _snap.canonical_cover_variant(cover)
    except Exception as exc:  # noqa: BLE001
        return {**row, "action": "skipped", "reason": str(exc), "bad_input": True}
    if variant not in spec.raw_cover_files and variant not in spec.cover_files:
        return {**row, "action": "skipped", "bad_input": True,
                "reason": f"{network} has no {cover!r} cover; its covers: "
                          + ", ".join(_snap.cover_variant_label(v) for v in spec.raw_cover_files)}
    base = _snap._dataset_dir(Path(network_root).expanduser(), spec)
    if _snap._first_existing(base, spec.raw_cover_files.get(variant, ())) is None and \
            _snap._first_existing(base, spec.cover_files.get(variant, ())) is None:
        return {**row, "action": "skipped", "reason": f"no {variant} cover file for {network} under {base}"}
    streamed = gtd.stream_covered_wanted(network, network_root, cover_variant=variant)
    need = estimate_bytes(network, variant, streamed)
    budget = capacity.budget_bytes()
    row["stream"] = False
    if not streamed and need is not None and budget is not None and need > budget:
        # the whole graph does not fit: cut the covered subgraph from a stream of the raw edge file instead, if that fits
        raw_edges = _snap._first_existing(base, spec.raw_edge_files)
        alternative = estimate_bytes(network, variant, True)
        if raw_edges is not None and alternative is not None and alternative <= budget:
            streamed, need, row["stream"] = True, alternative, True
    row["loader"] = "streamed covered subgraph" if streamed else "whole graph, then covered-induced"
    row["estimated_peak_bytes"] = need
    row["budget_bytes"] = budget
    if Path(row["path"]).is_file() and not overwrite:
        return {**row, "action": "exists"}
    if need is None:
        row["note"] = "size of the covered subgraph unknown: the worker's memory guard is the only limit"
    elif budget is not None and need > budget and not force:
        return {**row, "action": "skipped",
                "reason": f"needs about {capacity.gib(need)} but {capacity.gib(budget)} are free above the floor "
                          f"({'streamed' if streamed else 'whole-graph'} load; --force tries anyway)"}
    return {**row, "action": "write"}


# --------------------------------------------------------------------------- the worker (one pair, own process)
def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS reports bytes, Linux kilobytes


def _start_guard(label: str, partial: Path | None) -> None:
    from hedonic.experiments.overlapping import parallel

    floor = parallel.memory_floor()
    if not floor or parallel.free_memory_percent() is None:
        return

    def watch() -> None:
        while True:
            time.sleep(GUARD_SECONDS)
            free = parallel.free_memory_percent()
            if free is not None and free < floor:
                if partial is not None:
                    partial.unlink(missing_ok=True)
                print(json.dumps({"pair": label, "status": "failed",
                                  "reason": f"free memory {free:.0f}% fell below the {floor:.0f}% floor"}), flush=True)
                os._exit(75)

    threading.Thread(target=watch, name="memory-guard", daemon=True).start()


def build_game(network: str, cover: str, network_root: Path, cache_dir: Path | None = None):
    """The ``Game`` of a covered-induced pair with its reference cover attached; also returns the load report."""
    from hedonic import Game
    from hedonic.experiments.overlapping import codeseg_reproduction as cr

    dataset = cr._load_dataset(network, cover_variant=cover, network_root=Path(network_root).expanduser(), max_nodes=None,
                               smoke=False, allow_catalog=False, snap_cache_dir=cache_dir, induce="covered")
    graph = cr._undirected_graph(dataset.graph) if dataset.graph.is_directed() else dataset.graph
    gt, _gt_nodes, _report = cr._cover_for_paper_metrics(dataset)
    game = Game(graph)
    del graph
    game["network"], game["cover_variant"] = network, cover
    game["cover"] = [list(map(int, community)) for community in gt]
    game["report"] = json.loads(json.dumps(dataset.report, default=str))
    game["pickle_schema"] = SCHEMA
    return game


def load_game(path: str | Path):
    """Open a pickle written by ``hedonic pickle`` (a :class:`hedonic.Game` with ``cover`` and ``report`` attributes)."""
    with Path(path).expanduser().open("rb") as handle:
        return pickle.load(handle)


def write_pair(network: str, cover: str, network_root: Path, out_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    label = pair_label(network, cover)
    path = pickle_path(out_dir, network, cover)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    _start_guard(label, partial)
    t0 = time.perf_counter()
    game = build_game(network, cover, network_root, cache_dir)
    built = time.perf_counter()
    with partial.open("wb") as handle:
        pickle.dump(game, handle, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)
    done = time.perf_counter()
    return {"pair": label, "status": "written", "path": str(path), "bytes": path.stat().st_size, "n": game.vcount(),
            "m": game.ecount(), "communities": len(game["cover"]), "directed_source": game["report"].get("directed"),
            "build_seconds": round(built - t0, 2), "write_seconds": round(done - built, 2),
            "build_peak_rss_bytes": _peak_rss_bytes()}


def check_pair(network: str, cover: str, out_dir: Path) -> dict[str, Any]:
    """Reopen a written pickle in a fresh process: the time and peak memory a later run pays to use it."""
    label = pair_label(network, cover)
    path = pickle_path(out_dir, network, cover)
    _start_guard(label, None)
    t0 = time.perf_counter()
    game = load_game(path)
    seconds = time.perf_counter() - t0
    return {"pair": label, "status": "checked", "load_seconds": round(seconds, 2), "load_peak_rss_bytes": _peak_rss_bytes(),
            "n": game.vcount(), "m": game.ecount(), "communities": len(game["cover"]), "is_game": type(game).__name__ == "Game"}


def _worker_call(mode: str, pair: tuple[str, str], args: argparse.Namespace, stream: bool = False) -> dict[str, Any]:
    argv = [sys.executable, "-m", "hedonic.experiments.overlapping.snap_pickle", f"--{mode}", pair_label(*pair),
            "--out-dir", str(args.out_dir), "--network-root", str(args.network_root)]
    if args.cache_dir:
        argv += ["--cache-dir", str(args.cache_dir)]
    env = {**os.environ, "HEDONIC_SNAP_STREAM_COVERED": "1"} if stream else None
    run = subprocess.run(argv, stdout=subprocess.PIPE, text=True, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    if lines:
        return json.loads(lines[-1])
    return {"pair": pair_label(*pair), "status": "failed", "reason": f"worker exited with code {run.returncode}"}


def _gib(value: float | None) -> str:
    return "-" if value is None else f"{value / 1024**3:.2f} GiB"


def format_rows(rows: list[dict[str, Any]]) -> str:
    out = [f"{'pair':<22}{'result':<10}{'pickle':>11}{'n':>13}{'m':>14}{'build':>9}{'build RSS':>12}{'reopen':>9}{'reopen RSS':>12}"]
    for r in rows:
        if r.get("status") in ("written", "checked") or r.get("action") in ("exists",):
            def count(key):  # a pickle kept as it is was not opened: its sizes are unknown here, not zero
                return f"{r[key]:,}" if r.get(key) is not None else "-"

            def secs(key):
                return f"{r[key]:.0f}s" if r.get(key) is not None else "-"
            out.append(f"{r['pair']:<22}{r.get('result', r.get('status', r.get('action'))):<10}{_gib(r.get('bytes')):>11}"
                       f"{count('n'):>13}{count('m'):>14}{secs('build_seconds'):>9}"
                       f"{_gib(r.get('build_peak_rss_bytes')):>12}{secs('load_seconds'):>9}"
                       f"{_gib(r.get('load_peak_rss_bytes')):>12}")
        elif r.get("action") == "write":
            need = r.get("estimated_peak_bytes")
            out.append(f"{r['pair']:<22}{'to write':<10}  {r.get('loader', '')}; estimated peak "
                       f"{'unknown' if need is None else _gib(need)}")
        else:
            out.append(f"{r['pair']:<22}{r.get('action', r.get('status')):<10}  {r.get('reason', '')}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:  # `hedonic pickle run`: HOC-local on the pickles, time and memory per pair
        from hedonic.experiments.overlapping import pickle_run

        return pickle_run.main(argv[1:])
    parser = argparse.ArgumentParser(prog="hedonic pickle", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="*", type=parse_pair, metavar="NETWORK:COVER",
                        help="pairs to pickle (dblp:top5000 ...)")
    parser.add_argument("--all", action="store_true", help="every pair of the paper's Table 1 (11)")
    parser.add_argument("--out-dir", type=Path, help="folder of the pickles (default: pickle_dir of the user config, "
                                                     "else <data root>/Pickles)")
    parser.add_argument("--network-root", type=Path, help="folder of the SNAP networks (default: the configured one)")
    parser.add_argument("--cache-dir", type=Path, help="SNAP cache folder (default ~/.cache/hedonic/snap)")
    parser.add_argument("--overwrite", action="store_true", help="write again a pair whose pickle exists")
    parser.add_argument("--force", action="store_true", help="try a pair even when its estimated memory does not fit")
    parser.add_argument("--no-check", action="store_true", help="do not reopen each pickle to time the load")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written or skipped; load nothing")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    parser.add_argument("--write-one", metavar="NETWORK:COVER", type=parse_pair, help=argparse.SUPPRESS)
    parser.add_argument("--check-one", metavar="NETWORK:COVER", type=parse_pair, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from hedonic.experiments.config import NETWORKS_DIR

    args.network_root = (args.network_root or Path(NETWORKS_DIR)).expanduser()
    args.out_dir = (args.out_dir or default_out_dir()).expanduser()
    if args.write_one:
        print(json.dumps(write_pair(*args.write_one, args.network_root, args.out_dir, args.cache_dir)), flush=True)
        return 0
    if args.check_one:
        print(json.dumps(check_pair(*args.check_one, args.out_dir)), flush=True)
        return 0
    pairs = list(TABLE_PAIRS) if args.all else []
    pairs += [p for p in args.pairs if p not in pairs]
    if not pairs:
        parser.error("name at least one NETWORK:COVER, or use --all")
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        plan = plan_pair(*pair, args.network_root, args.out_dir, force=args.force, overwrite=args.overwrite)
        if args.dry_run or plan["action"] != "write":
            rows.append(plan)
            continue
        print(f"== {plan['pair']}: {plan['loader']}", file=sys.stderr, flush=True)
        row = _worker_call("write-one", pair, args, stream=plan["stream"])
        row = {**plan, **row, "result": row.get("status")}
        if row.get("status") == "written" and not args.no_check:
            row.update({k: v for k, v in _worker_call("check-one", pair, args).items() if k not in ("pair", "status")})
        rows.append(row)
    print(json.dumps(rows, indent=1) if args.json else format_rows(rows))
    if any(r.get("bad_input") for r in rows):  # a typo in a pair is an error, not a skipped pair
        return 2
    return 1 if any(r.get("status") == "failed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
