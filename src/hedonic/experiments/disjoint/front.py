"""``hedonic run disjoint``: the disjoint SBM benchmark through the durable front door.

    hedonic run disjoint                       wizard (in a terminal)
    hedonic run disjoint --preset paper        the paper's 100-seed grid (50,000 graphs)
    hedonic run disjoint --preset smoke        the whole pipeline in seconds
    hedonic run disjoint --dry-run [...]       plan, time and disk estimates; runs nothing
    hedonic run disjoint --communities 2,4 --seeds 0-9 --methods hedonic,leiden --nodes 600
    hedonic show disjoint                      presets, grids and estimates

A run executes detached (tmux, else a background process) under ``caffeinate``
on macOS, with one numerical thread per worker. It writes a columnar result
store (one Parquet shard per graph), then the logical-cell table, the five
paper figures and every statistic of the paper (``stats/``). ``hedonic run
list | status | attach | logs | stop | resume | check`` manage it like any
other run; ``hedonic run check NAME`` replays sampled rows from their seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from hedonic.experiments.config import DISJOINT_RUNS_ROOT, expand_path

# Preset name -> (sbm_sweep preset, one-line description)
PRESETS: dict[str, tuple[str, str]] = {
    "paper": ("sbm", "the paper's grid: 5 K × 10 p × 10 λ × 100 seeds = 50,000 graphs, 1,020 vertices"),
    "smoke": ("sbm-smoke", "the whole pipeline at small scale: 16 graphs of 120 vertices, seconds"),
}
PRESET_ALIASES = {"sbm": "paper", "sbm-smoke": "smoke", "full": "paper"}

METHOD_ALIASES = {
    "groundtruth": "GroundTruth", "gt": "GroundTruth", "ground-truth": "GroundTruth",
    "mirror": "Mirror", "onepass": "OnePass", "one-pass": "OnePass", "one_pass": "OnePass",
    "spectral": "Spectral", "leiden": "Leiden", "full": "Leiden", "hedonic": "Hedonic",
    "phase1": "Hedonic", "local": "Hedonic",
}
METHOD_LABELS = {
    "GroundTruth": "Ground truth", "Mirror": "Mirror", "OnePass": "One Pass", "Spectral": "Spectral",
    "Leiden": "Leiden", "Hedonic": "Hedonic",
}

# Cost model (M1 Max, 8 workers, measured on the paper grid): seconds per
# detector row = fixed overhead + a part linear in the vertex count.
SECONDS_PER_ROW_FIXED = 0.002
SECONDS_PER_ROW_PER_VERTEX = 0.0083 / 1020
BYTES_PER_ROW_HASH = 61.0
BYTES_PER_ROW_PER_VERTEX_FULL = 0.45


# --------------------------------------------------------------------------- settings
def _sweep():
    from hedonic.experiments.disjoint import sbm_sweep

    return sbm_sweep


def parse_floats(value: str) -> list[float]:
    """``0.1,0.25`` or a range ``0.01:0.10:0.01`` (start:stop:step, inclusive)."""
    out: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, stop, step = (float(x) for x in part.split(":"))
            if step <= 0 or stop < start:
                raise ValueError(f"invalid range {part!r}; use start:stop:step with step > 0")
            n = math.floor((stop - start) / step + 1e-9) + 1
            out.extend(round(start + i * step, 10) for i in range(n))
        else:
            out.append(float(part))
    if not out:
        raise ValueError(f"no values in {value!r}")
    return sorted(dict.fromkeys(out))


def parse_ints(value: str) -> list[int]:
    from hedonic.experiments.overlapping import quickstart as qs

    return qs.parse_seeds(value)


def parse_methods(value: str) -> list[str]:
    names = _sweep().METHODS
    if value.strip().lower() == "all":
        return list(names)
    out = []
    for part in value.split(","):
        key = part.strip()
        if not key:
            continue
        name = METHOD_ALIASES.get(key.lower(), key)
        if name not in names:
            raise ValueError(f"unknown method {part!r}; choose from: {', '.join(m.lower() for m in names)} (or all)")
        out.append(name)
    return list(dict.fromkeys(out))


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hedonic run disjoint", add_help=add_help,
        description="Disjoint SBM benchmark (durable). Without arguments, in a terminal, a wizard starts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("examples:\n"
                "  hedonic run disjoint --preset smoke\n"
                "  hedonic run disjoint --preset paper --dry-run\n"
                "  hedonic run disjoint --preset paper --name sbm100\n"
                "  hedonic run disjoint --communities 2-4 --p-in 0.05,0.1 --difficulty 0.3:0.7:0.2 \\\n"
                "      --seeds 0-9 --methods hedonic,leiden,spectral --nodes 600\n"
                "  hedonic show disjoint\n"))
    p.add_argument("--preset", default=None, help="paper | smoke (default: smoke); flags override it")
    g = p.add_argument_group("grid")
    g.add_argument("--nodes", type=int, help="vertices per graph (K must divide it)")
    g.add_argument("--communities", help="numbers of communities K, e.g. 2-6 or 2,4")
    g.add_argument("--p-in", dest="p_in", help="within-community edge probabilities, e.g. 0.01:0.10:0.01")
    g.add_argument("--difficulty", help="difficulty λ = q/p values, e.g. 0.1,0.5,0.75")
    g.add_argument("--seeds", help="network seeds, e.g. 0-99")
    g.add_argument("--noises", help="noise levels of the starting partitions, e.g. 0.1,0.25,0.5,0.75,1")
    g.add_argument("--partition-seeds", dest="partition_seeds", type=int, help="noisy starts per noise level")
    m = p.add_argument_group("methods")
    m.add_argument("--methods", help="comma list or all: groundtruth, mirror, onepass, spectral, leiden, hedonic")
    m.add_argument("--restarts", type=int, help="seeded restarts of Leiden and Hedonic per cell")
    r = p.add_argument_group("run")
    r.add_argument("--workers", type=int, help="graphs in parallel (default: CPUs − 2)")
    r.add_argument("--partitions", choices=("hash", "full"), help="store each partition or only its hash")
    r.add_argument("--output-dir", dest="output_dir", help="store directory (default: <disjoint_output_root>/<name>)")
    r.add_argument("--dry-run", action="store_true",
                   help="print the plan, estimates and what the store already holds per method; run nothing")
    s = p.add_argument_group("existing results")
    s.add_argument("--rerun", help="methods whose stored rows come from another build or setting: recompute them")
    s.add_argument("--reuse-from", dest="reuse_from",
                   help="a store (this one or another) whose rows of --reuse-methods are kept after a replay check")
    s.add_argument("--reuse-methods", dest="reuse_methods", help="methods to take from --reuse-from (comma list)")
    s.add_argument("--reuse-sample", dest="reuse_sample", type=int,
                   help="graphs replayed before reusing (default 5)")
    r.add_argument("--json", action="store_true", help="wait for the run and print its statistics as JSON")
    return p


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    """Preset + explicit flags -> validated settings."""
    sweep = _sweep()
    preset = PRESET_ALIASES.get(args.preset or "smoke", args.preset or "smoke")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {args.preset!r}; choose from: {', '.join(PRESETS)}")
    base = dict(sweep.PRESETS[PRESETS[preset][0]])
    settings = {
        "preset": preset,
        "nodes": int(args.nodes if args.nodes is not None else base["max_n_nodes"]),
        "communities": parse_ints(args.communities) if args.communities else list(base["n_communities"]),
        "p_in": parse_floats(args.p_in) if args.p_in else list(base["p_in"]),
        "difficulty": parse_floats(args.difficulty) if args.difficulty else list(base["difficulty"]),
        "seeds": parse_ints(args.seeds) if args.seeds else list(base["seeds"]),
        "noises": parse_floats(args.noises) if args.noises else list(base["noises"]),
        "partition_seeds": int(args.partition_seeds if args.partition_seeds is not None
                               else base["partition_seeds"]),
        "methods": parse_methods(args.methods) if args.methods else list(base["methods"]),
        "restarts": int(args.restarts if args.restarts is not None else base["n_runs"]),
        "partitions": args.partitions or base["partitions"],
        "workers": int(args.workers if args.workers is not None else default_workers()),
        "rerun": parse_methods(args.rerun) if getattr(args, "rerun", None) else [],
        "reuse_from": (str(expand_path(args.reuse_from).resolve())
                       if getattr(args, "reuse_from", None) else None),
        "reuse_methods": parse_methods(args.reuse_methods) if getattr(args, "reuse_methods", None) else [],
        "reuse_sample": int(getattr(args, "reuse_sample", None) or 5),
    }
    if bool(settings["reuse_from"]) != bool(settings["reuse_methods"]):
        raise ValueError("--reuse-from and --reuse-methods go together")
    stray = [m for m in settings["rerun"] + settings["reuse_methods"] if m not in settings["methods"]]
    if stray:
        raise ValueError(f"--rerun/--reuse-methods name methods the run does not: {', '.join(stray)}")
    if settings["workers"] < 1:
        raise ValueError("--workers must be at least 1")
    if settings["partition_seeds"] < 1 or settings["restarts"] < 1:
        raise ValueError("--partition-seeds and --restarts must be at least 1")
    sweep.validate_grid(
        max_n_nodes=settings["nodes"], n_communities=settings["communities"], p_in=settings["p_in"],
        difficulty=settings["difficulty"], noises=settings["noises"],
        partition_seeds=settings["partition_seeds"], methods=sweep._parse_methods(settings["methods"]),
        n_runs=settings["restarts"],
    )
    return settings


def default_workers() -> int:
    n = os.cpu_count() or 1
    return max(1, n - 2) if n > 3 else 1


def sweep_argv(settings: dict[str, Any], store_root: Path) -> list[str]:
    """The ``hedonic-exp disjoint`` arguments of these settings (store layout, resume)."""
    fmt = lambda xs: [repr(float(x)) if isinstance(x, float) else str(x) for x in xs]
    return [
        "--preset", PRESETS[settings["preset"]][0],
        "--max_n_nodes", str(settings["nodes"]),
        "--n_communities", *map(str, settings["communities"]),
        "--p_in", *fmt(settings["p_in"]),
        "--difficulty", *fmt(settings["difficulty"]),
        "--seeds", *map(str, settings["seeds"]),
        "--noises", *fmt(settings["noises"]),
        "--partition_seeds", str(settings["partition_seeds"]),
        "--methods", ",".join(settings["methods"]),
        "--n_runs", str(settings["restarts"]),
        "--layout", "store",
        "--partitions", settings["partitions"],
        "--workers", str(settings["workers"]),
        "--output_root", str(store_root.parent),
        "--folder_name", store_root.name,
        "--resume",
        *(["--rerun", ",".join(settings["rerun"])] if settings.get("rerun") else []),
        *(["--reuse-from", settings["reuse_from"], "--reuse-methods", ",".join(settings["reuse_methods"]),
           "--reuse-sample", str(settings.get("reuse_sample", 5))] if settings.get("reuse_from") else []),
    ]


def front_argv(settings: dict[str, Any]) -> list[str]:
    """Equivalent `hedonic run disjoint` flags (printed by the wizard and stored with the run)."""
    def floats(xs):
        return ",".join(f"{x:g}" for x in xs)

    def ints(xs):
        xs = list(xs)
        return f"{xs[0]}-{xs[-1]}" if len(xs) > 2 and xs == list(range(xs[0], xs[-1] + 1)) else ",".join(map(str, xs))

    return ["--preset", settings["preset"], "--nodes", str(settings["nodes"]),
            "--communities", ints(settings["communities"]), "--p-in", floats(settings["p_in"]),
            "--difficulty", floats(settings["difficulty"]), "--seeds", ints(settings["seeds"]),
            "--noises", floats(settings["noises"]), "--partition-seeds", str(settings["partition_seeds"]),
            "--methods", ",".join(m.lower() for m in settings["methods"]),
            "--restarts", str(settings["restarts"]), "--partitions", settings["partitions"],
            "--workers", str(settings["workers"]),
            *(["--rerun", ",".join(m.lower() for m in settings["rerun"])] if settings.get("rerun") else []),
            *(["--reuse-from", settings["reuse_from"],
               "--reuse-methods", ",".join(m.lower() for m in settings["reuse_methods"])]
              if settings.get("reuse_from") else []),
            *(["--reuse-sample", str(settings["reuse_sample"])]
              if settings.get("reuse_from") and settings.get("reuse_sample", 5) != 5 else [])]


def plan(settings: dict[str, Any]) -> dict[str, Any]:
    """Counts and estimates of a grid (no graph is generated)."""
    sweep = _sweep()
    methods = sweep._parse_methods(settings["methods"])
    cells = len(settings["noises"]) * settings["partition_seeds"]
    rows_per_graph = sum(
        sweep._method_n_runs(name, settings["restarts"]) * (cells if sweep._uses_initial_membership(info) else 1)
        for name, info in methods.items()
    )
    graphs = len(settings["communities"]) * len(settings["p_in"]) * len(settings["difficulty"]) * len(settings["seeds"])
    rows = graphs * rows_per_graph
    per_row = SECONDS_PER_ROW_FIXED + SECONDS_PER_ROW_PER_VERTEX * settings["nodes"]
    seconds = rows * per_row / settings["workers"]
    per_row_bytes = BYTES_PER_ROW_HASH + (BYTES_PER_ROW_PER_VERTEX_FULL * settings["nodes"]
                                          if settings["partitions"] == "full" else 0.0)
    return {"graphs": graphs, "rows_per_graph": rows_per_graph, "rows": rows,
            "logical_cells_per_graph": cells, "seconds": seconds, "bytes": rows * per_row_bytes}


def store_status(settings: dict[str, Any], output: str, *, replay: bool = False) -> dict[str, Any]:
    """What the store at ``output`` (and a ``--reuse-from`` source) already holds, per method.

    Read-only.  ``refused`` is the reason a run would stop before writing anything, ``state`` the
    store-level verdict per method (same / new / changed), ``graphs`` per method the graphs whose
    rows are current, of another build or setting, or missing; ``source`` the same counts in the
    reuse source (rows that would be carried after the replay check).
    """
    from hedonic.experiments.disjoint import store as store_mod

    sweep = _sweep()
    methods = sweep._parse_methods(settings["methods"])
    spec = store_mod.build_spec(methods=methods, n_runs=settings["restarts"],
                                partitions=settings["partitions"], generator_id=sweep.SBM_GENERATOR_ID)
    out: dict[str, Any] = {"refused": None, "state": None, "graphs": {}, "source": {}, "replay": None}
    try:
        out["state"] = store_mod.check_store(
            output, spec, rerun=set(settings.get("rerun", ())) | set(settings.get("reuse_methods", ())))
    except store_mod.StoreConflict as exc:
        out["refused"] = str(exc)

    def count(root: str) -> dict[str, dict[str, int]]:
        if store_mod.read_manifest(root) is None:
            return {}
        root = Path(root)
        found = sweep.preflight_grid(
            folder_name=root.name, max_n_nodes=settings["nodes"], n_communities=settings["communities"],
            seeds=settings["seeds"], p_in=settings["p_in"], difficulty=settings["difficulty"],
            noises=settings["noises"], partition_seeds=settings["partition_seeds"], methods=methods,
            n_runs=settings["restarts"], layout="store", output_root=root.parent, check_graphs=False,
            spec=spec)
        if Path(root).resolve() == Path(output).resolve():
            out["complete_graphs"] = found["complete_cells"] // max(1, found["cells_per_graph"])
            out["graphs_by_k"] = found["method_graphs_by_k"]
        return found["method_graphs"]

    out["complete_graphs"] = 0
    out["graphs_by_k"] = {}
    out["graphs"] = count(output)
    if settings.get("reuse_from"):
        if Path(settings["reuse_from"]).resolve() != Path(output).resolve():
            out["source"] = count(settings["reuse_from"])
        if replay and store_mod.read_manifest(settings["reuse_from"]) is not None:
            # The check the run makes before carrying, on a small sample (seconds).
            out["replay"] = store_mod.verify_carry(
                settings["reuse_from"], settings["reuse_methods"],
                sample_shards=min(2, settings.get("reuse_sample", 5)), sample_rows=5)
    return out


def render_store_status(settings: dict[str, Any], output: str, status: dict[str, Any] | None = None) -> str:
    from hedonic.experiments.disjoint import store as store_mod
    from hedonic.experiments.overlapping.tui import style

    status = status or store_status(settings, output)
    graphs = plan(settings)["graphs"]
    lines = [style("  existing results", "bold")]
    if not status["graphs"] and not status["refused"]:
        lines.append(f"  store       {output} is empty: every method runs")
    for title, table in (("store", status["graphs"]), ("reuse from", status["source"])):
        if not table:
            continue
        where = output if title == "store" else settings["reuse_from"]
        lines.append(f"  {title:<11} {where}")
        for name, c in table.items():
            if title == "reuse from" and name not in settings["reuse_methods"]:
                continue
            what = []
            if c["current"]:
                what.append(f"{c['current']:,} current")
            if c["other"]:
                what.append(f"{c['other']:,} from another build or setting")
            if c["missing"]:
                what.append(f"{c['missing']:,} missing")
            lines.append(f"    {METHOD_LABELS.get(name, name):<13} {' · '.join(what) or '-'}  (of {graphs:,} graphs)")
    replay = status.get("replay")
    if replay:
        clock = [p for p in replay["problems"] if "timed" in p]
        found = [p for p in replay["problems"] if p not in clock]
        verdict = (style(f"{replay['rows_checked']} replayed rows match", "green") if not found else
                   style(f"{len(found)} of {replay['rows_checked']} replayed rows differ", "red"))
        lines.append(f"  replay      {verdict} (sample of {replay['graphs_checked']} graphs, "
                     f"{', '.join(METHOD_LABELS[m] for m in replay['methods'])})")
        for problem in found[:3]:
            lines.append(style(f"              {problem}", "dim"))
        if replay["clock_differs"]:
            lines.append(style(f"  clock       the source timed its rows under {replay['timing_scope'] or 'the old rule'}, "
                               f"this build under {store_mod.TIMING_SCOPE}: a run will not carry them", "yellow"))
    if status["refused"]:
        lines.append(style(f"  refused     {status['refused']}", "red"))
    else:
        todo = [m for m in settings["methods"]
                if not status["graphs"] or status["graphs"].get(m, {}).get("current", 0) < graphs]
        blocked = bool(replay and (replay["problems"] or replay["clock_differs"]))
        carried = [] if blocked else [m for m in todo if m in settings.get("reuse_methods", [])]
        runs = [m for m in todo if m not in carried]
        if carried:
            lines.append(f"  reuse       {', '.join(METHOD_LABELS[m] for m in carried)} "
                         f"(after replaying {settings.get('reuse_sample', 5)} graphs with this build)")
        elif blocked and replay and not [p for p in replay["problems"] if "timed" not in p]:
            lines.append(style("  reuse       none: same results, other clock; recompute them (cheap) and keep the "
                               "replay as the evidence that they did not change", "dim"))
        lines.append(f"  to compute  {', '.join(METHOD_LABELS[m] for m in runs) or 'nothing (the store is complete)'}")
    return "\n".join(lines)


def describe(settings: dict[str, Any]) -> str:
    info = plan(settings)
    ks = settings["communities"]
    k_text = f"K {ks[0]}" if len(ks) == 1 else (f"K {ks[0]}–{ks[-1]}" if ks == list(range(ks[0], ks[-1] + 1))
                                                else "K " + ",".join(map(str, ks)))
    return (f"SBM {settings['preset']} · {info['graphs']:,} graphs of {settings['nodes']:,} vertices · "
            f"{k_text} · {len(settings['seeds'])} seeds · {settings['workers']} workers")


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def render_plan(settings: dict[str, Any], output: str | None = None) -> str:
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping.tui import style

    info = plan(settings)

    def span(xs, fmt="{:g}"):
        xs = list(xs)
        return ", ".join(fmt.format(x) for x in xs) if len(xs) <= 6 else \
            f"{fmt.format(xs[0])} … {fmt.format(xs[-1])} ({len(xs)} values)"

    lines = [
        style(f"  disjoint SBM · preset {settings['preset']}", "bold") + style(f"  {PRESETS[settings['preset']][1]}", "dim"),
        (f"  graphs      {info['graphs']:,}  = K {span(settings['communities'])} × p {span(settings['p_in'])} × "
        f"λ {span(settings['difficulty'])} × {len(settings['seeds'])} network seeds"),
        (f"  per graph   {settings['nodes']:,} vertices · noises {span(settings['noises'])} × "
        f"{settings['partition_seeds']} starts · {settings['restarts']} restarts of the CPM methods"),
        f"  methods     {', '.join(METHOD_LABELS[m] for m in settings['methods'])}",
        (f"  results     {info['rows']:,} detector runs ({info['rows_per_graph']:,} per graph) · "
        f"partitions {settings['partitions']}"),
        (f"  estimate    {'under a minute' if info['seconds'] < 60 else '~' + qs.fmt_time(info['seconds'])} with {settings['workers']} workers · "
        f"~{_human_bytes(info['bytes'])} on disk"),
    ]
    if output:
        lines.append(f"  output      {output}")
    lines.append(style("  then: logical-cell table → 5 figures → paper statistics (stats/summary.md)", "dim"))
    lines.append(style("  equivalent: hedonic run disjoint " + " ".join(front_argv(settings)), "dim"))
    return "\n".join(lines)


def render_presets() -> str:
    """``hedonic show disjoint``."""
    from hedonic.experiments.overlapping.tui import style

    out = [style("  disjoint SBM presets (`hedonic run disjoint --preset NAME`)", "bold"), ""]
    for name in PRESETS:
        settings = resolve(build_parser().parse_args(["--preset", name]))
        out.append(render_plan(settings))
        out.append("")
    out.append(style("  methods: " + ", ".join(f"{m.lower()} ({METHOD_LABELS[m]})" for m in _sweep().METHODS), "dim"))
    out.append(style("  every grid value is a flag; see `hedonic run disjoint --help`", "dim"))
    return "\n".join(out)


# --------------------------------------------------------------------------- front door
def default_output(name: str) -> str:
    from hedonic.experiments.overlapping import userconfig

    machine = userconfig.defaults()
    base = machine.get("disjoint_output_root") or DISJOINT_RUNS_ROOT
    return str(expand_path(base) / name)


def wizard(settings: dict[str, Any]) -> dict[str, Any]:
    from hedonic.experiments.overlapping import tui

    keys = list(PRESETS) + ["custom"]
    labels = [(k, PRESETS[k][1]) for k in PRESETS] + [("custom", "choose the grid, methods and workers")]
    choice = keys[tui.select("Which disjoint run?", labels, default=1)]
    args = build_parser().parse_args(["--preset", "smoke" if choice == "custom" else choice])
    settings = resolve(args)
    if choice == "custom":
        def ask(title, current, parse):
            return parse(tui.text(title, current))

        flags = dict(zip(front_argv(settings)[::2], front_argv(settings)[1::2]))
        settings["nodes"] = ask("Vertices per graph", flags["--nodes"], int)
        settings["communities"] = ask("Numbers of communities K", flags["--communities"], parse_ints)
        settings["p_in"] = ask("Within-community probabilities p", flags["--p-in"], parse_floats)
        settings["difficulty"] = ask("Difficulties λ = q/p", flags["--difficulty"], parse_floats)
        settings["seeds"] = ask("Network seeds", flags["--seeds"], parse_ints)
        settings["noises"] = ask("Noise levels", flags["--noises"], parse_floats)
        names = list(_sweep().METHODS)
        picked = tui.multiselect("Methods", [(n, METHOD_LABELS[n]) for n in names],
                                 selected=[names.index(m) for m in settings["methods"]])
        settings["methods"] = [names[i] for i in picked] or settings["methods"]
    settings["workers"] = int(tui.text(f"Workers (graphs in parallel; {os.cpu_count()} CPUs)", str(settings["workers"])))
    return resolve(build_parser().parse_args(front_argv(settings)))


def front_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.add_argument("-i", "--interactive", action="store_true", help="open the wizard (default when no arguments)")
    parser.add_argument("-y", "--yes", action="store_true", help="never prompt")
    parser.add_argument("--name", help="run name (default disjoint-YYYYmmdd-HHMMSS)")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    parser.add_argument("--foreground", action="store_true", help="run in this terminal (not recoverable)")
    return parser


def front_study(argv: list[str]) -> dict[str, Any]:
    """The settings, name and store folder of `hedonic run disjoint ARGS` (a queue item), nothing run or written.

    Raises ``ValueError`` on a bad argument; argparse errors exit as usual.  A queue item names its run (``--name``)
    and so its folder; without ``--output-dir`` the folder is ``<disjoint_output_root>/<name>``.
    """
    from hedonic.experiments.config import ensure_not_archived_v1020
    from hedonic.experiments.overlapping import quickstart as qs

    args = front_parser().parse_args(list(argv))
    if not args.name:
        raise ValueError("a queued disjoint run needs --name (its store folder follows it)")
    qs.parse_name(args.name)
    settings = resolve(args)
    output = str(expand_path(args.output_dir).resolve()) if args.output_dir else default_output(args.name)
    ensure_not_archived_v1020(output, label="disjoint output")
    return {"settings": settings, "name": args.name, "output": output}


def front_main(argv: list[str] | None = None) -> int:
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager, tui
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = front_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.name:
            qs.parse_name(args.name)
        settings = resolve(args)
        if (args.interactive or (not argv and tui.interactive())) and not args.yes:
            settings = wizard(settings)
            print(style("  equivalent: hedonic run disjoint " + " ".join(front_argv(settings)), "dim"))
        name = args.name or runmanager.new_name("disjoint")
        output = str(expand_path(args.output_dir).resolve()) if args.output_dir else default_output(name)
        from hedonic.experiments.config import ensure_not_archived_v1020

        ensure_not_archived_v1020(output, label="disjoint output")
    except ValueError as exc:
        print(f"hedonic run disjoint: error: {exc}\n(see `hedonic run disjoint --help`)", file=sys.stderr)
        return 2
    except (tui.Cancelled, KeyboardInterrupt):
        print(style("\n  cancelled", "yellow"))
        return 130
    print(render_plan(settings, output))
    status = store_status(settings, output, replay=args.dry_run)
    if args.dry_run or status["graphs"] or status["source"] or status["refused"]:
        print(render_store_status(settings, output, status))
    if args.dry_run:
        return 0
    if status["refused"]:
        # Refused before anything is written or registered.
        print(f"hedonic run disjoint: error: {status['refused']}", file=sys.stderr)
        return 2
    if args.foreground:
        entry = {"name": name, "kind": "disjoint", "settings": settings, "output_dir": output,
                 "argv": sweep_argv(settings, Path(output)), "command": "hedonic run disjoint " + " ".join(front_argv(settings))}
        return work(name, entry, {}, progress_file=Path(output) / "progress.json")
    entry = runmanager.launch_disjoint(settings, name, output)
    where = f"tmux session {entry['session']}" if entry["backend"] == "tmux" else f"background process (log {entry['log']})"
    out = sys.stderr if args.json else sys.stdout
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}", file=out)
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']} · morning check: hedonic run check {entry['name']}", "dim"),
          file=out)
    if args.json and not args.detach:
        return runmanager.wait_and_print_json(entry)
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


# --------------------------------------------------------------------------- worker
def work(name: str, entry: dict, prog: dict, progress_file: Path | None = None) -> int:
    """Run the sweep into the store, then table, figures and statistics; mirror progress."""
    from hedonic.experiments.disjoint import paper_stats
    from hedonic.experiments.disjoint import store as store_mod
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    settings = entry["settings"]
    store = Path(entry["output_dir"])
    store.mkdir(parents=True, exist_ok=True)
    ppath = progress_file or runmanager.progress_path(entry)
    prog.update({"name": name, "kind": "disjoint", "status": "running", "stage": "sweep", "pid": os.getpid(),
                 "started": time.time(), "updated": time.time(), "describe": describe(settings),
                 "command": entry["command"], "environment": qs.environment(), "plan": [],
                 "graphs": plan(settings)["graphs"], "done": 0, "written": 0, "skipped": 0,
                 "by_k": {}, "methods": {}, "recent": []})
    runmanager.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(settings))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    last = [0.0]

    def flush(force: bool = False) -> None:
        if force or time.time() - last[0] > 1.0:
            prog["updated"] = last[0] = time.time()
            runmanager.write_json(ppath, prog)

    def on_event(event: dict) -> None:
        if event["type"] == "plan":
            prog["graphs"], prog["workers"] = event["graphs"], event["workers"]
            prog["by_k"] = {str(k): [0, event["per_community"]] for k in event["communities"]}
            flush(True)
            return
        label, status = event["label"], event["status"]
        prog["done"] += 1
        prog[status] = prog.get(status, 0) + 1
        prog["by_k"][str(label["number_of_communities"])][0] += 1
        if status == "written":
            now = time.time()
            prog["recent"] = [t for t in prog["recent"] if now - t < 900] + [now]
            prog.setdefault("first_written", now)
        for method, sums in (event.get("summary") or {}).items():
            acc = prog["methods"].setdefault(method, {"n": 0})
            for key, value in sums.items():
                acc[key] = acc.get(key, 0) + value
        if prog["done"] % max(1, prog["graphs"] // 200) == 0 or prog["done"] == prog["graphs"]:
            print(f"  [{prog['done']:,}/{prog['graphs']:,}] graphs · {prog['written']:,} written, "
                  f"{prog['skipped']:,} already complete", flush=True)
        flush()

    code = _sweep().main(entry["argv"], progress=on_event)
    if code not in (0, None, True):
        raise RuntimeError(f"the sweep exited with code {code}")
    outputs: dict[str, str] = {"store": str(store)}
    notes: list[str] = []
    table = store / "plot_table.parquet"
    prog.update(stage="logical-cell table")
    flush(True)
    print("  writing the logical-cell table…", flush=True)
    if store_mod.main([str(store), "--export", str(table)]) == 0:
        outputs["table"] = str(table)
    prog.update(stage="figures")
    flush(True)
    print("  drawing the five figures…", flush=True)
    try:
        from hedonic.experiments.plots import paper_figures

        if paper_figures.main(["--data", str(table), "--output_dir", str(store / "figures"), "--no-persist"]) == 0:
            outputs["figures"] = str(store / "figures")
    except Exception as exc:
        notes.append(f"figures skipped: {type(exc).__name__}: {exc}")
    prog.update(stage="statistics")
    flush(True)
    print("  computing the paper statistics…", flush=True)
    try:
        if paper_stats.main(["--data", str(table), "--output_dir", str(store / "stats")]) == 0:
            outputs["stats"] = str(store / "stats" / "summary.md")
            stats = json.loads((store / "stats" / "paper_stats.json").read_text())
            prog["headline"] = {"grid": stats.get("grid"), "mean_ari": stats.get("tracking", {}).get("mean_ari")}
    except Exception as exc:
        notes.append(f"statistics skipped: {type(exc).__name__}: {exc}")
    prog.update(status="completed", stage="done", finished=time.time(), outputs=outputs, notes=notes)
    flush(True)
    print(style(f"\n  completed · store {store}", "green", "bold"))
    for key, value in outputs.items():
        print(f"  {key:<8}{value}")
    for note in notes:
        print(style(f"  {note}", "yellow"))
    print(style(f"  morning check: hedonic run check {name}", "dim"))
    return 0


# --------------------------------------------------------------------------- viewer and check
def snapshot(entry: dict, prog: dict, status: str, frame: str, table: bool) -> str:
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping import tui
    from hedonic.experiments.overlapping.tui import style

    G = tui.glyphs()
    colour = {"running": "cyan", "completed": "green", "interrupted": "red", "stopped": "yellow",
              "failed": "red"}.get(status, "dim")
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  "
             + style(f"{runmanager.status_mark(status, frame, G)} {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + (prog.get("describe") or describe(entry["settings"])), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · python-igraph {env.get('python_igraph')} · "
                           f"igraph C {env.get('igraph_c')} · Python {env.get('python')}", "dim"))
    total, done = prog.get("graphs") or plan(entry["settings"])["graphs"], prog.get("done", 0)
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status == "running" else prog.get("updated", time.time()))
    tail = f"  {done:,}/{total:,} graphs · elapsed {qs.fmt_time(end - started)}"
    recent = prog.get("recent") or []
    if status == "running" and prog.get("stage") == "sweep" and len(recent) >= 2:
        window = max(1.0, time.time() - recent[0]) if len(recent) > 1 else 1.0
        rate = len(recent) / window
        tail += f" · {rate * 3600:,.0f}/h · ETA ~{qs.fmt_time((total - done) / rate)}"
    lines.append("  " + runmanager.progress_bar(done, total, 36, G) + tail)
    if prog.get("skipped"):
        lines.append(style(f"  {prog['skipped']:,} graphs were already complete (resumed)", "dim"))
    if status == "running" and prog.get("stage") not in (None, "sweep"):
        lines.append(style(f"  {frame} {prog['stage']}…", "magenta"))
    if table and prog.get("by_k"):  # one line per K: the whole list is too wide for one terminal line
        lines.append(style("  by K", "dim"))
        for k, (d, n) in prog["by_k"].items():
            lines.append(f"    K={k:<4}{runmanager.progress_bar(d, n, 24, G)}  {d:,}/{n:,}")
    methods = prog.get("methods") or {}
    if table and methods:
        lines.append(style(f"\n  {'method':<18}{'runs':>12}{'ARI':>8}{'overlap':>9}{'NMI':>7}{'robust':>8}{'ms/run':>9}",
                           "bold"))
        for name in _sweep().METHODS:
            item = methods.get(name)
            if not item or not item.get("n"):
                continue
            n = item["n"]
            lines.append(f"  {METHOD_LABELS[name]:<18}{n:>12,}{item['adjusted_rand'] / n:>8.3f}"
                         f"{item['overlap_normalized'] / n:>9.3f}{item['nmi'] / n:>7.3f}"
                         f"{item['robustness'] / n:>8.3f}{1000 * item['duration'] / n:>9.2f}")
        lines.append(style("  running means over the graphs written by this run (every restart); "
                           "paper statistics come at the end", "dim"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}",
                           "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); continue with: hedonic run resume {entry['name']}",
                           "red"))
    elif status == "stopped":
        lines.append(style(f"\n  stopped · continue with: hedonic run resume {entry['name']}", "yellow"))
    if status in ("completed", "stopped", "failed", "interrupted"):
        for key, value in (prog.get("outputs") or {"store": entry["output_dir"]}).items():
            lines.append(style(f"  {key:<9}{value}", "dim"))
        for note in prog.get("notes") or []:
            lines.append(style(f"  {note}", "yellow"))
        if status == "completed":
            lines.append(style(f"  morning check: hedonic run check {entry['name']}", "dim"))
    return "\n".join(lines)


def check(entry: dict, shards: int, rows: int) -> int:
    """Replay sampled rows of a disjoint run's store from their seeds."""
    from hedonic.experiments.disjoint import store as store_mod
    from hedonic.experiments.overlapping.tui import style

    root = Path(entry["output_dir"])
    if store_mod.read_manifest(root) is None:
        print(f"  {entry['name']}: no store yet at {root}")
        return 1
    info = store_mod.progress(root)
    print(f"  store    {root}\n  shards   {info['completed']:,}/{info['expected']:,} graphs · "
          f"{'running' if info['running'] else 'not running'} · {info['bytes_on_disk'] / 1024**3:.2f} GiB")
    report = store_mod.verify_store(root, sample_shards=shards, sample_rows=rows)
    ok = report["ok"]
    print(style(f"  replay   {report['graphs_checked']} graphs regenerated, {report['rows_checked']:,} rows replayed, "
                f"{len(report['problems'])} problem(s)", "green" if ok else "red", "bold"))
    for problem in report["problems"][:20]:
        print(style(f"    - {problem}", "red"))
    return 0 if ok else 1
