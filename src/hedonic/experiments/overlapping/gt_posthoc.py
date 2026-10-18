"""Exact post-hoc stability of a supplied ground-truth cover, as a durable run.

For every vertex, the exact set of resolutions in [0, 1] where its ground-truth
membership is a best response (``overlapping.stability``: the C audit of
``hedonic._core``), under both isolation
policies. It then audits the locked resolution grid: the exact and grid stable
fractions side by side, the Nash window, and the vertices with several intervals.
No detector runs, and no spectrum output is touched.

    hedonic run posthoc --config FILE.toml --output-dir DIR   # durable: status, attach, ETA, resume
    hedonic-exp overlapping-gt-posthoc --config FILE.toml --workers 8

``--config`` takes a TOML path or a name resolved through
:func:`hedonic.experiments.resources.config_path` (``NAME`` tries
``overlapping-gt-posthoc-NAME.toml`` first). Without ``audit_resolutions`` the
registered 131-point audit grid (``gt_spectrum.AUDIT_GRID``) is used. A pair
missing from the networks folder is downloaded first (``snap.ensure_pair``).
The run folder is ``--output-dir``, else the config's ``output_dir``, else
``<output_dir>/NAME``; it holds ``summary.json``
(per-policy timings, agreement with the grid, Nash windows) and
``result.json.gz`` (every vertex interval).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from hedonic.experiments.config import RUNS_ROOT, expand_path
from hedonic.experiments.overlapping import ground_truth_data as gtd
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import snap
from hedonic.experiments.overlapping import stability as st
from hedonic.experiments.resources import config_path as resolve_config

CONFIG_PREFIX = "overlapping-gt-posthoc-"
POLICIES = (("fixed_labels", False), ("open_labels", True))
ATOL, RTOL = 1e-10, 1e-9
PROGRESS_EVERY = 400  # vertex updates per policy

Progress = Callable[[dict[str, Any]], None]


def _silent(event: dict[str, Any]) -> None:
    return None


# --------------------------------------------------------------------------- computation
VERTEX_ORDER_SEED = 0


def vertex_order(n_vertices: int) -> np.ndarray:
    """The fixed pseudo-random order the vertices are audited in.

    Not the index order, because on DBLP the low-index vertices are heavier: the first 1,304 have about three times the
    share of vertices with more than 320 candidate labels (16% against 6%), so an ETA taken from that prefix is far too
    pessimistic. Only the schedule changes; each vertex's intervals are the same, and the caller stores them by vertex.
    """
    return np.random.default_rng(VERTEX_ORDER_SEED).permutation(n_vertices)


def _vertex_stream(cover: st.PreparedCover, cap: int, allow: bool, *, atol: float, rtol: float, workers: int,
                   chunk: int | None = None) -> Iterator[tuple[list[int], list[list[list[float]]]]]:
    """``(vertices, intervals)`` chunk by chunk in ``vertex_order``; ``workers`` forked processes split the chunks
    (the prepared C state is read-only and inherited, so they share it)."""
    n = cover.n_vertices
    size = chunk or max(1, min(st.MAX_CHUNK, -(-n // PROGRESS_EVERY)))
    for vertices, intervals in st.stream_intervals(cover, cap, allow, atol=atol, rtol=rtol, workers=workers,
                                                   order=vertex_order(n), chunk=size):
        yield vertices.tolist(), [[[float(lo), float(hi)] for lo, hi in row] for row in intervals]


def _nash_window(per_vertex: list[list[list[float]]]) -> list[list[float]]:
    """Resolutions where every vertex is stable: the intersection of the per-vertex sets."""
    window = [[0.0, 1.0]]
    for intervals in per_vertex:
        if not intervals:
            return []
        merged = []
        for lo1, hi1 in window:
            for lo2, hi2 in intervals:
                lo, hi = max(lo1, lo2), min(hi1, hi2)
                if lo <= hi:
                    merged.append([lo, hi])
        window = merged
        if not window:
            return []
    return window


def _interval_arrays(per_vertex: list[list[list[float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi, owner = [], [], []
    for v, intervals in enumerate(per_vertex):
        for a, b in intervals:
            lo.append(a)
            hi.append(b)
            owner.append(v)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), np.asarray(owner, dtype=np.int64)


def analyse_state(cover: st.PreparedCover, *, cap: int, grid: list[float], workers: int = 1, atol: float = ATOL,
                  rtol: float = RTOL, progress: Progress = _silent) -> dict[str, Any]:
    """Exact intervals for every vertex and both policies, audited against ``grid`` (``cover`` from ``st.prepare``)."""
    n_vertices = cover.n_vertices
    total = n_vertices * len(POLICIES)
    done, start = 0, time.time()
    policies: dict[str, Any] = {}
    for policy, allow in POLICIES:
        progress({"type": "policy", "policy": policy, "vertices": n_vertices})
        per_vertex: list[list[list[float]] | None] = [None] * n_vertices
        policy_done = 0
        t = time.time()
        for vertices, intervals in _vertex_stream(cover, cap, allow, atol=atol, rtol=rtol, workers=workers):
            for v, found in zip(vertices, intervals):
                per_vertex[v] = found
            done += len(vertices)
            policy_done += len(vertices)
            progress({"type": "vertices", "policy": policy, "policy_done": policy_done,
                      "policy_total": n_vertices, "done": done, "total": total,
                      "elapsed": time.time() - start})
        exact_seconds = time.time() - t
        progress({"type": "stage", "text": f"{policy}: grid audit on {len(grid)} resolutions"})
        rows = st.spectrum_rows(cover, cap, grid, atol=atol, rtol=rtol, allow_isolation=allow, workers=workers)
        lo, hi, owner = _interval_arrays(per_vertex)
        per_gamma = []
        for gamma, row in zip(grid, rows):
            exact_count = int(np.unique(owner[(lo <= gamma) & (hi >= gamma)]).size)
            per_gamma.append({
                "gamma": gamma,
                "exact_stable_vertex_count": exact_count,
                "exact_stable_fraction": exact_count / n_vertices,
                "grid_stable_vertex_count": row["stable_vertex_count"],
                "grid_stable_fraction": row["stable_fraction"],
                "grid_is_nash": row["is_nash_equilibrium"],
                "grid_max_positive_regret": row["max_positive_regret"],
                "grid_mean_positive_regret": row["mean_positive_regret"],
            })
        policies[policy] = {
            "exact_seconds": round(exact_seconds, 2),
            "vertices_with_several_intervals": int(sum(len(intervals) > 1 for intervals in per_vertex)),
            "nash_window": _nash_window(per_vertex),
            "per_gamma": per_gamma,
            "intervals": per_vertex,
        }
    return policies


def analyse(options: dict[str, Any], progress: Progress = _silent) -> dict[str, Any]:
    """Load the configured SNAP pair and run :func:`analyse_state` on its ground-truth cover."""
    started = time.time()
    progress({"type": "stage", "text": f"loading {options['dataset']}/{options['cover']}"})
    files = snap.ensure_pair(options["dataset"], options["cover"], download=True,
                             say=lambda line: progress({"type": "stage", "text": line}))
    raw = snap.load_snap_dataset(files.network, cover_variant=files.cover, data_root=files.root,
                                 allow_catalog=False)
    prepared = gtd.prepare_dataset(raw, policy="covered-induced", max_nodes=options["max_nodes"])
    graph = prepared.dataset.graph
    gt_cover = rb.canonicalize_cover(prepared.dataset.cover, graph.vcount())
    memberships = rb.cover_to_vertex_memberships(gt_cover, graph.vcount(), require_covered=True)
    cap = max(2, max(len(labels) for labels in memberships))
    cover = st.prepare(graph, memberships)
    policies = analyse_state(cover, cap=cap, grid=options["grid"], workers=options["workers"], atol=options["atol"],
                             rtol=options["rtol"], progress=progress)
    return {"dataset": options["dataset"], "cover": options["cover"], "graph_identity": prepared.graph_identity,
            "n_vertices": graph.vcount(), "n_edges": graph.ecount(), "cap": cap,
            "atol": options["atol"], "rtol": options["rtol"], "grid": options["grid"], "engine": cover.engine,
            "policies": policies, "seconds": round(time.time() - started, 2)}


# --------------------------------------------------------------------------- outputs
def summarise(options: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    policies = {}
    for policy, block in result["policies"].items():
        policies[policy] = {
            "exact_seconds": block["exact_seconds"],
            "vertices_with_several_intervals": block["vertices_with_several_intervals"],
            "nash_window": block["nash_window"],
            "max_stable_fraction_difference": max(abs(row["exact_stable_fraction"] - row["grid_stable_fraction"])
                                                   for row in block["per_gamma"]),
            "per_gamma": block["per_gamma"],
        }
    return {"name": options["name"], "config": options["config"], "dataset": result["dataset"],
            "cover": result["cover"], "graph_identity": result["graph_identity"],
            "n_vertices": result["n_vertices"], "n_edges": result["n_edges"], "cap": result["cap"],
            "grid_points": len(result["grid"]), "atol": result["atol"], "rtol": result["rtol"],
            "seconds": result["seconds"], "engine": result.get("engine", "python"),
            "output_dir": options["output_dir"], "result_file": "result.json.gz",
            "policies": policies}


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def write_outputs(options: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    out = Path(options["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(options, result)
    _atomic_write(out / "result.json.gz", gzip.compress(json.dumps(result).encode()))
    _atomic_write(out / "summary.json", json.dumps(summary, indent=2).encode())
    return summary


def render_summary(summary: dict[str, Any]) -> list[str]:
    lines = []
    for policy, block in summary["policies"].items():
        lines.append(f"{policy}: exact {block['exact_seconds']}s, max |exact-grid| stable fraction "
                     f"{block['max_stable_fraction_difference']:.3e}, vertices with several intervals "
                     f"{block['vertices_with_several_intervals']}, Nash window {block['nash_window']}")
    lines.append(f"records {summary['output_dir']}")
    return lines


# --------------------------------------------------------------------------- configuration
def locked_grid() -> list[float]:
    """The registered 131-point audit grid of the spectrum (``gt_spectrum.AUDIT_GRID``)."""
    from hedonic.experiments.overlapping.gt_spectrum import AUDIT_GRID, _parse_grid

    return [float(g) for g in _parse_grid(AUDIT_GRID)]


def named_config(value: str) -> Path:
    """A config path, or a name: an existing file wins, then ``overlapping-gt-posthoc-NAME.toml``, then ``NAME``,
    both through :func:`hedonic.experiments.resources.config_path` (the shipped configs)."""
    given = expand_path(value)
    if given.is_file():
        return given
    for candidate in (f"{CONFIG_PREFIX}{value}", value):
        try:
            path = Path(resolve_config(candidate))
        except (FileNotFoundError, ValueError):
            continue
        if path.is_file():
            return path
    raise ValueError(f"no such config: {value} (neither a TOML file nor a shipped config "
                     f"{CONFIG_PREFIX}{value}.toml)")


def load_options(config: str, *, name: str | None = None, output_dir: str | None = None,
                 workers: int | None = None, default_output_root: str | None = None) -> dict[str, Any]:
    """Resolve ``[overlapping_gt_posthoc]`` from a TOML file; flags override the file."""
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping.gt_spectrum import _parse_grid

    path = named_config(config)
    section = tomllib.loads(path.read_text(encoding="utf-8")).get("overlapping_gt_posthoc", {})
    if "dataset" not in section:
        raise ValueError(f"{path}: [overlapping_gt_posthoc] needs `dataset`")
    dataset, cover = str(section["dataset"]), str(section.get("cover", "all"))
    name = name or str(section.get("name", f"posthoc-{dataset}-{cover}"))
    qs.parse_name(name)
    grid = _parse_grid(section["audit_resolutions"]) if "audit_resolutions" in section else locked_grid()
    default_root = Path(default_output_root) if default_output_root else RUNS_ROOT
    out = output_dir or section.get("output_dir") or default_root / name
    return {"config": str(path.resolve()), "name": name, "dataset": dataset, "cover": cover,
            "max_nodes": int(section.get("max_nodes", 0)) or None,
            "workers": int(workers or section.get("workers", 1)),
            "atol": float(section.get("atol", ATOL)), "rtol": float(section.get("rtol", RTOL)),
            "grid": grid, "output_dir": str(expand_path(out).resolve())}


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Exact post-hoc stability of a ground-truth cover.")
    parser.add_argument("--config", required=True,
                        help="a TOML path, or NAME (the shipped overlapping-gt-posthoc-NAME.toml, else NAME.toml)")
    parser.add_argument("--name", help="run name (default: the config's name)")
    parser.add_argument("--output-dir", help="run folder (default: the config's output_dir, else "
                                             "<output_dir>/NAME)")
    parser.add_argument("--workers", type=int, help="forked vertex workers (default: the config's workers, else 1)")
    return parser


def options_from_argv(argv: list[str]) -> dict[str, Any]:
    args = build_parser("hedonic run posthoc").parse_args(argv)
    return load_options(args.config, name=args.name, output_dir=args.output_dir, workers=args.workers)


def describe(options: dict[str, Any]) -> str:
    size = "complete graph" if options["max_nodes"] is None else f"{options['max_nodes']}-vertex induced subgraph"
    return (f"exact post-hoc stability · {options['dataset']}/{options['cover']} · {size} · "
            f"{len(options['grid'])} resolutions × {len(POLICIES)} policies")


# --------------------------------------------------------------------------- CLI entry points
def main(argv: list[str] | None = None) -> int:
    """``hedonic-exp overlapping-gt-posthoc``: foreground run; use ``hedonic run posthoc`` for a durable one."""
    import sys

    args = build_parser("hedonic-exp overlapping-gt-posthoc").parse_args(argv)
    options = load_options(args.config, name=args.name, output_dir=args.output_dir, workers=args.workers)

    def show(event: dict[str, Any]) -> None:
        if event["type"] in ("stage", "policy"):
            text = event["text"] if "text" in event else f"{event['policy']}: {event['vertices']:,} vertices"
            print("  " + text, file=sys.stderr, flush=True)

    summary = write_outputs(options, analyse(options, progress=show))
    for line in render_summary(summary):
        print(line)
    return 0


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run posthoc``: durable run with a live viewer, ETA, status, stop and resume."""
    import sys

    from hedonic.experiments.overlapping import runmanager, userconfig
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser("hedonic run posthoc")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    args = parser.parse_args(argv)
    machine = userconfig.defaults()
    options = load_options(args.config, name=args.name, output_dir=args.output_dir, workers=args.workers,
                           default_output_root=str(machine.get("output_dir") or RUNS_ROOT))
    entry = runmanager.launch_posthoc(
        ["--config", options["config"], "--name", options["name"], "--output-dir", options["output_dir"],
         "--workers", str(options["workers"])],
        options["name"], options["output_dir"])
    where = f"tmux session {entry['session']}" if entry["backend"] == "tmux" else f"background process (log {entry['log']})"
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


# --------------------------------------------------------------------------- durable run
def snapshot(entry: dict[str, Any], prog: dict[str, Any], frame: str) -> str:
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    status = rm.effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "interrupted": "red",
              "stopped": "yellow", "failed": "red"}.get(status, "dim")
    done, total = int(prog.get("done", 0)), int(prog.get("total", 0))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting")
                                   else prog.get("updated", time.time()))
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "exact post-hoc stability"), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = f"  {done:,}/{total:,} vertex checks · elapsed {qs.fmt_time(end - started)}"
        if status in ("running", "setup") and prog.get("eta_seconds") is not None:
            tail += f" · ETA ~{qs.fmt_time(prog['eta_seconds'])}"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    else:
        lines.append(style(f"  preparing… elapsed {qs.fmt_time(end - started)}", "dim"))
    if status in ("running", "setup") and prog.get("stage"):
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); restart with: hedonic run resume {entry['name']}", "red"))
    if status == "completed":
        for item in prog.get("policies", []):
            lines.append(f"  {item['policy']:<13} exact {item['exact_seconds']}s · max |exact−grid| "
                         f"{item['max_difference']:.1e} · several intervals {item['several']} · Nash window "
                         f"{item['nash_window']}")
        lines.append(style(f"\n  records  {entry['output_dir']}", "dim"))
    return "\n".join(lines)


def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body for a posthoc run: mirrors every analysis event into progress.json."""
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    options = options_from_argv(entry["argv"])
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "posthoc", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": describe(options), "command": entry["command"],
                 "environment": qs.environment(), "done": 0, "total": 0, "eta_seconds": None,
                 "stage": "loading", "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(options))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    last = [0.0]

    def flush(force: bool = False) -> None:
        if force or time.time() - last[0] > 0.5:
            prog["updated"] = last[0] = time.time()
            rm.write_json(ppath, prog)

    def on_event(event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "policy":
            prog.update(status="running", stage=f"{event['policy']}: exact stability of {event['vertices']:,} vertices")
            print("  " + prog["stage"], flush=True)
            flush(True)
        elif kind == "stage":
            prog.update(status="running", stage=event["text"])
            print("  " + event["text"], flush=True)
            flush(True)
        elif kind == "vertices":
            done, total = event["done"], event["total"]
            prog.update(done=done, total=total,
                        eta_seconds=event["elapsed"] / done * (total - done) if done else None,
                        stage=f"{event['policy']}: vertex {event['policy_done']:,}/{event['policy_total']:,}")
            flush()

    result = analyse(options, on_event)
    summary = write_outputs(options, result)
    prog.update(status="completed", done=prog["total"], eta_seconds=0, stage=None, finished=time.time(),
                updated=time.time(), policies=[{"policy": p, "exact_seconds": b["exact_seconds"],
                                                "max_difference": b["max_stable_fraction_difference"],
                                                "several": b["vertices_with_several_intervals"],
                                                "nash_window": b["nash_window"]}
                                               for p, b in summary["policies"].items()])
    rm.write_json(ppath, prog)
    for line in render_summary(summary):
        print(line, flush=True)
    return 0
