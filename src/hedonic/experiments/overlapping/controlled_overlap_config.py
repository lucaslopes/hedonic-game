"""The controlled-overlap study from a TOML table: ``hedonic run controlled`` (durable) and its foreground form.

``hedonic run controlled --config FILE.toml`` runs the study as a durable run: a registry entry, a ``progress.json``
with the instance count and an ETA, a live viewer, and stop and resume. ``hedonic-exp overlapping-controlled --config
FILE.toml`` runs the same study in the foreground; it is deprecated in favour of the durable run.

The table ``[overlapping_controlled]`` takes the option names of ``overlapping-controlled`` (hyphens or underscores).
Its values are passed as command-line arguments placed before the explicit flags, so a flag given on the command line
overrides the file (argparse keeps the last value). ``--config`` takes a file or the name of a shipped config
(``overlap-heatmap``; ``resources.config_path``). The study module itself is not changed here, so the source identity
recorded in its output is the module's own.

Cores. ``workers`` (a table key or ``--workers``; default: the performance cores) runs the instances side by side. The
study module generates and runs its instances one at a time, but each instance is a deterministic function of its
parameters, so this wrapper computes them in forked workers with the module's own ``generate_controlled_cover`` and
``run_ablations`` (one worker per (graph seed, mu) group, which shares one LFR base) and then lets the module's
``main`` assemble its output from those results: the same JSON, CSV, summaries and identity as a one-at-a-time run, in
the same order. Only a cell's ``elapsed_seconds`` differs (it is the wall-clock with that many cells sharing the
machine); the worker count is written next to the output as ``<output>.execution.json``.

A durable run writes into its run folder: ``--output-dir`` when given, else the folder of the table's ``output`` path,
else ``<user-config output_dir or ~/hedonic-runs>/<name>``. The run name defaults to the config file's name. Stop and
resume restart the study from its first instance, because the study does not checkpoint its instances.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import tomllib
import types
from pathlib import Path
from typing import Any, Callable

from hedonic.experiments.config import RUNS_ROOT, expand_path
from hedonic.experiments.overlapping import controlled_overlap, parallel
from hedonic.experiments.resources import config_path as resolve_config

JSON_NAME = "controlled_overlap.json"
# controlled_overlap.main prints one such line per finished instance; a durable run reads them for its progress.
PROGRESS_LINE = re.compile(r"completed instance (\d+)/(\d+): (.*)")


def config_arguments(path: str) -> list[str]:
    """Command-line arguments equivalent to the file's ``[overlapping_controlled]`` table (a file or a shipped name)."""
    file = resolve_config(path)
    if not file.is_file():
        raise SystemExit(f"no such config: {path} (neither a file nor a shipped config)")
    data = tomllib.loads(file.read_text())
    table = data.get("overlapping_controlled", data)
    arguments: list[str] = []
    for key, value in table.items():
        if isinstance(value, bool):
            raise SystemExit(f"{path}: {key} is a boolean; the study has no boolean options in a config file")
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        if key == "output":
            value = str(Path(str(value)).expanduser())
        arguments += ["--" + key.replace("_", "-"), str(value)]
    return arguments


def split_workers(arguments: list[str]) -> tuple[list[str], int | None]:
    """The arguments without ``--workers N`` (the study module does not know it), and N (the last one given)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--workers", type=int)
    known, rest = pre.parse_known_args(arguments)
    if known.workers is not None and known.workers < 1:
        raise SystemExit("--workers must be at least 1")
    return rest, known.workers


class _WithoutProgress(io.TextIOBase):
    """Forwards the study's output except its ``completed instance`` lines, which the wrapper has already printed."""

    def __init__(self, real: Any) -> None:
        super().__init__()
        self.real = real
        self.pending = ""

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not PROGRESS_LINE.match(line):
                self.real.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        self.real.flush()


def _grid(arguments: list[str]) -> tuple[Any, dict[str, list]]:
    args = controlled_overlap.build_parser().parse_args(arguments)
    numbers = controlled_overlap._csv_numbers
    return args, {"mus": numbers(args.mus, float), "fractions": numbers(args.overlap_fractions, float),
                  "memberships": numbers(args.overlap_memberships, int),
                  "probabilities": numbers(args.secondary_edge_probabilities, float),
                  "seeds": numbers(args.graph_seeds, int), "phases": numbers(args.phases, str),
                  "caps": numbers(args.max_memberships, str), "starts": numbers(args.starts, str),
                  "multipliers": numbers(args.resolution_multipliers, float)}


def _instance_key(seed: int, mu: float, fraction: float, memberships: int, probability: float) -> tuple:
    return (int(seed), float(mu), float(fraction), int(memberships), float(probability))


def compute_instances(arguments: list[str], workers: int,
                      emit: Callable[[str], Any] = print) -> dict[tuple, types.SimpleNamespace]:
    """Every instance of the grid (metadata and rows), computed by the module's own functions in forked workers.

    The calls are those of ``controlled_overlap.main``'s loops, with the same arguments. One task is one (graph seed,
    mu) group, so its instances share the LFR base the module caches. ``emit`` gets one ``completed instance`` line per
    instance as its group finishes, in the format the durable viewer reads."""
    args, grid = _grid(arguments)
    groups = [(seed, mu) for seed in grid["seeds"] for mu in grid["mus"]]
    total = len(groups) * len(grid["fractions"]) * len(grid["memberships"]) * len(grid["probabilities"])

    def task(group: tuple[int, float]) -> list[tuple]:
        seed, mu = group
        out = []
        with contextlib.redirect_stdout(io.StringIO()):  # the study prints a line per cell; eight at once would interleave
            for fraction in grid["fractions"]:
                for memberships in grid["memberships"]:
                    for probability in grid["probabilities"]:
                        instance = controlled_overlap.generate_controlled_cover(
                            n=args.n, mu=mu, overlap_fraction=fraction, memberships_per_overlapping_vertex=memberships,
                            secondary_edge_probability=probability, seed=seed, average_degree=args.average_degree,
                            min_community=args.min_community, max_community=args.max_community,
                            lfr_max_iters=args.lfr_max_iters)
                        rows = controlled_overlap.run_ablations(
                            instance, phases=grid["phases"], cap_specs=grid["caps"], starts=grid["starts"],
                            resolution_multipliers=grid["multipliers"], seed=seed, compute_omega=args.omega,
                            timeout_seconds=args.timeout_per_run)
                        out.append((_instance_key(seed, mu, fraction, memberships, probability), instance.metadata, rows))
        return out

    from hedonic.experiments.overlapping import parallel

    done: dict[tuple, types.SimpleNamespace] = {}
    for _, items in parallel.run_forked(task, groups, workers):
        for key, metadata, rows in items:
            done[key] = types.SimpleNamespace(metadata=metadata, rows=rows)
            seed, mu, fraction, memberships, probability = key
            emit(f"completed instance {len(done)}/{total}: seed={seed} mu={mu:g} overlap={fraction:g} "
                 f"memberships={memberships} secondary_p={probability:g}")
    return done


def run_study(arguments: list[str], workers: int | None = None, emit: Callable[[str], Any] = print) -> int:
    """The study of these arguments with its instances computed ``workers`` at a time (``None``: the default).

    With one worker, or a smoke run, this is exactly ``controlled_overlap.main``. Otherwise the instances are computed
    first (:func:`compute_instances`) and ``main`` assembles its output from them: its generator and runner are
    replaced for the call by lookups, so everything else it does (identity, protocol, summaries, files) is unchanged."""
    from hedonic.experiments.overlapping import parallel

    arguments, flag = split_workers(arguments)
    args, grid = _grid(arguments)
    groups = len(grid["seeds"]) * len(grid["mus"])
    count = max(1, min(parallel.default_workers(workers if workers is not None else flag), groups))
    if count == 1 or args.smoke:
        return controlled_overlap.main(arguments)
    done = compute_instances(arguments, count, emit)
    original = controlled_overlap.generate_controlled_cover, controlled_overlap.run_ablations

    def generate(**kw: Any) -> types.SimpleNamespace:
        return done[_instance_key(kw["seed"], kw["mu"], kw["overlap_fraction"],
                                  kw["memberships_per_overlapping_vertex"], kw["secondary_edge_probability"])]

    def run(instance: types.SimpleNamespace, **kw: Any) -> list[dict]:
        return instance.rows

    controlled_overlap.generate_controlled_cover, controlled_overlap.run_ablations = generate, run
    try:
        with contextlib.redirect_stdout(_WithoutProgress(sys.stdout)):
            code = controlled_overlap.main(arguments)
    finally:
        controlled_overlap.generate_controlled_cover, controlled_overlap.run_ablations = original
    if code == 0:
        output = expand_path(args.output)
        output.with_name(output.stem + ".execution.json").write_text(json.dumps(
            {"workers": count, "unit": "(graph seed, mu) group", "instances": len(done),
             "note": "elapsed_seconds of a cell is its wall-clock with this many workers sharing the machine"},
            indent=2) + "\n")
    return code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, rest = pre.parse_known_args(argv)
    if known.config is None:
        return run_study(rest)
    return run_study([*config_arguments(known.config), *rest])


# --------------------------------------------------------------------------- plan
def study_plan(arguments: list[str]) -> dict[str, Any]:
    """How large the study that ``arguments`` describe is, computed without running anything.

    The counts follow the loops of ``controlled_overlap.main`` (one instance per graph seed, mu, overlap fraction,
    membership count and secondary probability) and of ``run_ablations`` (one detector run per phase, distinct cap
    label, start and resolution multiplier).
    """

    def numbers(value: str, cast: Callable[[str], Any]) -> list:
        return controlled_overlap._csv_numbers(value, cast)

    arguments, flag = split_workers(arguments)
    args = controlled_overlap.build_parser().parse_args(arguments)
    instances = (len(numbers(args.graph_seeds, int)) * len(numbers(args.mus, float))
                 * len(numbers(args.overlap_fractions, float)) * len(numbers(args.overlap_memberships, int))
                 * len(numbers(args.secondary_edge_probabilities, float)))
    caps = {spec.strip().lower() for spec in numbers(args.max_memberships, str)}
    phases = controlled_overlap.check_phases(numbers(args.phases, str))  # a 1.x multiphase grid fails here, early
    per_instance = (len(phases) * len(caps) * len(numbers(args.starts, str))
                    * len(numbers(args.resolution_multipliers, float)))
    groups = len(numbers(args.graph_seeds, int)) * len(numbers(args.mus, float))
    return {"instances": instances, "runs_per_instance": per_instance, "runs": instances * per_instance,
            "timeout_per_run": float(args.timeout_per_run), "groups": groups,
            "workers": max(1, min(parallel.default_workers(flag), groups))}


def _front_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic run controlled",
        description=("LFR-derived controlled-overlap study (not canonical overlapping LFR) from a TOML table, as a "
                     "durable run with a live viewer, ETA, stop and resume"),
    )
    parser.add_argument("--config", required=True,
                        help="TOML file with an [overlapping_controlled] table, or the name of a shipped config "
                             "(overlap-heatmap)")
    parser.add_argument("--name", help="run name (default: the config file's name)")
    parser.add_argument("--output-dir", help="run folder (default: the folder of the table's output, else "
                                             "<user-config output_dir or ~/hedonic-runs>/NAME)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the paths; start nothing and write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the study's JSON if it already exists (a start is refused otherwise)")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    return parser


def resolve(argv: list[str]) -> dict[str, Any]:
    """What ``hedonic run controlled ARGV`` runs: config, run name, run folder, study JSON and plan. Writes nothing."""
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import userconfig

    args = _front_parser().parse_args(argv)
    config = resolve_config(args.config).resolve()
    if not config.is_file():
        raise ValueError(f"no such config: {args.config} (neither a file nor a shipped config)")
    arguments = config_arguments(str(config))
    table_output = arguments[arguments.index("--output") + 1] if "--output" in arguments else None
    name = qs.parse_name(args.name or config.stem)
    if args.output_dir:
        folder = expand_path(args.output_dir).resolve()
    elif table_output:
        folder = expand_path(table_output).resolve().parent
    else:
        folder = (expand_path(userconfig.defaults().get("output_dir") or RUNS_ROOT) / name).resolve()
    json_path = folder / (Path(table_output).name if table_output else JSON_NAME)
    return {"config": str(config), "name": name, "folder": str(folder), "json": str(json_path),
            "plan": study_plan([*arguments, "--output", str(json_path)])}


def _describe(run: dict[str, Any]) -> str:
    plan = run["plan"]
    return (f"controlled-overlap grid · {plan['instances']:,} instances × {plan['runs_per_instance']} detector run(s) "
            f"· at most {plan['timeout_per_run']:g} s per run")


def _plan_lines(run: dict[str, Any]) -> list[str]:
    plan = run["plan"]
    worst_case_hours = plan["runs"] * plan["timeout_per_run"] / 3600
    lines = [
        f"hedonic run controlled · {run['name']} · LFR-derived controlled overlap (not canonical overlapping LFR)",
        f"  config   {run['config']}",
        f"  grid     {plan['instances']:,} instances × {plan['runs_per_instance']} detector run(s) = {plan['runs']:,} runs",
        f"  workers  {plan['workers']} of {plan['groups']} (graph seed, mu) groups side by side",
        f"  timeout  {plan['timeout_per_run']:g} s per run; worst case {worst_case_hours / plan['workers']:,.1f} h "
        f"with {plan['workers']} workers if every run reached its timeout",
        f"  output   {run['json']} (and the sibling .csv)",
        f"  folder   {run['folder']} (progress.json, worker.log)",
    ]
    if Path(run["json"]).exists():
        lines.append("  exists   the output is already there: a start needs --overwrite (or another --output-dir)")
    return lines


# --------------------------------------------------------------------------- durable run
class ProgressTap(io.TextIOBase):
    """Passes the study's output through unchanged and hands every complete line to ``on_line``."""

    def __init__(self, real: Any, on_line: Callable[[str], None]) -> None:
        super().__init__()
        self.real = real
        self.on_line = on_line
        self.pending = ""

    def write(self, text: str) -> int:
        self.real.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.on_line(line)
        return len(text)

    def flush(self) -> None:
        self.real.flush()


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run controlled``: a durable run with a live viewer; ``--dry-run`` only prints the plan."""
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _front_parser().parse_args(argv)
        run = resolve(argv)
    except SystemExit as exc:  # argparse printed the help or a usage error, or the config file was refused
        if isinstance(exc.code, str):
            print(f"hedonic run controlled: {exc.code}", file=sys.stderr)
        return exc.code if isinstance(exc.code, int) else 2
    except (ValueError, argparse.ArgumentTypeError) as exc:
        print(f"hedonic run controlled: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("\n".join(_plan_lines(run)))
        print(style("  dry run: nothing started, nothing written", "dim"))
        return 0
    if Path(run["json"]).exists() and not args.overwrite:
        print(f"hedonic run controlled: {run['json']} already exists; pass --overwrite to replace it "
              "or --output-dir for another folder", file=sys.stderr)
        return 2
    entry = runmanager.launch_controlled(
        ["--config", run["config"], "--name", run["name"], "--output-dir", run["folder"]], run["name"], run["folder"])
    where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
             else f"background process (log {entry['log']})")
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body of a controlled run: the study through its config wrapper, its progress mirrored into progress.json."""
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    run = resolve(entry["argv"])
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "controlled", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": _describe(run), "command": entry["command"],
                 "environment": qs.environment(), "done": 0, "total": run["plan"]["instances"],
                 "runs_per_instance": run["plan"]["runs_per_instance"], "eta_seconds": None,
                 "stage": "starting the study", "output": run["json"], "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + _describe(run))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)

    def observe(line: str) -> None:
        found = PROGRESS_LINE.match(line)
        if not found:
            return
        done, seen_total = int(found[1]), int(found[2])
        elapsed = time.time() - prog["started"]
        prog.update(status="running", done=done, total=seen_total, last=found[3],
                    eta_seconds=elapsed / done * (seen_total - done) if done else None,
                    stage=f"instance {done:,}/{seen_total:,}", updated=time.time())
        rm.write_json(ppath, prog)

    prog.update(status="running", stage="first instance", updated=time.time())
    rm.write_json(ppath, prog)
    with contextlib.redirect_stdout(ProgressTap(sys.stdout, observe)):
        code = main(["--config", run["config"], "--output", run["json"]])
    if code != 0:
        raise RuntimeError(f"the controlled-overlap study returned {code}")
    prog.update(status="completed", eta_seconds=0, stage=None, finished=time.time(), updated=time.time())
    rm.write_json(ppath, prog)
    print(style(f"  wrote {run['json']}", "green"), flush=True)
    return 0


def snapshot(entry: dict[str, Any], prog: dict[str, Any], frame: str) -> str:
    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    status = rm.effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "interrupted": "red",
              "stopped": "yellow", "failed": "red"}.get(status, "dim")
    done, total = int(prog.get("done", 0)), int(prog.get("total", 0))
    per_instance = int(prog.get("runs_per_instance", 1))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting")
                                   else prog.get("updated", time.time()))
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "controlled-overlap grid"), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = (f"  {done:,}/{total:,} instances ({done * per_instance:,}/{total * per_instance:,} detector runs) "
                f"· elapsed {qs.fmt_time(end - started)}")
        if status in ("running", "setup") and prog.get("eta_seconds") is not None:
            tail += f" · ETA ~{qs.fmt_time(prog['eta_seconds'])}"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    else:
        lines.append(style(f"  preparing… elapsed {qs.fmt_time(end - started)}", "dim"))
    if status in ("running", "setup") and prog.get("stage"):
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    if status in ("running", "setup") and prog.get("last"):
        lines.append(style(f"  last finished: {prog['last']}", "dim"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: "
                           f"hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); restart with: hedonic run resume {entry['name']}",
                           "red"))
    elif status == "stopped":
        lines.append(style(f"\n  stopped by you; resume starts the study again from its first instance: "
                           f"hedonic run resume {entry['name']}", "yellow"))
    if status == "completed":
        lines.append(style(f"\n  records  {prog.get('output', entry['output_dir'])}", "dim"))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
