"""Durable, recoverable runs for ``hedonic run``.

A benchmark launched by ``hedonic run exp`` executes in a detached tmux session
(``hedonic-<name>``), or, without tmux, in a detached background process. The
terminal only shows a *viewer* that reads the run's ``progress.json``; closing
the terminal or pressing Ctrl-C closes the viewer, never the run. The worker
ignores SIGINT, so even Ctrl-C inside the tmux pane does not stop it; stopping
is explicit (``hedonic run stop NAME``). Every finished detector run is a cached
record, so ``hedonic run resume NAME`` continues where an interrupted run
(reboot, kill) stopped.

    hedonic run list                 all runs with status and progress
    hedonic run status [NAME]        snapshot of one run (default: latest)
    hedonic run attach [NAME]        live viewer again
    hedonic run logs [NAME]          raw tmux pane (tmux attach) or log file
    hedonic run stop NAME            stop a run (explicit)
    hedonic run resume NAME          restart an interrupted/stopped run; finished runs are reused
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from hedonic.experiments.config import expand_path
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import style

TERMINAL = {"completed", "completed_with_failures", "stopped", "failed"}


# --------------------------------------------------------------------------- registry
def registry_dir(cache_dir: str = "~/.cache/hedonic") -> Path:
    path = expand_path(os.environ.get("HEDONIC_RUNS_DIR", str(expand_path(cache_dir) / "runs")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def entry_path(name: str) -> Path:
    return registry_dir() / f"{name}.json"


def load_entry(name: str | None) -> dict:
    entries = all_entries()
    if not entries:
        raise SystemExit("no runs yet — start one with `hedonic run exp`")
    if name is None:
        return entries[-1]
    for e in entries:
        if e["name"] == name:
            return e
    partial = [e for e in entries if name in e["name"]]  # a unique fragment (e.g. the timestamp) is enough
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise SystemExit(f"{name!r} matches several runs ({', '.join(e['name'] for e in partial)}); use the full name")
    raise SystemExit(f"no run named {name!r}; see `hedonic run list`")


def all_entries() -> list[dict]:
    out = []
    for p in sorted(registry_dir().glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return out


def progress_path(entry: dict) -> Path:
    return Path(entry["output_dir"]) / "progress.json"


def read_progress(entry: dict) -> dict:
    try:
        return json.loads(progress_path(entry).read_text())
    except (OSError, ValueError):
        return {"status": "starting", "plan": [], "setup": []}


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False


def is_worker(pid: int | None, name: str | None = None) -> bool:
    """Is ``pid`` really a hedonic worker (and not an unrelated process that reused the number)?"""
    if not alive(pid):
        return False
    try:
        command = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True,
                                 check=False).stdout
    except (OSError, ValueError):
        return True  # cannot check: trust the pid
    return "runmanager" in command and " worker" in command and (name is None or name in command.split())


def tmux_session_exists(session: str) -> bool:
    return bool(session) and bool(shutil.which("tmux")) and subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"], capture_output=True).returncode == 0


def worker_alive(entry: dict, prog: dict) -> bool:
    return any(is_worker(pid, entry["name"]) for pid in {prog.get("pid"), entry.get("pid")} - {None})


def effective_status(entry: dict, prog: dict) -> str:
    status = prog.get("status", "starting")
    if status in TERMINAL:
        return status
    if worker_alive(entry, prog):
        return status
    # A tmux session alone is not proof of life: the pane deliberately stays open after the worker ends.
    # Only a run whose worker has not reported yet may still be starting up.
    if not prog.get("pid") and tmux_session_exists(entry.get("session", "")) \
            and time.time() - entry.get("created", 0) < 60:
        return status
    return "interrupted"


# --------------------------------------------------------------------------- launch
def new_name(prefix: str | None = None) -> str:
    return f"{prefix or 'exp'}-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# Environment forwarded into the detached worker. A tmux session inherits the tmux *server's* environment,
# not this shell's, so anything the run depends on (build SDK, config location, virtualenv) is passed on.
FORWARDED_ENV = ("HEDONIC_", "PATH", "CXX", "CC", "CFLAGS", "CPATH", "LIBRARY_PATH", "SDKROOT", "MACOSX_",
                 "HOME", "NO_COLOR", "XDG_", "TMPDIR", "LANG", "LC_", "VIRTUAL_ENV", "UV_", "OMP_", "TERM")


def launch(cfg: qs.Config, name: str | None = None, *, backend: str = "auto") -> dict:
    """Start the worker detached (tmux if available) and register the run."""
    name = qs.parse_name(name or new_name(cfg.profile))
    previous = entry_path(name)
    if previous.is_file():
        try:
            old = json.loads(previous.read_text())
        except ValueError:
            old = {}
        if old and effective_status(old, read_progress(old)) in ("running", "setup", "starting"):
            raise SystemExit(f"{name} is already running — reopen it with `hedonic run attach {name}` "
                             f"or pick another --name")
    cfg.output_dir = str(expand_path(cfg.output_dir) / name) if not cfg.output_dir.endswith(name) else cfg.output_dir
    cfg.output_dir = str(expand_path(cfg.output_dir).resolve())  # absolute: `hedonic run list` works from any directory
    cfg.cache_dir = str(expand_path(cfg.cache_dir).resolve())
    if cfg.network_root:
        cfg.network_root = str(expand_path(cfg.network_root).resolve())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "progress.json").unlink(missing_ok=True)  # a resumed run must not look finished
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "config": cfg.__dict__, "output_dir": cfg.output_dir, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(cfg.output_dir) / "worker.log"), "command": cfg.command()}
    return _spawn(entry)


def _spawn(entry: dict) -> dict:
    """Register ``entry`` and start its worker detached (tmux session, else a background process)."""
    name = entry["name"]
    write_json(entry_path(name), entry)
    worker = [sys.executable, "-m", "hedonic.experiments.overlapping.runmanager", "worker", name]
    env = {k: v for k, v in os.environ.items() if k.startswith(FORWARDED_ENV)}
    if entry["backend"] == "tmux":
        if tmux_session_exists(entry["session"]):  # the idle pane of an earlier, finished attempt of this run
            subprocess.run(["tmux", "kill-session", "-t", f"={entry['session']}"], capture_output=True, check=False)
        inner = (f"cd {shlex.quote(os.getcwd())} && env {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
                 f"{' '.join(map(shlex.quote, worker))} 2>&1 | tee -a {shlex.quote(entry['log'])}; "
                 "echo; echo '[hedonic] run finished — this pane stays open for inspection; type exit to close'; "
                 "exec $SHELL")
        started = subprocess.run(["tmux", "new-session", "-d", "-s", entry["session"], "-x", "220", "-y", "60", inner],
                                 capture_output=True, text=True, check=False)
        if started.returncode != 0:
            entry_path(name).unlink(missing_ok=True)
            raise SystemExit(f"could not start tmux session {entry['session']}: {started.stderr.strip() or 'tmux failed'}")
    else:
        with open(entry["log"], "ab") as log:
            proc = subprocess.Popen(worker, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, cwd=os.getcwd(), env={**os.environ, **env})
        entry["pid"] = proc.pid
        write_json(entry_path(name), entry)
    return entry


def launch_spectrum(argv: list[str], name: str | None, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the SNAP ground-truth robustness spectrum (``hedonic-exp overlapping-gt-spectrum ARGV``) durably."""
    name = qs.parse_name(name or new_name("spectrum"))
    previous = entry_path(name)
    if previous.is_file():
        try:
            old = json.loads(previous.read_text())
        except ValueError:
            old = {}
        if old and effective_status(old, read_progress(old)) in ("running", "setup", "starting"):
            raise SystemExit(f"{name} is already running — reopen it with `hedonic run attach {name}` "
                             f"or pick another --name")
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    argv = [a for a in argv if a not in ("--resume",)]
    argv = [*argv, "--resume"]
    if "--output-dir" in argv:
        argv[argv.index("--output-dir") + 1] = output
    else:
        argv += ["--output-dir", output]
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "spectrum", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic-exp overlapping-gt-spectrum " + " ".join(map(shlex.quote, argv))}
    return _spawn(entry)


# --------------------------------------------------------------------------- worker
def worker(name: str) -> int:
    """Executes the plan; writes progress.json after every step. Ignores Ctrl-C by design.

    An unexpected error is recorded as status ``failed`` (with its reason) instead of leaving the run looking
    alive; ``hedonic run resume`` then continues from the finished records.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    entry = json.loads(entry_path(name).read_text())
    prog: dict = {}
    try:
        return _work(name, entry, prog)
    except BaseException as exc:  # noqa: BLE001 - includes SIGTERM-driven SystemExit
        import traceback

        traceback.print_exc()
        for item in prog.get("plan", []):
            if item.get("status") == "running":
                item["status"] = "pending"
        prog.update(status="failed", reason=f"{type(exc).__name__}: {exc}", finished=time.time(), updated=time.time())
        try:
            write_json(progress_path(entry), prog)
        except OSError:
            pass
        print(style(f"\n  run failed: {type(exc).__name__}: {exc}\n  continue with: hedonic run resume {name}", "red"),
              flush=True)
        return 1


def _work_spectrum(name: str, entry: dict, prog: dict) -> int:
    """Run the ground-truth robustness spectrum, mirroring its events into progress.json."""
    from hedonic.experiments.overlapping import gt_spectrum as gs

    ppath = progress_path(entry)
    options = gs.resolve_options(gs.build_parser().parse_args(entry["argv"]))
    prog.update({"name": name, "kind": "spectrum", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": "SNAP ground-truth robustness spectrum", "command": entry["command"],
                 "environment": qs.environment(), "setup": [], "plan": []})
    write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · SNAP ground-truth robustness spectrum")
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    index: dict[tuple, int] = {}
    last_write = [0.0]

    def flush(force: bool = False) -> None:
        if force or time.time() - last_write[0] > 0.5:
            prog["updated"] = last_write[0] = time.time()
            write_json(ppath, prog)

    def on_event(event: dict) -> None:
        kind = event["type"]
        if kind == "plan":
            prog["plan"] = [{**item, "status": "pending", "network": item["dataset"], "label": item["job"]}
                            for item in event["conditions"]]
            index.update({(i["job"], i["policy"], i["gamma"], i["seed"]): n for n, i in enumerate(prog["plan"])})
            prog["jobs"] = event["jobs"]
            prog["describe"] = gs.describe(options, len(event["jobs"]))
            prog["status"] = "running"
            flush(True)
        elif kind in ("job_start", "audit_start"):
            prog["setup"] = [f"{'auditing' if kind == 'audit_start' else 'loading'} {event['job']}"]
            prog["stage"] = prog["setup"][0]
            flush(True)
        elif kind == "job_failed":
            for item in prog["plan"]:
                if item["job"] == event["job"].replace("/", "-") and item["status"] == "pending":
                    item.update(status="load_failed", reason=event["error"])
            flush(True)
        elif kind == "condition_start":
            n = index[(event["job"], event["policy"], event["gamma"], event["seed"])]
            prog["plan"][n].update(status="running", started=time.time())
            prog["current"], prog["stage"] = n, "detector"
            flush(True)
        elif kind == "condition":
            n = index[(event["job"], event["policy"], event["gamma"], event["seed"])]
            prog["plan"][n].update(status=event["status"] or "not_recorded", seconds=event["seconds"], f1=event.get("f1"))
            flush()
            print(f"  [{event['index']}/{len(prog['plan'])}] {event['job']} · {event['policy']} · γ={event['gamma']:g} · "
                  f"seed {event['seed']}: {event['status']}", flush=True)

    code = gs.run_study(options, progress=on_event)
    coverage = json.loads((Path(entry["output_dir"]) / "coverage_report.json").read_text()) \
        if (Path(entry["output_dir"]) / "coverage_report.json").is_file() else {}
    if code != 0 or not coverage:
        prog.update(status="failed", reason="no eligible graph/cover pair or the study did not finish",
                    finished=time.time(), updated=time.time())
    else:
        prog.update(status="completed" if coverage.get("complete") else "completed_with_failures",
                    coverage=coverage, finished=time.time(), updated=time.time(), current=None)
    write_json(ppath, prog)
    print(f"\n  records {entry['output_dir']}")
    return code


def _work(name: str, entry: dict, prog: dict) -> int:
    if entry.get("kind") == "spectrum":
        return _work_spectrum(name, entry, prog)
    cfg = qs.Config(**entry["config"])
    ppath = progress_path(entry)
    prog.update({"name": name, "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": qs.describe(cfg), "command": cfg.command(),
                 "environment": qs.environment(), "setup": [], "plan": []})
    write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + qs.describe(cfg))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)

    def say(text: str) -> None:
        print("  setup " + text, flush=True)
        prog["setup"].append(text)
        prog["updated"] = time.time()
        write_json(ppath, prog)

    binary, unavailable = qs.prepare(cfg, say)
    plan = qs.build_plan(cfg, unavailable)
    prog["plan"] = [{"network": net, "seed": seed, "method": m.key, "label": m.label, "status": "pending",
                     "expected": qs.expected_seconds(m, cfg, net)} for net, seed, m in plan]
    prog["unavailable"] = unavailable
    prog["status"] = "running"
    write_json(ppath, prog)
    records = []
    for i, (net, seed, m) in enumerate(plan):
        item = prog["plan"][i]
        item.update(status="running", started=time.time())
        prog["current"], prog["updated"] = i, time.time()
        write_json(ppath, prog)
        tag = f"[{i + 1}/{len(plan)}] {qs.NETWORKS[net][0]} · seed {seed} · {m.label}"
        print(f"  … {tag}", flush=True)
        rec = qs.run_one(cfg, net, seed, m, binary)
        records.append(rec)
        metrics = rec.get("metrics") or {}
        item.update(status=rec.get("status"), seconds=time.time() - item["started"],
                    detection_seconds=rec.get("detection_seconds"), reason=rec.get("reason"),
                    metrics={k: metrics.get(k) for k, _ in qs.COLUMNS})
        prog["updated"] = time.time()
        write_json(ppath, prog)
        print(qs.result_line(tag, rec), flush=True)
    summary = qs.finish(cfg, records, unavailable)
    prog.update(status="completed" if all(r.get("status") == "completed" for r in records) else "completed_with_failures",
                finished=time.time(), updated=time.time(), current=None)
    write_json(ppath, prog)
    print(qs.render(summary, cfg))
    print(f"\n  records {cfg.output_dir}\n  summary {Path(cfg.output_dir) / 'summary.json'}")
    return 0


# --------------------------------------------------------------------------- viewer
def _records_from_progress(prog: dict) -> list[dict]:
    recs = [{"network": p["network"], "method_key": p["method"], "seed": p["seed"], "status": p["status"],
             "reason": p.get("reason"), "metrics": p.get("metrics") or {}, "detection_seconds": p.get("detection_seconds")}
            for p in prog.get("plan", []) if p["status"] not in ("pending", "running")]
    for m, reason in (prog.get("unavailable") or {}).items():
        recs.append({"network": None, "method_key": m, "status": "unavailable", "reason": reason})
    return recs


def eta(prog: dict) -> float:
    plan = prog.get("plan", [])
    done = [p for p in plan if p.get("seconds") is not None]
    factor = sum(p["seconds"] for p in done) / max(1e-9, sum(p["expected"] for p in done)) if done else 1.0
    remaining = sum(p["expected"] for p in plan if p["status"] in ("pending", "running")) * factor
    running = [p for p in plan if p["status"] == "running"]
    if running:
        remaining -= min(running[0]["expected"] * factor, time.time() - running[0]["started"])
    return max(0.0, remaining)


def snapshot_spectrum(entry: dict, prog: dict, frame: str, table: bool) -> str:
    status = effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "completed_with_failures": "yellow",
              "interrupted": "red", "stopped": "yellow", "failed": "red"}.get(status, "dim")
    plan = prog.get("plan", [])
    finished_states = ("pending", "running")
    done = sum(p["status"] not in finished_states for p in plan)
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "SNAP ground-truth robustness spectrum"), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · lucas-igraph {env.get('lucas_igraph')} · "
                           f"Python {env.get('python')}", "dim"))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting") else prog.get("updated", time.time()))
    if plan:
        width = 36
        filled = int(width * done / len(plan))
        tail = f"  {done}/{len(plan)} detector runs · elapsed {qs.fmt_time(end - started)}"
        seconds = [p["seconds"] for p in plan if p.get("seconds") is not None]
        if status == "running" and seconds and done < len(plan):
            tail += f" · ETA ~{qs.fmt_time(statistics_mean(seconds) * (len(plan) - done))}"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    else:
        lines.append(style(f"  discovering the local SNAP cache… elapsed {qs.fmt_time(end - started)}", "dim"))
    if status in ("running", "setup") and prog.get("stage") and prog.get("stage") != "detector":
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    running = [p for p in plan if p["status"] == "running"]
    if running and status == "running":
        p = running[0]
        lines.append(f"  {style(frame, 'cyan')} [{plan.index(p) + 1}/{len(plan)}] {p['dataset']}/{p['cover']} · "
                     f"{p['policy']} · γ={p['gamma']:g} · seed {p['seed']}  "
                     + style(qs.fmt_time(time.time() - p['started']), "bold"))
    if plan and table:
        lines.append(style(f"\n  {'pair':<24} {'runs':>7} {'verified':>9} {'non-eq.':>8} {'timeout/fail':>13}", "bold"))
        for job in prog.get("jobs", []):
            key = job.replace("/", "-")
            items = [p for p in plan if p["job"] == key]
            n = sum(p["status"] not in finished_states for p in items)
            bad = sum(p["status"] in ("timeout", "failed", "load_failed", "invalid_worker_payload", "invalid_cover",
                                      "unsupported_cleanup") for p in items)
            lines.append(f"  {job:<24} {f'{n}/{len(items)}':>7} {sum(p['status'] == 'completed' for p in items):>9} "
                         f"{sum(p['status'] == 'completed_non_equilibrium' for p in items):>8} {bad:>13}")
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); continue with: hedonic run resume {entry['name']}", "red"))
    coverage = prog.get("coverage") or {}
    if coverage:
        lines.append(style(f"\n  {coverage.get('recorded_detector_conditions')}/{coverage.get('expected_detector_conditions')} "
                           f"conditions · {coverage.get('verified_equilibria')} verified equilibria · "
                           f"{coverage.get('non_equilibrium_returns')} non-equilibrium returns · {coverage.get('timeouts')} "
                           f"timeouts · {coverage.get('failures_and_unsupported')} failures · complete={coverage.get('complete')}", "bold"))
    if status in TERMINAL:
        out = Path(entry["output_dir"])
        lines.append(style(f"\n  records  {out}", "dim"))
        for rel in ("plots/gt_spectrum_stable_fraction.png", "coverage_report.json"):
            if (out / rel).is_file():
                lines.append(style(f"  {rel.split('/')[-1]:<34}{out / rel}", "dim"))
    return "\n".join(lines)


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def snapshot(entry: dict, frame: str = "●", table: bool = True) -> str:
    prog = read_progress(entry)
    if entry.get("kind") == "spectrum":
        return snapshot_spectrum(entry, prog, frame, table)
    cfg = qs.Config(**entry["config"])
    status = effective_status(entry, prog)
    plan = prog.get("plan", [])
    done = sum(p["status"] not in ("pending", "running") for p in plan)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "completed_with_failures": "yellow",
              "interrupted": "red", "stopped": "yellow", "failed": "red"}.get(status, "dim")
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", qs.describe(cfg)), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · lucas-igraph {env.get('lucas_igraph')} · "
                           f"Python {env.get('python')}", "dim"))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting") else prog.get("updated", time.time()))
    if plan:
        width = 36
        filled = int(width * done / len(plan))
        bar = style("█" * filled, "green") + style("░" * (width - filled), "dim")
        tail = f"  {done}/{len(plan)} runs · elapsed {qs.fmt_time(end - started)}"
        if status == "running":
            tail += f" · ETA ~{qs.fmt_time(eta(prog))}"
        lines.append(f"  {bar}{tail}")
    else:
        lines.append(style(f"  preparing… elapsed {qs.fmt_time(end - started)}", "dim"))
    for text in prog.get("setup", [])[-3:] if status == "setup" else []:
        lines.append(style(f"  setup {text}", "magenta"))
    current = [p for p in plan if p["status"] == "running"]
    if current and status == "running":
        p = current[0]
        i = plan.index(p) + 1
        lines.append(f"  {style(frame, 'cyan')} [{i}/{len(plan)}] {qs.NETWORKS[p['network']][0]} · seed {p['seed']} · "
                     f"{p['label']}  {style(qs.fmt_time(time.time() - p['started']), 'bold')}"
                     + style(f"  (expected ~{qs.fmt_time(p['expected'])})", "dim"))
    finished = [p for p in plan if p["status"] not in ("pending", "running")]
    if finished:
        lines.append(style("  recent", "dim"))
        for p in finished[-5:]:
            i = plan.index(p) + 1
            tag = f"[{i}/{len(plan)}] {qs.NETWORKS[p['network']][0]} · seed {p['seed']} · {p['label']}"
            lines.append(qs.result_line(tag, {"status": p["status"], "reason": p.get("reason"),
                                              "metrics": p.get("metrics"), "detection_seconds": p.get("detection_seconds")}))
        if table:
            recs = [r for r in _records_from_progress(prog) if r["network"] is not None]
            for m, reason in (prog.get("unavailable") or {}).items():
                recs += [{"network": n, "method_key": m, "status": "unavailable", "reason": reason} for n in cfg.networks]
            lines.append(qs.render(qs.summarise(recs, cfg), cfg))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); continue with: hedonic run resume {entry['name']}", "red"))
    if status in TERMINAL:
        lines.append(style(f"\n  records  {entry['output_dir']}", "dim"))
        if (Path(entry["output_dir"]) / "summary.json").is_file():
            lines.append(style(f"  summary  {Path(entry['output_dir']) / 'summary.json'}", "dim"))
    return "\n".join(lines)


def _frame(text: str, rows: int, cols: int) -> str:
    """Home the cursor and repaint in place (no full clear, so no flicker); each line is one terminal row."""
    return "\033[H" + "".join(tui.fit(line, cols) + "\033[K\n" for line in text.split("\n")) + "\033[J"


def view(entry: dict) -> int:
    """Live viewer. Ctrl-C closes the viewer only."""
    frames, i = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", 0
    live = sys.stdout.isatty()
    footer = (style("\n  Ctrl-C closes this view — the run keeps going.", "dim") + "\n"
              + style(f"  reopen: hedonic run attach {entry['name']} · stop: hedonic run stop {entry['name']}", "dim"))
    if live:
        sys.stdout.write("\033[2J\033[?25l")
    try:
        while True:
            status = effective_status(entry, read_progress(entry))
            final = status in TERMINAL or status == "interrupted"
            text = snapshot(entry, frames[i % len(frames)])
            if live and not final:
                size = shutil.get_terminal_size((100, 30))
                body = text + "\n" + footer
                if body.count("\n") + 2 > size.lines:  # too tall: keep progress visible, show the table at the end
                    body = (snapshot(entry, frames[i % len(frames)], table=False)
                            + style("\n  (results table hidden while the terminal is this short; "
                                    "it is shown when the run ends)", "dim") + "\n" + footer)
                lines = body.split("\n")
                if len(lines) > size.lines - 1:  # never taller than the screen, or every repaint scrolls
                    lines = lines[:size.lines - 2] + [style("  … enlarge the terminal to see more", "dim")]
                sys.stdout.write(_frame("\n".join(lines), size.lines, size.columns))
                sys.stdout.flush()
            if final:
                if live:
                    sys.stdout.write("\033[H\033[J")
                print(text)
                return 0 if status == "completed" else 1
            i += 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        print(style(f"\n  viewer closed; the run continues in the background.\n"
                    f"  reopen: hedonic run attach {entry['name']}   ·   stop: hedonic run stop {entry['name']}", "yellow"))
        return 0
    finally:
        if live:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


def wait_and_print_json(entry: dict) -> int:
    """Block until the run ends, then print its summary rows as JSON on stdout (for scripts)."""
    last = ""
    while True:
        prog = read_progress(entry)
        status = effective_status(entry, prog)
        done = sum(p["status"] not in ("pending", "running") for p in prog.get("plan", []))
        line = f"  {entry['name']}: {status} {done}/{len(prog.get('plan', []))}"
        if line != last and sys.stderr.isatty():
            print("\r\033[K" + line, end="", file=sys.stderr, flush=True)
            last = line
        if status in TERMINAL or status == "interrupted":
            break
        time.sleep(1.0)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    summary = Path(entry["output_dir"]) / ("coverage_report.json" if entry.get("kind") == "spectrum" else "summary.json")
    if not summary.is_file():
        print(f"  run {status}: {prog.get('reason', 'no summary was written')}; see `hedonic run logs {entry['name']}`",
              file=sys.stderr)
        return 1
    payload = json.loads(summary.read_text())
    print(json.dumps(payload if entry.get("kind") == "spectrum" else payload["results"], indent=2, default=str))
    return 0 if status == "completed" else 1


# --------------------------------------------------------------------------- commands
def cmd_list() -> int:
    entries = all_entries()
    if not entries:
        print("no runs yet — start one with `hedonic run exp`")
        return 0
    width = max(22, *(len(e["name"]) for e in entries))
    print(style(f"  {'name':<{width}} {'status':<24} {'progress':>9}  {'started':<16}  output", "bold"))
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "completed_with_failures": "yellow",
              "interrupted": "red", "stopped": "yellow", "failed": "red"}
    for e in entries:
        prog = read_progress(e)
        plan = prog.get("plan", [])
        done = sum(p["status"] not in ("pending", "running") for p in plan)
        started = dt.datetime.fromtimestamp(prog.get("started", e["created"])).strftime("%Y-%m-%d %H:%M")
        status = effective_status(e, prog)
        shown = style(f"{status:<24}", colour.get(status, "dim"))
        print(f"  {e['name']:<{width}} {shown} {f'{done}/{len(plan)}':>9}  {started:<16}  {e['output_dir']}")
    resumable = [e["name"] for e in entries if effective_status(e, read_progress(e)) in ("interrupted", "failed")]
    if resumable:  # a run you stopped yourself is not nagged about
        print(style(f"\n  not finished, not stopped by you: hedonic run resume {resumable[-1]}", "dim"))
    return 0


def cmd_stop(name: str) -> int:
    entry = load_entry(name)
    prog = read_progress(entry)
    finished = effective_status(entry, prog)
    if finished in TERMINAL or finished == "interrupted":  # nothing to stop; only an idle tmux pane may remain
        if tmux_session_exists(entry.get("session", "")):
            subprocess.run(["tmux", "kill-session", "-t", f"={entry['session']}"], check=False)
            print(f"  {entry['name']} already ended ({finished}); closed its idle tmux pane")
        else:
            print(f"  {entry['name']} is not running ({finished})")
        return 0
    if entry.get("session") and tmux_session_exists(entry["session"]):
        subprocess.run(["tmux", "kill-session", "-t", f"={entry['session']}"], check=False)
    for pid in {prog.get("pid"), entry.get("pid")} - {None}:
        if is_worker(pid, entry["name"]):  # never signal a process we cannot prove is this run's worker
            try:
                group = os.getpgid(int(pid))
                if group != os.getpgid(0):  # never signal our own terminal's group
                    os.killpg(group, signal.SIGTERM)
                else:
                    os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
    if prog.get("status") not in TERMINAL:
        prog.update(status="stopped", updated=time.time(), finished=time.time())
        for p in prog.get("plan", []):
            if p["status"] == "running":
                p["status"] = "pending"
        write_json(progress_path(entry), prog)
    print(style(f"  stopped {entry['name']}", "yellow") + f" — resume with: hedonic run resume {entry['name']}")
    return 0


def cmd_resume(name: str | None, *, attach: bool = True) -> int:
    entry = load_entry(name)
    status = effective_status(entry, read_progress(entry))
    if status in ("running", "setup", "starting"):
        print(f"  {entry['name']} is still running")
        return view(entry) if attach else 0
    if entry.get("kind") == "spectrum":
        entry = launch_spectrum(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    else:
        cfg = qs.Config(**entry["config"])
        entry = launch(cfg, entry["name"], backend=entry.get("backend", "auto"))
    print(style(f"  resumed {entry['name']}", "green") + " — finished runs are reused from their cached records")
    if not attach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return view(entry)


def cmd_logs(name: str | None) -> int:
    entry = load_entry(name)
    if entry.get("session") and tmux_session_exists(entry["session"]) and sys.stdout.isatty():
        print(style("  opening the run's tmux pane — detach with Ctrl-b d (the run is unaffected)", "dim"))
        return subprocess.run(["tmux", "attach", "-t", f"={entry['session']}"]).returncode
    log = Path(entry["log"])
    print(log.read_text(errors="replace") if log.is_file() else "no log yet")
    return 0


VERBS = {"list": "runs and their progress", "status": "snapshot of a run (default: latest)",
         "attach": "live view of a running benchmark (Ctrl-C closes only the view)",
         "logs": "raw output of a run (tmux pane, or the log file)", "stop": "stop a run (explicit; never automatic)",
         "resume": "continue an interrupted, stopped or failed run; finished runs are reused"}


def dispatch(argv: list[str]) -> int:
    """``hedonic run <verb> ...`` (verbs other than ``exp``)."""
    import argparse

    verb = argv[0]
    if verb not in VERBS:
        raise SystemExit(f"unknown command `hedonic run {verb}`; try `hedonic --help`")
    parser = argparse.ArgumentParser(prog=f"hedonic run {verb}", description=VERBS[verb])
    if verb != "list":
        parser.add_argument("name", nargs=None if verb == "stop" else "?",
                            help="run name (see `hedonic run list`)" + ("" if verb == "stop" else "; default: the latest run"))
    if verb == "status":
        parser.add_argument("--json", action="store_true", help="print progress.json instead of the formatted snapshot")
    if verb == "resume":
        parser.add_argument("--detach", action="store_true", help="restart in the background without opening the view")
    args = parser.parse_args(argv[1:])
    name = getattr(args, "name", None)
    if verb == "list":
        return cmd_list()
    if verb == "status":
        entry = load_entry(name)
        if args.json:
            prog = read_progress(entry)
            prog["status"] = effective_status(entry, prog)
            print(json.dumps(prog, indent=2, default=str))
        else:
            print(snapshot(entry))
        return 0
    if verb == "attach":
        return view(load_entry(name))
    if verb == "logs":
        return cmd_logs(name)
    if verb == "stop":
        return cmd_stop(name)
    return cmd_resume(name, attach=not args.detach)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(dispatch(sys.argv[1:]))
