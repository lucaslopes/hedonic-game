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
    hedonic run check [NAME]         morning check: replay a sample of the run's results
    hedonic run queue FILE|NAME      run a list of experiments one after another (see runqueue.py)

Nine kinds of run share this machinery: benchmark runs (``hedonic run exp`` /
``paper`` / ``paper-hoc``), the ground-truth spectrum (``hedonic run spectrum``),
the disjoint SBM benchmark (``hedonic run disjoint``), the exact post-hoc
stability of a ground-truth cover (``hedonic run posthoc``), the controlled-overlap
study from a TOML table (``hedonic run controlled``), the start-quality curve
(``hedonic run startcurve``), the robustness audit of saved benchmark covers
(``hedonic run audit``), the Hedonic and baseline calibrations (``hedonic run calibrate``,
``calibrate-baselines``) and queues of those runs
(``hedonic run queue``). On macOS every worker runs under ``caffeinate -i``
so the machine does not idle-sleep overnight.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from hedonic.experiments.config import expand_path
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import style

TERMINAL = {"completed", "completed_with_failures", "stopped", "failed"}
LIVE = ("running", "setup", "starting")  # a run that has a worker and is still going
BLOCK_WIDTH = 80  # the progress lines are read in a terminal this wide; longer lines are cut, never wrapped


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
    if not name.strip():
        raise SystemExit("an empty run name; give a name or a unique part of one (see `hedonic run list`), "
                         "or leave it out for the latest run")
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
    if entry.get("backend") == "foreground":  # the user's own `hedonic run exp --foreground` process
        return any(_is_foreground(pid) for pid in {prog.get("pid"), entry.get("pid")} - {None})
    return any(is_worker(pid, entry["name"]) for pid in {prog.get("pid"), entry.get("pid")} - {None})


def _is_foreground(pid: int | None) -> bool:
    if not alive(pid):
        return False
    try:
        command = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True,
                                 check=False).stdout
    except (OSError, ValueError):
        return True
    return "hedonic" in command


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
    entry = _benchmark_entry(cfg, name, backend)
    return _spawn(entry)


def run_foreground(cfg: qs.Config, name: str | None = None) -> int:
    """``--foreground``: the same worker, in this terminal, registered like a detached run (``hedonic run status NAME``,
    ``list``, ``resume`` know it). Ctrl-C stops it (recorded as ``stopped``); finished records are kept either way."""
    entry = _benchmark_entry(cfg, name, "foreground")
    entry["pid"] = os.getpid()
    write_json(entry_path(entry["name"]), entry)
    prog: dict = {}
    guard = start_memory_guard(entry["name"], entry, prog)
    try:
        return _work(entry["name"], entry, prog)
    except KeyboardInterrupt:
        prog.update(status="stopped", finished=time.time(), updated=time.time())
        write_json(progress_path(entry), prog)
        print(style(f"\n  stopped · continue with: hedonic run resume {entry['name']}", "yellow"), flush=True)
        return 130
    except BaseException as exc:  # noqa: BLE001 - recorded, then raised as before
        prog.update(status="failed", reason=f"{type(exc).__name__}: {exc}", finished=time.time(), updated=time.time())
        write_json(progress_path(entry), prog)
        raise
    finally:
        if guard is not None:
            guard.set()


def _benchmark_entry(cfg: qs.Config, name: str | None, backend: str) -> dict:
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
    return {"name": name, "config": cfg.__dict__, "output_dir": cfg.output_dir, "created": time.time(),
            "session": f"hedonic-{name}" if use_tmux else "",
            "backend": "foreground" if backend == "foreground" else "tmux" if use_tmux else "process",
            "log": str(Path(cfg.output_dir) / "worker.log"), "command": cfg.command()}


# One numerical thread per process: the disjoint worker runs one graph per CPU.
SINGLE_THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def _spawn(entry: dict) -> dict:
    """Register ``entry`` and start its worker detached (tmux session, else a background process).

    On macOS the worker runs under ``caffeinate -i`` (no idle sleep while it lives).
    """
    name = entry["name"]
    write_json(entry_path(name), entry)
    worker = [sys.executable, "-m", "hedonic.experiments.overlapping.runmanager", "worker", name]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        worker = ["caffeinate", "-i", *worker]
    env = {k: v for k, v in os.environ.items() if k.startswith(FORWARDED_ENV)}
    if entry.get("kind") == "disjoint":
        env.update(SINGLE_THREAD_ENV)
    if entry["backend"] == "tmux":
        if tmux_session_exists(entry["session"]):  # the idle pane of an earlier, finished attempt of this run
            subprocess.run(["tmux", "kill-session", "-t", f"={entry['session']}"], capture_output=True, check=False)
        # A tmux session starts from the tmux *server's* environment, which may hold HEDONIC_* settings of an
        # earlier shell (another registry, another output folder): unset those this shell does not have, and pin
        # the registry to the one this entry was written to, so the worker always finds its entry.
        env["HEDONIC_RUNS_DIR"] = str(registry_dir())
        stale = [k for k in _tmux_global_names() if k.startswith("HEDONIC_") and k not in env]
        unset = "".join(f"-u {shlex.quote(k)} " for k in stale)
        inner = (f"cd {shlex.quote(os.getcwd())} && env {unset}{' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
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


def _tmux_global_names() -> list[str]:
    """Names in the tmux server's global environment (empty when tmux has no server or cannot say)."""
    try:
        out = subprocess.run(["tmux", "show-environment", "-g"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    return [line.split("=", 1)[0] for line in out.splitlines() if line and not line.startswith("-")]


def _refuse_if_running(name: str) -> None:
    previous = entry_path(name)
    if previous.is_file():
        try:
            old = json.loads(previous.read_text())
        except ValueError:
            old = {}
        if old and effective_status(old, read_progress(old)) in ("running", "setup", "starting"):
            raise SystemExit(f"{name} is already running — reopen it with `hedonic run attach {name}` "
                             f"or pick another --name")


def launch_spectrum(argv: list[str], name: str | None, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the SNAP ground-truth robustness spectrum (``hedonic-exp overlapping-gt-spectrum ARGV``) durably."""
    name = qs.parse_name(name or new_name("spectrum"))
    _refuse_if_running(name)
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


def launch_disjoint(settings: dict, name: str | None, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the disjoint SBM benchmark (``hedonic run disjoint``) durably; the store is ``output_dir``.

    A run first started with ``--foreground`` resumes detached."""
    return _spawn(_disjoint_entry(settings, name, output_dir, "auto" if backend == "foreground" else backend))


def run_disjoint_foreground(settings: dict, name: str | None, output_dir: str) -> int:
    """``hedonic run disjoint --foreground``: the same worker in this terminal, registered like a detached run, so
    ``hedonic run status|list|check NAME`` know it. Ctrl-C stops it (recorded as ``stopped``); the store keeps its shards."""
    from hedonic.experiments.disjoint import front

    entry = _disjoint_entry(settings, name, output_dir, "foreground")
    entry["pid"] = os.getpid()
    write_json(entry_path(entry["name"]), entry)
    prog: dict = {}
    progress = Path(entry["output_dir"]) / "progress.json"
    try:
        return front.work(entry["name"], entry, prog, progress_file=progress)
    except KeyboardInterrupt:
        prog.update(status="stopped", finished=time.time(), updated=time.time())
        write_json(progress, prog)
        print(style(f"\n  stopped · continue with: hedonic run resume {entry['name']}", "yellow"), flush=True)
        return 130
    except BaseException as exc:  # noqa: BLE001 - recorded, then raised as before
        prog.update(status="failed", reason=f"{type(exc).__name__}: {exc}", finished=time.time(), updated=time.time())
        write_json(progress, prog)
        raise


def _disjoint_entry(settings: dict, name: str | None, output_dir: str, backend: str) -> dict:
    from hedonic.experiments.disjoint import front

    name = qs.parse_name(name or new_name("disjoint"))
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
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    return {"name": name, "kind": "disjoint", "settings": settings, "argv": front.sweep_argv(settings, Path(output)),
            "config": {}, "output_dir": output, "created": time.time(),
            "session": f"hedonic-{name}" if use_tmux else "",
            "backend": "foreground" if backend == "foreground" else "tmux" if use_tmux else "process",
            "log": str(Path(output) / "worker.log"),
            "command": "hedonic run disjoint " + " ".join(shlex.quote(a) for a in front.front_argv(settings))}


def launch_posthoc(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the exact post-hoc stability analysis (``hedonic run posthoc``) durably."""
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "posthoc", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run posthoc " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


def launch_controlled(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the controlled-overlap study (``hedonic run controlled``) durably; its records go to ``output_dir``."""
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "controlled", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run controlled " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


def launch_audit(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the robustness audit of saved benchmark covers (``hedonic run audit``) durably.

    The folder's records are kept: the worker skips the covers already audited.
    """
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "audit", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run audit " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


def launch_calibrate(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the HOC calibration (``hedonic run calibrate``) durably.

    The folder's results and ledger are kept: the worker skips the searches already recorded.
    """
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "calibrate", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run calibrate " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


def launch_baseline_calibration(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the baselines' calibration (``hedonic run calibrate-baselines``) durably.

    The folder's results and ledger are kept: the worker skips the calibrations already recorded.
    """
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "baseline_calibrate", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run calibrate-baselines " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


def launch_startcurve(argv: list[str], name: str, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the start-quality curve (``hedonic run startcurve``) durably; its records go to ``output_dir``.

    Unlike a controlled run, the folder's records are kept: the worker skips the conditions already recorded.
    """
    name = qs.parse_name(name)
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "startcurve", "argv": argv, "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run startcurve " + " ".join(shlex.quote(a) for a in argv)}
    return _spawn(entry)


# --------------------------------------------------------------------------- memory guard
MEMORY_GUARD_SECONDS = 0.5  # how often the worker reads the machine's free memory


def descendants(pid: int) -> list[int]:
    """Every process below ``pid`` (children, grandchildren, ...), from one ``ps`` listing."""
    try:
        listing = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def stop_for_memory(name: str, entry: dict, prog: dict, free: float, floor: float) -> None:
    """Free memory fell below the floor: kill this run's whole process tree, record why, and exit (never returns).

    A run whose workers each hold a large copy of its data can exhaust the machine in seconds (a spectrum audit of
    1.1 M vertices once reached 261 GB attributed to the terminal and rebooted the Mac). Stopping at the floor leaves
    the finished records intact, so ``hedonic run resume`` continues once the cause is fixed.
    """
    for pid in reversed(descendants(os.getpid())):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    reason = (f"stopped at the memory floor: {free:.0f}% of the machine's memory free, below {floor:.0f}% "
              f"(HEDONIC_MEMORY_FLOOR); this run's processes were killed. Finished records are kept")
    for item in prog.get("plan", []):
        if item.get("status") == "running":
            item["status"] = "pending"
    prog.update(status="failed", reason=reason, finished=time.time(), updated=time.time())
    try:
        write_json(progress_path(entry), prog)
    except OSError:
        pass
    print(style(f"\n  {reason}\n  continue with: hedonic run resume {name}", "red"), flush=True)
    os._exit(75)


def start_memory_guard(name: str, entry: dict, prog: dict):
    """Watch the machine's free memory from a daemon thread; the returned event stops the watch (``None``: no watch)."""
    import threading

    from hedonic.experiments.overlapping import parallel

    floor = parallel.memory_floor()
    if not floor or parallel.free_memory_percent() is None:
        return None
    done = threading.Event()

    def watch() -> None:
        while not done.wait(MEMORY_GUARD_SECONDS):
            free = parallel.free_memory_percent()
            if free is not None and free < floor:
                stop_for_memory(name, entry, prog, free, floor)

    threading.Thread(target=watch, name="memory-guard", daemon=True).start()
    return done


# --------------------------------------------------------------------------- worker
def worker(name: str) -> int:
    """Executes the plan; writes progress.json after every step. Ignores Ctrl-C by design.

    An unexpected error is recorded as status ``failed`` (with its reason) instead of leaving the run looking
    alive; ``hedonic run resume`` then continues from the finished records.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    entry = json.loads(entry_path(name).read_text())
    prog: dict = {}
    guard = start_memory_guard(name, entry, prog)
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
    finally:
        if guard is not None:  # a worker run in-process (tests, --foreground) must not leave a watcher behind
            guard.set()


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
            prog["tracked_files_sha256"] = gs.tracked_digest()  # the code these records are judged against
            prog["describe"] = gs.describe(options, len(event["jobs"]))
            prog["status"] = "running"
            flush(True)
        elif kind in ("job_start", "audit_start"):
            prog["setup"] = [f"{'auditing' if kind == 'audit_start' else 'loading'} {event['job']}"]
            prog["stage"] = prog["setup"][0]
            flush(True)
        elif kind == "stage":
            prog["stage"] = f"{event['text']} · {event['job']}"
            prog["setup"] = [prog["stage"]]
            flush(True)
        elif kind == "audit_progress":
            remaining = event["elapsed"] / event["done"] * (event["total"] - event["done"])
            prog["stage"] = (f"auditing {event['job']} · {event['policy']} · γ={event['gamma']:g}  "
                             f"{event['done']}/{event['total']} resolutions · ETA ~{qs.fmt_time(remaining)}")
            prog["setup"] = [prog["stage"]]
            flush()
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
    if entry.get("kind") == "posthoc":
        from hedonic.experiments.overlapping import gt_posthoc

        return gt_posthoc.work(name, entry, prog)
    if entry.get("kind") == "controlled":
        from hedonic.experiments.overlapping import controlled_overlap_config

        return controlled_overlap_config.work(name, entry, prog)
    if entry.get("kind") == "startcurve":
        from hedonic.experiments.overlapping import start_quality

        return start_quality.work(name, entry, prog)
    if entry.get("kind") == "audit":
        from hedonic.experiments.overlapping import cover_audit

        return cover_audit.work(name, entry, prog)
    if entry.get("kind") == "calibrate":
        from hedonic.experiments.overlapping import calibrate

        return calibrate.work(name, entry, prog)
    if entry.get("kind") == "baseline_calibrate":
        from hedonic.experiments.overlapping import baseline_run

        return baseline_run.work(name, entry, prog)
    if entry.get("kind") == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.work(name, entry, prog)
    if entry.get("kind") == "queue":
        from hedonic.experiments.overlapping import runqueue

        return runqueue.work(name, entry, prog)
    cfg = qs.Config(**entry["config"])
    ppath = progress_path(entry)
    prog.update({"name": name, "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": qs.describe(cfg), "command": cfg.command(),
                 "environment": qs.environment(), "setup": [], "plan": []})
    write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + qs.describe(cfg))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C stops this run (in the foreground); finished records are kept" if entry.get("backend") == "foreground"
                else "Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)

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
    stage = qs.open_stage(cfg)  # the third axis: each saved cover is audited right after its detector run
    audit_failed = False
    scan = _scan(cfg, plan, binary, stage, prog, ppath)
    for i, (net, seed, m) in enumerate(plan):
        item = prog["plan"][i]
        tag = f"[{i + 1}/{len(plan)}] {qs.NETWORKS[net][0]} · seed {seed} · {m.label}"
        if scan[i]["done"]:  # certified at the scan: nothing to run, nothing to print
            rec = qs.unit_record(cfg, net, seed, m)
            if stage is not None:
                qs.attach_robustness(stage, net, seed, m, rec)
            records.append(rec)
            continue
        item.update(status="running", started=time.time())
        prog["current"], prog["updated"] = i, time.time()
        write_json(ppath, prog)
        # The double check: the scan may be old by now (another process, or the user, may have put the record there).
        found = qs.unit_check(cfg, net, seed, m, binary, adopt=True)
        if found.state == "corrupt":
            moved = qs.quarantine_unit(cfg, net, seed, m, found.reason)
            print(style(f"  ! {tag}: {found.reason}; moved aside ({len(moved)} files), running it again", "yellow"), flush=True)
            found = qs.unit_check(cfg, net, seed, m, binary)
        if found.ok:
            origin = "audit only" if scan[i]["audit_pending"] else "found at its turn, certified"
            print(f"  … {tag} · {origin}", flush=True)
            rec = qs.unit_record(cfg, net, seed, m)
            item["origin"] = "scan" if scan[i]["audit_pending"] else "reach"
        else:
            print(f"  … {tag}", flush=True)
            rec = qs.run_one(cfg, net, seed, m, binary)
            item["origin"] = "run"
        if stage is not None:
            item.update(detection_seconds=rec.get("detection_seconds"))
            prog["stage"], prog["updated"] = f"robustness audit of the saved cover · {tag}", time.time()
            write_json(ppath, prog)
            stage.on_progress = _audit_reporter(prog, item, ppath, tag)
            failed = qs.attach_robustness(stage, net, seed, m, rec)
            stage.on_progress = None
            item.pop("audit", None)
            if failed:
                audit_failed = True
                print(style(f"    robustness audit failed: {rec['robustness_error']}", "red"), flush=True)
        records.append(rec)
        metrics = rec.get("metrics") or {}
        prog["stage"] = None
        item.update(status=rec.get("status"), detection_seconds=rec.get("detection_seconds"), reason=rec.get("reason"),
                    metrics={k: metrics.get(k) for k, _ in qs.COLUMNS}, robustness=rec.get("robustness"),
                    robustness_error=rec.get("robustness_error"))
        if item["origin"] == "run":  # only a unit that ran has a wall-clock time the ETA may learn from
            item["seconds"] = time.time() - item["started"]
        prog["updated"] = time.time()
        write_json(ppath, prog)
        print(qs.result_line(tag, rec), flush=True)
    uncertified = _final_check(cfg, plan, binary, prog, ppath)
    audit_failed = audit_failed or bool(uncertified)
    if stage is not None:
        stage.finish()
    summary = qs.finish(cfg, records, unavailable)
    prog.update(status="completed" if all(r.get("status") == "completed" for r in records) and not audit_failed
                else "completed_with_failures", finished=time.time(), updated=time.time(), current=None)
    write_json(ppath, prog)
    print(qs.render(summary, cfg))
    print(f"\n  records {cfg.output_dir}\n  summary {Path(cfg.output_dir) / 'summary.json'}")
    return 0


def _scan(cfg: qs.Config, plan: list, binary, stage, prog: dict, ppath: Path) -> list[dict]:
    """Before anything runs: certify what the disk holds for every unit, and mark the finished ones finished.

    A unit whose record is present, final, of the same settings and hash-sealed is done (its figures go on the plan
    at once; it has no ``seconds``, so the ETA learns only from units that really ran). A valid record without a seal
    is sealed now. A record whose bytes differ from its seal is moved aside and its unit runs again.
    """
    prog["status"] = "setup"
    prog["stage"], prog["updated"] = f"scanning {len(plan)} units: which records are already certified", time.time()
    write_json(ppath, prog)
    print(f"  scanning {len(plan)} units against the files on disk (identity, final status, cover, SHA-256 seal)…", flush=True)
    scan = qs.scan_plan(cfg, plan, binary, adopt=True, stage=stage)
    for i, (entry, (net, seed, m)) in enumerate(zip(scan, plan)):
        if entry["state"] == "corrupt":
            moved = qs.quarantine_unit(cfg, net, seed, m, entry["reason"])
            print(style(f"  ! {qs.NETWORKS[net][0]} · seed {seed} · {m.label}: {entry['reason']}; moved aside "
                        f"({len(moved)} files), it will run again", "yellow"), flush=True)
            entry.update(state="missing", done=False, audit_pending=False)
        if entry["done"]:
            rec = qs.unit_record(cfg, net, seed, m)
            metrics = rec.get("metrics") or {}
            prog["plan"][i].update(status=rec.get("status"), origin="scan", detection_seconds=rec.get("detection_seconds"),
                                   reason=rec.get("reason"), metrics={k: metrics.get(k) for k, _ in qs.COLUMNS},
                                   robustness=rec.get("robustness"))
        if entry.get("warning"):
            print(style(f"  ! {qs.NETWORKS[net][0]} · seed {seed} · {m.label}: {entry['warning']}", "yellow"), flush=True)
    counts = qs.summarize_scan(scan)
    prog["scan"], prog["stage"], prog["status"], prog["updated"] = counts, None, "running", time.time()
    write_json(ppath, prog)
    todo = counts["units"] - counts.get("done", 0)
    print(style(f"  scan: {counts.get('done', 0)} of {counts['units']} units are certified and will not run again; "
                f"{todo} remain" + "".join(f" · {counts[k]} {k.replace('_', ' ')}" for k in ("audit_pending", "stale", "missing")
                                           if counts.get(k)), "green"), flush=True)
    return scan


def _audit_reporter(prog: dict, item: dict, ppath: Path, tag: str) -> Callable[[dict], None]:
    """What the worker does with the audit process's progress while it waits: show it and keep it on the unit's row.

    The row's ``audit`` is what the queue view prices a running unit's remaining time from (``queuebuild.benchmark``).
    """
    def report(info: dict) -> None:
        text = info.get("detail") or info.get("text") or ""
        item["audit"] = {key: info.get(key) for key in ("policy", "number", "policies", "done", "total", "elapsed",
                                                        "eta_seconds", "updated", "since") if info.get(key) is not None}
        prog["stage"], prog["updated"] = f"robustness audit of the saved cover · {tag} · {text}", time.time()
        write_json(ppath, prog)

    return report


def _final_check(cfg: qs.Config, plan: list, binary, prog: dict, ppath: Path) -> list[str]:
    """After the last unit: check every completed unit's bytes against its seal once more, so the summary rests on them."""
    bad = []
    for (net, seed, m), item in zip(plan, prog["plan"]):
        if item.get("status") != "completed":
            continue
        result = qs.unit_check(cfg, net, seed, m, binary)
        if result.state != "verified":
            bad.append(f"{qs.NETWORKS[net][0]} · seed {seed} · {m.label}: {result.reason}")
            item.update(status="failed", reason=f"final verification failed: {result.reason}")
    prog["verified"] = {"checked": sum(it.get("status") in ("completed", "failed") for it in prog["plan"]),
                        "uncertified": bad, "at": time.time()}
    write_json(ppath, prog)
    for line in bad:
        print(style(f"  ! final verification: {line}", "red"), flush=True)
    return bad


# --------------------------------------------------------------------------- viewer
def _records_from_progress(prog: dict) -> list[dict]:
    recs = [{"network": p["network"], "method_key": p["method"], "seed": p["seed"], "status": p["status"],
             "reason": p.get("reason"), "metrics": p.get("metrics") or {}, "detection_seconds": p.get("detection_seconds"),
             "robustness": p.get("robustness")}
            for p in prog.get("plan", []) if p["status"] not in ("pending", "running")]
    for m, reason in (prog.get("unavailable") or {}).items():
        recs.append({"network": None, "method_key": m, "status": "unavailable", "reason": reason})
    return recs


def unit_finished(unit: dict) -> bool:
    """A plan unit is finished once it has a result: any status but pending or running."""
    return (unit.get("status") or "pending") not in ("pending", "running")


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _unit_key(unit: dict) -> tuple:
    """The row of a unit: its network and method. Times differ by network, so no row mixes networks."""
    return unit.get("network"), unit.get("method") or unit.get("label")


def method_rows(plan: list[dict]) -> list[dict]:
    """One row per (network, method) in plan order: its units, and the measured times of its finished units."""
    rows: dict[tuple, dict] = {}
    for unit in plan:
        key = _unit_key(unit)
        row = rows.setdefault(key, {"key": key, "network": key[0], "method": key[1],
                                    "label": unit.get("label") or str(key[1]), "units": [], "wall": [],
                                    "detection": [], "expected": None})
        row["units"].append(unit)
        if row["expected"] is None:
            row["expected"] = _number(unit.get("expected"))
        if unit_finished(unit):
            if (wall := _number(unit.get("seconds"))) is not None:
                row["wall"].append(wall)
            if (detection := _number(unit.get("detection_seconds"))) is not None:
                row["detection"].append(detection)
    return list(rows.values())


def row_seconds(row: dict) -> tuple[float | None, bool]:
    """Seconds one unit of a row should take, and whether that rests on the planning constant (True: estimated).

    Once a unit of the row has finished, the mean wall-clock time of its finished units is used (wall-clock, because
    that is what the queue spends: it includes loading the graph, not only the detector). Before that, the planning
    constant ``expected`` is the only figure there is, and it is labelled as an estimate.
    """
    if row["wall"]:
        return statistics.fmean(row["wall"]), False
    return row["expected"], True


def plan_remaining(plan: list[dict], now: float) -> tuple[float | None, bool]:
    """Seconds left in a plan, and whether that figure rests on planning constants (True: estimated).

    Each pending unit takes its row's ``row_seconds``; the running unit only the part it has not used yet. None when
    a remaining unit has no estimate at all.
    """
    rows = {row["key"]: row for row in method_rows(plan)}
    total, estimated = 0.0, False
    for unit in plan:
        state = unit.get("status") or "pending"
        if state not in ("pending", "running"):
            continue
        seconds, constant = row_seconds(rows[_unit_key(unit)])
        if seconds is None:
            return None, True
        if state == "running" and (started := _number(unit.get("started"))) is not None:
            seconds = max(0.0, seconds - (now - started))
        total += seconds
        estimated = estimated or constant
    return total, estimated


def running_unit(plan: list[dict], now: float) -> dict | None:
    """The unit in progress: its place in the plan, its time so far, its estimate and how far past it has run."""
    rows = {row["key"]: row for row in method_rows(plan)}
    for index, unit in enumerate(plan):
        if (unit.get("status") or "pending") != "running":
            continue
        estimate, constant = row_seconds(rows[_unit_key(unit)])
        started = _number(unit.get("started"))
        elapsed = None if started is None else max(0.0, now - started)
        over = elapsed - estimate if elapsed is not None and estimate is not None and elapsed > estimate else None
        detection = _number(unit.get("detection_seconds"))  # recorded once the detector returns: what follows is the audit
        audit = None if detection is None or elapsed is None else max(0.0, elapsed - detection)
        return {"index": index, "unit": unit, "elapsed": elapsed, "estimate": estimate, "constant": constant,
                "over": over, "detection": detection, "audit": audit}
    return None


def cell_glyph(unit: dict | None, live: bool, G: dict) -> str:
    """One plan unit as one glyph: done, running, pending, failed or stopped (the legend is under the grid)."""
    if unit is None:
        return " "
    state = unit.get("status") or "pending"
    if state == "completed":
        return G["done"]
    if state == "running":
        return G["running"] if live else G["stopped"]
    if state == "pending":  # a unit reset by a stop keeps its start time: it was interrupted, not never started
        return G["pending"] if live or "started" not in unit else G["stopped"]
    if state in ("stopped", "interrupted"):
        return G["stopped"]
    return G["failed"]


def _dur(seconds: float) -> str:
    """A duration in the progress lines: '6.0 s', '41 s', then m:ss or h:mm:ss."""
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    return qs.fmt_time(seconds)


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def _network_name(network) -> str:
    return qs.NETWORKS.get(network, (str(network),))[0]


def _row_labels(rows: list[dict]) -> tuple[list[str], int]:
    """Row labels (network · method when the plan spans several networks) and the width of their column."""
    multi = len({row["network"] for row in rows}) > 1
    texts = [f"{_network_name(row['network'])} · {row['label']}" if multi else str(row["label"]) for row in rows]
    width = min(24, max((len(text) for text in texts), default=0))
    return [_cut(text, width) for text in texts], width


def unit_grid(plan: list[dict], live: bool, indent: str, G: dict, width: int = BLOCK_WIDTH) -> list[str]:
    """The plan as a grid: one row per (network, method), one column per seed, in bands that fit the width."""
    rows = method_rows(plan)
    labels, label_width = _row_labels(rows)
    cells: dict[tuple, dict] = {}
    for unit in plan:
        cells.setdefault((*_unit_key(unit), unit.get("seed")), unit)
    seeds = sorted({unit.get("seed") for unit in plan if unit.get("seed") is not None},
                   key=lambda seed: (isinstance(seed, str), seed)) or [None]
    per_band = max(1, (width - len(indent) - label_width - 2) // 11) * 10  # ten glyphs and a gap per group
    lines = []
    for start in range(0, len(seeds), per_band):
        groups = [seeds[i:i + 10] for i in range(start, min(start + per_band, len(seeds)), 10)]
        lines.append((indent + "seed".ljust(label_width + 2) + "".join(str(g[0]).ljust(11) for g in groups)).rstrip())
        for row, label in zip(rows, labels):
            body = " ".join("".join(cell_glyph(cells.get((*row["key"], seed)), live, G) for seed in group)
                            for group in groups)
            lines.append(f"{indent}{label.ljust(label_width)}  {body}")
    lines.append(indent + style(f"{G['done']} done  {G['running']} running  {G['pending']} pending  "
                                f"{G['failed']} failed  {G['stopped']} stopped", "dim"))
    return [tui.fit(line, width) for line in lines]


def _method_line(row: dict, label: str, label_width: int, indent: str, G: dict) -> str:
    """Per method: finished units, their measured detection time and its spread, and the unit time the ETA uses."""
    text = f"{indent}{label.ljust(label_width)}  {sum(unit_finished(u) for u in row['units'])}/{len(row['units'])} finished"
    detection = row["detection"]
    if detection:
        text += f" · detection {_dur(statistics.fmean(detection))}"
        if len(detection) > 1:
            text += f" {G['pm']} {_dur(statistics.stdev(detection))}"
    if row["wall"]:
        text += f" · unit {_dur(statistics.fmean(row['wall']))}"
    else:
        text += " · none measured"
        if row["expected"] is not None:
            text += f" · estimated {_dur(row['expected'])} per unit"
    return tui.fit(text, BLOCK_WIDTH)


def last_log_line(path: str | None) -> str | None:
    """The last non-empty line of a run's log, without colour codes, read from the end of the file."""
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 8192))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        text = tui.strip_ansi(line).strip()
        if text:
            return text
    return None


def _home(path) -> str:
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text.startswith(home + "/") else text


def status_mark(status: str, frame: str, G: dict) -> str:
    """The glyph before a status word: the spinner while the run lives, a fixed mark once it has ended."""
    if status in LIVE:
        return frame
    return {"stopped": G["stopped"], "interrupted": G["stopped"], "failed": G["failed"]}.get(status, G["dot"])


def elapsed_seconds(entry: dict, prog: dict, status: str) -> float:
    """Seconds since the run started, or until it ended."""
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in LIVE else prog.get("updated", time.time()))
    return end - started


def progress_bar(done: int, total: int, width: int = 36, G: dict | None = None) -> str:
    G = G or tui.glyphs()
    filled = int(width * done / total) if total else 0
    return style(G["full"] * filled, "green") + style(G["empty"] * (width - filled), "dim")


def remaining_seconds(entry: dict, prog: dict, status: str, now: float | None = None) -> float | None:
    """Estimated seconds left while the run is running; None when the run has no estimate."""
    kind = entry.get("kind")
    if kind in ("posthoc", "controlled", "startcurve", "audit", "calibrate", "baseline_calibrate"):  # these publish their own estimate in their progress file
        return prog.get("eta_seconds") if status in ("running", "setup") else None
    plan = prog.get("plan", [])
    if status != "running" or not plan or kind not in (None, "spectrum"):
        return None
    if kind == "spectrum":
        done, total = progress_counts(entry, prog)
        seconds = [p["seconds"] for p in plan if p.get("seconds") is not None]
        return statistics_mean(seconds) * (total - done) if seconds and done < total else None
    return plan_remaining(plan, time.time() if now is None else now)[0]


def _left(entry: dict, prog: dict, status: str, now: float) -> tuple[float | None, bool]:
    """Seconds left of a run and whether that is an estimate. Benchmark runs derive it from their plan."""
    if entry.get("kind") is None:
        plan = prog.get("plan") or []
        return plan_remaining(plan, now) if status == "running" and plan else (None, False)
    return remaining_seconds(entry, prog, status, now), False


def progress_block(entry: dict, prog: dict, status: str, frame: str, *, indent: str = "  ",
                   G: dict | None = None, now: float | None = None) -> list[str]:
    """What a run is doing: the bar with counts, time and ETA, the unit in progress, the plan as a grid, a line per
    method with its measured times, the last log line and the records. Used by status, attach and queue items."""
    G = G or tui.glyphs()
    now = time.time() if now is None else now
    live = status in LIVE
    plan = prog.get("plan") or []
    benchmark = entry.get("kind") is None
    elapsed = qs.fmt_time(elapsed_seconds(entry, prog, status))
    done, total = progress_counts(entry, prog)
    lines: list[str] = []
    if benchmark and not plan:  # nothing planned yet: the setup steps, if any
        setup = prog.get("setup") or []
        if status == "setup" and setup:
            lines += [indent + style(f"setup {text}", "magenta") for text in setup[-3:]]
            lines.append(indent + style(f"elapsed {elapsed}", "dim"))
        else:
            lines.append(indent + style(f"starting… elapsed {elapsed}", "dim"))
    elif not total:
        lines.append(indent + style(f"starting… elapsed {elapsed}", "dim"))
    else:
        share = f"{done}/{total} · {100 * done // total}%"
        if live:
            lines.append(f"{indent}{progress_bar(done, total, 30, G)}  {share}")
            left, estimated = _left(entry, prog, status, now)
            tail = f"elapsed {elapsed}"
            if left is not None:
                tail += f" · ETA ~{qs.fmt_time(left)}" + (" (estimated)" if estimated else "")
            lines.append(f"{indent}{tail}")
        else:
            lines.append(f"{indent}{progress_bar(done, total, 30, G)}  {share} · elapsed {elapsed}")
        if benchmark:
            current = running_unit(plan, now) if live else None
            if current is not None:
                unit = current["unit"]
                head = (f"{G['running']} [{current['index'] + 1}/{len(plan)}] {_network_name(unit.get('network'))} · "
                        f"seed {unit.get('seed')} · {unit.get('label') or unit.get('method')}")
                if current["elapsed"] is not None:
                    head += f"  {qs.fmt_time(current['elapsed'])} elapsed"
                lines.append(indent + head)
                note = " (estimated)" if current["constant"] else ""
                if current["audit"] is not None:  # the detector has returned: the time since is the cover's exact audit
                    lines.append(f"{indent}  detection took {qs.fmt_time(current['detection'])} · the exact audit of its "
                                 f"cover has run {qs.fmt_time(current['audit'])}")
                reported = unit.get("audit") if isinstance(unit.get("audit"), dict) else None
                if reported and reported.get("total"):  # the audit's own count of vertices, from the audit process
                    left = reported.get("eta_seconds")
                    lines.append(f"{indent}  exact audit: policy {reported.get('number')} of {reported.get('policies')}, "
                                 f"{int(reported.get('done') or 0):,} of {int(reported['total']):,} vertices"
                                 + (f" · about {qs.fmt_time(left)} left" if isinstance(left, (int, float)) else ""))
                if current["audit"] is None or not current["constant"]:  # a planning constant prices the detection alone
                    if current["over"] is not None:
                        lines.append(f"{indent}  {qs.fmt_time(current['over'])} beyond its estimate of "
                                     f"~{qs.fmt_time(current['estimate'])}{note}")
                    elif current["estimate"] is not None:
                        lines.append(f"{indent}  estimate ~{qs.fmt_time(current['estimate'])}{note}")
            lines += unit_grid(plan, live, indent, G)
            rows = method_rows(plan)
            labels, label_width = _row_labels(rows)
            lines += [_method_line(row, label, label_width, indent, G) for row, label in zip(rows, labels)]
        else:
            unit = current_unit(entry, prog, status, frame)
            if unit:
                lines.append(indent + unit)
    if entry.get("kind") == "spectrum" and plan:  # which pairs are finished, so the last log line is not mistaken for the current one
        per: dict[str, list[int]] = {}
        for item in plan:
            counts = per.setdefault(str(item.get("job")), [0, 0])
            counts[1] += 1
            counts[0] += item.get("status") not in ("pending", "running")
        finished_pairs = sum(done_ == total_ for done_, total_ in per.values())
        partial = [f"{job.replace('-', '/', 1)} {d}/{t}" for job, (d, t) in per.items() if 0 < d < t]
        lines.append(f"{indent}pairs    {finished_pairs} of {len(per)} complete" + (f" · in progress: {', '.join(partial)}" if partial else ""))
    if live and not benchmark and prog.get("stage") and prog["stage"] != "detector":
        lines.append(f"{indent}now      {prog['stage']}")  # what the run says it is doing (calibration: which searches run)
    log = None if entry.get("kind") == "calibrate" else last_log_line(entry.get("log"))  # side-by-side searches interleave their lines
    if log:
        lines.append(f"{indent}log      {log}")
    if entry.get("output_dir") and status not in TERMINAL:  # a finished run prints its records in its own footer
        lines.append(f"{indent}records  {_home(entry['output_dir'])}")
    return [tui.fit(line, BLOCK_WIDTH) for line in lines]


def current_unit(entry: dict, prog: dict, status: str, frame: str = "●") -> str | None:
    """The plan unit in progress of a running benchmark or spectrum run: its place, what it is, how long it has run."""
    plan = prog.get("plan", [])
    running = [p for p in plan if p["status"] == "running"]
    if status != "running" or not running:
        return None
    p = running[0]
    i = plan.index(p) + 1
    took = style(qs.fmt_time(time.time() - p["started"]), "bold")
    if entry.get("kind") == "spectrum":
        return (f"{style(frame, 'cyan')} [{i}/{len(plan)}] {p['dataset']}/{p['cover']} · {p['policy']} · "
                f"γ={p['gamma']:g} · seed {p['seed']}  {took}")
    return (f"{style(frame, 'cyan')} [{i}/{len(plan)}] {qs.NETWORKS[p['network']][0]} · seed {p['seed']} · "
            f"{p['label']}  {took}" + style(f"  (expected ~{qs.fmt_time(p['expected'])})", "dim"))


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
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    elapsed = qs.fmt_time(elapsed_seconds(entry, prog, status))
    if plan:
        tail = f"  {done}/{len(plan)} detector runs · elapsed {elapsed}"
        left = remaining_seconds(entry, prog, status)
        if left is not None:
            tail += f" · ETA ~{qs.fmt_time(left)}"
        lines.append("  " + progress_bar(done, len(plan)) + tail)
    else:
        lines.append(style(f"  discovering the local SNAP cache… elapsed {elapsed}", "dim"))
    if status in ("running", "setup") and prog.get("stage") and prog.get("stage") != "detector":
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    unit = current_unit(entry, prog, status, frame)
    if unit:
        lines.append("  " + unit)
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


def snapshot(entry: dict, frame: str | None = None, table: bool = True, unicode_glyphs: bool | None = None) -> str:
    prog = read_progress(entry)
    G = tui.glyphs(unicode_glyphs)
    frame = frame or G["dot"]
    if entry.get("kind") == "spectrum":
        return snapshot_spectrum(entry, prog, frame, table)
    if entry.get("kind") == "queue":
        from hedonic.experiments.overlapping import runqueue

        return runqueue.snapshot(entry, prog, frame, table, unicode_glyphs=unicode_glyphs)
    if entry.get("kind") == "posthoc":
        from hedonic.experiments.overlapping import gt_posthoc

        return gt_posthoc.snapshot(entry, prog, frame)
    if entry.get("kind") == "controlled":
        from hedonic.experiments.overlapping import controlled_overlap_config

        return controlled_overlap_config.snapshot(entry, prog, frame)
    if entry.get("kind") == "startcurve":
        from hedonic.experiments.overlapping import start_quality

        return start_quality.snapshot(entry, prog, frame)
    if entry.get("kind") == "audit":
        from hedonic.experiments.overlapping import cover_audit

        return cover_audit.snapshot(entry, prog, frame)
    if entry.get("kind") == "calibrate":
        from hedonic.experiments.overlapping import calibrate

        return calibrate.snapshot(entry, prog, frame)
    if entry.get("kind") == "baseline_calibrate":
        from hedonic.experiments.overlapping import baseline_run

        return baseline_run.snapshot(entry, prog, frame)
    if entry.get("kind") == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.snapshot(entry, prog, effective_status(entry, prog), frame, table)
    cfg = qs.Config(**entry["config"])
    status = effective_status(entry, prog)
    plan = prog.get("plan", [])
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "completed_with_failures": "yellow",
              "interrupted": "red", "stopped": "yellow", "failed": "red"}.get(status, "dim")
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  "
             + style(f"{status_mark(status, frame, G)} {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", qs.describe(cfg)), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    lines += progress_block(entry, prog, status, frame, G=G)
    if status in ("running", "setup") and prog.get("stage"):  # the robustness audit of the unit that just ran
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    finished = [p for p in plan if unit_finished(p)]
    if finished:
        lines.append(style("  recent", "dim"))
        for p in finished[-5:]:
            i = plan.index(p) + 1
            tag = f"[{i}/{len(plan)}] {qs.NETWORKS[p['network']][0]} · seed {p['seed']} · {p['label']}"
            lines.append(qs.result_line(tag, {"status": p["status"], "reason": p.get("reason"),
                                              "metrics": p.get("metrics"), "detection_seconds": p.get("detection_seconds"),
                                              "robustness": p.get("robustness")}))
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
    elif status == "stopped":
        lines.append(style(f"\n  stopped · continue with: hedonic run resume {entry['name']}", "yellow"))
    if status in TERMINAL:
        lines.append(style(f"\n  records  {entry['output_dir']}", "dim"))
        if (Path(entry["output_dir"]) / "summary.json").is_file():
            lines.append(style(f"  summary  {Path(entry['output_dir']) / 'summary.json'}", "dim"))
    return "\n".join(lines)


def _frame(text: str, rows: int, cols: int) -> str:
    """Home the cursor and repaint in place (no full clear, so no flicker); each line is one terminal row."""
    return "\033[H" + "".join(tui.fit(line, cols) + "\033[K\n" for line in text.split("\n")) + "\033[J"


def view_queue(entry: dict) -> int:
    """The interactive tree of a queue (``queuetui``): open and close runs, pairs and loops, wheel scrolling, ETAs."""
    from hedonic.experiments.overlapping import queuetui, runqueue

    def build():
        return runqueue.tree(entry, read_progress(entry))

    def state() -> tuple[str, float]:
        prog = read_progress(entry)
        status = effective_status(entry, prog)
        return status, elapsed_seconds(entry, prog, status)

    def notice() -> list[str]:
        """Where and why the queue stopped: the viewer otherwise shows only a red mark."""
        prog = read_progress(entry)
        status = effective_status(entry, prog)
        return runqueue.stop_lines(entry, prog, status) if status in ("failed", "stopped", "interrupted") else []

    try:
        queuetui.run(build, state, notice)
    finally:
        status = effective_status(entry, read_progress(entry))
        print(style(f"queue {entry['name']} is {status}; closing the view does not stop it.\n"
                    f"  reopen: hedonic run attach {entry['name']}   ·   stop: hedonic run stop {entry['name']}", "dim"))
    return 1 if status in TERMINAL and status != "completed" else 0


def view_tree(entry: dict, build) -> int:
    """The interactive tree for a single run whose kind publishes one (``queuetui``): same keys and mouse as a queue's."""
    from hedonic.experiments.overlapping import queuetui

    def state() -> tuple[str, float]:
        prog = read_progress(entry)
        status = effective_status(entry, prog)
        return status, elapsed_seconds(entry, prog, status)

    def notice() -> list[str]:
        prog = read_progress(entry)
        status = effective_status(entry, prog)
        if status == "failed":
            return [f"failed: {prog.get('reason', 'unknown error')}", f"hedonic run logs {entry['name']} · hedonic run resume {entry['name']}"]
        if status in ("stopped", "interrupted"):
            return [f"{status}: hedonic run resume {entry['name']} keeps the finished calibrations and the ledger"]
        return []

    try:
        queuetui.run(lambda: build(entry, read_progress(entry)), state, notice)
    finally:
        status = effective_status(entry, read_progress(entry))
        print(style(f"{entry['name']} is {status}; closing the view does not stop it.\n"
                    f"  reopen: hedonic run attach {entry['name']}   ·   stop: hedonic run stop {entry['name']}", "dim"))
    return 1 if status in TERMINAL and status != "completed" else 0


def view(entry: dict) -> int:
    """Live viewer. Ctrl-C closes the viewer only. A queue opens as an interactive tree (``view_queue``) in a terminal."""
    if (entry.get("kind") == "queue" and sys.stdin.isatty() and sys.stdout.isatty() and tui.termios is not None
            and not os.environ.get("HEDONIC_PLAIN_VIEW")):
        return view_queue(entry)
    if (entry.get("kind") not in ("queue", "baseline_calibrate", "disjoint") and sys.stdin.isatty() and sys.stdout.isatty()
            and tui.termios is not None and not os.environ.get("HEDONIC_PLAIN_VIEW")):
        from hedonic.experiments.overlapping import runqueue

        try:  # any run started through `hedonic run VERB` is a one-item queue: the same tree, the same keys
            runqueue.single_tree(entry, read_progress(entry))
        except Exception:  # noqa: BLE001 - a run the tree cannot describe keeps its panel
            pass
        else:
            return view_tree(entry, runqueue.single_tree)
    if (entry.get("kind") == "baseline_calibrate" and sys.stdin.isatty() and sys.stdout.isatty()
            and tui.termios is not None and not os.environ.get("HEDONIC_PLAIN_VIEW")):
        from hedonic.experiments.overlapping import baseline_run

        return view_tree(entry, baseline_run.tree)
    frames, i = tui.glyphs()["spin"], 0
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
        done, total = progress_counts(entry, prog)
        line = f"  {entry['name']}: {status} {done}/{total}"
        if line != last and sys.stderr.isatty():
            print("\r\033[K" + line, end="", file=sys.stderr, flush=True)
            last = line
        if status in TERMINAL or status == "interrupted":
            break
        time.sleep(1.0)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    summary = Path(entry["output_dir"]) / {"spectrum": "coverage_report.json",
                                           "disjoint": "stats/paper_stats.json",
                                           "posthoc": "summary.json",
                                           "calibrate": "hoc_tuned.json",
                                           "baseline_calibrate": "baseline_tuned.json"}.get(entry.get("kind"), "summary.json")
    if not summary.is_file():
        print(f"  run {status}: {prog.get('reason', 'no summary was written')}; see `hedonic run logs {entry['name']}`",
              file=sys.stderr)
        return 1
    payload = json.loads(summary.read_text())
    print(json.dumps(payload if entry.get("kind") in ("spectrum", "disjoint", "posthoc", "audit", "calibrate", "baseline_calibrate") else payload["results"], indent=2,
                     default=str))
    return 0 if status == "completed" else 1


# --------------------------------------------------------------------------- commands
def progress_counts(entry: dict, prog: dict) -> tuple[int, int]:
    """Finished units and total units of a run (detector runs, or graphs for disjoint runs)."""
    if entry.get("kind") == "disjoint":
        return int(prog.get("done", 0)), int(prog.get("graphs", 0))
    if entry.get("kind") in ("posthoc", "controlled", "startcurve", "audit", "calibrate", "baseline_calibrate"):
        return int(prog.get("done", 0)), int(prog.get("total", 0))
    if entry.get("kind") == "queue":  # completed items of the queue, not detector runs
        from hedonic.experiments.overlapping import runqueue

        return runqueue.counts(entry)
    plan = prog.get("plan", [])
    return sum(unit_finished(p) for p in plan), len(plan)


def cmd_list() -> int:
    entries = all_entries()
    if not entries:
        print("no runs yet — start one with `hedonic run exp`")
        return 0
    width = max(22, *(len(e["name"]) for e in entries))
    print(style(f"  {'name':<{width}} {'status':<24} {'progress':>13}  {'started':<16}  output", "bold"))
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "completed_with_failures": "yellow",
              "interrupted": "red", "stopped": "yellow", "failed": "red"}
    for e in entries:
        prog = read_progress(e)
        done, total = progress_counts(e, prog)
        started = dt.datetime.fromtimestamp(prog.get("started", e["created"])).strftime("%Y-%m-%d %H:%M")
        status = effective_status(e, prog)
        shown = style(f"{status:<24}", colour.get(status, "dim"))
        print(f"  {e['name']:<{width}} {shown} {f'{done}/{total}':>13}  {started:<16}  {e['output_dir']}")
    resumable = [e["name"] for e in entries if effective_status(e, read_progress(e)) in ("interrupted", "failed")]
    if resumable:  # a run you stopped yourself is not nagged about
        print(style(f"\n  not finished, not stopped by you: hedonic run resume {resumable[-1]}", "dim"))
    return 0


def cmd_stop(name: str) -> int:
    entry = load_entry(name)
    if entry.get("kind") == "queue":  # the item in progress first, then the queue that waits for it
        from hedonic.experiments.overlapping import runqueue

        runqueue.stop_items(entry)
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
    elif entry.get("kind") == "posthoc":  # restarts the analysis: vertex intervals are not checkpointed
        entry = launch_posthoc(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "controlled":  # restarts the study (instances are not checkpointed) unless it finished
        if status == "completed":
            print(f"  {entry['name']} is completed; its records are in {entry['output_dir']}. To run it again: "
                  f"hedonic run controlled --config CONFIG --overwrite")
            return 0
        entry = launch_controlled(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "startcurve":  # the worker skips the conditions already in the records
        entry = launch_startcurve(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "audit":  # the worker skips the covers already audited
        entry = launch_audit(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "calibrate":  # finished searches are kept, and the search ledger memoizes probes
        entry = launch_calibrate(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "baseline_calibrate":  # finished calibrations are kept, and the ledger memoizes evaluations
        entry = launch_baseline_calibration(entry["argv"], entry["name"], entry["output_dir"], backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "queue":  # finished items are skipped; the first unfinished one is resumed
        from hedonic.experiments.overlapping import runqueue

        entry = runqueue.launch(entry, backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "disjoint":
        entry = launch_disjoint(entry["settings"], entry["name"], entry["output_dir"],
                                backend=entry.get("backend", "auto"))
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


def _cover_key(path: Path) -> list:
    import gzip

    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    cover = data.get("cover", data) if isinstance(data, dict) else data
    return sorted(sorted(int(v) for v in community) for community in cover)


def check_benchmark(entry: dict, sample: int) -> int:
    """Rerun the fastest ``sample`` completed detector runs and compare covers and scores exactly.

    Only planned units are candidates (the plan follows the run's replication), and a method whose runs are replicas
    (``replication.kind``: the program is stochastic and no seed reaches it) is left out, since a rerun is not expected to
    repeat its cover.
    """
    import tempfile

    from hedonic.experiments.overlapping import replication

    cfg = qs.Config(**entry["config"])
    prog = read_progress(entry)
    done = [p for p in prog.get("plan", []) if p.get("status") == "completed"]
    print(f"  run      {entry['name']} · {len(done)}/{len(prog.get('plan', []))} detector runs completed")
    replicas = sorted({p["method"] for p in done if replication.kind(p["method"]) == "replicas"})
    if replicas:
        print(f"  skipped  {', '.join(qs.BY_KEY[m].label for m in replicas)}: its runs are independent replicas "
              "(stochastic, no seed reaches the program), so a rerun need not repeat a cover")
    done = [p for p in done if p["method"] not in replicas]
    if not done:
        print("  nothing to check yet")
        return 1
    picked = sorted(done, key=lambda p: p.get("detection_seconds") or float("inf"))[:max(1, sample)]
    failures = 0
    for item in picked:
        m = qs.BY_KEY[item["method"]]
        dataset = qs.split_target(item["network"])[0]
        _, stored_dir = qs.unit_argv(cfg, item["network"], item["seed"], m, None)
        stored = json.loads((stored_dir / "runs" / dataset / f"{m.runner}.json").read_text())
        stored_cover = stored_dir / "covers" / dataset / f"{m.runner}.json.gz"
        with tempfile.TemporaryDirectory(prefix="hedonic-check-") as tmp:
            again = qs.Config(**{**cfg.__dict__, "output_dir": tmp, "save_covers": stored_cover.is_file()})
            print(f"  … rerunning {qs.NETWORKS[item['network']][0]} · seed {item['seed']} · {m.label} "
                  f"(~{qs.fmt_time(item.get('detection_seconds') or 0)})", flush=True)
            rec = qs.run_one(again, item["network"], item["seed"], m, None)
            same_metrics = all(rec.get("metrics", {}).get(k) == stored.get("metrics", {}).get(k)
                               for k in ("f1", "onmi") if k in stored.get("metrics", {}))
            same_cover = None
            fresh_cover = qs.unit_argv(again, item["network"], item["seed"], m, None)[1] / "covers" / dataset / \
                f"{m.runner}.json.gz"
            if stored_cover.is_file() and fresh_cover.is_file():
                same_cover = _cover_key(stored_cover) == _cover_key(fresh_cover)
        ok = rec.get("status") == "completed" and same_metrics and same_cover is not False
        failures += not ok
        what = "identical cover and scores" if same_cover else "identical scores (no saved cover to compare)"
        print(f"    {style('✓', 'green') if ok else style('✗', 'red')} {what if ok else 'DIFFERS: ' + str(rec.get('status'))}")
    print(style(f"  check    {len(picked) - failures}/{len(picked)} reruns identical", "green" if not failures else "red",
                "bold"))
    return 0 if not failures else 1


def cmd_check(name: str | None, sample: int | None, rows: int) -> int:
    entry = load_entry(name)
    if entry.get("kind") == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.check(entry, shards=sample or 20, rows=rows)
    if entry.get("kind") == "spectrum":
        print("  the spectrum audits every return itself; see its coverage_report.json")
        return 0
    if entry.get("kind") == "posthoc":
        print("  the post-hoc analysis is exact; summary.json holds each policy's comparison with the grid")
        return 0
    if entry.get("kind") == "controlled":
        print("  the controlled-overlap study writes every run with its seeds and implementation identity to its JSON; "
              "it has no sampled replay yet")
        return 0
    if entry.get("kind") == "startcurve":
        print("  the start-quality study writes one JSON line per condition with its seeds and runtime identity to "
              "start_quality.jsonl; it has no sampled replay yet")
        return 0
    if entry.get("kind") == "baseline_calibrate":
        print("  the baseline calibration records every method's rule, candidates and settings in baseline_tuned.json; "
              "its ledger/ holds every evaluation. It has no sampled replay")
        return 0
    if entry.get("kind") == "calibrate":
        print("  the calibration records every search's settings, scores and the build in hoc_tuned.json; "
              "its ledger/ holds every probe. It has no sampled replay")
        return 0
    if entry.get("kind") == "audit":
        print("  the audit is exact (interval arithmetic); each record names the graph's hash and, for the cover "
              "profile, the cover file's hash. It has no sampled replay yet")
        return 0
    if entry.get("kind") == "queue":
        names = ", ".join(item["name"] for item in entry["items"])
        print(f"  a queue has no results of its own; check each run: {names}")
        return 0
    return check_benchmark(entry, sample or 2)


VERBS = {"list": "runs and their progress", "status": "snapshot of a run (default: latest)",
         "attach": "live view of a running benchmark (Ctrl-C closes only the view)",
         "logs": "raw output of a run (tmux pane, or the log file)", "stop": "stop a run (explicit; never automatic)",
         "resume": "continue an interrupted, stopped or failed run; finished runs are reused",
         "check": "morning check: replay sampled results from their seeds and compare (default: latest run)",
         "queue": "run the experiments of a queue file one after another (hedonic run queue FILE|NAME)"}


def dispatch(argv: list[str]) -> int:
    """``hedonic run <verb> ...`` (verbs other than ``exp``)."""
    import argparse

    verb = argv[0]
    if verb not in VERBS:
        raise SystemExit(f"unknown command `hedonic run {verb}`; try `hedonic --help`")
    if verb == "queue":  # takes a queue file, not a run name
        from hedonic.experiments.overlapping import runqueue

        return runqueue.front_main(argv[1:])
    parser = argparse.ArgumentParser(prog=f"hedonic run {verb}", description=VERBS[verb])
    if verb != "list":
        parser.add_argument("name", nargs=None if verb == "stop" else "?",
                            help="run name (see `hedonic run list`)" + ("" if verb == "stop" else "; default: the latest run"))
    if verb == "status":
        parser.add_argument("--json", action="store_true", help="print progress.json instead of the formatted snapshot")
    if verb == "resume":
        parser.add_argument("--detach", action="store_true", help="restart in the background without opening the view")
        parser.add_argument("--again", action="store_true",
                            help="relaunch a completed run too: it checks its records against the files and redoes "
                                 "only what no longer certifies")
    if verb == "check":
        parser.add_argument("--sample", type=int, default=None,
                            help="graphs to regenerate (disjoint, default 20) or detector runs to rerun (default 2)")
        parser.add_argument("--rows", type=int, default=10, help="rows replayed per graph (disjoint)")
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
    if verb == "check":
        return cmd_check(name, args.sample, args.rows)
    if not args.again:  # the verb continues unfinished runs; a completed one only says where its results are
        entry = load_entry(name)
        if effective_status(entry, read_progress(entry)) == "completed" and entry.get("kind") != "queue":
            print(f"  {entry['name']} is completed; its results are in {entry.get('output_dir', '?')}")
            print(style(f"  to check its records against the files and redo anything missing: "
                        f"hedonic run resume {entry['name']} --again", "dim"))
            return 0
    return cmd_resume(name, attach=not args.detach)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(dispatch(sys.argv[1:]))
