"""Durable, recoverable runs for ``hedonic run``.

A run launched by ``hedonic run disjoint`` (or a queue, ``hedonic run queue``) executes in a detached tmux session
(``hedonic-<name>``), or, without tmux, in a detached background process. The
terminal only shows a *viewer* that reads the run's ``progress.json``; closing
the terminal or pressing Ctrl-C closes the viewer, never the run. The worker
ignores SIGINT, so even Ctrl-C inside the tmux pane does not stop it; stopping
is explicit (``hedonic run stop NAME``). Every finished graph is a cached
record of the result store, so ``hedonic run resume NAME`` continues where an
interrupted run (reboot, kill) stopped.

    hedonic run list                 all runs with status and progress
    hedonic run status [NAME]        snapshot of one run (default: latest)
    hedonic run attach [NAME]        live viewer again
    hedonic run logs [NAME]          raw tmux pane (tmux attach) or log file
    hedonic run stop NAME            stop a run (explicit)
    hedonic run resume NAME          restart an interrupted/stopped run; finished work is reused
    hedonic run check [NAME]         morning check: replay a sample of the run's results
    hedonic run queue FILE|NAME      run a list of experiments one after another (see runqueue.py)

Two kinds of run share this machinery: the disjoint SBM benchmark (``hedonic run disjoint``) and queues of such
runs (``hedonic run queue``). On macOS every worker runs under ``caffeinate -i`` so the machine does not idle-sleep
overnight.
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
from hedonic.experiments.overlapping import common
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
        raise SystemExit("no runs yet — start one with `hedonic run disjoint`")
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
    return f"{prefix or 'run'}-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


# Environment forwarded into the detached worker. A tmux session inherits the tmux *server's* environment,
# not this shell's, so anything the run depends on (build SDK, config location, virtualenv) is passed on.
FORWARDED_ENV = ("HEDONIC_", "PATH", "CXX", "CC", "CFLAGS", "CPATH", "LIBRARY_PATH", "SDKROOT", "MACOSX_",
                 "HOME", "NO_COLOR", "XDG_", "TMPDIR", "LANG", "LC_", "VIRTUAL_ENV", "UV_", "OMP_", "TERM")


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


def launch_disjoint(settings: dict, name: str | None, output_dir: str, *, backend: str = "auto") -> dict:
    """Start the disjoint SBM benchmark (``hedonic run disjoint``) durably; the store is ``output_dir``."""
    from hedonic.experiments.disjoint import front

    name = common.parse_name(name or new_name("disjoint"))
    _refuse_if_running(name)
    output = str(expand_path(output_dir).resolve())
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    entry = {"name": name, "kind": "disjoint", "settings": settings, "argv": front.sweep_argv(settings, Path(output)),
             "config": {}, "output_dir": output, "created": time.time(),
             "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
             "log": str(Path(output) / "worker.log"),
             "command": "hedonic run disjoint " + " ".join(shlex.quote(a) for a in front.front_argv(settings))}
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

    A run whose workers each hold a large copy of its data can exhaust the machine in seconds (one such run once
    reached 261 GB attributed to the terminal and rebooted the Mac). Stopping at the floor leaves the finished
    records intact, so ``hedonic run resume`` continues once the cause is fixed.
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


def _work(name: str, entry: dict, prog: dict) -> int:
    kind = entry.get("kind")
    if kind == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.work(name, entry, prog)
    if kind == "queue":
        from hedonic.experiments.overlapping import runqueue

        return runqueue.work(name, entry, prog)
    raise SystemExit(_unsupported(entry))


def _unsupported(entry: dict) -> str:
    """Why a registered run of another kind (an overlap run of hedonic 2) cannot be handled here."""
    return (f"{entry.get('name')} is a {entry.get('kind') or 'benchmark'} run, which hedonic 1 does not have: "
            "hedonic 1 runs the disjoint SBM benchmark and queues of it; overlapping runs are hedonic 2")


# --------------------------------------------------------------------------- viewer
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
    """Estimated seconds left while the run is running, when its progress file publishes one; else None."""
    return prog.get("eta_seconds") if status in ("running", "setup") else None


def progress_block(entry: dict, prog: dict, status: str, frame: str, *, indent: str = "  ",
                   G: dict | None = None, now: float | None = None) -> list[str]:
    """What a run is doing: the bar with counts, time and ETA, what it says it is doing now, the last log line and
    the records. Used by the items of a queue."""
    G = G or tui.glyphs()
    live = status in LIVE
    elapsed = common.fmt_time(elapsed_seconds(entry, prog, status))
    done, total = progress_counts(entry, prog)
    lines: list[str] = []
    if not total:
        lines.append(indent + style(f"starting… elapsed {elapsed}", "dim"))
    else:
        share = f"{done}/{total} · {100 * done // total}%"
        if live:
            lines.append(f"{indent}{progress_bar(done, total, 30, G)}  {share}")
            left = remaining_seconds(entry, prog, status, now)
            tail = f"elapsed {elapsed}"
            if left is not None:
                tail += f" · ETA ~{common.fmt_time(left)}"
            lines.append(f"{indent}{tail}")
        else:
            lines.append(f"{indent}{progress_bar(done, total, 30, G)}  {share} · elapsed {elapsed}")
    if live and prog.get("stage"):
        lines.append(f"{indent}now      {prog['stage']}")  # what the run says it is doing (sweep, figures, statistics)
    log = last_log_line(entry.get("log"))
    if log:
        lines.append(f"{indent}log      {log}")
    if entry.get("output_dir") and status not in TERMINAL:  # a finished run prints its records in its own footer
        lines.append(f"{indent}records  {_home(entry['output_dir'])}")
    return [tui.fit(line, BLOCK_WIDTH) for line in lines]


def snapshot(entry: dict, frame: str | None = None, table: bool = True, unicode_glyphs: bool | None = None) -> str:
    prog = read_progress(entry)
    G = tui.glyphs(unicode_glyphs)
    frame = frame or G["dot"]
    if entry.get("kind") == "queue":
        from hedonic.experiments.overlapping import runqueue

        return runqueue.snapshot(entry, prog, frame, table, unicode_glyphs=unicode_glyphs)
    if entry.get("kind") == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.snapshot(entry, prog, effective_status(entry, prog), frame, table)
    return style(f"  hedonic · {entry['name']}", "bold") + "\n" + style(f"  {_unsupported(entry)}", "yellow")


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


def view(entry: dict) -> int:
    """Live viewer. Ctrl-C closes the viewer only. A queue opens as an interactive tree (``view_queue``) in a terminal."""
    if (entry.get("kind") == "queue" and sys.stdin.isatty() and sys.stdout.isatty() and tui.termios is not None
            and not os.environ.get("HEDONIC_PLAIN_VIEW")):
        return view_queue(entry)
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
    if entry.get("kind") != "disjoint":
        print(f"  run {status}: a {entry.get('kind')} run prints no JSON summary; see `hedonic run status {entry['name']}`",
              file=sys.stderr)
        return 1
    summary = Path(entry["output_dir"]) / "stats" / "paper_stats.json"
    if not summary.is_file():
        print(f"  run {status}: {prog.get('reason', 'no summary was written')}; see `hedonic run logs {entry['name']}`",
              file=sys.stderr)
        return 1
    print(json.dumps(json.loads(summary.read_text()), indent=2, default=str))
    return 0 if status == "completed" else 1


# --------------------------------------------------------------------------- commands
def progress_counts(entry: dict, prog: dict) -> tuple[int, int]:
    """Finished units and total units of a run: graphs for a disjoint run, completed items for a queue."""
    if entry.get("kind") == "disjoint":
        return int(prog.get("done", 0)), int(prog.get("graphs", 0))
    if entry.get("kind") == "queue":  # completed items of the queue, not graphs
        from hedonic.experiments.overlapping import runqueue

        return runqueue.counts(entry)
    return int(prog.get("done", 0)), int(prog.get("total", 0))


def cmd_list() -> int:
    entries = all_entries()
    if not entries:
        print("no runs yet — start one with `hedonic run disjoint`")
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
    if entry.get("kind") == "queue":  # finished items are skipped; the first unfinished one is resumed
        from hedonic.experiments.overlapping import runqueue

        entry = runqueue.launch(entry, backend=entry.get("backend", "auto"))
    elif entry.get("kind") == "disjoint":
        entry = launch_disjoint(entry["settings"], entry["name"], entry["output_dir"],
                                backend=entry.get("backend", "auto"))
    else:
        raise SystemExit(_unsupported(entry))
    print(style(f"  resumed {entry['name']}", "green") + " — finished work is reused from its cached records")
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


def cmd_check(name: str | None, sample: int | None, rows: int) -> int:
    entry = load_entry(name)
    if entry.get("kind") == "disjoint":
        from hedonic.experiments.disjoint import front

        return front.check(entry, shards=sample or 20, rows=rows)
    if entry.get("kind") == "queue":
        names = ", ".join(item["name"] for item in entry["items"])
        print(f"  a queue has no results of its own; check each run: {names}")
        return 0
    print(f"  {_unsupported(entry)}")
    return 1


VERBS = {"list": "runs and their progress", "status": "snapshot of a run (default: latest)",
         "attach": "live view of a running run (Ctrl-C closes only the view)",
         "logs": "raw output of a run (tmux pane, or the log file)", "stop": "stop a run (explicit; never automatic)",
         "resume": "continue an interrupted, stopped or failed run; finished work is reused",
         "check": "morning check: replay sampled results from their seeds and compare (default: latest run)",
         "queue": "run the experiments of a queue file one after another (hedonic run queue FILE|NAME)"}


def dispatch(argv: list[str]) -> int:
    """``hedonic run <verb> ...`` for the durable runs themselves (``VERBS``)."""
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
    if verb == "check":
        parser.add_argument("--sample", type=int, default=None,
                            help="graphs to regenerate (default 20)")
        parser.add_argument("--rows", type=int, default=10, help="rows replayed per graph")
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
    return cmd_resume(name, attach=not args.detach)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(dispatch(sys.argv[1:]))
