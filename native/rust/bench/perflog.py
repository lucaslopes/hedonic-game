"""A perflog of hedonic's cores: where time and memory go, C against Rust.

For each workload and core (the Rust module hedonic._core, and hedonic 2.0's
C core where it is built beside it: native/c, HEDONIC_CORE=c, the paper
branch) it records, in fresh child processes, one heavy job at a
time:

* time, split by phase (median and spread of --reps runs). In process, around
  Game.community_hedonic: the Game's own checks and start (game_pre), the
  argument buffers of hedonic._native (native_pre), _core.detect, the igraph
  result objects (native_post) and the rows and provenance (game_post).
  Inside detect: the start rows read by the binding (detect minus the same
  call with start=None, when that is the same start), the capsule borrow
  (vcount, ecount, __graph_as_capsule and the three es[e].tuple probes, timed
  from Python), and the core alone, run by a native driver on the very same
  inputs (perflog_core.c, examples/perflog_core.rs; same edges, weights,
  start and Mersenne Twister words, checked by the result's hash): the input
  check, the adjacency build and the whole core call (check + start +
  adjacency + run + slack + result). What is left of detect (nostart minus
  core call minus capsule) is the binding's own: generator, buffers, result
  lists. The audit is split the same way (_prepare in Python, _audit_prepare,
  _audit_intervals; the driver's prepare and intervals).
* instructions and cycles of the whole call (proc_pid_rusage, macOS).
* memory: ru_maxrss of a fresh child per measurement (os.wait4), the growth
  of the high-water mark and of the peak physical footprint across the one
  call (proc_pid_rusage's interval maximum), the Python objects the call
  allocates (tracemalloc peak and what stays after it, a child of its own),
  and the core's own heap from the drivers (live peak, bytes, count).

    python native/rust/bench/perflog.py [--reps 5] [--filter NAME] [--cores rust,c]
        [--work DIR] [--table FILE] [--list]
    python native/rust/bench/perflog.py --profile NAME [--seconds 20]   # macOS sample

Run it under the memory guard, alone on the machine as far as possible:
    bash .agents/skills/hedonic-native-core/scripts/guard.sh 16 python native/rust/bench/perflog.py

Without the C core (no native/c, or no hedonic._core_c) --cores defaults to rust
alone.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import random
import resource
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUST = HERE.parent
REPO = RUST.parents[1]
C_CORE = RUST.parent / "c"  # hedonic 2.0's C core: the paper branch only
PICKLES = Path(os.environ.get("HEDONIC_PICKLE_DIR", "~/Databases/Hedonic/Pickles")).expanduser()
# The calibrated resolutions of the paper queue (uncapped Hedonic, hedonic_tuned.json), read only.
TUNED = {"dblp": 0.07692307692307693, "amazon": 0.0001212268153715602, "youtube": 2.6950545748551407e-05}
CORES = ("rust", "c")


# ---- Workloads ----------------------------------------------------------------------

def density(g):
    n = g.vcount()
    return 2 * g.ecount() / (n * (n - 1))


def _ba50k():
    import igraph
    random.seed(1)
    return igraph.Graph.Barabasi(50_000, 3)


def _sbm20k():
    import igraph
    random.seed(2)
    blocks, size = 40, 500
    pref = [[0.01 if i == j else 0.0002 for j in range(blocks)] for i in range(blocks)]
    return igraph.Graph.SBM(pref, [size] * blocks)


def _snap(name):
    from hedonic.experiments.overlapping import snap_pickle
    return snap_pickle.load_game(PICKLES / f"{name}-top5000.game.pkl")


def _game(g):
    from hedonic import Game
    return g if type(g).__name__ == "Game" else Game(g)


def detection(graph, inner=1, **kwargs):
    return lambda: {"kind": "detect", "game": _game(graph()), "kwargs": kwargs, "inner": inner}


def _weights(count, lo, hi, seed):
    import numpy as np
    return np.random.default_rng(seed).uniform(lo, hi, count)


def weighted_sbm():
    g = _sbm20k()
    return {"kind": "detect", "game": _game(g), "inner": 1,
            "kwargs": {"edge_weights": _weights(g.ecount(), 0.1, 2.0, 3), "seed": 0}}


def node_weighted_sbm():
    g = _sbm20k()
    return {"kind": "detect", "game": _game(g), "inner": 1,
            "kwargs": {"node_weights": _weights(g.vcount(), 0.5, 2.0, 4), "seed": 0}}


def ba_cover(cap):
    def build():
        g = _ba50k()
        return {"kind": "detect", "game": _game(g), "inner": 1,
                "kwargs": {"max_memberships": cap, "resolution": 4 * density(g), "seed": 0}}
    return build


def audit(source, cap):
    """The audit of the cover `source` detects (computed once, cached in the work dir)."""
    def build():
        built = WORKLOADS[source]()
        return {"kind": "audit", "game": built["game"], "inner": 1, "source": source, "cap": cap}
    return build


WORKLOADS = {
    "zachary partition x100": detection(lambda: __import__("igraph").Graph.Famous("Zachary"), inner=100, seed=0),
    "zachary cover M=3 x100": detection(lambda: __import__("igraph").Graph.Famous("Zachary"), inner=100,
                                        max_memberships=3, resolution=0.1, seed=0),
    "BA 50k partition": detection(_ba50k, seed=0),
    "BA 50k cover M=2": ba_cover(2),
    "BA 50k cover M=5": ba_cover(5),
    "BA 50k cover M=-1": ba_cover(-1),
    "SBM 20k weighted partition": weighted_sbm,
    "SBM 20k node-weighted partition": node_weighted_sbm,
    "SBM 20k random start K=40": detection(_sbm20k, max_communities=40, seed=7),
    "DBLP top5000 partition": detection(lambda: _snap("dblp"), seed=0),
    "DBLP top5000 cover M=-1": detection(lambda: _snap("dblp"), max_memberships=-1,
                                         resolution=TUNED["dblp"], seed=0),
    "Amazon top5000 partition": detection(lambda: _snap("amazon"), seed=0),
    "Amazon top5000 cover M=-1": detection(lambda: _snap("amazon"), max_memberships=-1,
                                           resolution=TUNED["amazon"], seed=0),
    "YouTube top5000 partition": detection(lambda: _snap("youtube"), seed=0),
    "YouTube top5000 cover M=-1": detection(lambda: _snap("youtube"), max_memberships=-1,
                                            resolution=TUNED["youtube"], seed=0),
    "DBLP top5000 cover M=3": detection(lambda: _snap("dblp"), max_memberships=3,
                                        resolution=TUNED["dblp"], seed=0),
    "audit DBLP M=3 cover, capped": audit("DBLP top5000 cover M=3", None),
    "audit DBLP M=3 cover, uncapped": audit("DBLP top5000 cover M=3", -1),
    "audit DBLP M=-1 cover, uncapped": audit("DBLP top5000 cover M=-1", -1),
}


# ---- Measuring inside a child -------------------------------------------------------

class Usage:
    """proc_pid_rusage (RUSAGE_INFO_V4) of this process: footprint, its maxima, instructions."""

    def __init__(self):
        self.buf = (ctypes.c_uint64 * 64)()
        try:
            self.lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        except OSError:
            self.lib = None

    def read(self):
        if self.lib is None or self.lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(self.buf)) != 0:
            return {}
        v = list(self.buf)[2:]  # after the 16-byte uuid
        return {"footprint": v[7], "lifetime_max_footprint": v[28], "instructions": v[29], "cycles": v[30],
                "interval_max_footprint": v[33]}

    def instructions(self):
        if self.lib is None or self.lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(self.buf)) != 0:
            return 0
        return self.buf[2 + 29]

    def reset_interval(self):
        if self.lib is not None:
            self.lib.proc_reset_footprint_interval(os.getpid())


def maxrss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def fnv(values, h=0xCBF29CE484222325):
    for v in values:
        v &= 0xFFFFFFFFFFFFFFFF
        for b in range(8):
            h = ((h ^ ((v >> (8 * b)) & 0xFF)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def double_bits(x):
    return struct.unpack("<q", struct.pack("<d", x))[0]


def result_record(kind, game, result):
    if kind == "audit":
        count = sum(len(row) for row in result)
        return {"count": count, "hash": "%016x" % fnv(double_bits(x) for row in result for pair in row for x in pair)}
    rows = game.memberships
    quality = getattr(result, "_hedonic_native_quality", None)
    slack = result._hedonic_provenance["equilibrium_slack"]
    if result._hedonic_provenance["effective_max_memberships"] == 1:
        h = fnv(r[0] for r in rows)
    else:
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + len(r))
        h = fnv((label for r in rows for label in r), fnv(offsets))
    count = len({label for r in rows for label in r})
    return {"count": count, "quality": quality, "slack": slack, "hash": "%016x" % h}


def cover_path(work, source):
    return Path(work) / "covers" / (source.replace(" ", "_").replace("=", "") + ".json")


def setup(name, work):
    built = WORKLOADS[name]()
    if built["kind"] == "audit":
        with open(cover_path(work, built["source"])) as f:
            built["rows"] = json.load(f)
    return built


class Marks:
    """Wrappers that timestamp the layers of one detector or audit call."""

    def __init__(self, kind, usage=None):
        import hedonic._audit as audit_mod
        import hedonic._native as native
        from hedonic import _core
        self.t, self.i, self.usage = {}, {}, usage
        self.captured = {}
        self.native, self.core = native, _core
        if kind == "detect":
            orig_native, orig_detect = native.community_hedonic, native._detect

            def wrapped_native(*a, **k):
                self.mark("native_in")
                try:
                    return orig_native(*a, **k)
                finally:
                    self.mark("native_out")

            def wrapped_detect(graph, rng, res, **k):
                if not self.captured:
                    self.captured = {"rng_state": rng.getstate(), "resolution": res, **k}
                self.mark("detect_in")
                try:
                    out = orig_detect(graph, rng, res, **k)
                finally:
                    self.mark("detect_out")
                self.raw = out
                return out

            native.community_hedonic = wrapped_native
            native._detect = wrapped_detect
            self.call_detect = orig_detect
        else:
            orig_prepare, orig_intervals = _core._audit_prepare, _core._audit_intervals

            def wrapped_prepare(graph, offsets, labels, weights=None):
                if not self.captured:
                    self.captured = {"offsets": offsets, "labels": labels}
                self.mark("prepare_in")
                try:
                    return orig_prepare(graph, offsets, labels, weights)
                finally:
                    self.mark("prepare_out")

            def wrapped_intervals(state, selected, cap, *a):
                self.captured.setdefault("cap", cap)
                self.captured.setdefault("rest", a)
                self.mark("intervals_in")
                try:
                    return orig_intervals(state, selected, cap, *a)
                finally:
                    self.mark("intervals_out")

            audit_mod._core = type("Proxy", (), {"_audit_prepare": staticmethod(wrapped_prepare),
                                                 "_audit_intervals": staticmethod(wrapped_intervals)})


def _mark(self, name):
    """Timestamps a layer boundary, with the instruction count when counting (a syscall: big calls only)."""
    if self.usage is not None:
        self.i[name] = self.usage.instructions()
    self.t[name] = time.perf_counter()


Marks.mark = _mark


def run_call(built):
    game = built["game"]
    if built["kind"] == "audit":
        cap = built["cap"]
        return game.stable_intervals(built["rows"], max_memberships=game.vcount() if cap == -1 else cap)
    return game.community_hedonic(**built["kwargs"])


def capsule_probe(game):
    t = time.perf_counter()
    game.vcount(), game.ecount()
    capsule = getattr(game, "__graph_as_capsule")()
    m = game.ecount()
    for e in (0, m // 2, m - 1):
        game.es[e].tuple
    del capsule
    return time.perf_counter() - t


def child_time(name, work, reps):
    built = setup(name, work)
    usage = Usage()
    inner = built["inner"]
    # Instructions per layer cost a syscall per boundary: only where a call is long.
    marks = Marks(built["kind"], usage if inner == 1 else None)
    out = []
    warm = 1 if inner > 1 else 0
    for rep in range(warm + reps):
        gc.collect()
        phases = {}
        u0 = usage.read()
        t0 = time.perf_counter()
        for _ in range(inner):
            marks.t.clear()
            marks.i.clear()
            marks.mark("start")
            result = run_call(built)
            marks.mark("end")
            for prefix, t in (("", marks.t), ("i_", marks.i)):
                if not t:
                    continue
                if built["kind"] == "detect":
                    parts = {"game_pre": t["native_in"] - t["start"], "native_pre": t["detect_in"] - t["native_in"],
                             "detect": t["detect_out"] - t["detect_in"],
                             "native_post": t["native_out"] - t["detect_out"], "game_post": t["end"] - t["native_out"]}
                else:
                    parts = {"py_prepare": t["prepare_in"] - t["start"],
                             "audit_prepare": t["prepare_out"] - t["prepare_in"],
                             "between": t["intervals_in"] - t["prepare_out"],
                             "audit_intervals": t["intervals_out"] - t["intervals_in"],
                             "py_post": t["end"] - t["intervals_out"]}
                for k, v in parts.items():
                    phases[prefix + k] = phases.get(prefix + k, 0.0) + v
        total = time.perf_counter() - t0
        u1 = usage.read()
        record = {k: v / inner for k, v in phases.items()}
        record["total"] = total / inner
        if u0 and u1:
            record["instructions"] = (u1["instructions"] - u0["instructions"]) / inner
            record["cycles"] = (u1["cycles"] - u0["cycles"]) / inner
        if built["kind"] == "detect":
            c = marks.captured
            default_start = "max_communities" not in built["kwargs"]
            if default_start:
                # The same call with start=None: the core's own singletons, the same game.
                times, counts = [], []
                for _ in range(inner):
                    rng = random.Random()
                    rng.setstate(c["rng_state"])
                    kw = {k: v for k, v in c.items() if k not in ("rng_state", "resolution")}
                    kw["start"] = None
                    i = usage.instructions() if inner == 1 else 0
                    s = time.perf_counter()
                    raw = marks.call_detect(built["game"], rng, c["resolution"], **kw)
                    times.append(time.perf_counter() - s)
                    counts.append(usage.instructions() - i if inner == 1 else 0)
                if inner == 1:
                    record["i_detect_nostart"] = sum(counts)
                    if rep == 0 and repr(raw) != repr(marks.raw):
                        raise SystemExit(f"{name}: detect(start=None) differs from the Game's start")
                record["detect_nostart"] = sum(times) / inner
            record["capsule"] = statistics.median(capsule_probe(built["game"]) for _ in range(5))
        if rep >= warm:
            out.append(record)
    return {"reps": out}


def child_memory(name, work, mode):
    built = setup(name, work)
    usage = Usage()
    gc.collect()
    before = {"maxrss": maxrss(), **usage.read()}
    if mode == "pymem":
        import tracemalloc
        tracemalloc.start()
        result = run_call(built)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"py_alloc_peak": peak, "py_alloc_kept": current}
    usage.reset_interval()
    result = run_call(built)
    after = {"maxrss": maxrss(), **usage.read()}
    del result
    return {"maxrss_before": before["maxrss"], "maxrss_after": after["maxrss"],
            "footprint_before": before.get("footprint"),
            "interval_max_footprint": after.get("interval_max_footprint"),
            "footprint_after": after.get("footprint")}


def child_dump(name, work):
    """One call with the arguments captured: the drivers' input and the expected result."""
    import numpy as np
    built = WORKLOADS[name]()
    if built["kind"] == "audit":
        path = cover_path(work, built["source"])
        if not path.exists():
            src = WORKLOADS[built["source"]]()
            run_call(src)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(src["game"].memberships, f)
        with open(path) as f:
            built["rows"] = json.load(f)
    marks = Marks(built["kind"])
    result = run_call(built)
    expected = result_record(built["kind"], built["game"], result)
    game, c = built["game"], marks.captured
    n, m = game.vcount(), game.ecount()
    edges = np.asarray(game.get_edgelist(), dtype=np.int64).reshape(-1)
    start_kind, start_arrays, start_len = 0, [], 0
    if built["kind"] == "detect":
        mode = 0 if c["max_memberships"] == 1 else 1
        cap = c["max_memberships"]
        ew, nw = c.get("edge_weights"), c.get("node_weights")
        start = c.get("start")
        if start is not None:
            if mode == 0:
                start_kind, start_arrays = 1, [np.asarray(start, dtype=np.int64)]
            else:
                offsets = np.cumsum([0] + [len(r) for r in start]).astype(np.int64)
                labels = np.fromiter((x for r in start for x in r), dtype=np.int64, count=int(offsets[-1]))
                start_kind, start_arrays, start_len = 2, [offsets, labels], int(offsets[-1])
        words = list(c["rng_state"][1])
        gamma, atol, rtol, isolation, prune = c["resolution"], 0.0, 0.0, bool(c.get("allow_isolation")), 1
        max_total = c.get("max_total_communities") or -1
        exact = c.get("n_communities") or -1
    else:
        mode, ew, nw = 2, None, None
        offsets = np.asarray(c["offsets"], dtype=np.int64)
        labels = np.asarray(c["labels"], dtype=np.int64)
        start_kind, start_arrays, start_len = 2, [offsets, labels], len(labels)
        cap = c["cap"]
        isolation, atol, rtol, prune = (bool(c["rest"][0]), c["rest"][1], c["rest"][2], int(bool(c["rest"][3])))
        words, gamma, max_total, exact = [0] * 625, 0.0, -1, -1
    header = [mode, n, m, cap, ew is not None, nw is not None, start_kind, isolation, max_total, exact,
              1, prune, start_len, 0, 0, 0]
    path = Path(work) / "inputs" / (name.replace(" ", "_").replace("=", "").replace(",", "") + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"HPLOG001")
        f.write(np.asarray(header, dtype=np.int64).tobytes())
        f.write(np.asarray([gamma, atol, rtol, 0.0], dtype=np.float64).tobytes())
        f.write(np.asarray(words, dtype=np.int64).tobytes())
        f.write(edges.tobytes())
        for w in (ew, nw):
            if w is not None:
                f.write(np.ascontiguousarray(w, dtype=np.float64).tobytes())
        for a in start_arrays:
            f.write(a.tobytes())
    return {"input": str(path), "expected": expected, "n": n, "m": m, "inner": built["inner"]}


def child_loop(name, work, seconds):
    built = setup(name, work)
    end = time.perf_counter() + seconds
    calls = 0
    while time.perf_counter() < end:
        run_call(built)
        calls += 1
    return {"calls": calls}


# ---- The parent ---------------------------------------------------------------------

def env_for(core):
    env = dict(os.environ)
    env.pop("HEDONIC_CORE", None)
    if core == "c":
        env["HEDONIC_CORE"] = "c"
    for knob in [k for k in env if k.startswith("HIG_")]:
        env.pop(knob)  # debug knobs count as set when present
    return env


def child(core, mode, name, work, extra=()):
    """Runs a child; returns (its JSON, its ru_maxrss in bytes)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode, "--workload", name, "--work", str(work),
           *extra]
    p = subprocess.Popen(cmd, env=env_for(core), stdout=subprocess.PIPE, text=True)
    out = p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)
    p.stdout.close()
    if status != 0:
        raise SystemExit(f"child {mode} {core} {name!r} failed (status {status})")
    rss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return json.loads(out.strip().splitlines()[-1]), rss


def c_available():
    """Whether hedonic 2.0's C core is here: its sources (native/c) and its build (hedonic._core_c)."""
    if not (C_CORE / "unity.c").is_file():
        return False
    probe = subprocess.run([sys.executable, "-c", "import hedonic._core_c"], capture_output=True)
    return probe.returncode == 0


def drivers(work, cores):
    found = {}
    if "c" in cores:
        c_bin = Path(work) / "perflog_core_c"
        src = HERE / "perflog_core.c"
        if not c_bin.exists() or c_bin.stat().st_mtime < max(src.stat().st_mtime, newest(C_CORE)):
            subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O3", "-DNDEBUG", "-ffp-contract=off",
                            "-fno-wrapv", f"-I{C_CORE / 'include'}", f"-I{C_CORE / 'core'}", str(src), "-o",
                            str(c_bin)], check=True)
        found["c"] = c_bin
    subprocess.run(["cargo", "build", "--release", "--quiet", "--example", "perflog_core"], cwd=RUST, check=True,
                   env={**os.environ, "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", "4")})
    found["rust"] = RUST / "target/release/examples/perflog_core"
    return found


def newest(folder):
    return max(p.stat().st_mtime for p in folder.rglob("*") if p.suffix in (".c", ".h"))


def run_driver(binary, path, reps):
    with open(path, "r+b") as f:
        f.seek(8 + 10 * 8)
        f.write(struct.pack("<q", reps))
    out = subprocess.run([str(binary), str(path)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def spread(xs):
    return (max(xs) - min(xs)) / statistics.median(xs) if xs and statistics.median(xs) else float("nan")


def summarize(record):
    reps = record["time"]["reps"]
    keys = sorted({k for r in reps for k in r})
    s = {k: {"median": med([r[k] for r in reps if k in r]), "min": min(r[k] for r in reps if k in r),
             "max": max(r[k] for r in reps if k in r)} for k in keys}
    d = record.get("driver")
    if d:
        for k in ("check_s", "adjacency_s", "prepare_s", "call_s", "call_instructions"):
            if d.get(k):
                key = "core_" + (k[:-2] if k.endswith("_s") else k)
                s[key] = {"median": med(d[k]), "min": min(d[k]), "max": max(d[k])}
    return s


def ms(x):
    if x != x:
        return "-"
    return f"{x * 1e3:.3f}" if x < 0.01 else f"{x * 1e3:.1f}" if x < 10 else f"{x * 1e3:,.0f}"


def mb(x):
    return "-" if x is None else f"{x / 2**20:.1f}"


def table(results, meta):
    lines = [f"# perflog baseline", "", meta, ""]
    lines += ["## Time (ms per call: median of the reps; spread = (max - min) / median of the whole call)", "",
              "Columns: total = Game call; game/native pre and post = the Python layers; detect = _core.detect; "
              "start read = detect - detect(start=None); capsule = vcount/ecount/capsule/3 probes from Python; "
              "core = the driver's whole core call on the same input (check and adjacency are its first steps, "
              "timed apart); binding = detect(start=None) - core - capsule (generator, buffers, result lists). "
              "Audit rows: game pre = _prepare in Python (normalize_membership, buffers), detect = _audit_prepare + "
              "_audit_intervals, core = driver prepare + intervals, of which the core adjacency column holds the "
              "prepare and the run column the intervals; binding = in-process intervals - driver intervals (the "
              "result lists); game post = after the intervals.", "",
              "| workload | core | total | spread | game pre | native pre | detect | start read | capsule | binding |"
              " core | core check | core adjacency | run (core - check - adj) | native post | game post | Ginstr |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, per_core in results.items():
        for core, rec in per_core.items():
            s = summarize(rec)
            g = lambda k: s.get(k, {}).get("median", float("nan"))
            spread_total = (s["total"]["max"] - s["total"]["min"]) / s["total"]["median"]
            ins = g("instructions")
            if "detect" in s:
                start_read = g("detect") - g("detect_nostart") if "detect_nostart" in s else float("nan")
                core_call = g("core_call")
                binding = (g("detect_nostart") if "detect_nostart" in s else g("detect")) - core_call - g("capsule")
                run = core_call - g("core_check") - g("core_adjacency")
                row = [g("total"), spread_total, g("game_pre"), g("native_pre"), g("detect"), start_read, g("capsule"),
                       binding, core_call, g("core_check"), g("core_adjacency"), run, g("native_post"), g("game_post")]
            else:
                core_both = g("core_prepare") + g("core_call")
                row = [g("total"), spread_total, g("py_prepare"), float("nan"), g("audit_prepare") + g("audit_intervals"),
                       float("nan"), float("nan"), g("audit_intervals") - g("core_call"), core_both,
                       float("nan"), g("core_prepare"), g("core_call"), float("nan"), g("py_post")]
            cells = [ms(x) if i != 1 else f"{x * 100:.0f}%" for i, x in enumerate(row)]
            cells.append(f"{ins / 1e9:.2f}" if ins == ins else "-")
            lines.append(f"| {name} | {core} | " + " | ".join(cells) + " |")
    lines += ["", "## Instructions (millions per call, medians; single-call workloads only, so x100 rows are blank)", "",
              "The same layers counted in retired instructions (proc_pid_rusage), which a loaded machine does not "
              "blur. core = the driver's whole call (for the audit, intervals only); binding = detect(start=None) - "
              "core (audit: in-process intervals - driver intervals).", "",
              "| workload | core | total | game pre | native pre | detect | start read | binding | core | native post |"
              " game post |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, per_core in results.items():
        for core, rec in per_core.items():
            s = summarize(rec)
            g = lambda k: s.get(k, {}).get("median", float("nan"))
            if "i_detect" in s:
                row = [g("instructions"), g("i_game_pre"), g("i_native_pre"), g("i_detect"),
                       g("i_detect") - g("i_detect_nostart"), g("i_detect_nostart") - g("core_call_instructions"),
                       g("core_call_instructions"), g("i_native_post"), g("i_game_post")]
            elif "i_audit_intervals" in s:
                row = [g("instructions"), g("i_py_prepare"), float("nan"), g("i_audit_prepare") + g("i_audit_intervals"),
                       float("nan"), g("i_audit_intervals") - g("core_call_instructions"), g("core_call_instructions"),
                       float("nan"), g("i_py_post")]
            else:
                continue
            cells = ["-" if x != x else f"{x / 1e6:,.1f}" for x in row]
            lines.append(f"| {name} | {core} | " + " | ".join(cells) + " |")
    lines += ["", "## Memory (MB; one fresh child per measurement, median of the memory reps)", "",
              "child peak = ru_maxrss of the child (wait4); before call = its high-water mark when the call starts; "
              "call +maxrss = growth of the high-water mark across the call; call peak footprint = the peak physical "
              "footprint during the call minus the footprint before it (proc_pid_rusage interval maximum); "
              "Python objects = tracemalloc peak during the call / what is still allocated after it; core heap = the "
              "driver's live peak above its inputs, bytes and count of allocations (one call).", "",
              "| workload | core | child peak | before call | call +maxrss | call peak footprint | Python objects peak / kept |"
              " core heap peak | core bytes allocated | core allocations | adjacency heap |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, per_core in results.items():
        for core, rec in per_core.items():
            mem = rec.get("memory", [])
            if not mem:
                continue
            peak = med([x["child_maxrss"] for x in mem])
            before = med([x["maxrss_before"] for x in mem])
            grow = med([x["maxrss_after"] - x["maxrss_before"] for x in mem])
            fp = [x["interval_max_footprint"] - x["footprint_before"] for x in mem if x.get("footprint_before")]
            py = rec.get("pymem") or {}
            d = (rec.get("driver") or {}).get("alloc_call") or {}
            da = (rec.get("driver") or {}).get("alloc_adjacency") or {}
            lines.append(f"| {name} | {core} | {mb(peak)} | {mb(before)} | {mb(grow)} | {mb(med(fp)) if fp else '-'} | "
                         f"{mb(py.get('py_alloc_peak'))} / {mb(py.get('py_alloc_kept'))} | {mb(d.get('peak_live_bytes'))} | "
                         f"{mb(d.get('total_bytes'))} | {d.get('count', '-'):,} | {mb(da.get('peak_live_bytes'))} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--memory-reps", type=int, default=3)
    p.add_argument("--filter", action="append", default=[], help="substring of a workload name (repeatable)")
    p.add_argument("--cores", default=None, help="rust,c (default: both where the C core is built, else rust)")
    p.add_argument("--work", default=os.environ.get("HEDONIC_PERFLOG_WORK",
                                                    os.path.join(os.environ.get("TMPDIR", "/tmp"), "hedonic-perflog")))
    p.add_argument("--table", help="write the markdown table here")
    p.add_argument("--json", help="write every measurement here")
    p.add_argument("--list", action="store_true")
    p.add_argument("--from-json", help="render the table of an earlier --json run")
    p.add_argument("--profile", help="sample a loop of this workload with macOS sample, per core")
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--profiles", default=str(HERE / "profiles"))
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--workload", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.child:
        out = {"time": lambda: child_time(a.workload, a.work, a.reps),
               "memory": lambda: child_memory(a.workload, a.work, "rss"),
               "pymem": lambda: child_memory(a.workload, a.work, "pymem"),
               "dump": lambda: child_dump(a.workload, a.work),
               "loop": lambda: child_loop(a.workload, a.work, a.seconds)}[a.child]()
        print(json.dumps(out))
        return
    if a.from_json:
        saved = json.loads(Path(a.from_json).read_text())
        text = table(saved["results"], saved["meta"])
        if a.table:
            Path(a.table).write_text(text)
        print(text)
        return
    if a.list:
        print("\n".join(WORKLOADS))
        return
    cores = [c for c in a.cores.split(",") if c] if a.cores else ["rust", "c"] if c_available() else ["rust"]
    names = [n for n in WORKLOADS if not a.filter or any(f in n for f in a.filter)]
    if a.profile:
        return profile(a, cores)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    binaries = drivers(work, cores)
    results = {}
    meta = (f"{platform.platform()}, {platform.machine()}, Python {platform.python_version()}; load average at start "
            f"{os.getloadavg()[0]:.1f}; reps {a.reps} (time), {a.memory_reps} (memory); "
            f"{time.strftime('%Y-%m-%d %H:%M')}")
    for name in names:
        dump, _ = child("rust", "dump", name, work)
        results[name] = {}
        for core in cores:
            rec = {}
            inner = dump["inner"]
            driver = run_driver(binaries[core], dump["input"], a.reps * inner + (1 if inner > 1 else 0))
            if inner > 1:  # the first call is cold
                for k in ("check_s", "adjacency_s", "prepare_s", "call_s", "call_instructions"):
                    driver[k] = driver.get(k, [])[1:]
            got = {k: driver["result"].get(k) for k in dump["expected"]}
            if any(got[k] != dump["expected"][k] for k in ("count", "hash")) or \
                    ("quality" in got and got["quality"] != dump["expected"]["quality"]):
                raise SystemExit(f"{name} {core}: the driver's result {got} differs from the call's {dump['expected']}")
            rec["driver"] = driver
            rec["time"], _ = child(core, "time", name, work, ["--reps", str(a.reps)])
            rec["memory"] = []
            for _ in range(a.memory_reps):
                m, rss = child(core, "memory", name, work)
                m["child_maxrss"] = rss
                rec["memory"].append(m)
            rec["pymem"], _ = child(core, "pymem", name, work)
            results[name][core] = rec
            s = summarize(rec)
            print(f"{name:34s} {core:4s} total {ms(s['total']['median']):>9s} ms  core {ms(s['core_call']['median']):>9s}"
                  f" ms  peak {mb(med([x['child_maxrss'] for x in rec['memory']]))} MB", flush=True)
    text = table(results, meta)
    if a.table:
        Path(a.table).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(text)


def profile(a, cores):
    """Samples a loop of one workload per core: macOS `sample`, the call tree as text."""
    name = next(n for n in WORKLOADS if a.profile in n)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if WORKLOADS[name]().get("kind") == "audit":
        child("rust", "dump", name, work)
    out = Path(a.profiles)
    out.mkdir(parents=True, exist_ok=True)
    slug = name.replace(" ", "_").replace("=", "").replace(",", "")
    for core in cores:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "loop", "--workload", name, "--work", str(work),
               "--seconds", str(a.seconds + 15)]
        proc = subprocess.Popen(cmd, env=env_for(core), stdout=subprocess.DEVNULL)
        time.sleep(8)  # imports and the workload's graph
        target = out / f"{slug}.{core}.sample.txt"
        subprocess.run(["sample", str(proc.pid), str(a.seconds), "1", "-mayDie", "-file", str(target)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()
        summary = out / f"{slug}.{core}.top.txt"
        summary.write_text(top_of_stack(target))
        print(target, summary)


def top_of_stack(path, lines=40):
    """The 'Sort by top of stack' table of a sample report: the functions where the samples landed."""
    text = Path(path).read_text(errors="replace").splitlines()
    try:
        first = next(i for i, line in enumerate(text) if line.startswith("Sort by top of stack"))
    except StopIteration:
        return "no samples\n"
    rows = [line for line in text[first + 1:] if line.strip()][:lines]
    total = sum(int(line.split()[-1]) for line in rows if line.split()[-1].isdigit())
    head = [f"{Path(path).name}: top {len(rows)} functions by samples at the top of the stack "
            f"(macOS sample, 1 ms interval; {total} samples listed)", ""]
    return "\n".join(head + rows) + "\n"


if __name__ == "__main__":
    main()
