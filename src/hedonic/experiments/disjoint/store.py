"""Columnar result store for the disjoint SBM sweep.

A store is a directory::

    STORE/
      manifest.json                         # schema, spec, grid, runtime
      runs/{k}C_{size}N/p_in-{p}/difficulty-{d}/network-{seed}.parquet

One Parquet shard holds every detector run on one generated graph, written
atomically and stamped (in the Parquet key-value metadata) with the run
specification hash, the graph fingerprint and its row count.  A shard either
exists complete or not at all, so ``--resume`` and parallel workers need no
coordination.

Graphs are **not** stored: ``sbm_sweep.generate_graph`` is a pure function of
``(n_communities, community_size, p_in, difficulty, network_seed)`` and each
row records ``graph_sha256`` so a regenerated graph can be checked.  Every
stochastic detector call runs under an explicit ``run_seed`` derived from the
row's keys, so on the same hedonic build any row can be replayed
(``hedonic-exp disjoint-store STORE --verify``).

Partitions are canonicalized (labels renumbered in order of first appearance)
and stored as ``list<uint16>`` (``uint32`` above 65,535 vertices) with zstd,
about 260 bytes for a 1,020-vertex partition instead of ~3 KB of JSON.  With
``partitions="hash"`` only a 128-bit ``partition_hash`` is kept, for
exploratory sweeps that can replay the partition when needed.  A restart
that repeats a partition already returned in the same cell
(``is_duplicate``) stores only its hash in either mode.

Query with DuckDB (``open_store``) or pandas/pyarrow (``load_runs``,
``plot_table``); ``plot_table`` returns the column layout the paper figures
read.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

STORE_SCHEMA = "hedonic-disjoint-store-v2"
# What the ``duration`` column measures: the detector call only (seeding of the generator, the
# membership copy, the error fallback and every metric are outside the clock). It is part of the
# spec, so rows timed under another rule are never mixed into one store.
TIMING_SCOPE = "detector-call-v1"
MANIFEST_NAME = "manifest.json"
RUNS_DIR = "runs"
_SHARD_METADATA_KEY = b"hedonic.disjoint_store"
PARTITION_MODES = ("full", "hash")

# Column order of a shard; partition is last because it dominates the bytes.
KEY_COLUMNS = (
    "number_of_communities",
    "community_size",
    "p_in",
    "difficulty",
    "network_seed",
    "method",
    "noise",
    "partition_seed",
    "run",
)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - experiments extra installs it
        raise SystemExit(
            "The disjoint result store needs pyarrow: "
            'pip install "hedonic[experiments]"'
        ) from exc
    return pa, pq


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - experiments extra installs it
        raise SystemExit(
            "SQL queries need duckdb: pip install \"hedonic[experiments]\""
        ) from exc
    return duckdb


# ---------------------------------------------------------------------------
# Keys, seeds and partition encoding
# ---------------------------------------------------------------------------


def decimal_text(value) -> str:
    """Shortest exact decimal text of a parameter (``0.05`` → ``"0.05"``)."""
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    return format(Decimal(repr(float(value))).normalize(), "f")


def shard_relative_path(
    number_of_communities: int,
    community_size: int,
    p_in,
    difficulty,
    network_seed: int,
) -> Path:
    return (
        Path(RUNS_DIR)
        / f"{int(number_of_communities)}C_{int(community_size)}N"
        / f"p_in-{decimal_text(p_in)}"
        / f"difficulty-{decimal_text(difficulty)}"
        / f"network-{int(network_seed):03d}.parquet"
    )


def run_seed(
    *,
    number_of_communities: int,
    community_size: int,
    p_in,
    difficulty,
    network_seed: int,
    method: str,
    noise,
    partition_seed,
    run: int,
) -> int:
    """Deterministic 63-bit detector seed for one row of the store."""
    key = json.dumps(
        [
            STORE_SCHEMA,
            int(number_of_communities),
            int(community_size),
            decimal_text(p_in),
            decimal_text(difficulty),
            int(network_seed),
            str(method),
            decimal_text(noise),
            None if partition_seed is None else int(partition_seed),
            int(run),
        ],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def canonical_partition(membership: Iterable[int]) -> np.ndarray:
    """Relabel communities in order of first appearance (0, 1, 2, ...)."""
    labels: dict[int, int] = {}
    out = [labels.setdefault(int(label), len(labels)) for label in membership]
    dtype = np.uint16 if len(out) <= 65_536 else np.uint32
    return np.asarray(out, dtype=dtype)


def partition_hash(canonical: np.ndarray) -> bytes:
    """128-bit identity of a canonical partition (independent of dtype)."""
    data = np.asarray(canonical, dtype="<u4").tobytes()
    return hashlib.sha256(data).digest()[:16]


# ---------------------------------------------------------------------------
# Specification / manifest
# ---------------------------------------------------------------------------


_EXTENSION_DIGESTS: dict[str, str] = {}


def _file_sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    if path not in _EXTENSION_DIGESTS:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _EXTENSION_DIGESTS[path] = digest.hexdigest()
    return _EXTENSION_DIGESTS[path]


def runtime_identity() -> dict[str, str | None]:
    """Versions that determine detector trajectories (resume must match).

    The hedonic release, the SHA-256 of its compiled core and of the loaded
    igraph extension (full Leiden, Spectral and graph building run there), and
    python-igraph's version. Paths are left out so the identity, and with it
    the spec hash, is the same on every machine that runs the same binaries;
    ``invocation_environment`` records where they were.
    """
    import igraph

    from hedonic.experiments.runtime import runtime_versions

    versions = runtime_versions()
    identity = {
        key: versions.get(key)
        for key in ("hedonic", "hedonic_core", "hedonic_core_sha256", "python_igraph", "igraph_c")
    }
    extension = versions.get("igraph_extension_path") or getattr(
        getattr(igraph, "_igraph", None), "__file__", None
    )
    identity["igraph_extension_sha256"] = _file_sha256(extension)
    return identity


def invocation_environment(workers: int) -> dict[str, Any]:
    """Context of one sweep invocation (recorded, not part of the spec)."""
    import platform
    import sys

    threads = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "started": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "workers": int(workers),
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "thread_controls": threads,
        "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
        "runtime_paths": _runtime_paths(),
    }


def _runtime_paths() -> dict[str, str | None]:
    from hedonic.experiments.runtime import runtime_versions

    versions = runtime_versions()
    return {key: versions.get(key) for key in versions if key.endswith("_path")}


def spec_hash(spec: dict[str, Any]) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_spec(
    *,
    methods: dict[str, dict],
    n_runs: int,
    partitions: str,
    generator_id: str,
) -> dict[str, Any]:
    """Everything that changes a row's content, apart from its grid keys."""
    if partitions not in PARTITION_MODES:
        raise ValueError(f"partitions must be one of {PARTITION_MODES}")
    return {
        "schema": STORE_SCHEMA,
        "graph_generator": generator_id,
        "methods": {
            name: {
                "method_call_name": info["method_call_name"],
                "parameters": {
                    key: value
                    for key, value in info["parameters"].items()
                    # Filled per graph/cell at run time.
                    if key not in {"initial_membership", "groundtruth", "clusters"}
                },
            }
            for name, info in sorted(methods.items())
        },
        "n_runs": int(n_runs),
        "partitions": partitions,
        "timing_scope": TIMING_SCOPE,
        "runtime": runtime_identity(),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def read_manifest(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_store(
    root: str | Path,
    spec: dict[str, Any],
    grid: dict[str, Any],
    invocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or reopen a store; refuse to mix rows from a different spec."""
    root = Path(root)
    digest = spec_hash(spec)
    manifest = read_manifest(root)
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    if manifest is None:
        manifest = {
            "schema": STORE_SCHEMA,
            "spec_hash": digest,
            "spec": spec,
            "grids": [],
            "created": now,
        }
    elif manifest.get("spec_hash") != digest:
        raise ValueError(
            f"{root} holds results of a different run specification "
            f"(spec {manifest.get('spec_hash', '?')[:12]} ≠ {digest[:12]}: "
            "methods, n_runs, partition mode, graph generator or runtime "
            "versions changed).  Use a new output folder."
        )
    if grid not in manifest["grids"]:
        manifest["grids"].append(grid)
    if invocation is not None:
        manifest.setdefault("invocations", []).append(dict(invocation, grid=grid))
    manifest["updated"] = now
    _atomic_write_bytes(
        root / MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


LOCK_NAME = ".writer.lock"


def acquire_writer_lock(root: str | Path):
    """Hold an exclusive lock on the store for this process's lifetime.

    Refuses a second concurrent sweep into the same store.  The lock is
    released by the OS when the process exits, however it exits, so a killed
    run never leaves a stale lock.  Once the lock is held no other writer can
    exist, so leftover temporary shard files of a killed run are removed.
    Returns the open lock file (keep a reference) and the number of temporary
    files removed.
    """
    import fcntl

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handle = open(root / LOCK_NAME, "a+")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "another process"
        handle.close()
        raise ValueError(
            f"{root} is being written by {owner}; refusing a second concurrent run"
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid {os.getpid()}\n")
    handle.flush()
    removed = 0
    for stale in (root / RUNS_DIR).glob("*/*/*/.*.tmp"):
        stale.unlink(missing_ok=True)
        removed += 1
    return handle, removed


def writer_lock_holder(root: str | Path) -> str | None:
    """Text of the lock file when a live process holds it, else ``None``."""
    import fcntl

    path = Path(root) / LOCK_NAME
    if not path.is_file():
        return None
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            return handle.read().strip() or "unknown"
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return None


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def _schema(pa, n_vertices: int, partitions: str):
    label_type = pa.uint16() if n_vertices <= 65_536 else pa.uint32()
    fields = [
        pa.field("number_of_communities", pa.int16()),
        pa.field("community_size", pa.int32()),
        pa.field("p_in", pa.float64()),
        pa.field("difficulty", pa.float64()),
        pa.field("p_out", pa.float64()),
        pa.field("network_seed", pa.int32()),
        pa.field("graph_sha256", pa.string()),
        pa.field("n_vertices", pa.int32()),
        pa.field("n_edges", pa.int64()),
        pa.field("method", pa.string()),
        pa.field("noise", pa.float64()),
        pa.field("partition_seed", pa.int32()),
        pa.field("run", pa.int16()),
        pa.field("run_seed", pa.int64()),
        pa.field("resolution", pa.float64()),
        pa.field("status", pa.string()),
        pa.field("error", pa.string()),
        pa.field("duration", pa.float64()),
        pa.field("rand_index", pa.float64()),
        pa.field("adjusted_rand", pa.float64()),
        pa.field("overlap", pa.float64()),
        pa.field("overlap_normalized", pa.float64()),
        pa.field("nmi", pa.float64()),
        pa.field("vi", pa.float64()),
        pa.field("robustness", pa.float64()),
        pa.field("n_found_communities", pa.int32()),
        pa.field("is_duplicate", pa.bool_()),
        pa.field("partition_hash", pa.binary(16)),
    ]
    if partitions == "full":
        fields.append(pa.field("partition", pa.list_(label_type)))
    return pa.schema(fields)


def write_shard(
    root: str | Path,
    *,
    rows: list[dict[str, Any]],
    shard_info: dict[str, Any],
    spec_digest: str,
    partitions: str,
) -> Path:
    """Write one graph's rows atomically; returns the shard path."""
    pa, pq = _require_pyarrow()
    root = Path(root)
    path = root / shard_relative_path(
        shard_info["number_of_communities"],
        shard_info["community_size"],
        shard_info["p_in"],
        shard_info["difficulty"],
        shard_info["network_seed"],
    )
    schema = _schema(pa, int(shard_info["n_vertices"]), partitions)
    columns = {}
    for field in schema:
        if field.name == "partition":
            columns["partition"] = pa.array(
                [row["partition"] for row in rows], type=field.type
            )
        else:
            columns[field.name] = pa.array(
                [row.get(field.name) for row in rows], type=field.type
            )
    metadata = dict(shard_info, spec_hash=spec_digest, n_rows=len(rows), complete=True)
    table = pa.table(columns, schema=schema).replace_schema_metadata(
        {_SHARD_METADATA_KEY: json.dumps(metadata, sort_keys=True).encode("utf-8")}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            use_dictionary=["graph_sha256", "method", "status", "error"],
        )
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return path


def shard_metadata(path: str | Path) -> dict[str, Any] | None:
    """Return a shard's store metadata, or ``None`` if unreadable/foreign."""
    _, pq = _require_pyarrow()
    try:
        meta = pq.read_metadata(str(path)).metadata or {}
    except (OSError, ValueError):
        return None
    raw = meta.get(_SHARD_METADATA_KEY)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def shard_is_complete(
    root: str | Path,
    *,
    number_of_communities: int,
    community_size: int,
    p_in,
    difficulty,
    network_seed: int,
    spec_digest: str,
    noises: Iterable[float],
    partition_seeds: Iterable[int],
) -> bool:
    """True when the graph's shard exists for this spec and covers the cells."""
    path = Path(root) / shard_relative_path(
        number_of_communities, community_size, p_in, difficulty, network_seed
    )
    if not path.is_file():
        return False
    meta = shard_metadata(path)
    if not meta or not meta.get("complete") or meta.get("spec_hash") != spec_digest:
        return False
    wanted_noises = {decimal_text(n) for n in noises}
    wanted_seeds = {int(s) for s in partition_seeds}
    return wanted_noises <= {decimal_text(n) for n in meta.get("noises", [])} and (
        wanted_seeds <= {int(s) for s in meta.get("partition_seeds", [])}
    )


def shard_paths(root: str | Path) -> list[Path]:
    return sorted((Path(root) / RUNS_DIR).glob("*/*/*/*.parquet"))


# ---------------------------------------------------------------------------
# Reading / querying
# ---------------------------------------------------------------------------


def load_runs(
    root: str | Path,
    *,
    columns: list[str] | None = None,
    with_partitions: bool = False,
):
    """Load every row of a store as a pandas DataFrame (no DuckDB needed)."""
    _require_pyarrow()
    import pyarrow.dataset as ds

    paths = shard_paths(root)
    if not paths:
        raise FileNotFoundError(f"No result shards under {Path(root) / RUNS_DIR}")
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    names = dataset.schema.names
    if columns is None:
        columns = [n for n in names if with_partitions or n != "partition"]
    return dataset.to_table(columns=columns).to_pandas()


def plot_table(root: str | Path, *, unique_only: bool = True):
    """Logical-cell table in the column layout read by ``hedonic-exp plots``.

    Aggregated in DuckDB, so the full grid (tens of millions of runs) never
    has to fit in memory: one row per logical cell (method, K, p, difficulty,
    network seed, noise, partition seed) holding the mean of its successful
    runs.  With ``unique_only`` (default, the manuscript's estimand) a restart
    that repeats a partition already returned in the same cell is dropped
    before averaging, as the archived pipeline did; ``unique_only=False``
    averages every restart.  ``outputs`` counts the averaged runs.

    ``accuracy`` is the Rand index and ``adjusted_rand`` the ARI, as in the
    archived ``resultados_ari`` table; ``multiplier`` is the difficulty.
    Methods that ignore the initial partition (GroundTruth, Spectral) run once
    per graph and are repeated across the store's noise levels so that every
    noise bar compares the same graphs.
    """
    con = open_store(root)
    duplicate_filter = "AND NOT is_duplicate" if unique_only else ""
    extra = "".join(f"avg({c}) AS {c},\n" for c in _optional_metrics(con))
    table = con.execute(
        f"""
        WITH cells AS (
            SELECT method, number_of_communities, community_size, p_in, p_out,
                   difficulty AS multiplier, network_seed, noise, partition_seed,
                   any_value(resolution) AS resolution,
                   avg(duration) AS duration,
                   avg(rand_index) AS accuracy,
                   avg(adjusted_rand) AS adjusted_rand,
                   {extra}
                   avg(robustness) AS robustness,
                   count(*) AS outputs
            FROM runs
            WHERE status = 'ok' {duplicate_filter}
            GROUP BY ALL
        ),
        noises AS (SELECT DISTINCT noise FROM runs WHERE noise IS NOT NULL)
        SELECT * FROM cells WHERE noise IS NOT NULL
        UNION ALL
        SELECT c.* REPLACE (n.noise AS noise)
        FROM cells c CROSS JOIN noises n
        WHERE c.noise IS NULL
        ORDER BY method, number_of_communities, p_in, multiplier, network_seed,
                 noise, partition_seed
        """
    ).df()
    table.attrs["logical_cells_collapsed"] = True
    return table


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


# Metric columns added after schema v1; older stores lack them and stay readable.
OPTIONAL_METRICS = ("overlap", "overlap_normalized", "nmi", "vi")
REPLAYED_METRICS = ("rand_index", "adjusted_rand", *OPTIONAL_METRICS, "robustness")


def _optional_metrics(con) -> list[str]:
    columns = {row[0] for row in con.execute("DESCRIBE runs").fetchall()}
    return [c for c in OPTIONAL_METRICS if c in columns]


def open_store(root: str | Path):
    """DuckDB connection with views ``runs`` (all rows) and ``cells``.

    ``cells`` averages the restarts of each logical cell, so each
    (graph, method, noise, partition_seed) condition has equal weight.
    """
    duckdb = _require_duckdb()
    root = Path(root).resolve()
    if not shard_paths(root):
        raise FileNotFoundError(f"No result shards under {root / RUNS_DIR}")
    pattern = _sql_literal(str(root / RUNS_DIR / "*" / "*" / "*" / "*.parquet"))
    con = duckdb.connect()
    con.execute(f"CREATE VIEW runs AS SELECT * FROM read_parquet({pattern}, union_by_name = true)")
    extra = "".join(f"avg({c}) AS {c},\n" for c in _optional_metrics(con) if c != "overlap")
    con.execute(
        f"""
        CREATE VIEW cells AS
        SELECT number_of_communities, community_size, p_in, difficulty, p_out,
               network_seed, method, noise, partition_seed,
               count(*) AS runs,
               count(DISTINCT partition_hash) AS distinct_partitions,
               avg(duration) AS duration,
               avg(rand_index) AS rand_index,
               avg(adjusted_rand) AS adjusted_rand,
               {extra}
               avg(robustness) AS robustness
        FROM runs
        GROUP BY ALL
        """
    )
    return con


def summarize(root: str | Path) -> dict[str, Any]:
    """Counts, sizes and per-method means of a store (aggregated in DuckDB)."""
    root = Path(root)
    paths = shard_paths(root)
    manifest = read_manifest(root) or {}
    disk = sum(p.stat().st_size for p in paths)
    summary: dict[str, Any] = {
        "store": str(root),
        "schema": manifest.get("schema"),
        "spec_hash": manifest.get("spec_hash"),
        "graph_generator": manifest.get("spec", {}).get("graph_generator"),
        "partitions": manifest.get("spec", {}).get("partitions"),
        "runtime": manifest.get("spec", {}).get("runtime"),
        "shards": len(paths),
        "rows": 0,
        "bytes_on_disk": int(disk),
        "bytes_per_row": None,
        "methods": {},
    }
    if not paths:
        return summary
    con = open_store(root)
    rows = con.execute(
        """
        SELECT method, count(*), avg(adjusted_rand), avg(robustness), avg(duration),
               sum((status <> 'ok')::INT)
        FROM runs GROUP BY method ORDER BY method
        """
    ).fetchall()
    total = sum(r[1] for r in rows)
    summary["rows"] = int(total)
    summary["bytes_per_row"] = round(disk / total, 1) if total else None
    summary["errors"] = int(sum(r[5] for r in rows))
    for method, n, ari, rob, dur, _ in rows:
        summary["methods"][str(method)] = {
            "rows": int(n),
            "adjusted_rand": round(float(ari), 4),
            "robustness": round(float(rob), 4),
            "duration_s": round(float(dur), 5),
        }
    return summary


def expected_graphs(manifest: dict[str, Any]) -> int:
    """Graph tasks of the most recent grid (one shard each)."""
    grids = manifest.get("grids") or []
    if not grids:
        return 0
    grid = grids[-1]
    return (
        len(grid["n_communities"]) * len(grid["p_in"]) * len(grid["difficulty"])
        * len(grid["seeds"])
    )


def progress(root: str | Path) -> dict[str, Any]:
    """Run progress from shard files only (cheap; safe while a run writes)."""
    import shutil
    import time

    root = Path(root)
    manifest = read_manifest(root) or {}
    expected = expected_graphs(manifest)
    paths = shard_paths(root)
    mtimes = sorted(p.stat().st_mtime for p in paths)
    disk = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    now = time.time()
    invocations = manifest.get("invocations") or []
    started = None
    if invocations:
        started = _dt.datetime.fromisoformat(invocations[-1]["started"]).timestamp()
    holder = writer_lock_holder(root)
    # A stopped run is measured up to its last shard, a running one up to now.
    end = now if holder or not mtimes else mtimes[-1]
    since_start = [m for m in mtimes if started is None or m >= started]
    rate = None
    if len(since_start) >= 2:
        span = max(end - (started or since_start[0]), 1.0)
        rate = len(since_start) / span
    # Recent window: the last 30 minutes, or since the latest start if shorter.
    window = 1800.0 if started is None else min(1800.0, max(end - started, 1.0))
    recent = [m for m in mtimes if m >= end - window]
    recent_rate = len(recent) / window if len(recent) >= 2 else None
    remaining = max(expected - len(paths), 0)
    use_rate = recent_rate or rate
    # No ETA unless a writer is running: a stopped run makes no progress.
    eta = remaining / use_rate if (use_rate and holder) else None
    return {
        "store": str(root),
        "completed": len(paths),
        "expected": expected,
        "fraction": round(len(paths) / expected, 4) if expected else None,
        "stale_tmp_files": len(list((root / RUNS_DIR).glob("*/*/*/.*.tmp"))),
        "writer": holder,
        "running": holder is not None,
        "started": invocations[-1]["started"] if invocations else None,
        "workers": invocations[-1].get("workers") if invocations else None,
        "graphs_per_hour_since_start": round(rate * 3600, 1) if rate else None,
        "graphs_per_hour_last_30min": round(recent_rate * 3600, 1) if recent_rate else None,
        "eta_hours": round(eta / 3600, 2) if eta else (0.0 if expected and not remaining else None),
        "rate_window_s": round(window),
        "eta_utc": (
            _dt.datetime.fromtimestamp(now + eta, _dt.UTC).isoformat(timespec="minutes")
            if eta else None
        ),
        "last_shard_utc": (
            _dt.datetime.fromtimestamp(mtimes[-1], _dt.UTC).isoformat(timespec="seconds")
            if mtimes else None
        ),
        "bytes_on_disk": int(disk),
        "disk_free_bytes": int(shutil.disk_usage(root).free),
        "runtime": manifest.get("spec", {}).get("runtime"),
        "partitions": manifest.get("spec", {}).get("partitions"),
        "spec_hash": manifest.get("spec_hash"),
    }


# ---------------------------------------------------------------------------
# Verification by regeneration and replay
# ---------------------------------------------------------------------------


def verify_store(
    root: str | Path,
    *,
    sample_shards: int | None = 5,
    sample_rows: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Regenerate sampled graphs and replay sampled rows from their seeds.

    Checks shard metadata against the manifest, graph fingerprints against
    freshly generated graphs, and, for a sample of rows per shard, that the
    replayed detector returns the stored partition (hash) and metrics.
    ``sample_shards=None`` checks every shard.
    """
    import pyarrow.parquet as pq

    from hedonic.experiments.disjoint import sbm_sweep

    root = Path(root)
    manifest = read_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"{root / MANIFEST_NAME} not found")
    spec = manifest["spec"]
    problems: list[str] = []
    if spec.get("runtime") != runtime_identity():
        problems.append(
            f"runtime differs from the store: stored {spec.get('runtime')}, "
            f"current {runtime_identity()} (replay needs the same build)"
        )
    paths = shard_paths(root)
    rng = random.Random(seed)
    chosen = paths if sample_shards is None else rng.sample(paths, min(sample_shards, len(paths)))
    graphs_checked = rows_checked = 0
    for path in chosen:
        meta = shard_metadata(path)
        if not meta or meta.get("spec_hash") != manifest["spec_hash"]:
            problems.append(f"{path}: missing or foreign shard metadata")
            continue
        table = pq.read_table(path)
        if table.num_rows != meta["n_rows"]:
            problems.append(f"{path}: {table.num_rows} rows, metadata says {meta['n_rows']}")
        g = sbm_sweep.generate_graph(
            meta["number_of_communities"],
            meta["community_size"],
            meta["p_in"],
            meta["difficulty"],
            meta["network_seed"],
        )
        graphs_checked += 1
        if sbm_sweep.graph_fingerprint(g) != meta["graph_sha256"]:
            problems.append(f"{path}: regenerated graph fingerprint differs")
            continue
        rows = table.to_pylist()
        for row in rng.sample(rows, min(sample_rows, len(rows))):
            replay = sbm_sweep.replay_store_row(g, row, spec)
            rows_checked += 1
            label = (
                f"{path.name} {row['method']} noise={row['noise']} "
                f"partition_seed={row['partition_seed']} run={row['run']}"
            )
            if replay["partition_hash"] != row["partition_hash"]:
                problems.append(f"{label}: replayed partition differs")
            for metric in REPLAYED_METRICS:
                if metric not in row:  # an older store without this column
                    continue
                if replay[metric] != row[metric]:
                    problems.append(
                        f"{label}: {metric} {replay[metric]!r} ≠ stored {row[metric]!r}"
                    )
            if row.get("partition") is not None and list(row["partition"]) != list(
                replay["partition"]
            ):
                problems.append(f"{label}: stored partition column differs")
    return {
        "store": str(root),
        "shards_total": len(paths),
        "graphs_checked": graphs_checked,
        "rows_checked": rows_checked,
        "problems": problems,
        "ok": not problems,
    }


# ---------------------------------------------------------------------------
# CLI: hedonic-exp disjoint-store
# ---------------------------------------------------------------------------


def _print_table(con, sql: str, limit: int) -> None:
    result = con.execute(sql)
    names = [d[0] for d in result.description]
    rows = result.fetchmany(limit)
    widths = [
        max(len(str(n)), *(len(_fmt(r[i])) for r in rows)) if rows else len(str(n))
        for i, n in enumerate(names)
    ]
    print("  ".join(str(n).ljust(w) for n, w in zip(names, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(_fmt(v).ljust(w) for v, w in zip(row, widths)))
    if len(rows) == limit:
        print(f"... (first {limit} rows; use --limit)")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp disjoint-store",
        description=(
            "Inspect, query, export and verify a disjoint SBM result store "
            "(written by `hedonic-exp disjoint --layout store`)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "views for --sql: runs (one row per detector run) and cells\n"
            "(restarts averaged per graph × method × noise × partition seed).\n"
            "\n"
            "examples:\n"
            "  hedonic-exp disjoint-store artifacts/disjoint/sbm-smoke\n"
            "  hedonic-exp disjoint-store artifacts/disjoint/sbm-smoke --sql \\\n"
            "      \"SELECT method, noise, avg(adjusted_rand) FROM cells GROUP BY ALL ORDER BY ALL\"\n"
            "  hedonic-exp disjoint-store artifacts/disjoint/sbm-smoke \\\n"
            "      --export artifacts/disjoint/sbm-smoke/plot_table.parquet\n"
            "  hedonic-exp disjoint-store artifacts/disjoint/sbm-smoke --verify --all\n"
        ),
    )
    parser.add_argument("store", help="Store directory (contains manifest.json)")
    parser.add_argument("--sql", help="Run a DuckDB query against the views")
    parser.add_argument("--limit", type=int, default=50, help="Rows printed by --sql")
    parser.add_argument(
        "--export",
        help="Write the plot table (.parquet, .csv or .csv.gzip) for `hedonic-exp plots`",
    )
    parser.add_argument(
        "--all-restarts",
        action="store_true",
        help=(
            "Export: average every restart in a cell (default keeps only the first "
            "run of each distinct partition, the manuscript's estimand)"
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Completed shards, rate, ETA, disk and runtime (cheap; safe during a run)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Regenerate graphs and replay rows from their seeds",
    )
    parser.add_argument("--all", action="store_true", help="With --verify: every shard")
    parser.add_argument(
        "--sample-shards", type=int, default=5, help="Shards checked by --verify"
    )
    parser.add_argument(
        "--sample-rows", type=int, default=10, help="Rows replayed per shard by --verify"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)

    root = Path(args.store).expanduser()
    if read_manifest(root) is None:
        print(f"Not a disjoint result store (no {MANIFEST_NAME}): {root}")
        return 1
    status = 0
    if args.sql:
        con = open_store(root)
        if args.json:
            frame = con.execute(args.sql).df()
            print(frame.to_json(orient="records"))
        else:
            _print_table(con, args.sql, args.limit)
    if args.export:
        out = Path(args.export).expanduser()
        table = plot_table(root, unique_only=not args.all_restarts)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            table.to_parquet(out, index=False, compression="zstd")
        elif out.name.endswith((".csv.gzip", ".csv.gz")):
            table.to_csv(out, index=False, compression="gzip")
        else:
            table.to_csv(out, index=False)
        print(f"Exported {len(table)} rows → {out}")
    if args.verify:
        report = verify_store(
            root,
            sample_shards=None if args.all else args.sample_shards,
            sample_rows=args.sample_rows,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"verify: {report['graphs_checked']}/{report['shards_total']} graphs "
                f"regenerated, {report['rows_checked']} rows replayed, "
                f"{len(report['problems'])} problem(s)"
            )
            for problem in report["problems"][:20]:
                print(f"  - {problem}")
        status = 0 if report["ok"] else 1
    if args.progress:
        info = progress(root)
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            gb = 1024**3
            print(f"store      {info['store']}")
            print(
                f"shards     {info['completed']}/{info['expected']} graphs"
                f" ({100 * (info['fraction'] or 0):.2f}%)"
                + (f", {info['stale_tmp_files']} stale tmp" if info["stale_tmp_files"] else "")
            )
            print(f"writer     {info['writer'] or 'none (not running)'}  workers={info['workers']}")
            print(
                f"rate       {info['graphs_per_hour_last_30min']} graphs/h (last "
                f"{info['rate_window_s'] // 60} min), "
                f"{info['graphs_per_hour_since_start']} since {info['started']}"
            )
            print(f"eta        {info['eta_hours']} h → {info['eta_utc']} UTC  (last shard {info['last_shard_utc']})")
            print(
                f"disk       {info['bytes_on_disk'] / gb:.2f} GiB used, "
                f"{info['disk_free_bytes'] / gb:.1f} GiB free  partitions={info['partitions']}"
            )
            print(f"runtime    {info['runtime']}")
    if not (args.sql or args.export or args.verify or args.progress):
        summary = summarize(root)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"store        {summary['store']}")
            print(f"schema       {summary['schema']}  spec {str(summary['spec_hash'])[:12]}")
            print(f"generator    {summary['graph_generator']}")
            print(f"runtime      {summary['runtime']}")
            print(
                f"shards       {summary['shards']} graphs, {summary['rows']} rows, "
                f"{summary['bytes_on_disk'] / 1e6:.2f} MB "
                f"({summary['bytes_per_row']} B/row, partitions={summary['partitions']})"
            )
            if summary.get("errors"):
                print(f"errors       {summary['errors']} rows with status != ok")
            print("method        rows    ARI     robustness  time(s)")
            for method, item in summary["methods"].items():
                print(
                    f"{method:<12}{item['rows']:>6}  {item['adjusted_rand']:>6.3f}  "
                    f"{item['robustness']:>10.3f}  {item['duration_s']:>8.5f}"
                )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
