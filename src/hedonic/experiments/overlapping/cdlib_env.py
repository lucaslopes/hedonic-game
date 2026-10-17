"""The isolated CDlib environment and the external methods that run in it.

CDlib 0.4.0 depends on python-igraph 1.0.0 and on pinned NumPy/NetworkX
versions; importing it into the hedonic process could replace that process's
igraph. Its methods (SLPA, DER, ANGEL, Ego-Splitting, Link Communities) run in
a locked uv project instead: its ``pyproject.toml`` and ``uv.lock`` ship with
the package (``hedonic/experiments/workers/slpa_env``) and ``codeseg-setup``
materialises them under the cache directory (``HEDONIC_SLPA_ENV`` overrides
the location). The worker scripts are always the shipped ones, located with
:mod:`hedonic.experiments.resources`, and run with that environment's
interpreter; each one reads a JSON request on stdin and prints one JSON object.

A missing or unprepared environment is :class:`MethodUnavailable`, never a
substitute detector.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import igraph as ig

from hedonic.experiments import resources
from hedonic.experiments.overlapping.robustness import canonicalize_cover

SLPA_PROTOCOL_VERSION = "cdlib-slpa-isolated-v1"
# The direct pins of the locked project (workers/slpa_env/pyproject.toml); a
# worker reports the versions it imported and any difference is refused.
SLPA_DIRECT_PINS: dict[str, str] = {
    "cdlib": "0.4.0",
    "networkx": "3.6.1",
    "numpy": "2.3.3",
    "python-igraph": "1.0.0",
}
PINNED_EXTERNAL_VERSIONS: dict[str, str | None] = {
    "demon": "2.0.6",
    "networkx": "3.6.1",
    "cdlib": "0.4.0",
}
# Where the shipped worker scripts live inside the package.
WORKER_SUBDIR = "slpa_env"
WORKERS = ("slpa_worker.py", "der_worker.py", "cdlib_worker.py", "link_clustering_worker.py")
ISOLATION = "dedicated uv project; CDlib is never imported in the hedonic process"


class MethodUnavailable(RuntimeError):
    """An external method cannot be executed (environment missing, drift, failure)."""


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _source_identity(function: Callable[..., Any]) -> dict[str, Any]:
    """The file name and SHA-256 of ``function``'s source (no absolute path)."""
    try:
        source = Path(inspect.getsourcefile(function) or "").resolve()
        return {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    except (OSError, TypeError):
        return {"path": None, "sha256": None}


def environment_root() -> Path:
    """The materialised CDlib project: ``HEDONIC_SLPA_ENV``, else ``<codeseg cache>/cdlib_env``."""
    configured = os.environ.get("HEDONIC_SLPA_ENV")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = os.environ.get("HEDONIC_CODESEG_CACHE", "~/.cache/hedonic/codeseg")
    return Path(cache).expanduser().resolve() / "cdlib_env"


# The 1.x name, kept for callers that resolve the root themselves.
_slpa_environment_root = environment_root


def shipped_project_dir() -> Path:
    """The locked project as shipped in the package (``pyproject.toml``, ``uv.lock`` and the workers)."""
    return resources.WORKER_DIR / WORKER_SUBDIR


def worker_script(name: str) -> Path:
    """The shipped worker script ``name`` (``slpa_worker.py``, ``cdlib_worker.py``, ...)."""
    return resources.worker_path(f"{WORKER_SUBDIR}/{name}")


def _worker_or_none(name: str) -> Path | None:
    try:
        return worker_script(name)
    except FileNotFoundError:
        return None


def slpa_environment_receipt() -> dict[str, Any]:
    """The files and pins of the isolated CDlib environment, read-only.

    ``ready`` needs the project and its lock under :func:`environment_root`,
    the shipped worker, and an interpreter: the project's materialised
    ``.venv`` (preferred) or ``uv`` with a resolved lock. Anything missing is
    ``ready=False``; an ambient CDlib is never used instead.
    """
    root = environment_root()
    project = root / "pyproject.toml"
    lock = root / "uv.lock"
    worker = _worker_or_none("slpa_worker.py")
    project_python = root / ".venv" / "bin" / "python"
    python_ready = project_python.is_file() and os.access(project_python, os.X_OK)
    uv = shutil.which("uv")
    files = {
        "project": {"path": str(project), "sha256": _sha256_file(project)},
        "lock": {"path": str(lock), "sha256": _sha256_file(lock)},
        "worker": {"path": str(worker) if worker else None, "sha256": _sha256_file(worker) if worker else None},
    }
    lock_is_materialized = False
    if lock.is_file():
        try:
            lock_is_materialized = not lock.read_text(encoding="utf-8").startswith(
                "# Generated by hedonic codeseg-setup"
            )
        except OSError:
            lock_is_materialized = False
    ready = bool((python_ready or (uv and lock_is_materialized)) and all(item["sha256"] for item in files.values()))
    return {
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "root": str(root),
        "worker_dir": str(shipped_project_dir()),
        "uv": str(Path(uv).resolve()) if uv else None,
        "project_python": str(project_python) if python_ready else None,
        "execution_preference": (
            "project_venv" if python_ready else "uv_locked" if uv and lock_is_materialized else None
        ),
        "files": files,
        "direct_pins": dict(SLPA_DIRECT_PINS),
        "lock_is_materialized": lock_is_materialized,
        "ready": ready,
        "isolation": ISOLATION,
    }


def link_clustering_environment_receipt() -> dict[str, Any]:
    """The CDlib receipt with the Link Communities worker (CDlib's ``hierarchical_link_community``)."""
    receipt = slpa_environment_receipt()
    worker = _worker_or_none("link_clustering_worker.py")
    files = dict(receipt.get("files", {}))
    files["link_clustering_worker"] = {
        "path": str(worker) if worker else None,
        "sha256": _sha256_file(worker) if worker else None,
    }
    return {
        **receipt,
        "files": files,
        "ready": bool(receipt.get("ready") and files["link_clustering_worker"]["sha256"]),
        "worker": str(worker) if worker else None,
        "api": "cdlib.algorithms.hierarchical_link_community",
        "output_conversion": "edge communities projected to unique node sets",
    }


def worker_command(receipt: Mapping[str, Any], worker: Path) -> tuple[list[str], str]:
    """The command that runs ``worker`` with the environment's interpreter, and how (venv or ``uv run``)."""
    project_python = receipt.get("project_python")
    if project_python:
        return [str(project_python), str(worker)], "project_venv"
    if receipt.get("uv"):
        return [str(receipt["uv"]), "run", "--project", str(receipt["root"]), "--locked", "python", str(worker)], "uv_locked"
    raise MethodUnavailable("the locked CDlib project has no interpreter; run `hedonic-exp codeseg-setup --all`")


def last_json_line(stdout: str, label: str = "isolated CDlib worker") -> dict[str, Any]:
    """The worker's last JSON object on stdout (third-party import chatter may precede it)."""
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise MethodUnavailable(f"{label} emitted no JSON result")


_last_json_line = last_json_line


def run_worker(
    worker_name: str,
    request: Mapping[str, Any],
    *,
    timeout_seconds: float | None = 90.0,
    label: str | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the shipped worker ``worker_name`` on ``request``; return its payload and how it ran.

    Raises :class:`MethodUnavailable` when the environment is not ready, the
    worker fails or times out, or it prints no JSON object.
    """
    label = label or f"isolated {worker_name}"
    receipt = dict(receipt or slpa_environment_receipt())
    worker = _worker_or_none(worker_name)
    if worker is None:
        raise MethodUnavailable(f"{label}: worker {worker_name} is not shipped with this hedonic")
    if not receipt.get("ready"):
        missing = [key for key, value in receipt.get("files", {}).items() if not value.get("sha256")]
        if not receipt.get("uv") and not receipt.get("project_python"):
            missing.append("uv executable or project Python")
        raise MethodUnavailable(
            f"{label}: the locked CDlib environment at {receipt.get('root')} is not ready"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + "; run `hedonic-exp codeseg-setup --all`"
        )
    command, mode = worker_command(receipt, worker)
    environment = os.environ.copy()
    # CDlib imports matplotlib; keep its cache out of a read-only home folder.
    matplotlib_config = Path(tempfile.gettempdir()) / "hedonic-cdlib-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_config)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=str(receipt["root"]),
            input=json.dumps(dict(request), sort_keys=True),
            text=True,
            capture_output=True,
            env=environment,
            timeout=None if timeout_seconds is None else float(timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MethodUnavailable(f"{label} could not complete: {exc}") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise MethodUnavailable(f"{label} exited {process.returncode}: " + (detail[-1] if detail else "no diagnostic"))
    payload = last_json_line(process.stdout, label)
    return payload, {
        "worker_seconds": time.perf_counter() - started,
        "worker_command": command,
        "execution_mode": mode,
        "worker_source_identity": {"path": worker.name, "sha256": _sha256_file(worker)},
    }


def check_pins(metadata: Mapping[str, Any], pins: Mapping[str, str], label: str) -> dict[str, Any]:
    """The versions a worker reported, refused when one differs from ``pins``."""
    actual = {name: metadata.get(f"{name.replace('-', '_')}_version") for name in pins}
    mismatches = {name: (actual[name], expected) for name, expected in pins.items() if actual[name] != expected}
    if mismatches:
        raise MethodUnavailable(f"{label} dependency drift: {mismatches}")
    return actual


def _cover_of(communities: Any, n_vertices: int, label: str) -> list[list[int]]:
    if not isinstance(communities, list):
        raise MethodUnavailable(f"{label} returned an invalid payload")
    return canonicalize_cover([sorted(int(v) for v in community) for community in communities], n_vertices)


def run_slpa(
    graph: ig.Graph,
    *,
    iterations: int = 20,
    threshold: float = 0.10,
    seed: int = 0,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """CDlib's SLPA (``algorithms.slpa``) in the locked environment; the cover and its provenance.

    CDlib has no seed argument: the worker seeds NumPy's global generator
    with ``seed`` right before the call.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    receipt = slpa_environment_receipt()
    request = {
        "n": int(graph.vcount()),
        "edges": [[int(left), int(right)] for left, right in graph.get_edgelist()],
        "iterations": int(iterations),
        "threshold": float(threshold),
        "seed": int(seed),
    }
    payload, run = run_worker("slpa_worker.py", request, timeout_seconds=timeout_seconds,
                              label="isolated SLPA worker", receipt=receipt)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise MethodUnavailable("isolated SLPA worker returned an invalid payload")
    versions = check_pins(metadata, SLPA_DIRECT_PINS, "SLPA worker")
    cover = _cover_of(payload.get("communities"), graph.vcount(), "isolated SLPA worker")
    return cover, {
        "method": "slpa",
        "family": "label_propagation",
        "implementation": "cdlib.algorithms.slpa",
        "implementation_kind": "external_pinned_adapter",
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "dependency": {
            "distribution": "cdlib",
            "version": versions["cdlib"],
            "direct_pins": dict(SLPA_DIRECT_PINS),
            "lock_sha256": receipt["files"]["lock"]["sha256"],
        },
        "iterations": int(iterations),
        "threshold": float(threshold),
        "seed": int(seed),
        "seed_support": False,
        "rng_semantics": "numpy.random.seed(seed) in isolated worker",
        "graph_semantics": "undirected, unweighted NetworkX conversion; no planted cover/cap",
        "warm_start": False,
        "metadata_free": True,
        "availability": "external",
        "information_budget": "graph topology only; no planted cover/cap tuning",
        **run,
        "environment_identity": receipt,
        "source_identity": _source_identity(run_slpa),
    }


def run_link_clustering(
    graph: ig.Graph,
    *,
    timeout_seconds: float = 90.0,
) -> tuple[list[list[int]], dict[str, Any]]:
    """CDlib's Ahn--Bagrow--Lehmann Link Communities (``hierarchical_link_community``), as node sets."""
    receipt = link_clustering_environment_receipt()
    request = {
        "n": int(graph.vcount()),
        "edges": [[int(left), int(right)] for left, right in graph.get_edgelist()],
    }
    payload, run = run_worker("link_clustering_worker.py", request, timeout_seconds=timeout_seconds,
                              label="isolated Link Clustering worker", receipt=receipt)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise MethodUnavailable("isolated Link Clustering worker returned an invalid payload")
    pins = {"cdlib": PINNED_EXTERNAL_VERSIONS["cdlib"], "networkx": PINNED_EXTERNAL_VERSIONS["networkx"]}
    versions = check_pins(metadata, pins, "Link Clustering worker")
    cover = _cover_of(payload.get("communities"), graph.vcount(), "isolated Link Clustering worker")
    return cover, {
        "method": "link_clustering",
        "family": "edge_link_communities",
        "implementation": "cdlib.algorithms.hierarchical_link_community",
        "implementation_kind": "external_pinned_adapter",
        "protocol_version": SLPA_PROTOCOL_VERSION,
        "dependency": {
            "distribution": "cdlib",
            "version": versions["cdlib"],
            "direct_pins": pins,
            "lock_sha256": receipt["files"]["lock"]["sha256"],
        },
        "seed": None,
        "seed_support": False,
        "graph_semantics": "undirected, unweighted NetworkX conversion; no planted cover/cap",
        "warm_start": False,
        "metadata_free": True,
        "availability": "external",
        "information_budget": "graph topology only; no planted cover/cap tuning",
        **run,
        "environment_identity": receipt,
        "source_identity": _source_identity(run_link_clustering),
        "output_conversion": metadata.get("output_conversion"),
    }


def run_cdlib_method(
    graph: ig.Graph,
    *,
    method: str,
    parameters: Mapping[str, Any],
    seed: int,
    timeout_seconds: float | None = 90.0,
) -> list[list[int]]:
    """One method of ``cdlib_worker.py`` (``angel``, ``ego_splitting``, ...) on ``graph``; its communities.

    Parameters whose name starts with ``_`` are private to the caller and not sent.
    """
    request = {
        "method": method,
        "n": int(graph.vcount()),
        "edges": [[int(left), int(right)] for left, right in graph.get_edgelist()],
        "seed": int(seed),
        **{key: value for key, value in parameters.items() if not str(key).startswith("_")},
    }
    payload, _run = run_worker("cdlib_worker.py", request, timeout_seconds=timeout_seconds,
                               label="isolated CDlib worker")
    if not isinstance(payload.get("communities"), list):
        raise MethodUnavailable("isolated CDlib worker emitted no valid cover")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise MethodUnavailable("isolated CDlib worker omitted dependency metadata")
    check_pins(metadata, SLPA_DIRECT_PINS, "CDlib worker")
    return [list(map(int, community)) for community in payload["communities"]]


__all__ = [
    "MethodUnavailable",
    "PINNED_EXTERNAL_VERSIONS",
    "SLPA_DIRECT_PINS",
    "SLPA_PROTOCOL_VERSION",
    "WORKERS",
    "environment_root",
    "link_clustering_environment_receipt",
    "run_cdlib_method",
    "run_link_clustering",
    "run_slpa",
    "run_worker",
    "shipped_project_dir",
    "slpa_environment_receipt",
    "worker_script",
]
