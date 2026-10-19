"""Prepare the data and the external runtimes of the overlapping benchmark.

``hedonic setup`` (this module's :func:`setup_main`) is the one command that
makes a fresh machine ready for ``hedonic run queue overlap``:

* ``--pairs dblp:top5000,...`` downloads the official SNAP graph and cover of
  each network:cover pair into the network root (``hedonic show covers`` lists
  what is there); ``--all`` names the paper's eight pairs;
* ``--methods codeseg,fox,...`` prepares each baseline's runtime: CoDeSEG and
  FOX are built from the authors' CoDeSEG repository with CMake, BigCLAM from
  the SNAP repository and NEO-K-Means from the authors' archive with make,
  NcGame runs the authors' ``NCG.py`` through the shipped worker, and ANGEL
  and Ego-Splitting run in a locked CDlib project (``workers/slpa_env``, with
  its ``uv.lock``) that is materialised under the cache directory and synced
  with ``uv``. Clique percolation and Hedonic need nothing.

The system tools a selection needs (a C++ compiler, CMake, an OpenMP-capable
compiler for FOX, make, git, uv) are checked before anything is fetched; a
missing tool stops the setup with what to install. ``--dry-run`` prints the
plan and the tool check without downloading or building.

Every result goes into a JSON manifest (``<cache>/setup_manifest.json``) that
the runners read to find the executables; a later setup merges its records
into it. ``hedonic-exp codeseg-doctor`` is the read-only readiness audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments import resources
from hedonic.experiments.config import NETWORKS_DIR, codeseg_cache_dir, expand_path
from hedonic.experiments.overlapping.cdlib_env import (
    SLPA_DIRECT_PINS,
    environment_root,
    shipped_project_dir,
    slpa_environment_receipt,
)
from hedonic.experiments.overlapping.snap import (
    MAPEQUATION_SNAP_NAMES,
    SnapLoadError,
    UnsupportedCoverVariant,
    canonical_cover_variant,
    prepare_snap_dataset,
)


SETUP_SCHEMA_VERSION = 1
# Parallel compile jobs of a native build (HEDONIC_BUILD_JOBS overrides).
BUILD_JOBS = max(1, int(os.environ.get("HEDONIC_BUILD_JOBS") or min(6, os.cpu_count() or 1)))
def default_cache_dir() -> Path:
    """The runtime cache (:func:`hedonic.experiments.config.codeseg_cache_dir`): ``HEDONIC_CODESEG_CACHE``, else
    ``codeseg/`` under the user config's ``cache_dir``, else ``~/.cache/hedonic/codeseg``. ``hedonic run exp`` uses
    the same folder, so what ``hedonic setup`` prepares is what the runs find."""
    return codeseg_cache_dir()


SNAP_DATASETS: tuple[str, ...] = (
    "amazon",
    "youtube",
    "dblp",
    "livejournal",
    "orkut",
    "friendster",
    "wikipedia",
)
DEFAULT_DATASETS: tuple[str, ...] = ("dblp",)
# The network:cover pairs of the overlap paper's queue, in its order.
PAPER_PAIRS: tuple[str, ...] = (
    "dblp:top5000",
    "amazon:top5000",
    "youtube:top5000",
    "friendster:top5000",
    "livejournal:top5000",
    "dblp:all",
    "amazon:all",
    "youtube:all",
)
# The eight baselines of the overlap paper (``hedonic run exp`` keys), cheapest first.
PAPER_BASELINES: tuple[str, ...] = (
    "codeseg",
    "ncgame",
    "ego_splitting",
    "fox",
    "clique_percolation",
    "angel",
    "bigclam",
    "neo_kmeans",
)
# Methods that run inside the hedonic process: nothing to prepare.
IN_PROCESS_METHODS: frozenset[str] = frozenset(
    {"hedonic", "hedonic_local", "community_hedonic", "louvain", "leiden", "flpa", "disjoint_oracle"}
)
# Methods that run in the locked CDlib project.
CDLIB_METHODS: frozenset[str] = frozenset({"angel", "ego_splitting", "slpa", "der", "link_clustering"})
# In-process Python adapters (NetworkX, igraph, the demon package): nothing to download or build.
ADAPTER_METHODS: frozenset[str] = frozenset({"clique_percolation", "infomap", "demon"})
# Every method this setup knows: the paper's baselines first, then the other registered runtimes.
DEFAULT_METHODS: tuple[str, ...] = (
    *PAPER_BASELINES,
    "slpa",
    "der",
    "link_clustering",
    "infomap",
    "demon",
    "oslom",
    "nise",
    "sse",
    "qoce",
    "svi",
    "essc",
    "hedonic",
    "hedonic_local",
    "community_hedonic",
    "louvain",
    "leiden",
    "flpa",
)
METHOD_ALIASES: dict[str, str] = {
    "hoc": "hedonic",
    "hoc_local": "hedonic",
    "lazyfox": "fox",
    "neo-k-means": "neo_kmeans",
    "ego-splitting": "ego_splitting",
    "cliqueperc": "clique_percolation",
}
# The CoDeSEG paper's nine-method catalogue (kept for the runner's records).
PAPER_METHODS: tuple[str, ...] = (
    "codeseg",
    "slpa",
    "bigclam",
    "ncgame",
    "fox",
    "louvain",
    "der",
    "leiden",
    "flpa",
)

UPSTREAM_CODESEG_URL = "https://github.com/SELGroup/CoDeSEG.git"
UPSTREAM_SNAP_URL = "https://github.com/snap-stanford/snap.git"
UPSTREAM_OSLOM_URL = "https://github.com/eXascaleInfolab/oslom2.git"
UPSTREAM_OSLOM_REF = "master"
UPSTREAM_NEO_URL = "https://bdi-lab.kaist.ac.kr/assets/down/neo_k_means_graph.zip"
UPSTREAM_NEO_FALLBACK_URL = "https://bdi-lab.kaist.ac.kr/down/assets/down/neo_k_means_graph.zip"
UPSTREAM_SVI_URL = "https://github.com/premgopalan/svinet.git"
UPSTREAM_SVI_REF = "master"
UPSTREAM_ESSC_URL = "https://github.com/jdwilson4/ESSC.git"
UPSTREAM_ESSC_REF = "master"
UPSTREAM_QOCE_URL = "https://github.com/PanShi2016/QOCE.git"
UPSTREAM_QOCE_REF = "master"
UPSTREAM_NISE_URL = "https://bdi-lab.kaist.ac.kr/assets/down/nise.htm"


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_selection(
    value: str | None,
    allowed: Iterable[str],
    *,
    option: str,
    default: tuple[str, ...],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(default)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    selected = [(aliases or {}).get(item, item) for item in selected]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(
            f"unknown {option}: {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    if not selected:
        raise ValueError(f"{option} must contain at least one value")
    return list(dict.fromkeys(selected))


def _default_worker_path() -> Path:
    """The NcGame worker shipped with the package (``hedonic/experiments/workers``)."""
    return resources.worker_path("ncgame_worker.py")


def _manifest_default_path(cache_dir: Path) -> Path:
    configured = os.environ.get("HEDONIC_CODESEG_MANIFEST")
    return expand_path(configured) if configured else cache_dir / "setup_manifest.json"


def load_setup_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the most recent setup manifest, if one exists."""
    candidate = expand_path(path) if path else _manifest_default_path(default_cache_dir())
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _executable(value: str | Path | None) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        resolved = shutil.which(str(candidate))
        if resolved:
            candidate = Path(resolved)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    return None


def _find_named(root: Path | None, names: Iterable[str]) -> Path | None:
    if root is None or not root.exists():
        return None
    wanted = {name.lower() for name in names}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in wanted and os.access(path, os.X_OK):
            return path.resolve()
    return None


def _fox_openmp_compiler(configured: str | Path | None = None) -> Path | None:
    """Find a C++ compiler that accepts Fox's hard-coded ``-fopenmp`` flag.

    The upstream Fox CMake file unconditionally adds ``-fopenmp``.  Apple's
    system Clang rejects that flag, while Homebrew GCC (or an explicitly
    configured compiler) supports it.  Keep this discovery local to the
    optional native build; the core package never depends on Homebrew.
    """
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    if sys.platform == "darwin":
        for name in ("g++-16", "g++-15", "g++-14", "g++-13", "g++-12"):
            resolved = shutil.which(name)
            if resolved:
                candidates.append(Path(resolved))
        for prefix in (Path("/opt/homebrew/opt/gcc/bin"), Path("/usr/local/opt/gcc/bin")):
            if prefix.is_dir():
                candidates.extend(sorted(prefix.glob("g++-*"), reverse=True))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _git_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _ensure_checkout(
    *,
    url: str,
    destination: Path,
    ref: str,
    offline: bool,
    dry_run: bool,
) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    result: dict[str, Any] = {
        "url": url,
        "ref": ref,
        "path": str(destination),
        "status": "missing",
        "commit": _git_commit(destination),
    }
    if (destination / ".git").is_dir():
        result["status"] = "ready"
        result["commit"] = _git_commit(destination)
        return result
    if dry_run or offline:
        result["reason"] = "checkout absent (dry-run/offline)"
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(destination)],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["reason"] = f"git clone failed: {exc}"
        return result
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        result["reason"] = "git clone failed: " + (detail[-1] if detail else "unknown error")
        return result
    result["status"] = "ready"
    result["commit"] = _git_commit(destination)
    return result


def _ensure_neo_source(
    *, cache_dir: Path, offline: bool, dry_run: bool
) -> dict[str, Any]:
    """Fetch and safely unpack the official NEO-K-Means graph archive."""
    source_root = cache_dir / "upstream" / "neo_k_means_graph"
    archive = cache_dir / "downloads" / "neo_k_means_graph.zip"
    result: dict[str, Any] = {
        "url": UPSTREAM_NEO_URL,
        "fallback_urls": [UPSTREAM_NEO_FALLBACK_URL],
        "archive": str(archive),
        "path": str(source_root),
        "status": "missing",
        "archive_sha256": _sha256_file(archive),
    }
    if (source_root / "programs" / "neo.cpp").is_file():
        result["status"] = "ready"
        result["compatibility_patch"] = "int-main-return-type"
        return result
    if dry_run or offline:
        result["reason"] = "archive absent (dry-run/offline)"
        return result
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not archive.is_file():
            # The KAIST host rejects urllib's default User-Agent with HTTP
            # 403, while accepting a normal browser request.  Download to a
            # sidecar and rename only after the response is complete so a
            # truncated archive can never be mistaken for a valid cache.
            temporary = archive.with_name(archive.name + ".part")
            last_error: Exception | None = None
            for url in (UPSTREAM_NEO_URL, UPSTREAM_NEO_FALLBACK_URL):
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
                        "Referer": "https://bigdata.oden.utexas.edu/software/neo-k-means/",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                    temporary.replace(archive)
                    result["url"] = url
                    break
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
                    temporary.unlink(missing_ok=True)
            else:
                raise last_error or urllib.error.URLError("NEO archive download failed")
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            # The official archive has no top-level directory (``programs/``,
            # ``metisLib/`` and ``Makefile`` are at its root).  Extract inside
            # the cache-specific source directory rather than beside it;
            # otherwise the subsequent ``programs/neo.cpp`` probe misses the
            # files and every rerun reports a false-unavailable NEO runtime.
            root = source_root
            for member in bundle.infolist():
                target = (root / member.filename).resolve()
                if not str(target).startswith(str(root.resolve()) + os.sep):
                    raise ValueError(f"unsafe NEO archive member: {member.filename}")
            bundle.extractall(root)
        # The official archive is a legacy C++ source tree whose entry point
        # omits its return type.  Keep the upstream checkout intact except for
        # this mechanical C++17 compatibility repair, and record it in the
        # manifest so the build remains auditable.
        source_file = source_root / "programs" / "neo.cpp"
        text = source_file.read_text(encoding="utf-8")
        if "\nmain(int argc, char *argv[])" in text:
            source_file.write_text(
                text.replace("\nmain(int argc, char *argv[])", "\nint main(int argc, char *argv[])", 1),
                encoding="utf-8",
            )
        result["status"] = "ready"
        result["archive_sha256"] = _sha256_file(archive)
        result["compatibility_patch"] = "int-main-return-type"
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        result["reason"] = f"NEO source preparation failed: {exc}"
    return result


def _run_build(
    command: list[str], *, cwd: Path, log_path: Path, timeout: float,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return {"status": "failed", "command": command, "reason": str(exc)}
    log_path.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    return {
        "status": "ready" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def _ensure_svi_source(
    *, cache_dir: Path, offline: bool, dry_run: bool, build_native: bool, timeout: float
) -> dict[str, Any]:
    """Fetch/build the authors' SVI implementation.

    The upstream tree is an old autotools C++ project.  Current Homebrew GSL
    is detected through ``gsl-config`` and the source is compiled as GNU C++98
    because the release uses adjacent string literals that are rejected by
    newer language modes.
    """
    source = cache_dir / "upstream" / "svinet"
    install = cache_dir / "build" / "svinet-install"
    record = _ensure_checkout(
        url=UPSTREAM_SVI_URL,
        destination=source,
        ref=UPSTREAM_SVI_REF,
        offline=offline,
        dry_run=dry_run,
    )
    record["install_prefix"] = str(install)
    binary = source / "src" / "svinet"
    if build_native and not dry_run and record.get("status") == "ready" and not binary.is_file():
        configure = source / "configure"
        gsl_config = shutil.which("gsl-config")
        if gsl_config:
            try:
                gsl_cflags = subprocess.check_output([gsl_config, "--cflags"], text=True).strip()
                gsl_libs = subprocess.check_output([gsl_config, "--libs"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                gsl_cflags, gsl_libs = "", ""
        else:
            gsl_cflags, gsl_libs = "", ""
        environment = os.environ.copy()
        if gsl_cflags:
            environment["CPPFLAGS"] = (environment.get("CPPFLAGS", "") + " " + gsl_cflags).strip()
        if gsl_libs:
            environment["LDFLAGS"] = (environment.get("LDFLAGS", "") + " " + gsl_libs).strip()
        environment["CXXFLAGS"] = (environment.get("CXXFLAGS", "") + " -O2 -std=gnu++98").strip()
        configure_result = _run_build(
            [str(configure), f"--prefix={install}"],
            cwd=source,
            log_path=cache_dir / "logs" / "svi-configure.log",
            timeout=timeout,
            environment=environment,
        )
        build_result = {"status": "failed", "reason": "configure failed"}
        if configure_result.get("status") == "ready":
            build_result = _run_build(
                ["make", "-j2"],
                cwd=source,
                log_path=cache_dir / "logs" / "svi-build.log",
                timeout=timeout,
                environment=environment,
            )
        record["build_svi"] = {"configure": configure_result, "build": build_result}
    record["binary"] = str(binary) if binary.is_file() else None
    record["status"] = "ready" if binary.is_file() else record.get("status", "missing")
    record["compatibility"] = "GNU C++98 plus GSL autodetected via gsl-config"
    return record


def _ensure_r_package(
    *, cache_dir: Path, offline: bool, dry_run: bool, install: bool, timeout: float
) -> dict[str, Any]:
    """Fetch and install the ESSC R package into an isolated cache library."""
    source = cache_dir / "upstream" / "ESSC"
    library = cache_dir / "build" / "r-lib"
    record = _ensure_checkout(
        url=UPSTREAM_ESSC_URL,
        destination=source,
        ref=UPSTREAM_ESSC_REF,
        offline=offline,
        dry_run=dry_run,
    )
    record["library"] = str(library)
    rscript = shutil.which("Rscript")
    if not rscript:
        record.update({"status": "unavailable", "reason": "Rscript is not installed"})
        return record
    if install and not dry_run and not offline and record.get("status") == "ready":
        library.mkdir(parents=True, exist_ok=True)
        # Matrix ships with R on current platforms. Rlab is the only CRAN
        # dependency missing from a clean base installation; install it into
        # the cache library so the global R library remains untouched.
        dependency_script = (
            "lib <- Sys.getenv('ESSC_R_LIB'); dir.create(lib, recursive=TRUE, showWarnings=FALSE); "
            "if (!requireNamespace('Rlab', quietly=TRUE, lib.loc=lib)) "
            "install.packages('Rlab', lib=lib, repos='https://cloud.r-project.org', quiet=TRUE)"
        )
        dependency = subprocess.run(
            [rscript, "-e", dependency_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "ESSC_R_LIB": str(library)},
        )
        install_result = _run_build(
            ["R", "CMD", "INSTALL", "--library=" + str(library), str(source)],
            cwd=source,
            log_path=cache_dir / "logs" / "essc-install.log",
            timeout=timeout,
            environment={**os.environ, "R_LIBS_USER": str(library)},
        )
        record["install"] = {
            "status": "ready" if install_result.get("status") == "ready" else "failed",
            "dependency_returncode": dependency.returncode,
            "result": install_result,
        }
    if not dry_run:
        try:
            probe = subprocess.run(
                [rscript, "-e", "quit(status=ifelse(requireNamespace('ESSC', quietly=TRUE),0,1))"],
                capture_output=True, text=True, check=False,
                env={**os.environ, "R_LIBS_USER": str(library)},
            )
        except OSError as exc:
            probe = None
            record["probe_error"] = str(exc)
    else:
        probe = None
    probe_ok = bool(dry_run or (probe is not None and probe.returncode == 0))
    ready = bool(record.get("status") == "ready" and probe_ok)
    record["status"] = "ready" if ready else record.get("status", "missing")
    record["rscript"] = rscript
    return record


def _ensure_octave_sources(
    *, cache_dir: Path, offline: bool, dry_run: bool, build_native: bool = False
) -> dict[str, Any]:
    """Cache QOCE and the official NISE/SSE archive for the Octave adapters."""
    qoce = _ensure_checkout(
        url=UPSTREAM_QOCE_URL,
        destination=cache_dir / "upstream" / "QOCE",
        ref=UPSTREAM_QOCE_REF,
        offline=offline,
        dry_run=dry_run,
    )
    # The NISE download is an HTML page with a stable zip href.  Setup records
    # the explicit code source and extracts it into the cache, allowing doctor
    # to fail closed instead of silently substituting another seed-expansion
    # implementation.
    nise_root = cache_dir / "upstream" / "nise"
    archive = cache_dir / "downloads" / "nise_codes.zip"
    nise_record: dict[str, Any] = {"url": UPSTREAM_NISE_URL, "path": str(nise_root), "archive": str(archive), "status": "missing"}
    if (nise_root / "src" / "nise.m").is_file() or (nise_root / "nise.m").is_file():
        nise_record["status"] = "ready"
    elif not dry_run and not offline:
        configured = os.environ.get("HEDONIC_NISE_SOURCE")
        if configured and Path(configured).exists():
            source = Path(configured).expanduser().resolve()
            nise_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, nise_root / "src", dirs_exist_ok=True)
            nise_record["status"] = "ready"
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_name(archive.name + ".part")
            try:
                request = urllib.request.Request(
                    "https://bdi-lab.kaist.ac.kr/assets/down/nise_codes.zip",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                temporary.replace(archive)
                nise_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive) as bundle:
                    root = nise_root / "src"
                    for member in bundle.infolist():
                        target = (root / member.filename).resolve()
                        if not str(target).startswith(str(root.resolve()) + os.sep):
                            raise ValueError(f"unsafe NISE archive member: {member.filename}")
                    bundle.extractall(root)
                nise_record["status"] = "ready"
                nise_record["archive_sha256"] = _sha256_file(archive)
            except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as exc:
                temporary.unlink(missing_ok=True)
                nise_record["reason"] = f"NISE source preparation failed: {exc}"
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if octave and not dry_run:
        nise_src = nise_root / "src"
        if (nise_src / "nise.m").is_file():
            for source_name in ("pprgrow_mex.cc", "vpprgrow_mex.cc", "cutcond_mex.cc", "triangleclusters_mex.cc"):
                source_file = nise_src / source_name
                if source_file.is_file():
                    if sys.platform == "darwin":
                        text = source_file.read_text(encoding="utf-8")
                        patched = text.replace("#include <tr1/unordered_set>", "#include <unordered_set>").replace("#include <tr1/unordered_map>", "#include <unordered_map>").replace("#define tr1ns std::tr1", "#define tr1ns std")
                        patched = patched.replace(
                            "maxdeg = std::max(maxdeg, sr_degree(G,set[i]));",
                            "if (sr_degree(G,set[i]) > maxdeg) maxdeg = sr_degree(G,set[i]);",
                        ).replace(
                            "std::vector< size_t > cluster;",
                            "std::vector<mwIndex> cluster;",
                        ).replace(
                            "const std::vector<size_t >& cluster",
                            "const std::vector<mwIndex>& cluster",
                        )
                        if patched != text:
                            source_file.write_text(patched, encoding="utf-8")
                    # Build the C++ MEX sources locally when requested. The
                    # MATLAB wrapper files below are interpreted by Octave.
                    target_exists = any(nise_src.glob(source_file.stem + ".mex")) or any(nise_src.glob(source_file.stem + ".oct"))
                    if build_native and not target_exists:
                        nise_record.setdefault("build", {})[source_name] = _run_build(
                            ["mkoctfile", "--mex", "-O", str(source_file)],
                            cwd=nise_src,
                            log_path=cache_dir / "logs" / f"nise-{source_file.stem}.log",
                            timeout=900,
                        )
            for source_name in ("pprgrow.m", "vpprgrow.m"):
                source_file = nise_src / source_name
                if source_file.is_file():
                    text = source_file.read_text(encoding="utf-8")
                    patched = text.replace("p.addOptional('nruns'", "p.addParameter('nruns'").replace("p.addOptional('alpha'", "p.addParameter('alpha'").replace("p.addOptional('expands'", "p.addParameter('expands'").replace("p.addOptional('maxexpand'", "p.addParameter('maxexpand'")
                    if patched != text:
                        source_file.write_text(patched, encoding="utf-8")
    nise_record["octave"] = octave
    q = {"qoce": qoce, "nise": nise_record}
    q["qoce"]["octave"] = octave
    qoce_root = Path(str(qoce.get("path", "")))
    clique = qoce_root / "QOCE_codes" / "GetMaxCliques" / "build" / "CliqueFinder"
    if qoce.get("status") == "ready" and build_native and not dry_run and octave:
        build_dir = clique.parent
        makefile = build_dir / "makefile"
        if makefile.is_file():
            graph_loader = qoce_root / "QOCE_codes" / "GetMaxCliques" / "graph_loading.cpp"
            if graph_loader.is_file():
                text = graph_loader.read_text(encoding="utf-8")
                patched = text.replace("struct stat64", "struct stat").replace("fstat64", "fstat")
                if patched != text:
                    graph_loader.write_text(patched, encoding="utf-8")
            q["qoce"]["build_clique_finder"] = _run_build(
                [
                    "g++", "-O3", "-o", str(clique),
                    "../Clique_Finder.cpp", "../aaron_utils.cpp", "../cliques.cpp",
                    "../find_cliques.cpp", "../graph_loading.cpp", "../graph_representation.cpp",
                    "-I..",
                ],
                cwd=build_dir,
                log_path=cache_dir / "logs" / "qoce-clique-build.log",
                timeout=900,
            )
        mex_source = qoce_root / "QOCE_codes" / "rwvec_mex.cpp"
        mex_target = qoce_root / "QOCE_codes" / "rwvec_mex.mex"
        if mex_source.is_file() and not mex_target.is_file():
            q["qoce"]["build_rwvec_mex"] = _run_build(
                ["mkoctfile", "--mex", "-O", str(mex_source)],
                cwd=mex_source.parent,
                log_path=cache_dir / "logs" / "qoce-rwvec-mex.log",
                timeout=900,
            )
        qoce_main = qoce_root / "QOCE_codes" / "QOCE.m"
        if qoce_main.is_file():
            text = qoce_main.read_text(encoding="utf-8")
            marker = "[~, I] = sort(v, 'descend');"
            patched = text.replace(
                marker,
                marker + "\n    if length(I) <= w, detectedComms{i} = I'; continue; end\n    w = min(w, length(I)-1);",
                1,
            )
            if patched != text:
                qoce_main.write_text(patched, encoding="utf-8")
    q["qoce"]["clique_finder"] = str(clique) if clique.is_file() else None
    q["qoce"]["status"] = "ready" if q["qoce"].get("status") == "ready" and octave and q["qoce"].get("clique_finder") else "unavailable"
    return q


def _native_runtime(
    *,
    cache_dir: Path,
    offline: bool,
    dry_run: bool,
    build_native: bool,
    codeseg_bin: str | None,
    bigclam_bin: str | None,
    fox_bin: str | None,
    oslom_bin: str | None = None,
    neo_bin: str | None = None,
    fox_cxx: str | None,
    upstream_url: str,
    upstream_ref: str,
    snap_url: str,
    snap_ref: str,
    timeout: float,
    methods: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fetch (and with ``build_native``, build) the sources the selected ``methods`` need; find their executables.

    Only what a selected method needs is cloned, downloaded or built (all of it when ``methods`` is None)."""
    wanted = set(DEFAULT_METHODS if methods is None else methods)
    source_dir = cache_dir / "upstream" / "CoDeSEG"
    snap_dir = cache_dir / "upstream" / "snap"
    oslom_dir = cache_dir / "upstream" / "oslom2"

    def skipped() -> dict[str, Any]:
        return {"status": "skipped", "reason": "not selected"}

    upstream = (
        _ensure_checkout(url=upstream_url, destination=source_dir, ref=upstream_ref, offline=offline, dry_run=dry_run)
        if wanted & {"codeseg", "fox"} else skipped()
    )
    snap_source = (
        _ensure_checkout(url=snap_url, destination=snap_dir, ref=snap_ref, offline=offline, dry_run=dry_run)
        if "bigclam" in wanted else skipped()
    )
    oslom_source = (
        _ensure_checkout(url=UPSTREAM_OSLOM_URL, destination=oslom_dir, ref=UPSTREAM_OSLOM_REF,
                         offline=offline, dry_run=dry_run)
        if "oslom" in wanted else skipped()
    )
    neo_source = (
        _ensure_neo_source(cache_dir=cache_dir, offline=offline, dry_run=dry_run)
        if "neo_kmeans" in wanted else skipped()
    )
    svi_source = (
        _ensure_svi_source(cache_dir=cache_dir, offline=offline, dry_run=dry_run, build_native=build_native,
                           timeout=timeout)
        if "svi" in wanted else skipped()
    )
    essc_source = (
        _ensure_r_package(cache_dir=cache_dir, offline=offline, dry_run=dry_run, install=build_native,
                          timeout=timeout)
        if "essc" in wanted else skipped()
    )
    octave_sources = (
        _ensure_octave_sources(cache_dir=cache_dir, offline=offline, dry_run=dry_run, build_native=build_native)
        if wanted & {"qoce", "nise", "sse"} else {}
    )

    if build_native and not dry_run:
        cmake = shutil.which("cmake")
        make = shutil.which("make")
        if cmake and upstream["status"] == "ready":
            for name, source in (
                ("codeseg", source_dir / "code_c++" / "CoDeSEG"),
                ("fox", source_dir / "code_c++" / "fox"),
            ):
                if name in wanted and source.is_dir():
                    cxx = _fox_openmp_compiler(fox_cxx) if name == "fox" else None
                    build_dir_name = (
                        f"{name}-{cxx.name}"
                        if name == "fox" and cxx is not None
                        else name
                    )
                    build_dir = cache_dir / "build" / build_dir_name
                    # The upstream Fox CMakeLists only adds ``-O3`` through
                    # CMAKE_CXX_FLAGS_RELEASE.  Without an explicit build
                    # type CMake silently configures an unoptimised binary,
                    # which is especially costly on the bounded DBLP run.
                    configure = [
                        cmake,
                        "-S",
                        str(source),
                        "-B",
                        str(build_dir),
                        "-DCMAKE_BUILD_TYPE=Release",
                    ]
                    if cxx is not None:
                        configure.append(f"-DCMAKE_CXX_COMPILER={cxx}")
                    result = _run_build(
                        configure,
                        cwd=source,
                        log_path=cache_dir / "logs" / f"{name}-cmake.log",
                        timeout=timeout,
                    )
                    if cxx is not None:
                        result["cxx_compiler"] = str(cxx)
                    result["build_type"] = "Release"
                    if result["status"] == "ready":
                        result.update(
                            _run_build(
                                [cmake, "--build", str(build_dir), "--parallel", str(BUILD_JOBS)],
                                cwd=source,
                                log_path=cache_dir / "logs" / f"{name}-build.log",
                                timeout=timeout,
                            )
                        )
                    upstream[f"build_{name}"] = result
        if make and snap_source["status"] == "ready":
            bigclam_source = snap_dir / "examples" / "bigclam"
            if bigclam_source.is_dir():
                snap_source["build_bigclam"] = _run_build(
                    [make, "-C", str(bigclam_source), "--jobs", "2"],
                    cwd=bigclam_source,
                    log_path=cache_dir / "logs" / "bigclam-build.log",
                    timeout=timeout,
                )
        if make and oslom_source["status"] == "ready":
            source = oslom_dir / "main_undirected.cpp"
            if source.is_file():
                target = cache_dir / "build" / "oslom2" / "oslom_undir"
                target.parent.mkdir(parents=True, exist_ok=True)
                # OSLOM has no maintained Makefile.  Compile the single
                # undirected driver directly; a previous placeholder
                # ``make -f -`` call blocked forever waiting for stdin.
                oslom_source["build_oslom"] = _run_build(
                    ["g++", "-O3", "-Wall", "-o", str(target), str(source)],
                    cwd=oslom_dir,
                    log_path=cache_dir / "logs" / "oslom-build.log",
                    timeout=timeout,
                )
        if make and neo_source["status"] == "ready":
            neo_dir = Path(str(neo_source["path"]))
            clean = _run_build(
                [make, "realclean"],
                cwd=neo_dir,
                log_path=cache_dir / "logs" / "neo-clean.log",
                timeout=timeout,
            )
            build = _run_build(
                [make, "-j2"],
                cwd=neo_dir,
                log_path=cache_dir / "logs" / "neo-build.log",
                timeout=timeout,
            )
            neo_source["build_neo"] = {**build, "clean": clean}

    codeseg = _executable(codeseg_bin or os.environ.get("HEDONIC_CODESEG_BIN"))
    bigclam = _executable(bigclam_bin or os.environ.get("HEDONIC_BIGCLAM_BIN"))
    fox = _executable(fox_bin or os.environ.get("HEDONIC_FOX_BIN"))
    oslom = _executable(oslom_bin or os.environ.get("HEDONIC_OSLOM_BIN"))
    neo = _executable(neo_bin or os.environ.get("HEDONIC_NEO_BIN"))
    codeseg = codeseg or _find_named(cache_dir / "build", ("CoDeSEG", "codeseg"))
    bigclam = bigclam or _find_named(cache_dir / "upstream" / "snap", ("bigclam",))
    fox = fox or _find_named(cache_dir / "build", ("LazyFox", "fox"))
    oslom = oslom or _find_named(cache_dir / "build", ("oslom_undir",))
    neo = neo or _find_named(cache_dir / "upstream" / "neo_k_means_graph", ("neo",))
    svi = _find_named(cache_dir / "upstream" / "svinet" / "src", ("svinet",))

    def executable_record(path: Path | None, label: str) -> dict[str, Any]:
        if path is None:
            return {"status": "unavailable", "reason": f"missing {label} executable"}
        return {
            "status": "ready",
            "path": str(path),
            "sha256": _sha256_file(path),
        }

    return {
        "codeseg": executable_record(codeseg, "CoDeSEG"),
        "bigclam": executable_record(bigclam, "Bigclam"),
        "fox": executable_record(fox, "LazyFox"),
        "oslom": executable_record(oslom, "OSLOM"),
        "neo_kmeans": executable_record(neo, "NEO-K-Means"),
        "svi": executable_record(svi, "SVI"),
        "essc": {
            **essc_source,
            "status": "ready" if essc_source.get("status") == "ready" else "unavailable",
        },
        "qoce": octave_sources.get("qoce", {"status": "unavailable"}),
        "nise": octave_sources.get("nise", {"status": "unavailable"}),
        "sse": octave_sources.get("nise", {"status": "unavailable"}),
        "upstream_codeseg": upstream,
        "upstream_snap": snap_source,
        "upstream_oslom": oslom_source,
        "upstream_neo": neo_source,
        "upstream_svi": svi_source,
        "upstream_essc": essc_source,
        "upstream_qoce": octave_sources.get("qoce"),
        "upstream_nise": octave_sources.get("nise"),
    }


def _ncgame_runtime(
    *,
    cache_dir: Path,
    offline: bool,
    dry_run: bool,
    upstream_url: str,
    upstream_ref: str,
) -> dict[str, Any]:
    configured = os.environ.get("HEDONIC_CODESEG_UPSTREAM")
    root = Path(configured).expanduser().resolve() if configured else cache_dir / "upstream" / "CoDeSEG"
    checkout = _ensure_checkout(
        url=upstream_url,
        destination=root,
        ref=upstream_ref,
        offline=offline or bool(configured),
        dry_run=dry_run,
    )
    script = root / "code_py" / "NCG.py"
    ready = script.is_file()
    python = os.environ.get("HEDONIC_NCGAME_PYTHON") or sys.executable
    worker = _default_worker_path()
    command = (
        f"{shlex.quote(python)} {shlex.quote(str(worker))} "
        f"--upstream-root {shlex.quote(str(root))} --input {{input}} --ground-truth {{ground_truth}} "
        f"--output {{output}} --dataset {{dataset}} --timing-output {{timing}}"
    )
    return {
        "status": "ready" if ready else "unavailable",
        "reason": None if ready else f"missing NcGame script: {script}",
        "upstream_root": str(root),
        "script": str(script),
        "script_sha256": _sha256_file(script),
        "command": command if ready else None,
        "python": python,
        "worker": str(worker),
        "worker_sha256": _sha256_file(worker),
        "checkout": checkout,
    }


def _materialize_cdlib_project(project: Path) -> dict[str, Any]:
    """Copy the shipped locked CDlib project (``pyproject.toml``, ``uv.lock``) into ``project``.

    The worker scripts are never copied: they always run from the package (``cdlib_env.worker_script``). A file that
    already has the shipped bytes is left alone, so a synced ``.venv`` stays valid until the lock changes."""
    shipped = shipped_project_dir()
    copied = []
    project.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "uv.lock"):
        source, target = shipped / name, project / name
        if not source.is_file():
            raise FileNotFoundError(f"the shipped CDlib project lacks {name} ({shipped})")
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
            copied.append(name)
    return {"shipped": str(shipped), "copied": copied}


def _isolated_runtime(*, install: bool, offline: bool, dry_run: bool) -> dict[str, Any]:
    """The locked CDlib project under :func:`cdlib_env.environment_root`, materialised and synced with ``uv``."""
    project = environment_root()
    uv = shutil.which("uv")
    if dry_run:
        receipt = slpa_environment_receipt()
        return {**receipt, "status": "ready" if receipt.get("ready") else "planned",
                "plan": f"copy {shipped_project_dir()}/{{pyproject.toml,uv.lock}} to {project}; "
                        f"uv sync --project {project} --locked"}
    try:
        if os.environ.get("HEDONIC_SLPA_ENV") and (project / "pyproject.toml").is_file():
            # A project the user points at is theirs: never overwrite it, only sync it when it has no environment.
            materialized = {"shipped": str(shipped_project_dir()), "copied": [], "user_project": True}
        else:
            materialized = _materialize_cdlib_project(project)
    except OSError as exc:
        receipt = slpa_environment_receipt()
        return {**receipt, "status": "unavailable", "reason": f"could not materialise the CDlib project: {exc}"}
    os.environ["HEDONIC_SLPA_ENV"] = str(project)
    sync: dict[str, Any] = {"status": "skipped"}
    venv_python = project / ".venv" / "bin" / "python"
    if install and not offline and (materialized["copied"] or not venv_python.exists()):
        if not uv:
            sync = {"status": "failed", "reason": "uv is not installed (https://docs.astral.sh/uv/)"}
        else:
            try:
                completed = subprocess.run(
                    [uv, "sync", "--project", str(project), "--locked", "--no-install-project"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1800,
                )
                sync = {
                    "status": "ready" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "stdout_tail": (completed.stdout or "")[-1000:],
                    "stderr_tail": (completed.stderr or "")[-1000:],
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                sync = {"status": "failed", "reason": str(exc)}
    receipt = slpa_environment_receipt()
    receipt.update({"materialized": materialized, "sync": sync, "direct_pins": dict(SLPA_DIRECT_PINS)})
    if not receipt.get("ready"):
        receipt["reason"] = (sync.get("reason") or sync.get("stderr_tail") or
                             "the CDlib environment is not synced; run `hedonic setup --methods angel`")
    return receipt


def _data_records(
    datasets: list[str],
    *,
    cover: str,
    network_root: Path,
    cache_dir: Path,
    allow_download: bool,
    dry_run: bool,
    max_download_bytes: int | None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in datasets:
        record: dict[str, Any] = {"dataset": name, "cover": cover}
        if dry_run:
            record.update(
                {
                    "status": "planned",
                    "local_root": str(network_root),
                    "catalog_name": MAPEQUATION_SNAP_NAMES[name],
                }
            )
            records[name] = record
            continue
        try:
            prepared = prepare_snap_dataset(
                name,
                cover_variant=cover,
                data_root=network_root,
                cache_dir=cache_dir,
                allow_catalog=allow_download,
                max_download_bytes=max_download_bytes,
            )
        except (SnapLoadError, UnsupportedCoverVariant, OSError) as exc:
            record.update({"status": "unavailable", "reason": str(exc)})
        else:
            record.update(prepared)
        records[name] = record
    return records


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, int]:
    datasets = manifest.get("datasets") or {}
    methods = manifest.get("methods") or {}
    return {
        "datasets_ready": sum(value.get("status") == "ready" for value in datasets.values()),
        "datasets_total": len(datasets),
        "methods_ready": sum(value.get("status") == "ready" for value in methods.values()),
        "methods_total": len(methods),
    }


def _method_record(
    method: str, *, native: dict[str, Any], ncgame: dict[str, Any], isolated: dict[str, Any]
) -> dict[str, Any]:
    """The manifest record of one method's runtime (``status`` ready or unavailable, and where it is)."""
    if method in IN_PROCESS_METHODS:
        return {"status": "ready", "runtime": "hedonic-process"}
    if method in CDLIB_METHODS:
        return {
            "status": "ready" if isolated.get("ready") else "unavailable",
            "runtime": "isolated-cdlib",
            "reason": None if isolated.get("ready") else isolated.get("reason"),
            "receipt": isolated,
        }
    if method == "ncgame":
        return {**ncgame, "runtime": "packaged-worker"}
    if method in ADAPTER_METHODS:
        try:
            from hedonic.experiments.overlapping.methods import method_availability

            availability = method_availability().get(method, {})
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            availability = {"available": False, "reason": str(exc)}
        return {
            "status": "ready" if availability.get("available") else "unavailable",
            "runtime": "hedonic-overlapping-adapter",
            "reason": availability.get("reason"),
            "dependency": availability.get("dependency"),
        }
    if method in {"qoce", "nise", "sse"}:
        runtime = native.get(method, {})
        return {**runtime, "status": "ready" if runtime.get("status") == "ready" else "unavailable",
                "runtime": "official-octave-source"}
    if method in {"svi", "essc"}:
        runtime = native.get(method, {})
        return {**runtime, "status": "ready" if runtime.get("status") == "ready" else "unavailable",
                "runtime": "official-source"}
    return {**native.get(method, {"status": "unavailable"}), "runtime": "external-executable"}


def _build_manifest(
    *,
    datasets: list[str],
    methods: list[str],
    data: dict[str, dict[str, Any]],
    native: dict[str, Any],
    ncgame: dict[str, Any],
    isolated: dict[str, Any],
    network_root: Path,
    cache_dir: Path,
    cover: str,
    pairs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    method_records = {
        method: _method_record(method, native=native, ncgame=ncgame, isolated=isolated) for method in methods
    }
    return {
        "schema": SETUP_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "hedonic_version": importlib.metadata.version("hedonic")
        if _package_installed()
        else None,
        "datasets": data,
        "pairs": dict(pairs or {}),
        "methods": method_records,
        "configuration": {
            "datasets": datasets,
            "methods": methods,
            "cover": cover,
            "network_root": str(network_root),
            "cache_dir": str(cache_dir),
        },
        "upstream": {
            "codeseg_repository": UPSTREAM_CODESEG_URL,
            "snap_repository": UPSTREAM_SNAP_URL,
            "codeseg_checkout": native.get("upstream_codeseg"),
            "snap_checkout": native.get("upstream_snap"),
        },
        # Keep the complete source/build receipt, not only the final binary
        # paths embedded in individual method records.  This is the evidence
        # needed to recreate a result after an upstream checkout changes.
        "native_runtime": native,
        "summary": _manifest_summary(
            {"datasets": data, "methods": method_records}
        ),
    }


def _merge_manifest(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """``current`` laid over ``previous``: records of methods, datasets and pairs set up earlier are kept."""
    if not previous or previous.get("schema") != current.get("schema"):
        return current
    merged = dict(current)
    for key in ("methods", "datasets", "pairs"):
        merged[key] = {**(previous.get(key) or {}), **(current.get(key) or {})}
    merged["summary"] = _manifest_summary(merged)
    return merged


def _package_installed() -> bool:
    try:
        importlib.metadata.version("hedonic")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


# --------------------------------------------------------------------------- system tools
def _cxx_compiler() -> Path | None:
    for name in (os.environ.get("CXX"), "c++", "clang++", "g++"):
        found = _executable(name) if name else None
        if found:
            return found
    return None


def needs_setup(method: str) -> bool:
    """Does ``method`` have an external runtime for ``hedonic setup`` to prepare?"""
    method = METHOD_ALIASES.get(method, method)
    return method not in IN_PROCESS_METHODS and method not in ADAPTER_METHODS


def _ready_before(manifest: dict[str, Any] | None, method: str) -> bool:
    """Was ``method``'s runtime ready in an earlier setup, and is its executable still there?"""
    record = ((manifest or {}).get("methods") or {}).get(method) or {}
    if record.get("status") != "ready":
        return False
    if method in CDLIB_METHODS:
        return bool(slpa_environment_receipt().get("ready"))
    path = record.get("path") or record.get("script")
    return not path or Path(str(path)).exists()


def required_tools(
    methods: Iterable[str],
    *,
    build: bool = True,
    fox_cxx: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The system tools the selected methods' runtimes need, found or not.

    Each row is ``{"tool", "for", "path", "hint"}`` with ``path`` None when the tool is missing. A method whose runtime
    an earlier setup already prepared needs nothing."""
    pending = [m for m in methods if needs_setup(m) and not _ready_before(manifest, m)]
    needs: dict[str, list[str]] = {}

    def need(tool: str, method: str) -> None:
        needs.setdefault(tool, [])
        if method not in needs[tool]:
            needs[tool].append(method)

    for method in pending:
        if method in {"codeseg", "fox", "ncgame", "bigclam", "oslom", "svi"}:
            need("git", method)
        if build and method in {"codeseg", "fox", "bigclam", "neo_kmeans", "oslom", "svi"}:
            need("c++", method)
        if build and method in {"codeseg", "fox"}:
            need("cmake", method)
        if build and method in {"bigclam", "neo_kmeans", "svi"}:
            need("make", method)
        if build and method == "fox":
            need("openmp-c++", method)
        if method in CDLIB_METHODS:
            need("uv", method)
        if method == "essc":
            need("Rscript", method)
        if method in {"qoce", "nise", "sse"}:
            need("octave", method)
    mac = sys.platform == "darwin"
    hints = {
        "git": "install git (macOS: xcode-select --install; Debian/Ubuntu: apt install git)",
        "c++": "install a C++ compiler (macOS: xcode-select --install; Debian/Ubuntu: apt install g++) or set CXX",
        "cmake": "install CMake (macOS: brew install cmake; Debian/Ubuntu: apt install cmake)",
        "make": "install make (macOS: xcode-select --install; Debian/Ubuntu: apt install make)",
        "openmp-c++": ("FOX needs a compiler that accepts -fopenmp; Apple clang does not: brew install gcc "
                       "(or pass --fox-cxx PATH)") if mac else
                      "FOX needs a compiler that accepts -fopenmp (apt install g++) or --fox-cxx PATH",
        "uv": "install uv (https://docs.astral.sh/uv/getting-started/installation/)",
        "Rscript": "install R (https://www.r-project.org)",
        "octave": "install GNU Octave (https://octave.org)",
    }
    rows = []
    for tool, users in needs.items():
        if tool == "c++":
            path = _cxx_compiler()
        elif tool == "openmp-c++":
            path = _fox_openmp_compiler(fox_cxx) if mac or fox_cxx else _executable("g++")
        else:
            path = _executable(tool)
        rows.append({"tool": tool, "for": users, "path": str(path) if path else None, "hint": hints[tool]})
    return rows


def _say(quiet: bool, *parts: Any) -> None:
    if not quiet:
        print(*parts, flush=True)


def _parse_pairs(value: str | None, *, all_pairs: bool) -> list[tuple[str, str]]:
    from hedonic.experiments.overlapping import snap

    if all_pairs and not value:
        value = "paper"
    if value is None:
        return []
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--pairs names no pair; e.g. --pairs dblp:top5000,amazon:top5000 (or --all)")
    out: list[tuple[str, str]] = []
    for name in names:
        for target in (PAPER_PAIRS if name.lower() == "paper" else (name,)):
            pair = snap.split_pair(target)
            if pair not in out:
                out.append(pair)
    return out


def _parse_methods(value: str | None, *, all_methods: bool) -> list[str]:
    if value is None:
        return list(PAPER_BASELINES) if all_methods else []
    lowered = value.strip().lower()
    if lowered in {"", "all", "paper"}:
        return list(PAPER_BASELINES)
    return _parse_selection(value, DEFAULT_METHODS, option="--methods", default=PAPER_BASELINES,
                            aliases=METHOD_ALIASES)


def _setup_parser(prog: str = "hedonic setup") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Download the SNAP network:cover pairs and prepare the external methods' runtimes for `hedonic run`. "
            "Without options it prepares the paper's eight baselines; --all also downloads the paper's eight pairs."
        ),
        epilog=(
            "examples:\n"
            "  hedonic setup --dry-run --all                 # the plan and the tool check, nothing fetched\n"
            "  hedonic setup --all                           # everything `hedonic run queue overlap` needs\n"
            "  hedonic setup --pairs dblp:top5000,amazon:top5000\n"
            "  hedonic setup --methods codeseg,fox,angel\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="the paper's eight pairs and its eight baselines (" + ", ".join(PAPER_BASELINES) + ")")
    parser.add_argument("--pairs", default=None,
                        help="network:cover pairs to download, comma-separated (dblp:top5000, amazon:all, ...; "
                             "'paper' = the paper's eight)")
    parser.add_argument("--methods", default=None,
                        help="methods whose runtimes to prepare, comma-separated ('all' = the paper's baselines; "
                             "known: " + ", ".join(DEFAULT_METHODS) + ")")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the tool check; download, build and write nothing")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR),
                        help="where the SNAP files go (default: the user config's network_root, "
                             "HEDONIC_NETWORKS_DIR, else ~/Databases/Hedonic/Networks)")
    parser.add_argument("--cache-dir", default=None,
                        help="the cache folder, as for `hedonic run exp --cache-dir` and the user config's cache_dir: "
                             "sources, builds and the CDlib environment go to CACHE_DIR/codeseg, where the runs look "
                             "(default: the user config's cache_dir, else ~/.cache/hedonic)")
    parser.add_argument("--manifest", default=None, help="setup manifest (default <cache-dir>/setup_manifest.json)")
    parser.add_argument("--offline", action="store_true", help="never download or clone; inspect existing resources only")
    parser.add_argument("--no-build", action="store_true", help="fetch sources but do not compile them")
    parser.add_argument("--json", action="store_true", help="print the manifest summary as one JSON line")
    parser.add_argument("--max-download-bytes", type=int, default=None)
    parser.add_argument("--fox-cxx", default=None, help="C++ compiler for FOX (auto-detects Homebrew GCC on macOS)")
    # Whole SNAP datasets through the catalogue cache (the runner's --auto-download path) and older flags.
    parser.add_argument("--datasets", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cover", type=canonical_cover_variant, choices=("all", "top5000"), default="all",
                        help=argparse.SUPPRESS)
    parser.add_argument("--skip-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-native", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-native", action="store_true", help=argparse.SUPPRESS)  # building is the default
    parser.add_argument("--codeseg-bin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bigclam-bin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fox-bin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--oslom-bin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--neo-bin", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--upstream-url", default=UPSTREAM_CODESEG_URL, help=argparse.SUPPRESS)
    parser.add_argument("--upstream-ref", default="main", help=argparse.SUPPRESS)
    parser.add_argument("--snap-url", default=UPSTREAM_SNAP_URL, help=argparse.SUPPRESS)
    # SNAP's upstream repository still names its default branch ``master``.
    parser.add_argument("--snap-ref", default="master", help=argparse.SUPPRESS)
    parser.add_argument("--build-timeout", type=float, default=1800.0, help=argparse.SUPPRESS)
    return parser


def _pair_plan(pairs: list[tuple[str, str]], network_root: Path) -> list[dict[str, Any]]:
    from hedonic.experiments.overlapping import snap

    rows = []
    for network, cover in pairs:
        found = snap.pair_files(network, cover, root=network_root)
        downloads = [] if found else snap.pair_downloads(network, cover, root=network_root)
        rows.append({"target": f"{network}:{cover}", "present": bool(found),
                     "graph": str(found.graph) if found else None,
                     "downloads": [{"url": url, "path": str(dest)} for url, dest in downloads]})
    return rows


def _method_plan(method: str) -> str:
    if not needs_setup(method):
        return "nothing to prepare (runs in the hedonic process)"
    if method in CDLIB_METHODS:
        return f"locked CDlib project -> {environment_root()} (uv sync)"
    return {
        "codeseg": "clone SELGroup/CoDeSEG, build code_c++/CoDeSEG with CMake",
        "fox": "clone SELGroup/CoDeSEG, build code_c++/fox (LazyFox) with CMake and an OpenMP compiler",
        "ncgame": "clone SELGroup/CoDeSEG; run code_py/NCG.py through the shipped worker",
        "bigclam": "clone snap-stanford/snap, build examples/bigclam with make",
        "neo_kmeans": "download the authors' NEO-K-Means archive, build it with make",
    }.get(method, "fetch and build its official source")


def setup_main(argv: list[str] | None = None, *, prog: str = "hedonic setup") -> int:
    parser = _setup_parser(prog)
    args = parser.parse_args(argv)
    try:
        pairs = _parse_pairs(args.pairs, all_pairs=args.all)
        methods = _parse_methods(args.methods, all_methods=args.all)
        datasets = (_parse_selection(args.datasets, SNAP_DATASETS, option="--datasets", default=DEFAULT_DATASETS)
                    if args.datasets is not None and not args.skip_data else [])
    except (ValueError, SnapLoadError, UnsupportedCoverVariant) as exc:  # one line, not the usage block
        print(f"{prog}: error: {exc}\n(see `{prog} --help`)", file=sys.stderr)
        return 2
    if not pairs and not methods and not datasets and args.methods is None and args.pairs is None:
        methods = list(PAPER_BASELINES)  # a bare `hedonic setup`: the paper's runtimes
    if args.skip_native:
        methods = [m for m in methods if m in IN_PROCESS_METHODS or m in CDLIB_METHODS or m == "ncgame"]
    quiet = bool(args.json)
    network_root = expand_path(args.network_root)
    cache_dir = codeseg_cache_dir(args.cache_dir) if args.cache_dir else default_cache_dir()
    build = not args.no_build
    # The CDlib project lives under the same cache, so the runners and this setup agree on it.
    os.environ["HEDONIC_CODESEG_CACHE"] = str(cache_dir)
    manifest_path = expand_path(args.manifest) if args.manifest else _manifest_default_path(cache_dir)
    previous = load_setup_manifest(manifest_path)

    tools = required_tools(methods, build=build, fox_cxx=args.fox_cxx, manifest=previous) if methods else []
    missing = [row for row in tools if not row["path"]]
    _say(quiet, f"{prog}: network root {network_root}, cache {cache_dir}")
    if pairs:
        _say(quiet, "pairs:")
        listed: set[str] = set()
        for row in _pair_plan(pairs, network_root):
            if row["present"]:
                _say(quiet, f"  {row['target']:<22} present ({row['graph']})")
            for item in row["downloads"]:
                shared = " (shared with an earlier pair)" if item["url"] in listed else ""
                listed.add(item["url"])
                _say(quiet, f"  {row['target']:<22} download {item['url']}{shared}")
    if methods:
        _say(quiet, "methods:")
        for method in methods:
            state = "ready" if _ready_before(previous, method) else _method_plan(method)
            _say(quiet, f"  {method:<22} {state}")
    if tools:
        _say(quiet, "tools:")
        for row in tools:
            mark = row["path"] or "MISSING"
            _say(quiet, f"  {row['tool']:<22} {mark}   (for {', '.join(row['for'])})")
    if args.dry_run:
        for row in missing:
            _say(quiet, f"missing {row['tool']}: {row['hint']}")
        if args.json:
            print(json.dumps({"dry_run": True, "pairs": [f"{n}:{c}" for n, c in pairs], "methods": methods,
                              "missing_tools": [row["tool"] for row in missing]}, sort_keys=True))
        else:
            print("dry run: nothing was downloaded, built or written")
        return 0
    if missing and not args.offline:
        lines = [f"{prog}: cannot prepare {', '.join(sorted({m for row in missing for m in row['for']}))}: "
                 f"missing {', '.join(row['tool'] for row in missing)}"]
        lines += [f"  {row['tool']}: {row['hint']}" for row in missing]
        print("\n".join(lines), file=sys.stderr)
        return 2

    pair_records: dict[str, dict[str, Any]] = {}
    if pairs:
        from hedonic.experiments.overlapping import snap

        for network, cover in pairs:
            target = f"{network}:{cover}"
            try:
                found = snap.ensure_pair(network, cover, root=network_root, download=not args.offline,
                                         max_download_bytes=args.max_download_bytes,
                                         say=None if quiet else (lambda text: print(f"  {text}", flush=True)))
            except (SnapLoadError, UnsupportedCoverVariant, OSError) as exc:
                pair_records[target] = {"status": "unavailable", "reason": str(exc)}
                _say(quiet, f"  {target}: unavailable ({exc})")
            else:
                pair_records[target] = {"status": "ready", "graph": str(found.graph),
                                        "cover": str(found.cover_file), "downloaded": list(found.downloaded)}
                _say(quiet, f"  {target}: ready")
    data = (
        _data_records(
            datasets,
            cover=args.cover,
            network_root=network_root,
            cache_dir=cache_dir,
            allow_download=not args.offline,
            dry_run=False,
            max_download_bytes=args.max_download_bytes,
        )
        if datasets
        else {}
    )
    native_methods = [m for m in methods if m not in IN_PROCESS_METHODS and m not in CDLIB_METHODS]
    if methods:
        _say(quiet, "preparing runtimes (sources and builds go to the cache; this can take several minutes)")
    native = (
        _native_runtime(
            cache_dir=cache_dir,
            offline=args.offline,
            dry_run=False,
            build_native=build,
            codeseg_bin=args.codeseg_bin,
            bigclam_bin=args.bigclam_bin,
            fox_bin=args.fox_bin,
            oslom_bin=args.oslom_bin,
            neo_bin=args.neo_bin,
            fox_cxx=args.fox_cxx,
            upstream_url=args.upstream_url,
            upstream_ref=args.upstream_ref,
            snap_url=args.snap_url,
            snap_ref=args.snap_ref,
            timeout=args.build_timeout,
            methods=native_methods,
        )
        if native_methods
        else {}
    )
    ncgame = (
        _ncgame_runtime(
            cache_dir=cache_dir,
            offline=args.offline,
            dry_run=False,
            upstream_url=args.upstream_url,
            upstream_ref=args.upstream_ref,
        )
        if "ncgame" in methods
        else {"status": "skipped"}
    )
    isolated = (
        _isolated_runtime(install=True, offline=args.offline, dry_run=False)
        if any(m in CDLIB_METHODS for m in methods)
        else slpa_environment_receipt()
    )
    manifest = _build_manifest(
        datasets=datasets,
        methods=methods,
        data=data,
        native=native,
        ncgame=ncgame,
        isolated=isolated,
        network_root=network_root,
        cache_dir=cache_dir,
        cover=args.cover,
        pairs=pair_records,
    )
    manifest["configuration"].update(
        {
            "offline": bool(args.offline),
            "dry_run": False,
            "build_native": build,
            "max_download_bytes": args.max_download_bytes,
            "manifest_path": str(manifest_path),
        }
    )
    manifest = _merge_manifest(previous, manifest)
    _atomic_json(manifest_path, manifest)
    failed = [name for name in methods if manifest["methods"][name].get("status") != "ready"]
    failed += [name for name, record in pair_records.items() if record.get("status") != "ready"]
    if args.json:
        print(json.dumps({"manifest": str(manifest_path), **manifest["summary"]}, sort_keys=True))
    else:
        for method in methods:
            record = manifest["methods"][method]
            reason = "" if record.get("status") == "ready" else f": {record.get('reason') or 'see the manifest'}"
            print(f"  {method:<22} {record.get('status')}{reason}")
        print(f"manifest {manifest_path}")
        if failed:
            print(f"not ready: {', '.join(failed)}; the logs are in {cache_dir / 'logs'}", file=sys.stderr)
    return 1 if failed else 0


def _doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp codeseg-doctor",
        description="Read-only readiness audit of the SNAP data and the external method runtimes.",
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--methods", default="all", help="comma-separated methods ('all' = the paper's baselines)")
    parser.add_argument("--cover", type=canonical_cover_variant, choices=("all", "top5000"),
                        metavar="{all,top5k}", default="all")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR))
    parser.add_argument("--cache-dir", default=str(default_cache_dir()))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--offline", action="store_true", help="do not consult the catalogue or modify anything")
    parser.add_argument("--require-all", action="store_true", help="return status 1 unless every selected dataset and method is ready")
    return parser


def _doctor_method(method: str, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest and isinstance(manifest.get("methods"), dict):
        value = manifest["methods"].get(method)
        if isinstance(value, dict):
            return dict(value)
    if method in IN_PROCESS_METHODS:
        return {"status": "ready", "runtime": "hedonic-process"}
    if method in CDLIB_METHODS:
        receipt = slpa_environment_receipt()
        return {"status": "ready" if receipt.get("ready") else "unavailable", "receipt": receipt}
    if method == "ncgame":
        return {"status": "unavailable", "reason": "run `hedonic setup --methods ncgame` first"}
    if method in ADAPTER_METHODS:
        # These adapters are in-process Python methods rather than binaries; ask the runner's availability probe.
        try:
            from .methods import method_availability

            availability = method_availability().get(method, {})
            return {
                "status": "ready" if availability.get("available") else "unavailable",
                "runtime": availability.get("runtime"),
                "reason": availability.get("reason"),
            }
        except Exception as exc:  # pragma: no cover - defensive doctor path
            return {"status": "unavailable", "reason": f"availability probe failed: {exc}"}
    return {"status": "unavailable",
            "reason": f"missing {method} executable; run `hedonic setup --methods {method}` or configure HEDONIC_*_BIN"}


def doctor_main(argv: list[str] | None = None) -> int:
    parser = _doctor_parser()
    args = parser.parse_args(argv)
    try:
        datasets = _parse_selection(args.datasets, SNAP_DATASETS, option="--datasets", default=DEFAULT_DATASETS)
        methods = _parse_methods(args.methods, all_methods=True)
    except ValueError as exc:
        parser.error(str(exc))
    network_root = expand_path(args.network_root)
    cache_dir = expand_path(args.cache_dir)
    manifest_path = expand_path(args.manifest) if args.manifest else _manifest_default_path(cache_dir)
    manifest = load_setup_manifest(manifest_path)
    data: dict[str, Any] = {}
    for name in datasets:
        try:
            prepared = prepare_snap_dataset(
                name,
                cover_variant=args.cover,
                data_root=network_root,
                cache_dir=cache_dir,
                allow_catalog=False,
            )
        except (SnapLoadError, UnsupportedCoverVariant, OSError) as exc:
            prior = (manifest or {}).get("datasets", {}).get(name, {})
            # A setup manifest can prove a previous catalogue download even
            # when the local archive root is intentionally different.
            cached_locator = prior.get("cache_root") or prior.get("network_locator")
            cached_ready = bool(
                prior.get("status") == "ready"
                and cached_locator
                and Path(str(cached_locator)).expanduser().exists()
            )
            status = "ready" if cached_ready else "unavailable"
            data[name] = {"status": status or "unavailable", "reason": str(exc)}
        else:
            if prepared.get("status") == "unavailable":
                prior = (manifest or {}).get("datasets", {}).get(name, {})
                cached_locator = prior.get("cache_root") or prior.get("network_locator")
                if (
                    prior.get("status") == "ready"
                    and cached_locator
                    and Path(str(cached_locator)).expanduser().exists()
                ):
                    prepared = {
                        "status": "ready",
                        "dataset": name,
                        "cover_variant": args.cover,
                        "source_kind": "snap_catalog",
                        "cache_root": str(cached_locator),
                    }
            data[name] = dict(prepared)
    methods_report = {method: _doctor_method(method, manifest) for method in methods}
    report = {
        "schema": SETUP_SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "datasets": data,
        "methods": methods_report,
        "ready": all(item.get("status") == "ready" for item in data.values())
        and all(item.get("status") == "ready" for item in methods_report.values()),
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["ready"] or not args.require_all else 1


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_METHODS",
    "PAPER_BASELINES",
    "PAPER_METHODS",
    "PAPER_PAIRS",
    "SNAP_DATASETS",
    "SETUP_SCHEMA_VERSION",
    "doctor_main",
    "load_setup_manifest",
    "needs_setup",
    "required_tools",
    "setup_main",
]
