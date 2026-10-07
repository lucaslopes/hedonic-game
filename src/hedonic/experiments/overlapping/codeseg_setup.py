"""Bootstrap and audit the external runtimes used by the CoDeSEG protocol.

The nine-method experiment deliberately keeps third-party implementations at
runtime boundaries:

* native ``community_hedonic``, Louvain, Leiden and FLPA stay in hedonic's
  pinned igraph environment;
* SLPA and DER run through the locked CDlib worker project;
* NcGame runs through the packaged worker against the authors' checkout; and
* CoDeSEG, Bigclam and LazyFox are discovered as executables, with optional
  source checkouts/builds but no unverified substitute implementation.

``codeseg-setup`` is resumable and writes a JSON manifest.  ``codeseg-doctor``
is read-only and can be used in CI to fail before a long experiment when a
dataset, executable, or isolated environment is missing.
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

from hedonic.experiments.config import NETWORKS_DIR, expand_path
from hedonic.experiments.overlapping.baselines import slpa_environment_receipt
from hedonic.experiments.overlapping.snap import (
    MAPEQUATION_SNAP_NAMES,
    SPECS,
    SnapLoadError,
    UnsupportedCoverVariant,
    prepare_snap_dataset,
)


SETUP_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = expand_path(
    os.environ.get("HEDONIC_CODESEG_CACHE", "~/.cache/hedonic/codeseg")
)
# The first reproducibility target is the full SNAP ``com-DBLP`` instance.
# Keep the complete catalogue below for explicit multi-network runs, but do
# not make a fresh ``codeseg-setup --all`` download several additional
# multi-gigabyte archives that are outside the current benchmark scope.
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
DEFAULT_METHODS: tuple[str, ...] = (
    "codeseg",
    "slpa",
    "bigclam",
    "ncgame",
    "fox",
    "louvain",
    "der",
    "leiden",
    "flpa",
    "community_hedonic",
    "hedonic_local",
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "angel",
    "infomap",
    "demon",
    "cpm",
    "link_clustering",
    "oslom",
    "neo_kmeans",
    "nise",
    "sse",
    "qoce",
    "svi",
    "essc",
)
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
    value: str | None, allowed: Iterable[str], *, option: str, default: tuple[str, ...]
) -> list[str]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(default)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(
            f"unknown {option}: {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    if not selected:
        raise ValueError(f"{option} must contain at least one value")
    return list(dict.fromkeys(selected))


def _repo_root() -> Path:
    # codeseg_setup.py -> overlapping -> experiments -> hedonic -> src -> repo
    return Path(__file__).resolve().parents[4]


def _default_worker_path() -> Path:
    return Path(__file__).resolve().parent / "ncgame_worker.py"


def _manifest_default_path(cache_dir: Path) -> Path:
    configured = os.environ.get("HEDONIC_CODESEG_MANIFEST")
    return expand_path(configured) if configured else cache_dir / "setup_manifest.json"


def load_setup_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the most recent setup manifest, if one exists."""
    candidate = expand_path(path) if path else _manifest_default_path(DEFAULT_CACHE_DIR)
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
) -> dict[str, Any]:
    source_dir = cache_dir / "upstream" / "CoDeSEG"
    snap_dir = cache_dir / "upstream" / "snap"
    oslom_dir = cache_dir / "upstream" / "oslom2"
    upstream = _ensure_checkout(
        url=upstream_url,
        destination=source_dir,
        ref=upstream_ref,
        offline=offline,
        dry_run=dry_run,
    )
    snap_source = _ensure_checkout(
        url=snap_url,
        destination=snap_dir,
        ref=snap_ref,
        offline=offline,
        dry_run=dry_run,
    )
    oslom_source = _ensure_checkout(
        url=UPSTREAM_OSLOM_URL,
        destination=oslom_dir,
        ref=UPSTREAM_OSLOM_REF,
        offline=offline,
        dry_run=dry_run,
    )
    neo_source = _ensure_neo_source(cache_dir=cache_dir, offline=offline, dry_run=dry_run)
    svi_source = _ensure_svi_source(
        cache_dir=cache_dir,
        offline=offline,
        dry_run=dry_run,
        build_native=build_native,
        timeout=timeout,
    )
    essc_source = _ensure_r_package(
        cache_dir=cache_dir,
        offline=offline,
        dry_run=dry_run,
        install=build_native,
        timeout=timeout,
    )
    octave_sources = _ensure_octave_sources(
        cache_dir=cache_dir, offline=offline, dry_run=dry_run, build_native=build_native
    )

    if build_native and not dry_run:
        cmake = shutil.which("cmake")
        make = shutil.which("make")
        if cmake and upstream["status"] == "ready":
            for name, source in (
                ("codeseg", source_dir / "code_c++" / "CoDeSEG"),
                ("fox", source_dir / "code_c++" / "fox"),
            ):
                if source.is_dir():
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
                                [cmake, "--build", str(build_dir), "--parallel"],
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
    command = (
        f"{shlex.quote(python)} -m hedonic.experiments.overlapping.ncgame_worker "
        f"--upstream-root {shlex.quote(str(root))} --input {{input}} --ground-truth {{ground_truth}} "
        f"--output {{output}} --dataset {{dataset}}"
    )
    return {
        "status": "ready" if ready else "unavailable",
        "reason": None if ready else f"missing NcGame script: {script}",
        "upstream_root": str(root),
        "script": str(script),
        "script_sha256": _sha256_file(script),
        "command": command if ready else None,
        "python": python,
        "checkout": checkout,
    }


def _isolated_runtime(*, install: bool, offline: bool, dry_run: bool) -> dict[str, Any]:
    receipt = slpa_environment_receipt()
    project = Path(receipt["root"])
    configured_environment = os.environ.get("HEDONIC_SLPA_ENV")
    repository_project = _repo_root() / "tools" / "slpa_env"
    # A wheel does not contain the repository's ``tools/`` tree.  Materialize
    # the same pinned worker project below the setup cache in that case so the
    # external CDlib dependency remains isolated on fresh machines as well.
    if not (project / "pyproject.toml").is_file() and (
        not (repository_project / "pyproject.toml").is_file()
        or project != repository_project
    ):
        if not configured_environment and project.name != "cdlib_env":
            project = project / "cdlib_env"
        if dry_run:
            return {
                "protocol_version": "cdlib-slpa-isolated-v1",
                "root": str(project),
                "uv": shutil.which("uv"),
                "project_python": None,
                "execution_preference": None,
                "files": {
                    name: {"path": str(project / name), "sha256": None}
                    for name in (
                        "pyproject.toml",
                        "uv.lock",
                        "slpa_worker.py",
                        "der_worker.py",
                        "cdlib_worker.py",
                        "link_clustering_worker.py",
                    )
                },
                "direct_pins": {
                    "cdlib": "0.4.0",
                    "networkx": "3.6.1",
                    "numpy": "2.3.3",
                    "python-igraph": "1.0.0",
                },
                "ready": False,
                "isolation": "dedicated uv project; never imported in the lucas-igraph process",
                "status": "planned",
            }
        project.mkdir(parents=True, exist_ok=True)
        pyproject = project / "pyproject.toml"
        if not pyproject.exists():
            pyproject.write_text(
                """[project]
name = \"hedonic-codeseg-cdlib\"
version = \"0.0.0\"
requires-python = \">=3.12,<3.13\"
dependencies = [
  \"cdlib==0.4.0\",
  \"networkx==3.6.1\",
  \"numpy==2.3.3\",
  \"python-igraph==1.0.0\",
]

[tool.uv]
package = false
""",
                encoding="utf-8",
            )
        package_root = Path(__file__).resolve().parent
        for worker_name in (
            "slpa_worker.py",
            "der_worker.py",
            "cdlib_worker.py",
            "link_clustering_worker.py",
        ):
            target = project / worker_name
            source = package_root / worker_name
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copyfile(source, target)
        lock = project / "uv.lock"
        if not lock.exists() and not dry_run and shutil.which("uv"):
            try:
                subprocess.run(
                    [shutil.which("uv") or "uv", "lock", "--project", str(project)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=900,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        if not lock.exists():
            lock.write_text(
                "# Generated by hedonic codeseg-setup; direct pins are in pyproject.toml.\n",
                encoding="utf-8",
            )
        project_python = project / ".venv" / "bin" / "python"
        if install and not dry_run and not offline and not project_python.exists():
            uv = shutil.which("uv")
            if uv:
                try:
                    subprocess.run(
                        [uv, "sync", "--project", str(project), "--locked"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=900,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if not project_python.exists():
                try:
                    subprocess.run(
                        [sys.executable, "-m", "venv", str(project / ".venv")],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=180,
                    )
                    pip = project_python.parent / "pip"
                    if pip.exists():
                        subprocess.run(
                            [
                                str(pip),
                                "install",
                                "cdlib==0.4.0",
                                "networkx==3.6.1",
                                "numpy==2.3.3",
                                "python-igraph==1.0.0",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=900,
                        )
                except (OSError, subprocess.TimeoutExpired):
                    pass
        os.environ["HEDONIC_SLPA_ENV"] = str(project)
        receipt = slpa_environment_receipt()
    else:
        project = Path(receipt["root"])
    if install and not dry_run and not offline and receipt.get("uv") and (project / "pyproject.toml").is_file():
        try:
            completed = subprocess.run(
                [str(receipt["uv"]), "sync", "--project", str(project), "--locked"],
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
            receipt["sync"] = {
                "status": "ready" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-1000:],
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            receipt["sync"] = {"status": "failed", "reason": str(exc)}
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
) -> dict[str, Any]:
    method_records: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method in {
            "louvain",
            "leiden",
            "flpa",
            "community_hedonic",
            "hedonic_local",
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
        }:
            method_records[method] = {
                "status": "ready",
                "runtime": "hedonic-pinned-igraph",
            }
        elif method in {"slpa", "der"}:
            method_records[method] = {
                "status": "ready" if isolated.get("ready") else "unavailable",
                "runtime": "isolated-cdlib",
                "receipt": isolated,
            }
        elif method == "ncgame":
            method_records[method] = {**ncgame, "runtime": "packaged-worker"}
        elif method in {"angel", "infomap", "demon", "cpm", "link_clustering"}:
            try:
                from hedonic.experiments.overlapping.methods import method_availability

                availability = method_availability().get(method, {})
            except Exception as exc:
                availability = {"available": False, "reason": str(exc)}
            method_records[method] = {
                "status": "ready" if availability.get("available") else "unavailable",
                "runtime": "hedonic-overlapping-adapter",
                "reason": availability.get("reason"),
                "dependency": availability.get("dependency"),
            }
        elif method in {"qoce", "nise", "sse"}:
            runtime = native.get(method, {})
            method_records[method] = {
                **runtime,
                "status": "ready" if runtime.get("status") == "ready" else "unavailable",
                "runtime": "official-octave-source",
            }
        elif method in {"svi", "essc"}:
            runtime = native.get(method, {})
            method_records[method] = {
                **runtime,
                "status": "ready" if runtime.get("status") == "ready" else "unavailable",
                "runtime": "official-source",
            }
        else:
            method_records[method] = {
                **native.get(method, {"status": "unavailable"}),
                "runtime": "external-executable",
            }
    return {
        "schema": SETUP_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "hedonic_version": importlib.metadata.version("hedonic")
        if _package_installed()
        else None,
        "datasets": data,
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


def _package_installed() -> bool:
    try:
        importlib.metadata.version("hedonic")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp codeseg-setup",
        description="Download/cache SNAP data and prepare the CoDeSEG method runtimes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="prepare the default full-DBLP dataset and all registered methods",
    )
    parser.add_argument("--datasets", default="all", help="comma-separated datasets or all")
    parser.add_argument("--methods", default="all", help="comma-separated methods or all")
    parser.add_argument("--cover", choices=("all", "top5000"), default="all")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-download-bytes", type=int, default=None)
    parser.add_argument("--offline", action="store_true", help="never download or clone; inspect existing resources only")
    parser.add_argument("--dry-run", action="store_true", help="write a planned manifest without downloading, building, or changing environments")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--build-native", action="store_true", help="configure/build CoDeSEG, LazyFox, and Bigclam source checkouts")
    parser.add_argument("--codeseg-bin", default=None)
    parser.add_argument("--bigclam-bin", default=None)
    parser.add_argument("--fox-bin", default=None)
    parser.add_argument("--oslom-bin", default=None)
    parser.add_argument("--neo-bin", default=None)
    parser.add_argument(
        "--fox-cxx",
        default=None,
        help="C++ compiler for Fox (auto-detects Homebrew GCC on macOS)",
    )
    parser.add_argument("--upstream-url", default=UPSTREAM_CODESEG_URL)
    parser.add_argument("--upstream-ref", default="main")
    parser.add_argument("--snap-url", default=UPSTREAM_SNAP_URL)
    # SNAP's upstream repository still names its default branch ``master``;
    # keep this explicit so ``--build-native`` works without a manual ref.
    parser.add_argument("--snap-ref", default="master")
    parser.add_argument("--build-timeout", type=float, default=1800.0)
    return parser


def setup_main(argv: list[str] | None = None) -> int:
    parser = _setup_parser()
    args = parser.parse_args(argv)
    try:
        datasets = _parse_selection(args.datasets, SNAP_DATASETS, option="--datasets", default=DEFAULT_DATASETS)
        methods = _parse_selection(args.methods, DEFAULT_METHODS, option="--methods", default=DEFAULT_METHODS)
    except ValueError as exc:
        parser.error(str(exc))
    # ``--all`` is intentionally explicit in the public docs; selecting no
    # subset has the same semantics for scriptable callers.
    if args.all:
        datasets, methods = list(DEFAULT_DATASETS), list(DEFAULT_METHODS)
    network_root = expand_path(args.network_root)
    cache_dir = expand_path(args.cache_dir)
    # Let the baseline adapter resolve the same cache when the repository's
    # ``tools/slpa_env`` project is absent (for example from an installed
    # wheel).  This is process-local; the manifest records the durable path.
    os.environ["HEDONIC_CODESEG_CACHE"] = str(cache_dir)
    manifest_path = expand_path(args.manifest) if args.manifest else _manifest_default_path(cache_dir)
    data = (
        _data_records(
            datasets,
            cover=args.cover,
            network_root=network_root,
            cache_dir=cache_dir,
            allow_download=not args.offline,
            dry_run=args.dry_run or args.skip_data,
            max_download_bytes=args.max_download_bytes,
        )
        if not args.skip_data
        else {name: {"dataset": name, "status": "skipped"} for name in datasets}
    )
    native = (
        _native_runtime(
            cache_dir=cache_dir,
            offline=args.offline,
            dry_run=args.dry_run,
            build_native=args.build_native,
            codeseg_bin=args.codeseg_bin,
            bigclam_bin=args.bigclam_bin,
            fox_bin=args.fox_bin,
            oslom_bin=getattr(args, "oslom_bin", None),
            neo_bin=getattr(args, "neo_bin", None),
            fox_cxx=args.fox_cxx,
            upstream_url=args.upstream_url,
            upstream_ref=args.upstream_ref,
            snap_url=args.snap_url,
            snap_ref=args.snap_ref,
            timeout=args.build_timeout,
        )
        if not args.skip_native
        else {name: {"status": "skipped"} for name in ("codeseg", "bigclam", "fox", "oslom", "neo_kmeans", "svi", "essc", "qoce", "nise", "sse")}
    )
    ncgame = _ncgame_runtime(
        cache_dir=cache_dir,
        offline=args.offline,
        dry_run=args.dry_run,
        upstream_url=args.upstream_url,
        upstream_ref=args.upstream_ref,
    )
    isolated = _isolated_runtime(
        install=not args.skip_native,
        offline=args.offline,
        dry_run=args.dry_run,
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
    )
    manifest["configuration"].update(
        {
            "offline": bool(args.offline),
            "dry_run": bool(args.dry_run),
            "build_native": bool(args.build_native),
            "max_download_bytes": args.max_download_bytes,
            "manifest_path": str(manifest_path),
        }
    )
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), **manifest["summary"]}, sort_keys=True))
    return 0


def _doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp codeseg-doctor",
        description="Read-only readiness audit for the CoDeSEG reproduction.",
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--cover", choices=("all", "top5000"), default="all")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--offline", action="store_true", help="do not consult the catalogue or modify anything")
    parser.add_argument("--require-all", action="store_true", help="return status 1 unless every selected dataset and method is ready")
    return parser


def _doctor_method(method: str, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest and isinstance(manifest.get("methods"), dict):
        value = manifest["methods"].get(method)
        if isinstance(value, dict):
            return dict(value)
    if method in {
        "louvain",
        "leiden",
        "flpa",
        "community_hedonic",
        "hedonic_local",
        "hedonic_multiphase",
        "hedonic_multiphase_x10",
        "hedonic_multiphase_x100",
    }:
        return {"status": "ready", "runtime": "hedonic-pinned-igraph"}
    if method in {"slpa", "der"}:
        receipt = slpa_environment_receipt()
        return {"status": "ready" if receipt.get("ready") else "unavailable", "receipt": receipt}
    if method == "ncgame":
        return {"status": "unavailable", "reason": "run codeseg-setup first"}
    if method in {"angel", "infomap", "demon", "cpm", "link_clustering"}:
        # Keep the doctor useful even when the manifest was moved or deleted.
        # These adapters are local Python/CDlib methods rather than binaries;
        # ask the same availability probe used by the runner instead of
        # incorrectly reporting a missing executable.
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
    return {"status": "unavailable", "reason": f"missing {method} executable; run codeseg-setup or configure HEDONIC_*_BIN"}


def doctor_main(argv: list[str] | None = None) -> int:
    parser = _doctor_parser()
    args = parser.parse_args(argv)
    try:
        datasets = _parse_selection(args.datasets, SNAP_DATASETS, option="--datasets", default=DEFAULT_DATASETS)
        methods = _parse_selection(args.methods, DEFAULT_METHODS, option="--methods", default=DEFAULT_METHODS)
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
    "PAPER_METHODS",
    "SNAP_DATASETS",
    "SETUP_SCHEMA_VERSION",
    "doctor_main",
    "load_setup_manifest",
    "setup_main",
]
