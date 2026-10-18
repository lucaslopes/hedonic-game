"""Shared configuration paths for hedonic experiments.

Resolution order for path settings (first present wins per field when
resolving, after applying sources in this order):

1. Explicit function / CLI arguments
2. Environment variables (``HEDONIC_DBLP_DIR``, ``HEDONIC_SYNTHETIC_DIR``,
   ``HEDONIC_NETWORKS_DIR``, ``HEDONIC_OUTPUT_DIR``, ``HEDONIC_DATA_DIR``,
   ``HEDONIC_CONFIG``)
3. The user config (``~/.config/hedonic/config.toml``): ``[defaults]
   network_root`` names the SNAP networks folder
4. Built-in defaults: one folder, ``~/Databases/Hedonic`` (``HEDONIC_DATA_DIR``),
   holds the networks (``Networks``), the caches (``cache``), the overlap runs
   (``Overlap``, ``HEDONIC_OUTPUT_DIR``) and the disjoint runs (``Disjoint``,
   ``HEDONIC_DISJOINT_OUTPUT_DIR``)

A pip-installed ``hedonic`` has no repository next to it, so no default points
into one. A relative path that starts with ``artifacts`` is anchored at
``~/Databases/Hedonic/Overlap`` (``HEDONIC_OUTPUT_DIR`` moves it), whatever the working
directory.

An optional experiment TOML (``--config path.toml``) may set the same paths::

    [paths]
    networks_dir = "~/Databases/Hedonic/Networks"
    output_dir = "~/Databases/Hedonic/Overlap"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TOML config requires Python 3.11+ (tomllib) or the tomli package"
        ) from exc


def expand_path(value: str | Path) -> Path:
    """Expand ``~`` and anchor ``artifacts/...`` at the run-output root.

    Dataset paths remain user-relative or explicitly supplied paths. A
    relative path beginning with ``artifacts`` is anchored at
    :data:`RUNS_ROOT`, so a default output behaves the same from any working
    directory and never lands inside the installed package.
    """
    path = Path(os.path.expanduser(str(value).strip())).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] == "artifacts":
        path = RUNS_ROOT / path
    return path


def _env_path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.getenv(name) or default)).expanduser()


def _user_network_root() -> str | None:
    """``[defaults] network_root`` of the user config, when it sets one (read without importing the CLI)."""
    try:
        from hedonic.experiments.overlapping import userconfig

        value = userconfig.defaults().get("network_root")
    except (Exception, SystemExit):  # noqa: BLE001 - an unreadable user config never breaks an import
        return None
    return str(value) if value else None


# One user folder holds the data (downloaded networks, normalized caches) and the run outputs, for every user and
# every verb; nothing defaults to a repository or to the working directory. A run only adds its own folder there.
DATA_ROOT = _env_path("HEDONIC_DATA_DIR", "~/Databases/Hedonic")
RUNS_ROOT = _env_path("HEDONIC_OUTPUT_DIR", str(DATA_ROOT / "Overlap"))
DISJOINT_RUNS_ROOT = _env_path("HEDONIC_DISJOINT_OUTPUT_DIR", str(DATA_ROOT / "Disjoint"))
ARTIFACTS_DIR = RUNS_ROOT / "artifacts"
DISJOINT_ARTIFACTS_DIR = ARTIFACTS_DIR / "disjoint"
OVERLAPPING_ARTIFACTS_DIR = ARTIFACTS_DIR / "overlapping"
PAPER_ARTIFACTS_DIR = ARTIFACTS_DIR / "papers" / "overlapping_communities"
EVIDENCE_DIR = ARTIFACTS_DIR / "evidence"
SNAP_CACHE_DIR = DATA_ROOT / "cache" / "snap"
CODESEG_CACHE_DEFAULT = "~/.cache/hedonic/codeseg"


def codeseg_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Where ``hedonic setup`` builds the external methods and every run looks for them.

    An explicit ``cache_dir`` (a ``--cache-dir`` flag) gives ``cache_dir/codeseg``; otherwise
    ``HEDONIC_CODESEG_CACHE``, else ``codeseg/`` under the user config's ``cache_dir`` (``hedonic config set
    cache_dir DIR``), else ``~/.cache/hedonic/codeseg``. Read when called, never at import.
    """
    if cache_dir:
        return expand_path(cache_dir) / "codeseg"
    configured = os.environ.get("HEDONIC_CODESEG_CACHE")
    if configured:
        return expand_path(configured)
    try:
        from hedonic.experiments.overlapping import userconfig

        base = userconfig.defaults().get("cache_dir")
    except Exception:  # noqa: BLE001 - an unreadable user config is reported by the commands that read it
        base = None
    return expand_path(base) / "codeseg" if base else expand_path(CODESEG_CACHE_DEFAULT)


DEFAULT_NETWORKS_DIR = expand_path(_user_network_root() or DATA_ROOT / "Networks")
DEFAULT_DBLP_DIR = DEFAULT_NETWORKS_DIR / "DBLP"
DEFAULT_SYNTHETIC_DIR = DATA_ROOT / "PHYSA" / "Synthetic_Networks" / "V1020"
# The historical V1020 tree (the PHYSA archive) is an input archive. Experiment
# writers never place generated output in it (or in a descendant directory).
# Keep this in the shared config module so every disjoint entry point applies
# the same guard instead of comparing one exact path string.
ARCHIVED_V1020_DIR = expand_path("~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020").resolve()
# ``OUTPUT_DIR`` is the canonical output root. Individual experiments add
# their own subdirectory below it.
DEFAULT_OUTPUT_DIR = RUNS_ROOT

# Default experiment-TOML search order (relative to cwd).
DEFAULT_TOML_NAMES: tuple[str, ...] = (
    "configs/hedonic.toml",
    "hedonic.toml",  # legacy cwd root
    "config/hedonic.toml",  # legacy singular
    ".hedonic.toml",
)

# Module-level paths (reloaded by reload_paths / load_config_file).
DBLP_DIR = expand_path(os.getenv("HEDONIC_DBLP_DIR", str(DEFAULT_DBLP_DIR)))
SYNTHETIC_DIR = expand_path(
    os.getenv("HEDONIC_SYNTHETIC_DIR", str(DEFAULT_SYNTHETIC_DIR))
)
OUTPUT_DIR = expand_path(os.getenv("HEDONIC_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
NETWORKS_DIR = expand_path(
    os.getenv("HEDONIC_NETWORKS_DIR", str(DEFAULT_NETWORKS_DIR))
)


def ensure_not_archived_v1020(
    path: str | Path,
    *,
    label: str = "output",
) -> Path:
    """Return a resolved path unless it would write inside archived V1020.

    The archive is intentionally read-only.  Checking ancestry (rather than
    only equality) prevents an easy-to-miss ``archive/recovery`` write from
    mutating the source data while still allowing the archive to be used as a
    read-only input path.
    """
    resolved = expand_path(path).resolve()
    try:
        resolved.relative_to(ARCHIVED_V1020_DIR)
    except ValueError:
        return resolved
    raise ValueError(
        f"Refusing to use archived V1020 as {label}: {resolved}. "
        "Choose a separate output root (for example "
        "~/Databases/Hedonic/Overlap/disjoint/v1020)."
    )

# Last successfully loaded TOML (full table) and its path; for experiments.
_LOADED_TOML: dict[str, Any] = {}
_LOADED_TOML_PATH: Path | None = None


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return expand_path(s)


def find_config_file(
    explicit: str | Path | None = None,
    *,
    search_cwd: bool = True,
) -> Path | None:
    """Locate a TOML config file.

    Order:
      1. ``explicit`` argument
      2. ``HEDONIC_CONFIG`` env
      3. ``configs/hedonic.toml`` (and legacy names) under cwd
    """
    if explicit is not None:
        p = expand_path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        return p.resolve()

    env = os.getenv("HEDONIC_CONFIG")
    if env:
        p = expand_path(env)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"HEDONIC_CONFIG not a file: {p}")

    if not search_cwd:
        return None

    cwd = Path.cwd()
    for name in DEFAULT_TOML_NAMES:
        candidate = cwd / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_toml(path: str | Path) -> dict[str, Any]:
    """Parse a TOML file into a plain dict (empty dict if missing tables)."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return data


def _paths_from_toml(data: dict[str, Any]) -> dict[str, Path | None]:
    """Extract known path keys from ``[paths]`` (and legacy top-level aliases)."""
    paths_tbl = data.get("paths") if isinstance(data.get("paths"), dict) else {}
    merged = {
        **{
            k: data.get(k)
            for k in ("dblp_dir", "synthetic_dir", "networks_dir", "output_dir")
        },
        **paths_tbl,
    }
    return {
        "dblp_dir": _as_path(merged.get("dblp_dir") or merged.get("dblp")),
        "synthetic_dir": _as_path(
            merged.get("synthetic_dir") or merged.get("synthetic")
        ),
        "networks_dir": _as_path(
            merged.get("networks_dir") or merged.get("networks")
        ),
        "output_dir": _as_path(
            merged.get("output_dir") or merged.get("artifacts_dir")
        ),
    }


def apply_path_overrides(
    *,
    dblp_dir: str | Path | None = None,
    synthetic_dir: str | Path | None = None,
    networks_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Set module-level path globals from explicit values (non-None only)."""
    global DBLP_DIR, SYNTHETIC_DIR, NETWORKS_DIR, OUTPUT_DIR
    if dblp_dir is not None:
        DBLP_DIR = expand_path(dblp_dir)
    if synthetic_dir is not None:
        SYNTHETIC_DIR = expand_path(synthetic_dir)
    if networks_dir is not None:
        NETWORKS_DIR = expand_path(networks_dir)
    if output_dir is not None:
        OUTPUT_DIR = expand_path(output_dir)
    return DBLP_DIR, SYNTHETIC_DIR, OUTPUT_DIR


def load_config_file(
    path: str | Path | None = None,
    *,
    search_cwd: bool = True,
    apply: bool = True,
) -> dict[str, Any]:
    """Load TOML config; optionally apply ``[paths]`` into module globals.

    Env vars still override TOML when applied. Returns the full parsed table
    (may be empty if no file found).
    """
    global _LOADED_TOML, _LOADED_TOML_PATH
    found = find_config_file(path, search_cwd=search_cwd) if path or search_cwd else None
    if path is not None and found is None:
        found = find_config_file(path, search_cwd=False)

    if found is None:
        _LOADED_TOML = {}
        _LOADED_TOML_PATH = None
        return {}

    data = read_toml(found)
    _LOADED_TOML = data
    _LOADED_TOML_PATH = found

    if apply:
        extracted = _paths_from_toml(data)
        if extracted["dblp_dir"] is not None:
            apply_path_overrides(dblp_dir=extracted["dblp_dir"])
        if extracted["synthetic_dir"] is not None:
            apply_path_overrides(synthetic_dir=extracted["synthetic_dir"])
        if extracted["networks_dir"] is not None:
            apply_path_overrides(networks_dir=extracted["networks_dir"])
        if extracted["output_dir"] is not None:
            apply_path_overrides(output_dir=extracted["output_dir"])
        _apply_env_overrides()

    return data


def _apply_env_overrides() -> None:
    """Apply env vars on top of current module paths (env wins)."""
    global DBLP_DIR, SYNTHETIC_DIR, NETWORKS_DIR, OUTPUT_DIR
    if "HEDONIC_DBLP_DIR" in os.environ:
        DBLP_DIR = expand_path(os.environ["HEDONIC_DBLP_DIR"])
    if "HEDONIC_SYNTHETIC_DIR" in os.environ:
        SYNTHETIC_DIR = expand_path(os.environ["HEDONIC_SYNTHETIC_DIR"])
    if "HEDONIC_NETWORKS_DIR" in os.environ:
        NETWORKS_DIR = expand_path(os.environ["HEDONIC_NETWORKS_DIR"])
    if "HEDONIC_OUTPUT_DIR" in os.environ:
        OUTPUT_DIR = expand_path(os.environ["HEDONIC_OUTPUT_DIR"])


def reload_paths(
    *,
    config_path: str | Path | None = None,
    search_cwd: bool = False,
) -> tuple[Path, Path]:
    """Re-read path settings (tests + after mutating the environment).

    Returns ``(DBLP_DIR, SYNTHETIC_DIR)`` for backward compatibility.
    When ``config_path`` is set or ``search_cwd`` is True, TOML is applied
    first, then env overrides.
    """
    global DBLP_DIR, SYNTHETIC_DIR, NETWORKS_DIR, OUTPUT_DIR
    DBLP_DIR = expand_path(DEFAULT_DBLP_DIR)
    SYNTHETIC_DIR = expand_path(DEFAULT_SYNTHETIC_DIR)
    NETWORKS_DIR = expand_path(DEFAULT_NETWORKS_DIR)
    OUTPUT_DIR = expand_path(DEFAULT_OUTPUT_DIR)
    if config_path is not None or search_cwd:
        load_config_file(config_path, search_cwd=search_cwd, apply=True)
    else:
        _apply_env_overrides()
    return DBLP_DIR, SYNTHETIC_DIR


def get_toml_section(name: str) -> dict[str, Any]:
    """Return a table from the last loaded TOML (empty dict if missing)."""
    sec = _LOADED_TOML.get(name)
    return dict(sec) if isinstance(sec, dict) else {}


def get_loaded_config_path() -> Path | None:
    """Path of the last TOML loaded via ``load_config_file``, if any."""
    return _LOADED_TOML_PATH


def resolve_experiment_paths(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    experiment_section: str | None = None,
    search_cwd: bool = True,
) -> dict[str, Any]:
    """Resolve data/output dirs for an experiment CLI.

    Priority per field: explicit CLI arg → env → experiment TOML section →
    ``[paths]`` TOML → module defaults.

    Returns dict with keys: ``data_dir``, ``output_dir``, ``config_path``,
    ``toml`` (full table), ``section`` (experiment table).
    """
    toml_data = load_config_file(config_path, search_cwd=search_cwd, apply=True)
    section: dict[str, Any] = {}
    if experiment_section:
        raw = toml_data.get(experiment_section)
        if isinstance(raw, dict):
            section = dict(raw)

    resolved_data = data_dir
    if resolved_data is None:
        resolved_data = section.get("data_dir") or section.get("dblp_dir")
    if resolved_data is None:
        resolved_data = DBLP_DIR
    else:
        resolved_data = expand_path(resolved_data)

    resolved_out = output_dir
    if resolved_out is None:
        resolved_out = section.get("output_dir")
    if resolved_out is None:
        resolved_out = OUTPUT_DIR
    else:
        resolved_out = expand_path(resolved_out)

    return {
        "data_dir": Path(resolved_data),
        "output_dir": Path(resolved_out),
        "config_path": get_loaded_config_path(),
        "toml": toml_data,
        "section": section,
    }
