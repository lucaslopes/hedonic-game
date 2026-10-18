"""Files shipped inside the package: run configs and external-method workers.

A pip-installed ``hedonic`` has no repository next to it, so every config a
queue item names (``configs/start-quality-hoc-local.toml``) and every worker
script an external method runs in its own environment is package data under
``hedonic/experiments/configs`` and ``hedonic/experiments/workers``. A path
that exists as given (relative to the working directory, or absolute) always
wins, so a user's own copy of a config is read instead of the shipped one.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
WORKER_DIR = PACKAGE_DIR / "workers"


def config_path(name: str | Path) -> Path:
    """Resolve a config the way a run reads it.

    ``name`` is tried as given (``~`` expanded), then inside the shipped
    configs: as a relative path with a leading ``configs/`` removed, then as a
    bare name with ``.toml`` added. The first that exists is returned; when
    none does, the path as given is returned so the caller's error names it.
    """
    raw = Path(str(name)).expanduser()
    if raw.exists():
        return raw
    parts = raw.parts
    if parts and parts[0] == "configs":
        parts = parts[1:]
    candidates = []
    if parts and not raw.is_absolute():
        candidates.append(CONFIG_DIR.joinpath(*parts))
    if raw.suffix == "" and not raw.is_absolute():
        candidates.append(CONFIG_DIR / f"{raw.name}.toml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def shipped_configs() -> list[Path]:
    """Every config shipped with the package, sorted by name."""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p for p in CONFIG_DIR.rglob("*") if p.is_file() and p.suffix in {".toml", ".json"})


def worker_path(name: str) -> Path:
    """The shipped worker script ``name`` (for instance ``cdlib_worker.py``)."""
    path = WORKER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"worker {name!r} is not shipped in {WORKER_DIR}")
    return path
