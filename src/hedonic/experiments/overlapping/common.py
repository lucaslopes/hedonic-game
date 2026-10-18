"""Small helpers shared by the durable-run machinery (``runmanager``, ``runqueue``) and the disjoint front door.

They used to live in the overlap benchmark's front door (``quickstart``); hedonic 1 keeps only these, with the same
behaviour: run names, seed lists, durations and the software provenance a run records in its ``progress.json``.
"""

from __future__ import annotations

import sys

from hedonic.experiments.config import expand_path

__all__ = ["environment", "environment_line", "expand_path", "fmt_time", "parse_name", "parse_seeds", "visible"]


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    try:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                if int(hi) < int(lo):
                    raise ValueError
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(part))
    except ValueError:
        raise ValueError(f"invalid seeds {value!r}; use e.g. 0, 0-4 or 0,3,7") from None
    if not seeds:
        raise ValueError("no seeds given")
    if min(seeds) < 0:
        raise ValueError(f"invalid seeds {value!r}; seeds are non-negative integers")
    return sorted(dict.fromkeys(seeds))


def parse_name(value: str) -> str:
    """Run names become directory and tmux session names, so keep them plain."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"invalid run name {value!r}; use letters, digits, '-' and '_' (at most 64, no dots or spaces)")
    return value


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


def fmt_duration(seconds: float) -> str:
    """A length of time that cannot be read as a clock time: ``45 s``, ``12 min``, ``19 h 42 min``, ``3 d 4 h``."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min" if minutes else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


def seeds_text(seeds: list[int]) -> str:
    """``seed 0`` or ``seeds 0,1,2``."""
    return ("seed " if len(seeds) == 1 else "seeds ") + ",".join(map(str, seeds))


def environment() -> dict:
    """Software provenance stored with every run (progress.json)."""
    import platform

    from hedonic.experiments.runtime import ALGORITHM_IDENTITY, runtime_versions

    try:  # the loaded native builds and their SHA-256 (a rebuild is a different runtime)
        from hedonic.experiments.disjoint.store import runtime_identity

        native = runtime_identity()
    except Exception:  # noqa: BLE001 - provenance only
        native = {}
    versions = runtime_versions()
    return {"hedonic": versions.get("hedonic"), "hedonic_core_sha256": versions.get("hedonic_core_sha256"),
            "python_igraph": versions.get("python_igraph"), "igraph_c": versions.get("igraph_c"),
            "igraph_extension_sha256": native.get("igraph_extension_sha256"),
            "igraph_extension_path": versions.get("igraph_extension_path"),
            "algorithm_identity": ALGORITHM_IDENTITY,
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "executable": sys.executable}


def environment_line() -> str:
    env = environment()
    return (f"hedonic {env['hedonic']} · python-igraph {env['python_igraph']} · Python {env['python']} · "
            f"{env['machine']}")


def visible(text: str) -> int:
    """Printed width of ``text`` without its colour codes."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))
