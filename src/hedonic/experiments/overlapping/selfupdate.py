"""``hedonic update``: compare the installed version with PyPI and, only if asked, upgrade.

    hedonic update            report installed vs latest (a single HTTPS GET to pypi.org; installs nothing)
    hedonic update --yes      upgrade with pip when a newer release exists
    hedonic update --json     machine-readable report

A development checkout (editable install) is never modified: it is told to use git and ``uv sync``.
The unreleased case (installed newer than PyPI) is reported as such.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from hedonic.experiments.overlapping.tui import style

PYPI_URL = "https://pypi.org/pypi/hedonic/json"


def _key(version: str) -> tuple:
    """Order versions without third-party code (numeric parts, then a pre-release marker)."""
    import re

    parts = [int(p) for p in re.findall(r"\d+", version.split("+")[0].split("rc")[0].split("a")[0].split("b")[0])]
    pre = bool(re.search(r"(a|b|rc|dev)\d*$", version))
    return (*parts, 0 if pre else 1)


def installed_version() -> str | None:
    try:
        return md.version("hedonic")
    except md.PackageNotFoundError:
        return None


def is_editable() -> bool:
    """True for a development checkout (PEP 610 editable marker, or a source tree with a .git next to it)."""
    try:
        text = md.distribution("hedonic").read_text("direct_url.json")
        if text and json.loads(text).get("dir_info", {}).get("editable"):
            return True
    except (md.PackageNotFoundError, ValueError):
        pass
    return any((parent / ".git").exists() for parent in Path(__file__).resolve().parents[:6])


def latest_version(timeout: float = 10.0) -> str:
    request = urllib.request.Request(PYPI_URL, headers={"Accept": "application/json", "User-Agent": "hedonic-update"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.load(response)["info"]["version"])


def report(latest: str | None, error: str | None = None) -> dict:
    current = installed_version()
    if current is None or latest is None:
        state = "unknown"
    elif _key(latest) > _key(current):
        state = "update_available"
    elif _key(latest) < _key(current):
        state = "ahead_of_pypi"
    else:
        state = "up_to_date"
    return {"installed": current, "latest": latest, "state": state, "editable": is_editable(), "error": error,
            "lucas_igraph": _lucas_igraph()}


def _lucas_igraph() -> str | None:
    try:
        return md.version("lucas-igraph")
    except md.PackageNotFoundError:
        return None


def upgrade() -> int:
    """Upgrade in the running interpreter's environment (pip, else uv)."""
    import shutil

    commands = [[sys.executable, "-m", "pip", "install", "--upgrade", "hedonic"]]
    if shutil.which("uv"):
        commands.append(["uv", "pip", "install", "--python", sys.executable, "--upgrade", "hedonic"])
    for command in commands:
        print(style("  $ " + " ".join(command), "dim"))
        if subprocess.run(command, check=False).returncode == 0:
            return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hedonic update", description="Check for, and optionally install, a newer hedonic.")
    parser.add_argument("-y", "--yes", action="store_true", help="upgrade with pip if a newer release exists")
    parser.add_argument("--check", action="store_true", help="only report (the default without --yes)")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)
    latest, error = None, None
    try:
        latest = latest_version()
    except Exception as exc:  # noqa: BLE001 - offline, blocked, malformed reply
        error = f"could not reach PyPI: {exc}"
    info = report(latest, error)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0 if error is None else 1
    print(style("\n  hedonic update", "bold", "magenta"))
    print(f"  installed  {info['installed'] or 'not installed as a package'}"
          + (f"   (lucas-igraph {info['lucas_igraph']})" if info["lucas_igraph"] else ""))
    print(f"  on PyPI    {info['latest'] or 'unknown'}")
    if error:
        print(style(f"\n  {error}", "yellow"))
        return 1
    messages = {
        "up_to_date": ("you are up to date", "green"),
        "ahead_of_pypi": ("this build is newer than the latest release (unreleased or development version)", "cyan"),
        "update_available": (f"a newer release is available: {info['latest']}", "yellow"),
        "unknown": ("could not compare versions", "yellow"),
    }
    text, colour = messages[info["state"]]
    print("\n  " + style(text, colour, "bold"))
    if info["state"] != "update_available":
        return 0
    if info["editable"]:
        print(style("  this is a development checkout, so it is not changed automatically: "
                    "git pull && uv sync", "dim"))
        return 0
    if not args.yes:
        print(style("  install it with: hedonic update --yes   (runs already in flight keep their old code until "
                    "`hedonic run resume`)", "dim"))
        return 0
    code = upgrade()
    print(style("  upgraded" if code == 0 else "  upgrade failed (see the pip output above)", "green" if code == 0 else "red"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
