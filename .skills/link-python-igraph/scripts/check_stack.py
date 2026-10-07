#!/usr/bin/env python3
"""Report (and optionally assert) which igraph stack this interpreter loads.

Run it with the interpreter of the environment under test, for example
``$HEDONIC_VENV/bin/python check_stack.py --expect-prefix "$C_PREFIX"``.
It prints one JSON document with

- the interpreter, python-igraph version and source path;
- the C core version compiled into the extension (``__igraph_version__``);
- the extension module file, the libigraph it links against and its rpaths;
- the libigraph the dynamic loader actually loads (macOS: DYLD_PRINT_LIBRARIES
  in a child process; Linux: /proc/self/maps);
- distribution metadata of lucas-igraph / python-igraph / igraph / hedonic
  (editable installs keep stale metadata until they are reinstalled);
- hedonic's source path and algorithm identity, if hedonic is importable;
- the loader variables that are set.

With --expect-* options it exits with status 1 when the stack does not match,
listing the failed checks. It never modifies anything.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def distribution_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("lucas-igraph", "python-igraph", "igraph", "hedonic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def linked_libraries(extension: Path) -> dict[str, list[str]]:
    if platform.system() == "Darwin":
        links = subprocess.run(["otool", "-L", str(extension)], capture_output=True, text=True).stdout
        load = subprocess.run(["otool", "-l", str(extension)], capture_output=True, text=True).stdout
        lines = load.splitlines()
        rpaths = [lines[i + 2].split()[1] for i, line in enumerate(lines)
                  if "LC_RPATH" in line and i + 2 < len(lines)]
        return {"links": [l.split()[0] for l in links.splitlines()[1:] if "igraph" in l],
                "rpaths": rpaths}
    if platform.system() == "Linux":
        links = subprocess.run(["ldd", str(extension)], capture_output=True, text=True).stdout
        return {"links": [l.strip() for l in links.splitlines() if "igraph" in l], "rpaths": []}
    return {"links": [], "rpaths": []}


def loaded_libigraph() -> list[str]:
    """Paths of libigraph images the loader maps when importing igraph."""
    if platform.system() == "Darwin":
        env = dict(os.environ, DYLD_PRINT_LIBRARIES="1")
        child = subprocess.run([sys.executable, "-c", "import igraph"], env=env,
                               capture_output=True, text=True)
        return sorted({line.split()[-1] for line in child.stderr.splitlines()
                       if "libigraph" in line})
    if platform.system() == "Linux":
        import igraph  # noqa: F401  (maps the library into this process)
        with open("/proc/self/maps") as fh:
            return sorted({line.split()[-1] for line in fh if "libigraph" in line})
    return []


def hedonic_identity() -> dict:
    try:
        import hedonic
        from hedonic import Game
    except Exception as exc:  # noqa: BLE001 - report, do not fail
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    import importlib
    # `hedonic.Game` is the exported class; the module lives in sys.modules.
    game_module = importlib.import_module("hedonic.Game")
    return {"importable": True, "path": hedonic.__file__,
            "algorithm_identity": getattr(game_module, "HEDONIC_ALGORITHM_IDENTITY", None),
            "game": Game.__module__}


def collect() -> dict:
    try:
        import igraph
    except Exception as exc:  # noqa: BLE001 - a broken stack is a finding
        return {"python": sys.executable, "import_error": f"{type(exc).__name__}: {exc}",
                "loaded_libigraph": [], "distributions": distribution_versions(),
                "loader_environment": {k: v for k, v in os.environ.items()
                                       if k in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")}}

    extension = Path(igraph._igraph.__file__).resolve()
    return {
        "python": sys.executable,
        "igraph_version": igraph.__version__,
        "igraph_c_version": getattr(igraph._igraph, "__igraph_version__", None),
        "igraph_path": str(Path(igraph.__file__).resolve()),
        "extension": str(extension),
        "extension_linkage": linked_libraries(extension),
        "loaded_libigraph": [str(Path(p).resolve()) for p in loaded_libigraph()],
        "distributions": distribution_versions(),
        "hedonic": hedonic_identity(),
        "loader_environment": {k: v for k, v in os.environ.items()
                               if k in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                                        "LD_LIBRARY_PATH", "PYTHONPATH")},
    }


def under(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).expanduser().resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--expect-prefix", help="C install prefix that must provide libigraph")
    ap.add_argument("--expect-igraph-source", help="python-igraph worktree that must provide igraph")
    ap.add_argument("--expect-hedonic-source", help="hedonic worktree that must provide hedonic")
    ap.add_argument("--expect-c-version", help="exact __igraph_version__ string")
    args = ap.parse_args()

    report = collect()
    failures = []
    if "import_error" in report:
        failures.append(f"igraph does not import: {report['import_error']}")
        report["failures"] = failures
        print(json.dumps(report, indent=1))
        return 1
    if args.expect_prefix:
        loaded = report["loaded_libigraph"]
        if not loaded or not all(under(p, args.expect_prefix) for p in loaded):
            failures.append(f"libigraph not loaded from {args.expect_prefix}: {loaded}")
    if args.expect_igraph_source and not under(report["igraph_path"], args.expect_igraph_source):
        failures.append(f"igraph imported from {report['igraph_path']}")
    if args.expect_hedonic_source:
        path = report["hedonic"].get("path")
        if not path or not under(path, args.expect_hedonic_source):
            failures.append(f"hedonic imported from {path}")
    if args.expect_c_version and report["igraph_c_version"] != args.expect_c_version:
        failures.append(f"C core is {report['igraph_c_version']}")
    metadata = report["distributions"].get("lucas-igraph") or report["distributions"].get("python-igraph")
    if metadata and metadata != report["igraph_version"]:
        report["warning"] = (f"distribution metadata {metadata} differs from igraph.__version__ "
                             f"{report['igraph_version']}; reinstall the editable package")
    report["failures"] = failures
    print(json.dumps(report, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
