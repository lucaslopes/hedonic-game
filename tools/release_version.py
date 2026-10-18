"""Sets the version a release tag names, before the distributions are built.

    python tools/release_version.py [REF_NAME]

pyproject.toml holds the version being released (1.0.0). A tag names that
version or one of its pre-releases:

    v1.0.0          the release                        -> PyPI
    v1.0.0rc1       a release candidate (pip --pre)    -> PyPI
    v1.0.0.dev1     a trial upload                     -> TestPyPI

On a tag (GITHUB_REF_TYPE=tag, or REF_NAME given) this checks the tag against
the version in pyproject.toml, writes the tag's version into pyproject.toml so
the sdist and every wheel carry it, and prints the index the tag publishes to
(`pypi` or `testpypi`). A candidate and the release are therefore built from
the same commit. Anywhere else it changes nothing and prints `none`.
"""

import os
import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
TAG = re.compile(r"v(?P<release>\d+\.\d+\.\d+)(?P<pre>rc\d+|\.dev\d+)?")


def target(tag, release):
    """The version a tag names and the index it goes to; ValueError if invalid."""
    match = TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"tag {tag!r} is not vX.Y.Z, vX.Y.ZrcN or vX.Y.Z.devN")
    if match["release"] != release:
        raise ValueError(f"tag {tag!r} does not name pyproject.toml's version {release}")
    pre = match["pre"] or ""
    return release + pre, "testpypi" if pre.startswith(".dev") else "pypi"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        tag = argv[0]
    elif os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ["GITHUB_REF_NAME"]
    else:
        print("none")
        return 0
    text = PYPROJECT.read_text(encoding="utf-8")
    release = tomllib.loads(text)["project"]["version"]
    try:
        version, index = target(tag, release)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if version != release:
        line = f'version = "{release}"'
        if text.count(line) != 1:
            print(f"error: pyproject.toml must hold exactly one {line!r}", file=sys.stderr)
            return 1
        PYPROJECT.write_text(text.replace(line, f'version = "{version}"'), encoding="utf-8")
    print(f"{version} -> {index}", file=sys.stderr)
    print(index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
