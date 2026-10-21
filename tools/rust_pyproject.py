"""pyproject.toml for hedonic built with the Rust core alone (maturin).

Usage: python tools/rust_pyproject.py [PYPROJECT] > pyproject.toml

Prints PYPROJECT (default: the repository's pyproject.toml) with its
setuptools build (setup.py compiling native/ into hedonic._core) replaced by
maturin compiling native/rust/ into the same module, hedonic._core (cargo feature
`core-module`). Everything above [build-system] — the name, version,
dependencies, extras and entry points — is kept as it is, so both builds are
one distribution. CI builds it in a checkout without native/ and setup.py and
runs the whole test suite on it: the C core can be removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

MATURIN = """\
[build-system]
# maturin: the package compiles its Rust core (native/rust/) into the extension
# hedonic._core.
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "native/rust/Cargo.toml"
python-source = "src"
module-name = "hedonic._core"
features = ["core-module"]
# An extension the C build left in the source tree is not this core.
exclude = ["src/hedonic/_core.*.so", "src/hedonic/_core.*.pyd", "src/hedonic/**/__pycache__/**"]
"""


def rust_pyproject(text: str) -> str:
    head, sep, _ = text.partition("\n[build-system]")
    if not sep:
        raise SystemExit("no [build-system] table to replace")
    head = head.replace('"Programming Language :: C",', '"Programming Language :: Rust",')
    head = head.replace("purpose-built C core", "purpose-built Rust core")
    return head.rstrip("\n") + "\n\n" + MATURIN


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[1] / "pyproject.toml"
    sys.stdout.write(rust_pyproject(path.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
