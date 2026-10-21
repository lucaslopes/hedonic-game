"""Is the loaded hedonic._core built from this checkout's Rust sources?

    python fresh.py [CHECKOUT]     (default: the current folder)

Compares hedonic._core.source_sha256 (the digest native/rust/build.rs compiled
into the extension) with the digest of the checkout's crate now: every
src/**/*.rs, build.rs, Cargo.toml and Cargo.lock, each as path, length and
bytes (CRLF as LF), in path order. Exit 0 when they are equal; 1, with both digests, when the
extension is stale (rebuild with ``maturin develop --release`` or ``pip install
-e .``), is another core (HEDONIC_CORE set: unset it) or comes from somewhere
else.
"""

import hashlib
import os
import sys
from pathlib import Path


def crate_digest(crate):
    lock = ["Cargo.lock"] if (crate / "Cargo.lock").is_file() else []
    paths = sorted([p.relative_to(crate).as_posix() for p in (crate / "src").rglob("*.rs")] + ["build.rs", "Cargo.toml"] + lock)
    digest = hashlib.sha256()
    for rel in paths:
        data = (crate / rel).read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest(), len(paths)


root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
now, count = crate_digest(root / "native" / "rust")

import hedonic._core as core  # noqa: E402

built = getattr(core, "source_sha256", None)
print(f"extension  {core.__file__}  (module {core.__spec__.name}, backend {getattr(core, 'backend', '?')})\n"
      f"built from {built}\ncheckout   {now}  ({count} files)")
if os.environ.get("HEDONIC_CORE"):
    print(f"HEDONIC_CORE={os.environ['HEDONIC_CORE']} is set: unset it to check the package's own core")
sys.exit(0 if built == now and core.__spec__.name == "hedonic._core" else 1)
