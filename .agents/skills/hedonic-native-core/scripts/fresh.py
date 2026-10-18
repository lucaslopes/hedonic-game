"""Is the loaded hedonic._core built from this checkout's C sources?

    python fresh.py [CHECKOUT]     (default: the current folder)

Compares hedonic._core.source_sha256 (the digest setup.py compiled into the
extension) with the digest of the checkout's sources now. Exit 0 when they are
equal; 1, with both digests, when the extension is stale (rebuild with
``python setup.py build_ext --inplace --force``) or comes from somewhere else.
"""

import runpy
import sys
import unittest.mock
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
with unittest.mock.patch("setuptools.setup"):   # load setup.py's functions without building
    build = runpy.run_path(str(root / "setup.py"))

import hedonic._core as core  # noqa: E402

built = getattr(core, "source_sha256", None)
now = build["source_digest"](str(root))
print(f"extension  {core.__file__}\nbuilt from {built}\ncheckout   {now}  ({len(build['source_paths'](str(root)))} files)")
sys.exit(0 if built == now else 1)
