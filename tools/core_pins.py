"""The core's bit pins through the installed extension alone: no igraph, no NumPy.

    python tools/core_pins.py

Loads hedonic._core from the installed hedonic without importing the package
(whose Game needs igraph) and runs the pins of native/tests/test_core.c and
native/rust/tests/pins.rs (Zachary, random.Random(seed)'s order) through its plain
calls, partition() and cover() over array buffers: count, quality, slack and
result, bit for bit, once with the generator run natively and once called
from Python, with the generator's end state compared. For wheels on platforms
where igraph or NumPy have no wheel to install (32-bit Linux, ppc64le, s390x,
armv7l, riscv64, free-threaded CPython); everywhere else tools/native_tests.py
runs these pins and the rest through the package.
"""

import hashlib
import importlib.machinery
import importlib.util
import random
import sys
import sysconfig
from array import array
from pathlib import Path

ZACHARY = [
    0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13, 0, 17, 0, 19, 0, 21, 0,
    31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21, 1, 30, 2, 3, 2, 7, 2, 27, 2, 28, 2, 32, 2, 9, 2,
    8, 2, 13, 3, 7, 3, 12, 3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5, 16, 6, 16, 8, 30, 8, 32, 8, 33, 9, 33,
    13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32, 18, 33, 19, 33, 20, 32, 20, 33, 22, 32, 22, 33,
    23, 25, 23, 27, 23, 32, 23, 33, 23, 29, 24, 25, 24, 27, 24, 31, 25, 31, 26, 29, 26, 33, 27, 33,
    28, 31, 28, 33, 29, 32, 29, 33, 30, 32, 30, 33, 31, 32, 31, 33, 32, 33,
]

# (seed, resolution or None for the density, cap, isolation, edge-weighted,
# node-weighted, max_total, exact) -> (k, quality, raw slack, membership in
# order of first appearance or the sorted bit masks of the communities).
PINS = [
    ((0, None, 1, False, False, False, None, None), (5, "0x1.b8f407b8f407dp-2", "0x0.0p+0", [
        0, 0, 0, 0, 1, 2, 2, 0, 3, 0, 1, 1, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 4, 4, 3, 4, 4, 3, 3,
        4, 3, 3])),
    ((0, 0.1, 3, False, False, False, None, None), (9, "0x1.0fc298892fe39p-1", "0x1.f1d4abc914b81p-44", [
        0x10060, 0x10470, 0x10470, 0x2a388f, 0x10000204, 0x402a318f, 0x8b800000, 0x3f4d4c100,
        0x3f4d4c100])),
    ((1, 0.2, 1, True, True, False, None, None), (10, "0x1.96f439aec498ep-2", "0x1.0c0000000000ap-50", [
        0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 1, 4, 0, 0, 5, 5, 1, 0, 6, 0, 6, 7, 5, 8, 9, 9, 8, 6, 6, 8, 2,
        6, 5, 6])),
    ((2, -0.02, 2, False, False, False, None, None), (5, "0x1.c89a7b611699ep-1", "0x1.1078050fee7cap-43", [
        0x10060, 0x2a388f, 0xdb2b3fff, 0x364d4c100, 0x3bfd4c200])),
    ((3, 0.1, 1, False, False, False, None, 3), (3, "0x1.01f81f81f81f8p-1", "0x0.0p+0", [
        0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 1, 2, 2,
        1, 2, 2])),
    ((5, 0.05, 4, True, True, True, 6, None), (6, "0x1.523b9503a6befp-1", "0x1.7c5aa71afd091p-43", [
        0x800, 0x10470, 0x2a309f, 0x930b34ff, 0x3fcd4c304, 0x3fcd4c304])),
    ((7, 0.5, 1, True, False, False, None, None), (19, "0x1.20d20d20d20d2p-4", "0x0.0p+0", [
        0, 0, 0, 0, 1, 2, 2, 0, 3, 4, 1, 5, 6, 0, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17,
        15, 18, 17, 3, 16, 3, 3])),
]


class Called(random.Random):
    """random.Random whose draws the core makes through getrandbits()."""


def load_core():
    """hedonic._core from the installed package, without running its __init__."""
    spec = importlib.util.find_spec("hedonic")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("hedonic is not installed")
    folder = Path(next(iter(spec.submodule_search_locations)))
    found = [folder / f"_core{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES
             if (folder / f"_core{suffix}").exists()]
    if not found:
        raise SystemExit(f"no hedonic._core extension in {folder}")
    loader = importlib.machinery.ExtensionFileLoader("hedonic._core", str(found[0]))
    core = importlib.util.module_from_spec(importlib.util.spec_from_loader("hedonic._core", loader))
    loader.exec_module(core)
    return core


def rust_source_digest(crate):
    """build.rs's digest of the crate at `crate` (setup.py's format)."""
    lock = ["Cargo.lock"] if (crate / "Cargo.lock").is_file() else []
    files = sorted([(p.relative_to(crate).as_posix(), p) for p in (crate / "src").rglob("*.rs")]
                   + [(name, crate / name) for name in ["build.rs", "Cargo.toml"] + lock])
    stream = hashlib.sha256()
    for rel, path in files:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        stream.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return stream.hexdigest()


def run(core, pin, rng):
    seed, resolution, cap, isolation, weighted, node_weighted, max_total, exact = pin
    kwargs = dict(
        edge_weights=array("d", [0.5 + 0.25 * (e % 7) for e in range(78)]) if weighted else None,
        node_weights=array("d", [0.5 * (1 + v % 4) for v in range(34)]) if node_weighted else None,
        resolution=78 / 561 if resolution is None else resolution,
        max_total_communities=max_total, n_communities=exact, allow_isolation=isolation)
    edges = array("q", ZACHARY)
    if cap == 1:
        membership, k, quality, record = core._trace_partition(34, edges, rng, **kwargs)
        names = {}
        result = [names.setdefault(c, len(names)) for c in membership]
    else:
        rows, k, quality, record = core._trace_cover(34, edges, rng, max_memberships=cap, **kwargs)
        masks = [0] * k
        for v, row in enumerate(rows):
            for label in row:
                masks[label] |= 1 << v
        result = sorted(masks)
    return k, quality.hex(), record["raw_slack"].hex(), result


def main():
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    core = load_core()
    print(f"hedonic._core from {core.__file__}", flush=True)
    if free_threaded:
        # Informational until the module declares gil_used = false: CPython then
        # enables the GIL on import, with a RuntimeWarning.
        print(f"GIL {'enabled' if sys._is_gil_enabled() else 'disabled'} after loading the module", flush=True)
    fma = core.fma_path() if hasattr(core, "fma_path") else "?"
    print(f"backend {getattr(core, 'backend', 'c')}, fma {fma}, "
          f"{sysconfig.get_platform()} {sys.byteorder}-endian, Python {sys.version.split()[0]}"
          f"{' free-threaded' if free_threaded else ''}", flush=True)
    failed = 0
    crate = Path(__file__).resolve().parent.parent / "native" / "rust"
    if getattr(core, "backend", "c") == "rust" and (crate / "Cargo.toml").exists():
        expected = rust_source_digest(crate)
        same = core.source_sha256 == expected
        failed += not same
        print(f"{'ok  ' if same else 'FAIL'} source_sha256 {core.source_sha256}"
              f"{'' if same else f', the checkout {expected}'}", flush=True)
    for pin, expected in PINS:
        states = []
        for kind in (random.Random, Called):
            rng = kind(pin[0])
            got = run(core, pin, rng)
            states.append(rng.getstate())
            ok = got == expected and states[-1] == states[0]
            if states[-1] != states[0]:
                print("  the generator ends in another state when called from Python", flush=True)
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} seed {pin[0]} cap {pin[2]} ({kind.__name__}): k {got[0]}, "
                  f"quality {got[1]}, slack {got[2]}", flush=True)
            if not ok:
                print(f"  expected k {expected[0]}, quality {expected[1]}, slack {expected[2]}\n"
                      f"  result   {got[3]}\n  expected {expected[3]}", flush=True)
    print("all agree" if not failed else f"{failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
