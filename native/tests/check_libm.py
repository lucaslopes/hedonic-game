"""The C core calls no libm function but sqrt.

Usage: python check_libm.py OBJECT [OBJECT ...]

Lists each object's undefined symbols (``nm -u``; ``dumpbin /symbols`` for a
``.obj`` built by MSVC) and fails if one is a <math.h> function other than
sqrt, or any float/long double variant (sqrtf, fmal, ...). Pass every object
of the core: unity.c, and on x86_64 dispatch targets unity_fma.c and
dispatch.c (elsewhere these compile to empty objects).

Why this pins the results across platforms: IEEE 754 requires +, -, *, /
and sqrt to be correctly rounded, and the hardware of every target does
them (sqrt is an instruction everywhere; a libm sqrt computes the same
correctly rounded result). The fused multiply-add is also correctly rounded
by IEEE 754, but the core does not rely on a C library for it: it uses the
instruction where the target has one at compile time and otherwise its own
soft fma in integer arithmetic (core/softfma.h; x86_64 wheels choose between
a build with the instruction and one with the soft fma from CPUID), so fma
must not appear here either. With the core's other guarantees (no FP
contraction, double everywhere, fixed summation orders, no threads) every
operation then gives the same bits on every conforming platform. Functions
such as exp, log or pow carry no such requirement and differ between C
libraries; the core uses none of them. x87 double rounding (32-bit x86
without SSE2) is the one exception to the argument, and 32-bit x86 is not a
wheel target.
"""

from __future__ import annotations

import re
import subprocess
import sys

BASE = """acos asin atan atan2 cos sin tan acosh asinh atanh cosh sinh tanh exp exp2
expm1 frexp ilogb ldexp log log10 log1p log2 logb modf scalbn scalbln cbrt fabs
hypot pow sqrt erf erfc lgamma tgamma ceil floor nearbyint rint lrint llrint round
lround llround trunc fmod remainder remquo copysign nan nextafter nexttoward fdim
fmax fmin fma sincos exp10 pow10 significand gamma j0 j1 jn y0 y1 yn""".split()
ALLOWED = {"sqrt"}
LIBM = {name + suffix for name in BASE for suffix in ("", "f", "l")}


def undefined(path: str) -> list[str]:
    if path.lower().endswith(".obj"):
        out = subprocess.run(["dumpbin", "/symbols", path], check=True, capture_output=True,
                             text=True).stdout
        return [line.split("|")[-1].strip().split()[0] for line in out.splitlines()
                if "UNDEF" in line and "External" in line and "|" in line]
    out = subprocess.run(["nm", "-u", path], check=True, capture_output=True, text=True).stdout
    return [line.split()[-1] for line in out.splitlines() if line.strip()]


def libm_name(symbol: str) -> str | None:
    """The <math.h> function a symbol stands for (_sqrt, __fma_finite, sqrt@GLIBC_2.2.5, ...)."""
    name = symbol.split("@")[0].lstrip("_")
    name = re.sub(r"_finite$", "", name)
    return name if name in LIBM else None


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = []
    for path in argv:
        used = sorted({n for n in map(libm_name, undefined(path)) if n})
        bad += [(path, n) for n in used if n not in ALLOWED]
        print(f"{path}: libm functions used: {', '.join(used) or 'none'}")
    for path, name in bad:
        print(f"{path}: calls {name}, which is not correctly rounded everywhere", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
