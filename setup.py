"""Builds the C core (``native/``: hedonic local moving) into ``hedonic._core``.

Everything else about the package is in pyproject.toml. The core is compiled
as one translation unit (``native/unity.c``) with -O3 -DNDEBUG. Fused
multiply-adds are explicit (``hig_fma``: the instruction, or the core's own
correctly rounded soft fma, never the C library's) and contraction is off, so
the rounding of every expression is fixed by the source rather than by the
optimizer. On x86_64 without a baseline FMA instruction (the wheels),
``native/unity_fma.c`` compiles the core a second time with the instruction
and ``native/dispatch.c`` picks one of the two builds from CPUID; elsewhere
both files compile to nothing, so the source list is the same on every
platform (a cross or universal2 build decides per target, in the
preprocessor).
"""

from glob import glob

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# GCC and Clang. -ffp-contract=off: every fused multiply-add is an explicit
# hig_fma() in the source. Python's own CFLAGS come first and carry
# -fno-strict-overflow (wrapping signed arithmetic); -fno-wrapv turns it back
# off, as the core checks its overflows explicitly. They also carry -g on Linux:
# -g0 keeps debug information out of the wheel (it never changes the code).
UNIX_FLAGS = ["-std=c11", "-O3", "-DNDEBUG", "-ffp-contract=off", "-fno-wrapv",
              "-fno-wrapv-pointer", "-g0"]
# MSVC. /fp:strict: no contraction (as -ffp-contract=off), no reassociation,
# and no other value-changing transformation of floating-point code, whatever
# the compiler version's defaults for /fp:precise.
MSVC_FLAGS = ["/std:c11", "/O2", "/DNDEBUG", "/fp:strict"]


class BuildExt(build_ext):
    """Gives the extension the flags of the compiler the build actually uses."""

    def build_extensions(self):
        msvc = self.compiler.compiler_type == "msvc"
        for ext in self.extensions:
            ext.extra_compile_args = MSVC_FLAGS if msvc else UNIX_FLAGS
            ext.libraries = [] if msvc else ["m"]
        super().build_extensions()


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension(
            "hedonic._core",
            sources=["native/unity.c", "native/unity_fma.c", "native/dispatch.c",
                     *sorted(glob("native/binding/*.c"))],
            include_dirs=["native/include", "native/core", "native/binding"],
        )
    ],
)
