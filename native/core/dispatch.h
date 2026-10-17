/*
   hedonic -- which builds of the C core a target compiles.
   Copyright (C) 2026  Lucas Lopes Felipe

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * HIG_DISPATCH is 1 on x86_64 targets whose baseline has no fused
 * multiply-add instruction (no -mfma, no /arch:AVX2: the wheels'). There the
 * core is compiled twice, as unity.c (hig_fma is the soft fma of softfma.h,
 * exported names prefixed hig_x86soft_) and as unity_fma.c (the FMA
 * instruction, prefixed hig_x86fma_), and dispatch.c defines the public
 * entry points, which call one build or the other according to the CPU
 * (CPUID: FMA, AVX and the OS saving the AVX state). The two builds round
 * every operation the same way (hig_fma is correctly rounded in both), so
 * their results are identical bit for bit; the instruction is only faster.
 * Everywhere else (arm64, an x86_64 baseline with FMA, other targets) there
 * is one build, unity.c, and unity_fma.c and dispatch.c compile to nothing.
 * HIG_FORCE_SOFT_FMA (tests) makes every target one soft build;
 * HIG_NO_DISPATCH makes x86_64 one soft build.
 *
 * Included first by internal.h and by unity_fma.c, before the latter turns
 * the FMA instruction on, so the decision is taken on the baseline target.
 */

#ifndef HIG_DISPATCH_H
#define HIG_DISPATCH_H

#if !defined(HIG_FORCE_SOFT_FMA) && !defined(HIG_NO_DISPATCH) && !defined(__FMA__) && \
    !defined(__AVX2__) && \
    ((defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))) || \
     (defined(_M_X64) && !defined(_M_ARM64EC) && defined(_MSC_VER)))
#define HIG_DISPATCH 1
#else
#define HIG_DISPATCH 0
#endif

#endif
