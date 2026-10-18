/*
   hedonic -- the C core with the FMA instruction, on x86_64 dispatch targets.
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
 * On an x86_64 target without a baseline FMA instruction (core/dispatch.h),
 * this is unity.c compiled a second time with the instruction: every
 * function of the core gets target("fma") (GCC, Clang), hig_fma is the
 * instruction (MSVC: the _mm_fmadd_sd intrinsic, which needs no /arch), and
 * the entry points are named hig_x86fma_*. dispatch.c calls them only on a
 * CPU that has FMA. The results are those of unity.c, bit for bit: hig_fma
 * is correctly rounded in both, contraction is off in both, and nothing else
 * changes how an operation rounds. On every other target this file is empty.
 */

#include "dispatch.h"

#if HIG_DISPATCH

#define HIG_FMA_VARIANT

/* The C library's headers first, outside the region with the instruction. */
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("fma")
#endif

#include "unity.c"

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#else

typedef int hig_unity_fma_unused;   /* ISO C: a translation unit declares something */

#endif
