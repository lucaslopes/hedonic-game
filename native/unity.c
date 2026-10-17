/*
   hedonic -- the C core as one translation unit.
   Copyright (C) 2025-2026  Lucas Lopes Felipe

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
 * The extension compiles the core as one unit, so the helpers of common.c
 * (static here, see HIG_SHARED) inline into the engine's loops. Floating-point
 * contraction is off and every fma is an explicit hig_fma, so this changes no
 * result. On an x86_64 dispatch target (core/dispatch.h) this is the build
 * with the soft fma, its entry points named hig_x86soft_*; unity_fma.c
 * includes this file again for the build with the FMA instruction.
 */

#define HIG_UNITY_BUILD

#include "common.c"
#include "partition.c"
#include "check.c"
