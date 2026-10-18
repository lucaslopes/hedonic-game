/*
   hedonic -- random numbers, and the fused multiply-add a build runs.
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

#include "internal.h"

/* ---- Random numbers --------------------------------------------------------- */

HIG_SHARED hig_error_t hig_random_below(const hig_rng_t *rng, uint32_t range, uint32_t *value) {
    const uint32_t threshold = (0u - range) % range;
    uint64_t product;

    for (;;) {
        int failed = 0;
        const uint32_t word = rng->next32(rng->state, &failed);
        if (failed) {
            return hig_error(HIG_ECALLBACK, "The random number generator failed.");
        }
        product = (uint64_t) word * range;
        if ((uint32_t) product >= threshold) {
            break;
        }
    }
    *value = (uint32_t) (product >> 32);
    return HIG_SUCCESS;
}

/* PCG32 (XSH-RR, 64-bit state), O'Neill 2014. */
void hig_pcg32_seed(hig_pcg32_t *pcg, uint64_t seed, uint64_t stream) {
    int unused = 0;

    pcg->state = 0;
    pcg->increment = (stream << 1) | 1u;
    hig_pcg32_next32(pcg, &unused);
    pcg->state += seed;
    hig_pcg32_next32(pcg, &unused);
}

uint32_t hig_pcg32_next32(void *state, int *failed) {
    hig_pcg32_t *pcg = (hig_pcg32_t *) state;
    const uint64_t old = pcg->state;
    const uint32_t xorshifted = (uint32_t) (((old >> 18) ^ old) >> 27);
    const uint32_t rotation = (uint32_t) (old >> 59);

    (void) failed;
    pcg->state = old * 6364136223846793005ULL + pcg->increment;
    return (xorshifted >> rotation) | (xorshifted << ((0u - rotation) & 31));
}

/* ---- The fused multiply-add (hedonic_check.h) -------------------------------- */

/* A dispatch target defines these in dispatch.c. */
#if !HIG_DISPATCH
const char *hig_fma_path(void) {
    return HIG_FMA_INSTRUCTION ? "instruction" : "soft";
}

int hig_fma_cpu(void) {
    return HIG_FMA_INSTRUCTION;
}

int hig_fma_select(int instruction) {
    (void) instruction;
    return 0;
}
#endif
