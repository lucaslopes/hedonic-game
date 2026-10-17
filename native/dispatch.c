/*
   hedonic -- the entry points of the C core on x86_64 dispatch targets.
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
 * On an x86_64 target without a baseline FMA instruction (core/dispatch.h)
 * the core is built twice: unity.c with the soft fma (hig_x86soft_*) and
 * unity_fma.c with the instruction (hig_x86fma_*). The public functions here
 * call the second when the CPU has FMA (CPUID leaf 1: FMA, AVX and OSXSAVE,
 * and XGETBV: the OS saves the XMM and YMM state), else the first. The choice
 * is made once per process, at the first call. Both builds give the same
 * results bit for bit; the choice only changes the speed. On every other
 * target this file is empty.
 */

#include "dispatch.h"

#if HIG_DISPATCH

#include "hedonic_core.h"
#include "check/hedonic_check.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif

#define HIG_DECLARE_BUILD(p)                                                                 \
    hig_error_t p##audit_prepare(const hig_graph_t *, const hig_cover_t *, const hig_interrupt_t *, hig_audit_state_t **); \
    void p##audit_free(hig_audit_state_t *); \
    hig_error_t p##audit_intervals(const hig_audit_state_t *, const hig_audit_options_t *, hig_audit_selection_t, hig_audit_intervals_t *); \
    void p##audit_intervals_free(hig_audit_intervals_t *); \
    hig_error_t p##audit_regrets(const hig_audit_state_t *, const hig_audit_options_t *, hig_audit_selection_t, const double *, hig_int, double *, double *); \
    void p##pcg32_seed(hig_pcg32_t *pcg, uint64_t seed, uint64_t stream);                    \
    uint32_t p##pcg32_next32(void *pcg, int *failed);                                       \
    void p##cover_free(hig_cover_t *cover);                                                 \
    hig_error_t p##partition(const hig_graph_t *graph, const hig_options_t *options,         \
                             int start, hig_int *membership, hig_int *nb_communities,       \
                             double *quality, double *equilibrium_slack);                   \
    hig_error_t p##cover(const hig_graph_t *graph, const hig_options_t *options,             \
                         const hig_cover_t *start, hig_cover_t *result,                     \
                         hig_int *nb_communities, double *quality,                          \
                         double *equilibrium_slack);                                        \
    const char *p##error_message(void);                                                     \
    hig_error_t p##partition_traced(const hig_graph_t *graph, const hig_options_t *options,  \
                                    int start, hig_int *membership,                         \
                                    hig_int *nb_communities, double *quality,               \
                                    double *raw_slack, const hig_trace_t *trace);           \
    hig_error_t p##cover_traced(const hig_graph_t *graph, const hig_options_t *options,      \
                                const hig_cover_t *start, hig_cover_t *result,              \
                                hig_int *nb_communities, double *quality,                   \
                                double *raw_slack, const hig_trace_t *trace);               \
    hig_error_t p##verify_partition(const hig_graph_t *graph, const hig_options_t *options,  \
                                    const hig_int *membership, hig_verify_t *report);       \
    hig_error_t p##verify_cover(const hig_graph_t *graph, const hig_options_t *options,      \
                                const hig_cover_t *cover, hig_verify_t *report);

HIG_DECLARE_BUILD(hig_x86soft_)
HIG_DECLARE_BUILD(hig_x86fma_)

/* Does this CPU run FMA3 with the AVX state enabled by the OS? */
static int hig_cpu_has_fma(void) {
    unsigned ecx, xcr0;
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4];

    __cpuid(r, 0);
    if (r[0] < 1) {
        return 0;
    }
    __cpuid(r, 1);
    ecx = (unsigned) r[2];
#else
    unsigned eax, ebx, edx;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
#endif
    /* FMA (bit 12), OSXSAVE (bit 27: XGETBV is available), AVX (bit 28) */
    if (!(ecx & (1u << 12)) || !(ecx & (1u << 27)) || !(ecx & (1u << 28))) {
        return 0;
    }
#if defined(_MSC_VER) && !defined(__clang__)
    xcr0 = (unsigned) _xgetbv(0);
#else
    {
        unsigned high;
        __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0), "=d"(high) : "c"(0)); /* xgetbv */
        (void) high;
    }
#endif
    return (xcr0 & 6u) == 6u;   /* the OS saves the XMM (bit 1) and YMM (bit 2) state */
}

/* 0: not chosen yet, 1: the soft build, 2: the FMA build. Every thread that
 * finds it unset computes the same value, so a relaxed atomic suffices. */
static int hig_build;

static int hig_load_build(void) {
#if defined(_MSC_VER) && !defined(__clang__)
    return *(volatile int *) &hig_build;
#else
    return __atomic_load_n(&hig_build, __ATOMIC_RELAXED);
#endif
}

static void hig_store_build(int build) {
#if defined(_MSC_VER) && !defined(__clang__)
    *(volatile int *) &hig_build = build;
#else
    __atomic_store_n(&hig_build, build, __ATOMIC_RELAXED);
#endif
}

static int hig_fma_build(void) {
    int build = hig_load_build();

    if (!build) {
        build = hig_cpu_has_fma() ? 2 : 1;
        hig_store_build(build);
    }
    return build == 2;
}

const char *hig_fma_path(void) {
    return hig_fma_build() ? "dispatch: instruction" : "dispatch: soft";
}

int hig_fma_cpu(void) {
    return hig_cpu_has_fma();
}

int hig_fma_select(int instruction) {
    if (instruction && !hig_cpu_has_fma()) {
        return 0;
    }
    hig_store_build(instruction ? 2 : 1);
    return 1;
}

/* Neither depends on the build (no floating point). */
void hig_pcg32_seed(hig_pcg32_t *pcg, uint64_t seed, uint64_t stream) {
    hig_x86soft_pcg32_seed(pcg, seed, stream);
}

uint32_t hig_pcg32_next32(void *pcg, int *failed) {
    return hig_x86soft_pcg32_next32(pcg, failed);
}

/* Both builds allocate with the same malloc. */
void hig_cover_free(hig_cover_t *cover) {
    hig_x86soft_cover_free(cover);
}

/* Each build keeps its own message: the one the calls go to. */
const char *hig_error_message(void) {
    return hig_fma_build() ? hig_x86fma_error_message() : hig_x86soft_error_message();
}

#define HIG_DISPATCH_CALL(name, ...) \
    (hig_fma_build() ? hig_x86fma_##name(__VA_ARGS__) : hig_x86soft_##name(__VA_ARGS__))

hig_error_t hig_partition(const hig_graph_t *graph, const hig_options_t *options, int start,
                          hig_int *membership, hig_int *nb_communities, double *quality,
                          double *equilibrium_slack) {
    return HIG_DISPATCH_CALL(partition, graph, options, start, membership, nb_communities,
                             quality, equilibrium_slack);
}

hig_error_t hig_cover(const hig_graph_t *graph, const hig_options_t *options,
                      const hig_cover_t *start, hig_cover_t *result, hig_int *nb_communities,
                      double *quality, double *equilibrium_slack) {
    return HIG_DISPATCH_CALL(cover, graph, options, start, result, nb_communities, quality,
                             equilibrium_slack);
}

hig_error_t hig_partition_traced(const hig_graph_t *graph, const hig_options_t *options, int start,
                                 hig_int *membership, hig_int *nb_communities, double *quality,
                                 double *raw_slack, const hig_trace_t *trace) {
    return HIG_DISPATCH_CALL(partition_traced, graph, options, start, membership,
                             nb_communities, quality, raw_slack, trace);
}

hig_error_t hig_cover_traced(const hig_graph_t *graph, const hig_options_t *options,
                             const hig_cover_t *start, hig_cover_t *result,
                             hig_int *nb_communities, double *quality, double *raw_slack,
                             const hig_trace_t *trace) {
    return HIG_DISPATCH_CALL(cover_traced, graph, options, start, result, nb_communities,
                             quality, raw_slack, trace);
}

hig_error_t hig_audit_prepare(const hig_graph_t *g, const hig_cover_t *c,
                              const hig_interrupt_t *i, hig_audit_state_t **s) {
    return HIG_DISPATCH_CALL(audit_prepare, g, c, i, s);
}
void hig_audit_free(hig_audit_state_t *s) {
    HIG_DISPATCH_CALL(audit_free, s);
}
hig_error_t hig_audit_intervals(const hig_audit_state_t *s, const hig_audit_options_t *o,
                                hig_audit_selection_t v, hig_audit_intervals_t *r) {
    return HIG_DISPATCH_CALL(audit_intervals, s, o, v, r);
}
void hig_audit_intervals_free(hig_audit_intervals_t *r) {
    HIG_DISPATCH_CALL(audit_intervals_free, r);
}
hig_error_t hig_audit_regrets(const hig_audit_state_t *s, const hig_audit_options_t *o,
                              hig_audit_selection_t v, const double *g, hig_int n,
                              double *r, double *t) {
    return HIG_DISPATCH_CALL(audit_regrets, s, o, v, g, n, r, t);
}

hig_error_t hig_verify_partition(const hig_graph_t *graph, const hig_options_t *options,
                                 const hig_int *membership, hig_verify_t *report) {
    return HIG_DISPATCH_CALL(verify_partition, graph, options, membership, report);
}

hig_error_t hig_verify_cover(const hig_graph_t *graph, const hig_options_t *options,
                             const hig_cover_t *cover, hig_verify_t *report) {
    return HIG_DISPATCH_CALL(verify_cover, graph, options, cover, report);
}

#else

typedef int hig_dispatch_unused;   /* ISO C: a translation unit declares something */

#endif
