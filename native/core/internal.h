/*
   hedonic -- internal declarations of the C core.
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
 * What both engines share (shared/): errors and checked allocation, input
 * checks, the adjacency, the random visiting order and the queue that decides
 * when every vertex is settled, the mass-ordered heap of communities, the
 * double-double masses and the certified arithmetic of the move rule.
 * partition/ and cover/ are the two engines, check/ the independent
 * verifier of a result (hedonic_check.h). Conventions: a function that can fail returns
 * hig_error_t; a workspace is zero-initialized and released by one call
 * whatever step failed.
 */

#ifndef HIG_INTERNAL_H
#define HIG_INTERNAL_H

#include "dispatch.h"

/* The two builds of an x86_64 dispatch target (dispatch.h) export their
 * entry points under prefixed names; dispatch.c defines the public ones. */
#if HIG_DISPATCH
#ifdef HIG_FMA_VARIANT
#define HIG_VARIANT(name) hig_x86fma_##name
#else
#define HIG_VARIANT(name) hig_x86soft_##name
#endif
#define hig_pcg32_seed HIG_VARIANT(pcg32_seed)
#define hig_pcg32_next32 HIG_VARIANT(pcg32_next32)
#define hig_cover_free HIG_VARIANT(cover_free)
#define hig_partition HIG_VARIANT(partition)
#define hig_cover HIG_VARIANT(cover)
#define hig_error_message HIG_VARIANT(error_message)
#define hig_partition_traced HIG_VARIANT(partition_traced)
#define hig_cover_traced HIG_VARIANT(cover_traced)
#define hig_verify_partition HIG_VARIANT(verify_partition)
#define hig_verify_cover HIG_VARIANT(verify_cover)
#define hig_audit_prepare HIG_VARIANT(audit_prepare)
#define hig_audit_free HIG_VARIANT(audit_free)
#define hig_audit_intervals HIG_VARIANT(audit_intervals)
#define hig_audit_intervals_free HIG_VARIANT(audit_intervals_free)
#define hig_audit_regrets HIG_VARIANT(audit_regrets)
#endif

#include "hedonic_core.h"
#include "check/hedonic_check.h"
#include "softfma.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Vertices, communities and labels. */
typedef int32_t hig_id;
#define HIG_ID_MAX INT32_MAX

#if defined(_MSC_VER) && !defined(__clang__)
#define HIG_THREAD_LOCAL __declspec(thread)
#else
#define HIG_THREAD_LOCAL _Thread_local
#endif

/* Functions one file calls in another: static in the one-unit build
 * (unity.c), so the compiler inlines them as freely as within a file. */
#ifdef HIG_UNITY_BUILD
#define HIG_SHARED static
#else
#define HIG_SHARED
#endif

/* The rare paths of a visit (a move's certificate, the slack) stay out of
 * line, so the scan every visit runs keeps its registers. */
#if defined(__GNUC__) || defined(__clang__)
#define HIG_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define HIG_NOINLINE __declspec(noinline)
#else
#define HIG_NOINLINE
#endif
/* ... and the steps of a move stay inline even where they are used twice. */
#if defined(__GNUC__) || defined(__clang__)
#define HIG_ALWAYS_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define HIG_ALWAYS_INLINE static __forceinline
#else
#define HIG_ALWAYS_INLINE static inline
#endif

/* ---- Errors and memory --------------------------------------------------- */

HIG_SHARED hig_error_t hig_error(hig_error_t code, const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* Stores the status of `expr` and jumps to `done` on failure. */
#define HIG_TRY(expr) \
    do { \
        status = (expr); \
        if (status != HIG_SUCCESS) { \
            goto done; \
        } \
    } while (0)

/* *p = an array of count elements of `size` bytes (zeroed if `zero`). */
HIG_SHARED hig_error_t hig_alloc(void *p, int64_t count, size_t size, bool zero);
/* Grows *p (holding *capacity elements) to at least `needed`, doubling. */
HIG_SHARED hig_error_t hig_grow(void *p, int64_t *capacity, int64_t needed, size_t size);

/* ---- Inputs and adjacency ------------------------------------------------ */

/* Entry h of the edge list read as one array: edge e joins 2e and 2e + 1. */
static inline hig_int hig_edge_end(const hig_graph_t *graph, int64_t h) {
    if (graph->edges) {
        return graph->edges[h];
    }
    return (h & 1 ? graph->edge_second : graph->edge_first)[h >> 1];
}

/* Checks the graph and the options both engines share; cover = 1 adds the
 * cover's domain (non-negative edge weights, no loops, at least one edge).
 * Normalizes the count limits (<= 0 becomes -1). */
HIG_SHARED hig_error_t hig_check_input(const hig_graph_t *graph, hig_options_t *options, bool cover);

static inline bool hig_limit_set(hig_int limit) {
    return limit > 0;
}

/* The adjacency of the game: v's neighbours are adj[start[v] .. start[v+1]),
 * in increasing edge id, self-loops left out. */
typedef struct {
    hig_id n;
    int64_t *start;
    hig_id *adj;
    double *weight;           /* NULL when every edge weight is 1 */
    const double *node_weight; /* NULL when every node weight is 1 */
    double total_node_weight; /* N */
    int64_t max_degree;
} hig_adjacency_t;

HIG_SHARED hig_error_t hig_adjacency_build(hig_adjacency_t *adjacency, const hig_graph_t *graph);
HIG_SHARED void hig_adjacency_free(hig_adjacency_t *adjacency);

/* ---- Labels of a start ---------------------------------------------------- */

/* Maps count non-negative labels to 0 .. *distinct - 1, preserving their
 * order (out may alias nothing; in is read only). */
HIG_SHARED hig_error_t hig_compact_labels(const hig_int *in, int64_t count, hig_id *out,
                                          int64_t *distinct);

/* ---- The visiting queue -------------------------------------------------- */

/* A FIFO of vertices (each at most once, `queued` marks them), the random
 * order they were first queued in, the random source, and the settlement
 * record: settled[v].at
 * is the number of moves made when a visit of v last found no improving move
 * (-1 if v has moved since, or was never visited), settled[v].bound what
 * that visit recorded (cover/visit.c: a bound of v's gain; together, because both
 * are written at every visit and the run is bound by memory latency). A
 * vertex is settled when settled[v].at equals the move count. When the FIFO runs dry, the unsettled
 * vertices are queued again in a fresh random order; when none is left, every
 * vertex is a best response to the others and the run stops. */
typedef struct {
    int64_t at;
    double bound;
} hig_settled_t;

typedef struct {
    hig_id n;
    hig_id *ring;
    uint8_t *queued;
    hig_id *order;
    hig_settled_t *settled;
    int64_t head, count;
    int64_t moves;
    int64_t visits;
    const hig_rng_t *rng;
    const hig_interrupt_t *interrupt;
} hig_queue_t;

HIG_SHARED hig_error_t hig_queue_init(hig_queue_t *queue, hig_id n, const hig_rng_t *rng,
                                      const hig_interrupt_t *interrupt);
HIG_SHARED void hig_queue_free(hig_queue_t *queue);
/* The next vertex to visit; *done when every vertex is settled. */
HIG_SHARED hig_error_t hig_queue_next(hig_queue_t *queue, hig_id *v, bool *done);

static inline void hig_queue_push(hig_queue_t *queue, hig_id u) {
    if (!queue->queued[u]) {
        int64_t tail = queue->head + queue->count;
        if (tail >= queue->n) {
            tail -= queue->n;
        }
        queue->ring[tail] = u;
        queue->queued[u] = 1;
        queue->count++;
    }
}

static inline void hig_queue_settled(hig_queue_t *queue, hig_id v) {
    queue->settled[v].at = queue->moves;
}

static inline void hig_queue_settled_bound(hig_queue_t *queue, hig_id v, double bound) {
    queue->settled[v].at = queue->moves;
    queue->settled[v].bound = bound;
}

static inline void hig_queue_moved(hig_queue_t *queue, hig_id v) {
    queue->moves++;
    queue->settled[v].at = -1;
}

/* ---- Certified arithmetic -------------------------------------------------- */

/* A move is made only when its computed improvement exceeds a rigorous bound
 * on the rounding error of that computation (native/README.md, "Certified
 * moves"). The bounds use the standard model of IEEE double arithmetic with
 * rounding to nearest: u = 2^-53, a sum (or difference) is off by at most
 * u |its rounded value| (and is exact when that value is subnormal), a
 * product or fma by at most u |its rounded value| + eta / 2, eta the smallest
 * subnormal. A bound is a non-negative double no smaller than the exact
 * quantity it bounds; the helpers below round every operation upward, so an
 * expression built from them is itself a bound. Every helper returns exactly
 * 0 when all its arguments are 0: exact computations get exact bounds. */

#ifndef DBL_TRUE_MIN
#define DBL_TRUE_MIN 4.9406564584124654e-324
#endif
#define HIG_U (DBL_EPSILON / 2.0)
#define HIG_ETA DBL_TRUE_MIN
/* 2^-968: the error of a product at least this large is a double, and
 * fma(a, b, -fl(a b)) computes it exactly. */
#define HIG_EXACT_PRODUCT (DBL_MIN * 18014398509481984.0)
/* A community's mass is rebuilt from its members when its accumulated error
 * exceeds this fraction of it (only cancellation can get there). */
#define HIG_REBUILD_RATIO (16.0 * HIG_U)

/* The fused multiply-add: x y + z rounded once. Every one in the core is
 * this, never the C library's fma(): the instruction where the target has it
 * at compile time (arm64; x86_64 with -mfma or /arch:AVX2, or the FMA build
 * of a dispatch target, HIG_FMA_VARIANT), otherwise the soft fma of
 * softfma.h, which computes the same correctly rounded result in integer
 * arithmetic. HIG_FORCE_SOFT_FMA (tests) takes the soft one everywhere.
 * HIG_FMA_INSTRUCTION says which this build is. MSVC on arm64 takes the soft
 * one: whether its fma() is the instruction is not documented. */
#if !defined(HIG_FORCE_SOFT_FMA) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__aarch64__) || defined(__FMA__) || defined(HIG_FMA_VARIANT))
#define HIG_FMA_INSTRUCTION 1
#define hig_fma(x, y, z) __builtin_fma(x, y, z)
#elif !defined(HIG_FORCE_SOFT_FMA) && defined(_MSC_VER) && defined(_M_X64) && \
    !defined(_M_ARM64EC) && (defined(__AVX2__) || defined(HIG_FMA_VARIANT))
#include <immintrin.h>
#define HIG_FMA_INSTRUCTION 1
static inline double hig_fma(double x, double y, double z) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(x), _mm_set_sd(y), _mm_set_sd(z)));
}
#else
#define HIG_FMA_INSTRUCTION 0
#define hig_fma(x, y, z) hig_soft_fma(x, y, z)
#endif

/* x >= 0 rounded up: >= x (1 + u)^2, and >= x. */
static inline double hig_up(double x) {
    return hig_fma(x, 4.0 * HIG_U, x);
}

/* Upper bounds of a + b and a b for bounds a, b >= 0. */
static inline double hig_add_up(double a, double b) {
    return hig_up(a + b);
}

static inline double hig_mul_up(double a, double b) {
    const double p = a * b;
    return p >= DBL_MIN ? hig_up(p) : (a > 0.0 && b > 0.0 ? p + HIG_ETA : 0.0);
}

/* gamma_k = (1 + u)^k - 1 bounds the relative error of k roundings; this is
 * an upper bound of it (with room for the roundings of its own use) for any
 * k below 2^40. */
static inline double hig_gamma(double k) {
    return k * (1.125 * HIG_U);
}

/* a + b = *sum + *error exactly (Knuth's TwoSum; no overflow assumed). */
static inline void hig_two_sum(double a, double b, double *sum, double *error) {
    const double s = a + b;
    const double bb = s - a;
    *error = (a - (s - bb)) + (b - bb);
    *sum = s;
}

/* The improvement a - b of a computed payoff a over a computed payoff b,
 * their errors bounded together by `bound`: *d is the computed difference and
 * *e bounds |exact improvement - *d|. The exact improvement is certainly
 * positive when *d > *e (both doubles: d - e > 0 holds exactly), and
 * hig_gain_up(d, e) bounds it from above. */
static inline void hig_difference(double a, double b, double bound, double *d, double *e) {
    double de;
    hig_two_sum(a, -b, d, &de);   /* a - b = d + de exactly */
    *e = hig_add_up(bound, fabs(de));
}

static inline double hig_gain_up(double d, double e) {
    const double t = d + e;       /* t <= 0 only if d + e <= 0 exactly */
    return t > 0.0 ? hig_up(t) : 0.0;
}

/* |exact a b - p| for p = fl(a b). */
static inline double hig_product_error(double a, double b, double p) {
    if (fabs(p) >= HIG_EXACT_PRODUCT || a == 0.0 || b == 0.0) {
        return fabs(hig_fma(a, b, -p));
    }
    return hig_add_up(hig_mul_up(HIG_U, fabs(p)), HIG_ETA);
}

/* ---- Communities and labels ----------------------------------------------- */

/* One community (or label): its mass (the high half of a double-double sum
 * of member weights, or of n_u f_u for labels), its number of members, and
 * its slot in the current visit's accumulator (-1 outside a visit). */
typedef struct {
    double mass;
    hig_id members;
    hig_id slot;
} hig_group_t;

/* The rest of a mass, next to each other because a move reads both: the
 * low half (the mass is hi + lo, hi in hig_group_t) and a bound of the
 * accumulated error. */
typedef struct {
    double lo, err;
} hig_tail_t;

/* hi + lo += x, both two-sums exact; the one rounded step (e + lo) adds its
 * exact error to *err, a bound of |exact sum - (hi + lo)|. */
HIG_ALWAYS_INLINE void hig_mass_add(double *hi, double *lo, double *err, double x) {
    double s, e, e2, r;
    hig_two_sum(*hi, x, &s, &e);
    hig_two_sum(e, *lo, &e2, &r);
    hig_two_sum(s, e2, hi, lo);
    if (r != 0.0) {
        *err = hig_add_up(*err, fabs(r));
    }
}

/* ---- The heap of communities by mass ------------------------------------- */

/* Occupied communities ordered by mass, lightest first (descending = false)
 * or heaviest first, ties by smaller id: the extreme community a vertex has
 * no edge to is found by walking it best-first and skipping the vertex's
 * candidates. pos[c] is c's index in heap[] or -1. */
typedef struct {
    hig_id *heap;
    hig_id *pos;
    int64_t size;
    int64_t capacity;          /* of pos (ids) and heap */
    bool descending;
    int64_t *frontier;         /* the walk's own small heap of heap indices */
    int64_t frontier_size, frontier_capacity;
} hig_heap_t;

HIG_SHARED hig_error_t hig_heap_reserve(hig_heap_t *heap, int64_t ids);
HIG_SHARED void hig_heap_free(hig_heap_t *heap);
HIG_SHARED void hig_heap_insert(hig_heap_t *heap, const hig_group_t *groups, hig_id c);
HIG_SHARED void hig_heap_remove(hig_heap_t *heap, const hig_group_t *groups, hig_id c);
HIG_SHARED void hig_heap_update(hig_heap_t *heap, const hig_group_t *groups, hig_id c);
/* Empties the heap (its ids stay reserved). */
HIG_SHARED void hig_heap_clear(hig_heap_t *heap);
/* Best-first walk: begin, then next until it returns -1. */
HIG_SHARED void hig_heap_walk_begin(hig_heap_t *heap);
HIG_SHARED hig_error_t hig_heap_walk_next(hig_heap_t *heap, const hig_group_t *groups,
                                          hig_id *c);

/* ---- Mass ratios ---------------------------------------------------------- */

/* An upper bound of err / hi (hi > 0), at least eta. */
static inline double hig_ratio_up(double err, double hi) {
    return hig_up(err / hi) + HIG_ETA;
}

/* The raw slack divided by |W| (rounded up), as the quality is: NaN when
 * W = 0 (partition/partition.c). */
HIG_SHARED double hig_normalized_slack(double slack, double total);

/* ---- Random numbers -------------------------------------------------------- */

/* A uniform integer in [0, range), range >= 1 (Lemire's multiply-and-reject
 * on 32-bit words). */
HIG_SHARED hig_error_t hig_random_below(const hig_rng_t *rng, uint32_t range, uint32_t *value);

#define HIG_VISITS_PER_INTERRUPTION_CHECK (1 << 14)

#endif
