/*
   hedonic: best-response local moving for the CPM hedonic game.
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
 * The public interface of the C core: two entry points over plain arrays,
 * hig_partition (each vertex in one community) and hig_cover (each vertex in
 * at most max_memberships communities: the unit-l2 overlapping game). The
 * core links against nothing but the C library. native/README.md explains
 * the design; in short:
 *
 * THE GAME. Vertex v has weight n_v >= 0 (1 by default), gamma is the
 * resolution and W the total edge weight. In a partition, v in community C
 * earns u_v(C) = w(v, C \ v) - gamma n_v N(C \ v), with N(S) the total
 * weight of S. In a cover, v holds k_v labels with intensity f_v = 1/sqrt(k_v)
 * on each and earns U_v(sigma) = sum_{c in sigma} g_c(v) / sqrt(|sigma|),
 * g_c(v) = sum_{u ~ v holding c} w_uv f_u - gamma n_v sum_{u != v holding c}
 * n_u f_u. Both games have an exact potential Phi (a sum over pairs of
 * vertices); the returned quality is (Phi - gamma sum_v n_v^2 / 2) / W, the
 * constant being each vertex's own crowding.
 *
 * THE DYNAMICS. Vertices are visited from a queue, at first all of them in a
 * random order, and again, in a fresh random order, the vertices not
 * settled when the queue runs dry (shuffles: the only use of the random
 * source). A visit
 * computes the vertex's best response over every feasible action and adopts
 * it only when the move certainly raises the vertex's exact payoff: the
 * computed improvement exceeds a rigorous bound of its rounding error, built
 * operation by operation under the IEEE model (native/README.md, "Certified
 * moves"). Every accepted move therefore raises the exact potential: no
 * state repeats, and the dynamics stops on the finite state space. The call
 * returns when every vertex has been visited, without moving, after the last
 * move. Those visits also bound, from above, the exact gain of every feasible
 * unilateral deviation; the largest bound, divided by |W| as the quality is,
 * is the equilibrium slack: no deviation improves a vertex's exact payoff by
 * more than slack |W|. A gain the arithmetic cannot certify is at most twice
 * its own error bound. For a partition with unit weights at a resolution
 * whose products are exact (such as 0, 1/2 or 1) every bound is 0, so the
 * slack is 0 and the result is an exact Nash equilibrium; a cover's rounded
 * intensities 1/sqrt(k) keep its slack positive.
 *
 * DETERMINISM. No threads, fixed summation orders, explicit fused
 * multiply-adds that are the FMA instruction or the core's own correctly
 * rounded soft fma (never the C library's), no libm function but sqrt
 * (correctly rounded by IEEE 754) and fabs (exact), and the core is compiled with
 * -ffp-contract=off (/fp:strict on MSVC): given the inputs and the random
 * words, results are identical on every platform. On x86_64 the core is
 * built with and without the FMA instruction and each call takes the build
 * the CPU can run; both give the same results.
 */

#ifndef HEDONIC_CORE_H
#define HEDONIC_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HIG_VERSION_STRING "2.0.0"

typedef int64_t hig_int;

typedef enum {
    HIG_SUCCESS = 0,
    HIG_EINVAL,       /* invalid argument or input outside the model's domain */
    HIG_ENOMEM,       /* out of memory */
    HIG_EOVERFLOW,    /* a size, or the arithmetic of the game, would overflow */
    HIG_INTERRUPTED,  /* the interruption callback asked to stop */
    HIG_ECALLBACK     /* the random-number callback failed */
} hig_error_t;

/* A source of random 32-bit words. A callback sets *failed to a nonzero
 * value to abort the call (with HIG_ECALLBACK). */
typedef struct {
    uint32_t (*next32)(void *state, int *failed);
    void *state;
} hig_rng_t;

/* A built-in source (PCG32, XSH-RR), for C callers and tests. */
typedef struct {
    uint64_t state;
    uint64_t increment;
} hig_pcg32_t;

void hig_pcg32_seed(hig_pcg32_t *pcg, uint64_t seed, uint64_t stream);
uint32_t hig_pcg32_next32(void *pcg, int *failed);

/* Polled every few thousand visits; a nonzero return stops the call with
 * HIG_INTERRUPTED. */
typedef struct {
    int (*check)(void *state);
    void *state;
} hig_interrupt_t;

/* An undirected graph: edge e joins edges[2e] and edges[2e + 1], or, when
 * edges is NULL, edge_first[e] and edge_second[e] (two arrays, read without a
 * copy). Multi-edges are allowed; self-loops are allowed for partitions (they
 * add a constant to the potential) and rejected for covers. NULL weights mean
 * all ones. Edge weights are finite (and non-negative for covers); node
 * weights are finite and non-negative. */
typedef struct {
    hig_int vertex_count;
    hig_int edge_count;
    const hig_int *edges;
    const hig_int *edge_first;
    const hig_int *edge_second;
    const double *edge_weights;
    const double *node_weights;
} hig_graph_t;

typedef struct {
    double resolution;              /* gamma, finite */
    hig_int max_memberships;        /* labels per vertex at most: 1 for partitions */
    hig_int max_total_communities;  /* occupied communities at most; <= 0: no limit */
    hig_int n_communities;          /* exactly this many communities; <= 0: no limit */
    int allow_isolation;            /* may a vertex move to an empty community */
    const hig_rng_t *rng;           /* required */
    const hig_interrupt_t *interrupt; /* NULL: never interrupted */
} hig_options_t;

/* A cover: row v is labels[offset[v] .. offset[v + 1] - 1]. */
typedef struct {
    hig_int vertex_count;
    hig_int *offset;   /* vertex_count + 1 entries */
    hig_int *labels;
} hig_cover_t;

/* Releases a cover returned by hig_cover (arrays from malloc). */
void hig_cover_free(hig_cover_t *cover);

/* Local moving of a partition. With `start`, membership[] (vertex_count
 * non-negative labels) is the start partition; otherwise singletons, or,
 * under a count limit, vertex v in community v mod K. On success
 * membership[] holds communities 0 .. *nb_communities - 1 in order of first
 * appearance, *quality (if not NULL) the potential divided by W and
 * *equilibrium_slack (if not NULL) the equilibrium slack (NaN when W = 0). */
hig_error_t hig_partition(const hig_graph_t *graph, const hig_options_t *options, int start,
                          hig_int *membership, hig_int *nb_communities, double *quality,
                          double *equilibrium_slack);

/* Local moving of a cover. `start` (NULL: singletons, or, under a count
 * limit, a round-robin start that occupies K labels) gives each vertex a
 * non-empty row of non-negative labels; a repeated label counts once. On
 * success *result holds sorted rows with labels 0 .. *nb_communities - 1 in
 * order of first appearance (release it with hig_cover_free), *quality
 * (if not NULL) the potential divided by W and *equilibrium_slack (if not
 * NULL) the equilibrium slack. */
hig_error_t hig_cover(const hig_graph_t *graph, const hig_options_t *options,
                      const hig_cover_t *start, hig_cover_t *result, hig_int *nb_communities,
                      double *quality, double *equilibrium_slack);

/* Fixed-cover stability audit on [0, 1], with the paper's plain-double
 * arithmetic and heuristic tolerance (NOT the detector's certificate).
 * Prepared state owns its adjacency and normalized rows. No per-(vertex,
 * label) support table is stored. Concurrent readers may share the state. */
typedef struct hig_audit_state hig_audit_state_t;
typedef struct {
    hig_int max_memberships; /* 0: largest row; otherwise >= largest row */
    int allow_isolation;    /* add a fresh, zero-score label */
    int prune;              /* 1: relevant labels; 0: reference superset */
    double atol, rtol;      /* finite, non-negative */
    const hig_interrupt_t *interrupt;
} hig_audit_options_t;
#define HIG_AUDIT_OPTIONS_INIT { 0, 0, 1, 1e-10, 1e-9, NULL }

/* vertices == NULL selects [begin, end); otherwise selects vertices[begin:end]
 * in the given order (duplicates allowed). All counts/offsets are int64. */
typedef struct {
    hig_int begin, end;
    const hig_int *vertices;
} hig_audit_selection_t;
typedef struct { double lo, hi; } hig_interval_t;
typedef struct {
    hig_int vertex_count, interval_count;
    hig_int *offset;       /* vertex_count + 1 entries */
    hig_interval_t *intervals;
} hig_audit_intervals_t;

/* Nonempty rows, non-negative contiguous labels; duplicate row labels count
 * once and rows are sorted in the owned copy. Loopless graph, finite
 * non-negative edge weights, unit node weights only. Empty graphs allowed.
 * *state is NULL on failure. Free with hig_audit_free. */
hig_error_t hig_audit_prepare(const hig_graph_t *graph, const hig_cover_t *cover,
                              const hig_interrupt_t *interrupt, hig_audit_state_t **state);
void hig_audit_free(hig_audit_state_t *state);
hig_error_t hig_audit_intervals(const hig_audit_state_t *state,
                                const hig_audit_options_t *options,
                                hig_audit_selection_t vertices,
                                hig_audit_intervals_t *result);
void hig_audit_intervals_free(hig_audit_intervals_t *result);
/* Caller-owned row-major arrays: (end-begin) * resolution_count doubles each.
 * All resolutions must be finite and in [0, 1]. On error outputs may be
 * partially written. Regrets <= 1e-15 are clamped to zero. */
hig_error_t hig_audit_regrets(const hig_audit_state_t *state,
                              const hig_audit_options_t *options,
                              hig_audit_selection_t vertices,
                              const double *resolutions, hig_int resolution_count,
                              double *regret, double *tolerance);

/* The message of the last error raised on this thread. */
const char *hig_error_message(void);

#ifdef __cplusplus
}
#endif

#endif
