/*
   hedonic -- the checking interface of the C core (tests only).
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
 * Not part of the public interface: what the C tests and the private
 * functions _trace_partition, _trace_cover, _verify_partition and
 * _verify_cover of hedonic._core use to check the engines against exact
 * arithmetic.
 *
 * A traced run is hig_partition / hig_cover that also reports its start
 * state, every accepted move and the raw (unnormalized) equilibrium slack.
 * Labels are the engine's internal ids: the start's labels compacted in
 * order, then the ids of emptied communities reused (a stack) or new ones.
 *
 * The verifier (check.c) takes a result and, with no queue, no settlement
 * record and masses summed afresh, scores every vertex's every feasible
 * deviation with the engines' certified arithmetic (internal.h).
 */

#ifndef HEDONIC_CHECK_H
#define HEDONIC_CHECK_H

#include "hedonic_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Row v of the state: once per vertex at the start, then after each
     * accepted move for the vertex that moved. */
    void (*row)(void *state, hig_int v, const int32_t *labels, hig_int size);
    /* An accepted move of v (called before its new row): the computed
     * improvement and the bound of its rounding error (improvement > bound). */
    void (*move)(void *state, hig_int v, double improvement, double bound);
    void *state;
} hig_trace_t;

/* hig_partition / hig_cover with a trace (NULL: none). *raw_slack (if not
 * NULL) receives the slack before it is divided by |W|. */
hig_error_t hig_partition_traced(const hig_graph_t *graph, const hig_options_t *options, int start,
                                 hig_int *membership, hig_int *nb_communities, double *quality,
                                 double *raw_slack, const hig_trace_t *trace);
hig_error_t hig_cover_traced(const hig_graph_t *graph, const hig_options_t *options,
                             const hig_cover_t *start, hig_cover_t *result,
                             hig_int *nb_communities, double *quality, double *raw_slack,
                             const hig_trace_t *trace);

typedef struct {
    hig_int vertices;      /* vertices checked (a vertex alone under an exact count has no deviation) */
    hig_int improvable;    /* vertices with a certainly improving deviation */
    double max_certified;  /* the largest certified lower bound of a deviation's gain (raw) */
    double max_bound;      /* the largest upper bound of a deviation's gain (raw) */
} hig_verify_t;

/* Checks a partition (membership: vertex_count labels in 0 .. K - 1) or a
 * cover (rows of distinct labels) under `options` (resolution, cap, count
 * limits, isolation; the random source is not used). */
hig_error_t hig_verify_partition(const hig_graph_t *graph, const hig_options_t *options,
                                 const hig_int *membership, hig_verify_t *report);
hig_error_t hig_verify_cover(const hig_graph_t *graph, const hig_options_t *options,
                             const hig_cover_t *cover, hig_verify_t *report);

/* hig_audit_intervals by the exhaustive search: every cell of every pairwise
 * crossing. The default search skips cells it proves irrelevant and must
 * return the same intervals, bit for bit (audit.c). */
hig_error_t hig_audit_intervals_exhaustive(const hig_audit_state_t *state,
                                           const hig_audit_options_t *options,
                                           hig_audit_selection_t vertices,
                                           hig_audit_intervals_t *result);

/* The fused multiply-add the core runs (internal.h, hig_fma): "instruction"
 * or "soft" for a build of one kind, and on an x86_64 dispatch target
 * (dispatch.h) "dispatch: instruction" or "dispatch: soft", the build the
 * calls go to. */
const char *hig_fma_path(void);
/* Whether this CPU runs the FMA instruction: 1 for a build that uses it, from
 * CPUID on a dispatch target, 0 otherwise. */
int hig_fma_cpu(void);
/* Dispatch targets, for tests: sends every later call to the build with the
 * instruction (1; refused when the CPU lacks it) or to the soft one (0).
 * Returns 1 when done, 0 when refused or on a build of one kind. Call it
 * while no other call of the core is running. */
int hig_fma_select(int instruction);

#ifdef __cplusplus
}
#endif

#endif
