/*
   hedonic -- the state of the overlapping engine, shared by its files.
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
 * cover.c holds the result, the quality and the entry points (hig_cover,
 * hig_cover_traced); rows.c the labels and the rows; visit.c a visit, the
 * run and the slack; start.c the start and the setup. Functions one of them
 * calls in another are HIG_SHARED (static in the one-unit build, unity.c).
 */

#ifndef HIG_COVER_H
#define HIG_COVER_H

#include "internal.h"

typedef struct {
    int64_t start;      /* in the pool */
    hig_id size;
    hig_id capacity;
} hig_row_ref_t;

typedef struct {
    double gain;
    hig_id label;
    hig_id index;       /* the candidate's index in the visit */
} hig_scored_t;

#define HIG_FRESH HIG_ID_MAX      /* a fresh label's sort key: last among equal gains */
#define HIG_SHORT_SELECTION 32

typedef struct {
    hig_adjacency_t graph;
    double gamma;
    hig_id cap;                 /* max_memberships */
    bool isolation, exact;
    hig_int max_total;          /* -1: no limit */
    double sqrt_cap_up;         /* >= sqrt(cap) */

    hig_row_ref_t *row;
    hig_id *pool;
    int64_t pool_size, pool_capacity, pool_live;

    hig_group_t *group;
    hig_tail_t *tail;           /* the masses' low halves and error bounds */
    double theta;               /* >= (err + |lo|) / mass of every label */
    double theta_below;         /* theta (1 - 16u), or 0 (hig_cover_track) */
    bool rebuild;               /* a mass lost too much of itself to cancellation */
    double tiny_mass;           /* n eta when a term n_u f_u can underflow, else 0 */
    bool tiny_support;          /* a term w f_u can underflow */
    int64_t label_capacity;
    hig_id label_count;         /* ids in use are below this */
    hig_id *free_ids;
    int64_t free_count, free_capacity;
    hig_id occupied;

    double *inv_sqrt;           /* inv_sqrt[k] = 1 / sqrt(k), k <= cap */
    bool use_heap, heap_ready;
    hig_heap_t heap;

    hig_id *candidate;
    double *value;              /* support, then gain, of each candidate */
    int64_t candidate_capacity;
    hig_scored_t *top;
    int64_t top_capacity;
    hig_id *chosen;             /* cap entries */
    hig_queue_t queue;
    double *own_support;        /* cap entries: the supports of v's labels */
    double slack;               /* the equilibrium slack (hig_cover_slack) */
    const hig_trace_t *trace;
} hig_cover_state_t;

/* rows.c */
HIG_SHARED hig_id hig_row_capacity(const hig_cover_state_t *s, hig_id size);
HIG_SHARED hig_error_t hig_cover_reserve_labels(hig_cover_state_t *s, int64_t ids);
HIG_SHARED hig_error_t hig_cover_fresh_label(hig_cover_state_t *s, hig_id *label);
HIG_SHARED void hig_cover_rebuild(hig_cover_state_t *s);
HIG_SHARED hig_error_t hig_cover_leave(hig_cover_state_t *s, hig_id c, double term);
HIG_SHARED void hig_cover_join(hig_cover_state_t *s, hig_id c, double term);
HIG_SHARED void hig_cover_rescale(hig_cover_state_t *s, hig_id c, double old_term, double new_term);
HIG_SHARED hig_error_t hig_cover_prepare_heap(hig_cover_state_t *s);
HIG_SHARED hig_error_t hig_cover_write_row(hig_cover_state_t *s, hig_id v, const hig_id *labels,
                                           hig_id size);

/* visit.c */
HIG_SHARED hig_error_t hig_cover_reserve_candidates(hig_cover_state_t *s, int64_t needed);
HIG_SHARED hig_error_t hig_cover_run(hig_cover_state_t *s);
HIG_SHARED hig_error_t hig_cover_slack(hig_cover_state_t *s);

/* start.c */
HIG_SHARED void hig_sort_ids(hig_id *a, int64_t count);
HIG_SHARED hig_error_t hig_cover_init(hig_cover_state_t *s, const hig_graph_t *graph,
                                      const hig_options_t *options, const hig_cover_t *start);

#endif
