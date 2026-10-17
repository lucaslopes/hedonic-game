/*
   hedonic -- the state of the disjoint engine, shared by its files.
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
 * partition.c holds the start, the quality and the entry points (hig_partition,
 * hig_partition_traced); visit.c the masses, the payoff bounds, a visit and
 * the run. Functions one of them calls in the other are HIG_SHARED (static in
 * the one-unit build, unity.c).
 */

#ifndef HIG_PARTITION_H
#define HIG_PARTITION_H

#include "internal.h"

typedef struct {
    hig_adjacency_t graph;
    double gamma;
    bool isolation, exact;
    hig_int max_total;          /* -1: no limit */
    hig_id *membership;
    hig_group_t *group;         /* n ids */
    hig_tail_t *tail;           /* the masses' low halves and error bounds */
    double theta;               /* >= (err + |lo|) / mass of every community */
    double crowd;               /* 4 total_up (theta + 5u): coarse's |gamma n_v| term */
    bool rebuild;               /* a mass lost too much of itself to cancellation */
    double total_up;            /* an upper bound of every community's mass, with room */
    hig_id *free_ids;           /* empty community ids, a stack */
    hig_id free_count;
    hig_id occupied;
    bool use_heap, heap_ready;
    hig_heap_t heap;
    hig_id *candidate;          /* the communities of v's neighbours */
    double *candidate_weight;   /* w(v, c) for each */
    hig_queue_t queue;
    double slack;               /* the largest gain bound found since the last move */
    const hig_trace_t *trace;
} hig_partition_state_t;

/* visit.c */
HIG_SHARED void hig_partition_rebuild(hig_partition_state_t *s);
HIG_SHARED hig_error_t hig_partition_run(hig_partition_state_t *s);

#endif
