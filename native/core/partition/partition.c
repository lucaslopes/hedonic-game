/*
   hedonic -- best-response local moving of a partition.
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
 * The disjoint game. A visit of v scores, with v taken out of its community
 * c0, the payoff g(c) = w(v, c) - gamma n_v N(c) of every action: staying,
 * each community holding a neighbour (the weights w(v, c) gathered in a
 * sparse accumulator), an empty community when isolation is allowed and the
 * count limits permit one more, and the best community v has no edge to
 * (g = -gamma n_v N(c): read off the heap of communities by mass, needed only
 * when gamma n_v < 0 or every other action loses). v moves to the best action
 * when the move certainly raises its exact payoff: the computed improvement
 * exceeds a rigorous bound of its rounding error (internal.h; the bound of
 * each payoff is hig_partition_bound). A visit that does not move v also
 * bounds, from above, the exact gain of each of v's deviations: the largest
 * such bound after the last move is the equilibrium slack.
 */

#include "partition.h"

#include <stdlib.h>
#include <string.h>

static void hig_partition_state_free(hig_partition_state_t *s) {
    hig_adjacency_free(&s->graph);
    hig_heap_free(&s->heap);
    hig_queue_free(&s->queue);
    free(s->membership);
    free(s->group);
    free(s->tail);
    free(s->free_ids);
    free(s->candidate);
    free(s->candidate_weight);
}

/* The start: the caller's labels or the default, compacted, checked against
 * the count limits. */
static hig_error_t hig_partition_start(hig_partition_state_t *s, const hig_options_t *options,
                                       hig_id n, int start, hig_int *labels) {
    hig_error_t status = HIG_SUCCESS;
    int64_t distinct;

    if (!start) {
        const hig_int target = hig_limit_set(options->n_communities) ? options->n_communities
            : hig_limit_set(options->max_total_communities) &&
                      options->max_total_communities < n ? options->max_total_communities
            : n;
        for (hig_id v = 0; v < n; v++) {
            labels[v] = v % target;
        }
    }
    for (hig_id v = 0; v < n; v++) {
        if (labels[v] < 0) {
            return hig_error(HIG_EINVAL, "Initial membership labels must be non-negative.");
        }
    }
    HIG_TRY(hig_alloc(&s->membership, n, sizeof(hig_id), false));
    HIG_TRY(hig_compact_labels(labels, n, s->membership, &distinct));
    if (hig_limit_set(options->max_total_communities) &&
        distinct > options->max_total_communities) {
        return hig_error(HIG_EINVAL, "Initial membership exceeds max_total_communities.");
    }
    if (hig_limit_set(options->n_communities) && distinct != options->n_communities) {
        return hig_error(HIG_EINVAL,
                         "Initial membership does not contain exactly n_communities communities.");
    }
    s->occupied = (hig_id) distinct;
done:
    return status;
}

static hig_error_t hig_partition_init(hig_partition_state_t *s, const hig_graph_t *graph,
                                      const hig_options_t *options) {
    const hig_id n = (hig_id) graph->vertex_count;
    hig_error_t status = HIG_SUCCESS;

    s->gamma = options->resolution;
    s->isolation = options->allow_isolation != 0;
    s->exact = hig_limit_set(options->n_communities);
    s->max_total = options->max_total_communities;
    s->use_heap = s->gamma < 0.0 || !s->isolation || hig_limit_set(s->max_total) || s->exact;
    s->heap.descending = s->gamma < 0.0;

    HIG_TRY(hig_adjacency_build(&s->graph, graph));
    /* N is a sum of n non-negative terms: N(1 + gamma_n) bounds the exact
     * one, and every community's mass stays below twice that. */
    s->total_up = hig_up(2.0 * hig_add_up(s->graph.total_node_weight,
                                          hig_mul_up(hig_gamma((double) n),
                                                     s->graph.total_node_weight)));
    HIG_TRY(hig_alloc(&s->group, n, sizeof(hig_group_t), false));
    HIG_TRY(hig_alloc(&s->tail, n, sizeof(hig_tail_t), true));
    for (hig_id c = 0; c < n; c++) {
        s->group[c] = (hig_group_t) { 0.0, 0, -1 };
    }
    for (hig_id v = 0; v < n; v++) {
        s->group[s->membership[v]].members++;
    }
    HIG_TRY(hig_alloc(&s->free_ids, n, sizeof(hig_id), false));
    for (hig_id c = n - 1; c >= s->occupied; c--) {
        s->free_ids[s->free_count++] = c;
    }
    hig_partition_rebuild(s);   /* the masses, theta and the heap */
    HIG_TRY(hig_alloc(&s->candidate, s->graph.max_degree, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&s->candidate_weight, s->graph.max_degree, sizeof(double), false));
    HIG_TRY(hig_queue_init(&s->queue, n, options->rng, options->interrupt));
done:
    return status;
}

/* (2 sum_{internal} w - gamma sum_C N(C)^2) / 2W, in the caller's edge list. */
static hig_error_t hig_partition_quality(const hig_graph_t *graph, double gamma,
                                         const hig_int *membership, hig_int nb_communities,
                                         double *quality) {
    double *mass, q = 0.0, total = 0.0;
    hig_error_t status = HIG_SUCCESS;

    HIG_TRY(hig_alloc(&mass, nb_communities, sizeof(double), true));
    for (hig_int e = 0; e < graph->edge_count; e++) {
        const double w = graph->edge_weights ? graph->edge_weights[e] : 1.0;
        total += w;
        if (membership[hig_edge_end(graph, 2 * e)] == membership[hig_edge_end(graph, 2 * e + 1)]) {
            q += 2.0 * w;
        }
    }
    for (hig_int v = 0; v < graph->vertex_count; v++) {
        mass[membership[v]] += graph->node_weights ? graph->node_weights[v] : 1.0;
    }
    for (hig_int c = 0; c < nb_communities; c++) {
        q = hig_fma(-(gamma * mass[c]), mass[c], q);
    }
    free(mass);
    *quality = total != 0.0 ? q / (2.0 * total) : NAN;   /* W = 0: undefined */
done:
    return status;
}

/* The slack divided by |W| (rounded up), as the quality is: NaN when W = 0. */
HIG_SHARED double hig_normalized_slack(double slack, double total) {
    if (total == 0.0) {
        return NAN;
    }
    return slack > 0.0 ? hig_up(slack / fabs(total)) + HIG_ETA : 0.0;
}

hig_error_t hig_partition_traced(const hig_graph_t *graph, const hig_options_t *options_in,
                                 int start, hig_int *membership, hig_int *nb_communities,
                                 double *quality, double *raw_slack, const hig_trace_t *trace) {
    hig_options_t options = *options_in;
    hig_partition_state_t s;
    hig_error_t status = HIG_SUCCESS;
    hig_int next = 0;
    hig_id *relabel = NULL;

    memset(&s, 0, sizeof(s));
    s.trace = trace;
    HIG_TRY(hig_check_input(graph, &options, false));
    if (graph->vertex_count > 0 && !membership) {
        HIG_TRY(hig_error(HIG_EINVAL, "The membership array is missing."));
    }
    HIG_TRY(hig_partition_start(&s, &options, (hig_id) graph->vertex_count, start, membership));
    HIG_TRY(hig_partition_init(&s, graph, &options));
    if (trace && trace->row) {
        for (hig_id v = 0; v < s.graph.n; v++) {
            const int32_t label = s.membership[v];
            trace->row(trace->state, v, &label, 1);
        }
    }
    HIG_TRY(hig_partition_run(&s));

    /* Communities 0, 1, ... in order of first appearance. */
    HIG_TRY(hig_alloc(&relabel, graph->vertex_count, sizeof(hig_id), false));
    for (hig_int c = 0; c < graph->vertex_count; c++) {
        relabel[c] = -1;
    }
    for (hig_int v = 0; v < graph->vertex_count; v++) {
        const hig_id c = s.membership[v];
        if (relabel[c] < 0) {
            relabel[c] = (hig_id) next++;
        }
        membership[v] = relabel[c];
    }
    if (nb_communities) {
        *nb_communities = next;
    }
    if (quality) {
        HIG_TRY(hig_partition_quality(graph, options.resolution, membership, next, quality));
    }
    if (raw_slack) {
        *raw_slack = s.slack;
    }
done:
    free(relabel);
    hig_partition_state_free(&s);
    return status;
}

hig_error_t hig_partition(const hig_graph_t *graph, const hig_options_t *options, int start,
                          hig_int *membership, hig_int *nb_communities, double *quality,
                          double *equilibrium_slack) {
    double slack = 0.0, q = 0.0;
    hig_error_t status = hig_partition_traced(graph, options, start, membership, nb_communities,
                                              &q, &slack, NULL);

    if (status == HIG_SUCCESS) {
        if (quality) {
            *quality = q;
        }
        if (equilibrium_slack) {
            double total = 0.0;
            for (hig_int e = 0; e < graph->edge_count; e++) {
                total += graph->edge_weights ? graph->edge_weights[e] : 1.0;
            }
            *equilibrium_slack = hig_normalized_slack(slack, total);
        }
    }
    return status;
}
