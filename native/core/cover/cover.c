/*
   hedonic -- best-response local moving of a cover (the unit-l2 game).
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
 * The overlapping game. Vertex v holds a sorted row of k_v labels and puts
 * intensity f_v = 1 / sqrt(k_v) on each; label c has mass S_c = sum of n_u f_u
 * over its holders. With v's own contribution taken out, label c is worth
 *
 *     g_c(v) = sum_{u ~ v holding c} w_uv f_u - gamma n_v (S_c - n_v f_v [c in sigma_v])
 *
 * to v, and a set sigma is worth U_v(sigma) = sum_{c in sigma} g_c / sqrt(|sigma|).
 * The best set of at most M labels is therefore a prefix of the labels sorted
 * by g_c: the positive gains (the best j of them, j <= M, maximizing their sum
 * over sqrt(j)), or, if none is positive, the single best label. A visit
 * lists the candidates that can be in it: v's labels, its neighbours' labels
 * (supports gathered in a sparse accumulator), a fresh label (payoff 0) when
 * isolation is allowed and the count limits permit it, and the labels v has
 * no support from whose g_c = -gamma n_v S_c could win: the M heaviest when
 * gamma n_v < 0, the lightest when no other candidate reaches 0 (from the
 * heap of labels by mass). Under n_communities the labels v holds alone are
 * kept. v adopts the best set when that certainly raises its exact payoff:
 * the computed improvement exceeds a rigorous bound of its rounding error
 * (internal.h). A visit that does not move v bounds, from above, the exact
 * gain of every set v could hold instead: the largest such bound after the
 * last move is the equilibrium slack.
 */

#include "cover.h"

#include <stdlib.h>
#include <string.h>

static void hig_cover_state_free(hig_cover_state_t *s) {
    hig_adjacency_free(&s->graph);
    hig_heap_free(&s->heap);
    hig_queue_free(&s->queue);
    free(s->row);
    free(s->pool);
    free(s->group);
    free(s->tail);
    free(s->free_ids);
    free(s->inv_sqrt);
    free(s->candidate);
    free(s->value);
    free(s->top);
    free(s->chosen);
    free(s->own_support);
}

/* ---- Result and quality ----------------------------------------------------- */

void hig_cover_free(hig_cover_t *cover) {
    if (cover) {
        free(cover->offset);
        free(cover->labels);
        cover->offset = cover->labels = NULL;
        cover->vertex_count = 0;
    }
}

/* The rows with labels 0, 1, ... in order of first appearance, sorted. */
static hig_error_t hig_cover_result(const hig_cover_state_t *s, hig_cover_t *result,
                                    hig_int *nb_communities) {
    const hig_id n = s->graph.n;
    hig_id *relabel = NULL;
    int64_t next = 0, total = 0;
    hig_error_t status = HIG_SUCCESS;

    for (hig_id v = 0; v < n; v++) {
        total += s->row[v].size;
    }
    result->vertex_count = n;
    HIG_TRY(hig_alloc(&result->offset, (int64_t) n + 1, sizeof(hig_int), false));
    HIG_TRY(hig_alloc(&result->labels, total, sizeof(hig_int), false));
    HIG_TRY(hig_alloc(&relabel, s->label_count, sizeof(hig_id), false));
    for (hig_id c = 0; c < s->label_count; c++) {
        relabel[c] = -1;
    }
    result->offset[0] = 0;
    for (hig_id v = 0; v < n; v++) {
        const hig_row_ref_t r = s->row[v];
        hig_int *out = result->labels + result->offset[v];
        for (hig_id i = 0; i < r.size; i++) {
            const hig_id c = s->pool[r.start + i];
            if (relabel[c] < 0) {
                relabel[c] = (hig_id) next++;
            }
            out[i] = relabel[c];
        }
        for (hig_id i = 1; i < r.size; i++) {
            const hig_int x = out[i];
            hig_id j = i;
            while (j > 0 && out[j - 1] > x) {
                out[j] = out[j - 1];
                j--;
            }
            out[j] = x;
        }
        result->offset[v + 1] = result->offset[v] + r.size;
    }
    *nb_communities = next;
done:
    free(relabel);
    return status;
}

static hig_int hig_shared_labels(const hig_int *a, hig_int ka, const hig_int *b, hig_int kb) {
    hig_int i = 0, j = 0, shared = 0;

    while (i < ka && j < kb) {
        if (a[i] < b[j]) {
            i++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            shared++;
            i++;
            j++;
        }
    }
    return shared;
}

/* (sum_edges 2 w |sigma_a & sigma_b| f_a f_b - gamma sum_c S_c^2) / 2W. */
static hig_error_t hig_cover_quality(const hig_graph_t *graph, double gamma,
                                     const hig_cover_t *cover, hig_int nb_communities,
                                     const double *inv_sqrt, double *quality) {
    const hig_int *offset = cover->offset, *labels = cover->labels;
    double *mass, q = 0.0, total = 0.0;
    hig_error_t status = HIG_SUCCESS;

    HIG_TRY(hig_alloc(&mass, nb_communities, sizeof(double), true));
    for (hig_int e = 0; e < graph->edge_count; e++) {
        const hig_int a = hig_edge_end(graph, 2 * e), b = hig_edge_end(graph, 2 * e + 1);
        const hig_int ka = offset[a + 1] - offset[a], kb = offset[b + 1] - offset[b];
        const double w = graph->edge_weights ? graph->edge_weights[e] : 1.0;
        const hig_int shared = hig_shared_labels(labels + offset[a], ka, labels + offset[b], kb);
        total += w;
        if (shared > 0) {
            q += 2.0 * w * (double) shared * inv_sqrt[ka] * inv_sqrt[kb];
        }
    }
    for (hig_int v = 0; v < graph->vertex_count; v++) {
        const hig_int k = offset[v + 1] - offset[v];
        const double term = (graph->node_weights ? graph->node_weights[v] : 1.0) * inv_sqrt[k];
        for (hig_int i = offset[v]; i < offset[v + 1]; i++) {
            mass[labels[i]] += term;
        }
    }
    for (hig_int c = 0; c < nb_communities; c++) {
        q = hig_fma(-(gamma * mass[c]), mass[c], q);
    }
    free(mass);
    *quality = q / (2.0 * total);
done:
    return status;
}

hig_error_t hig_cover_traced(const hig_graph_t *graph, const hig_options_t *options_in,
                             const hig_cover_t *start, hig_cover_t *result,
                             hig_int *nb_communities, double *quality, double *raw_slack,
                             const hig_trace_t *trace) {
    hig_options_t options = *options_in;
    hig_cover_state_t s;
    hig_cover_t out = { 0, NULL, NULL };
    hig_int count = 0;
    hig_error_t status = HIG_SUCCESS;

    memset(&s, 0, sizeof(s));
    s.trace = trace;
    if (!result) {
        return hig_error(HIG_EINVAL, "The result cover is missing.");
    }
    HIG_TRY(hig_check_input(graph, &options, true));
    HIG_TRY(hig_cover_init(&s, graph, &options, start));
    if (trace && trace->row) {
        for (hig_id v = 0; v < s.graph.n; v++) {
            trace->row(trace->state, v, (const int32_t *) (s.pool + s.row[v].start),
                       s.row[v].size);
        }
    }
    HIG_TRY(hig_cover_run(&s));
    HIG_TRY(hig_cover_slack(&s));
    HIG_TRY(hig_cover_result(&s, &out, &count));
    if (quality) {
        HIG_TRY(hig_cover_quality(graph, options.resolution, &out, count, s.inv_sqrt, quality));
    }
    if (nb_communities) {
        *nb_communities = count;
    }
    if (raw_slack) {
        *raw_slack = s.slack;
    }
    *result = out;
    out.offset = out.labels = NULL;
done:
    hig_cover_free(&out);
    hig_cover_state_free(&s);
    return status;
}

hig_error_t hig_cover(const hig_graph_t *graph, const hig_options_t *options,
                      const hig_cover_t *start, hig_cover_t *result, hig_int *nb_communities,
                      double *quality, double *equilibrium_slack) {
    double slack = 0.0, q = 0.0;
    hig_error_t status = hig_cover_traced(graph, options, start, result, nb_communities, &q,
                                          &slack, NULL);

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
