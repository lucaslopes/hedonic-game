/*
   hedonic -- what both engines share: errors and memory, input checks, the
   adjacency, the labels of a start.
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Errors and memory --------------------------------------------------- */

static HIG_THREAD_LOCAL char hig_message[512];

HIG_SHARED hig_error_t hig_error(hig_error_t code, const char *format, ...) {
    va_list args;

    va_start(args, format);
    vsnprintf(hig_message, sizeof(hig_message), format, args);
    va_end(args);
    return code;
}

const char *hig_error_message(void) {
    return hig_message;
}

HIG_SHARED hig_error_t hig_alloc(void *p, int64_t count, size_t size, bool zero) {
    void **out = (void **) p;
    const size_t items = count > 0 ? (size_t) count : 1;

    *out = NULL;
    if (count < 0 || (uint64_t) count > SIZE_MAX / size) {
        return hig_error(HIG_EOVERFLOW, "Allocation size overflow.");
    }
    *out = zero ? calloc(items, size) : malloc(items * size);
    if (!*out) {
        return hig_error(HIG_ENOMEM, "Out of memory.");
    }
    return HIG_SUCCESS;
}

HIG_SHARED hig_error_t hig_grow(void *p, int64_t *capacity, int64_t needed, size_t size) {
    void **data = (void **) p;
    int64_t grown;
    void *resized;

    if (needed <= *capacity) {
        return HIG_SUCCESS;
    }
    grown = *capacity < 8 ? 16 : *capacity > INT64_MAX / 2 ? INT64_MAX : 2 * *capacity;
    if (grown < needed) {
        grown = needed;
    }
    if ((uint64_t) grown > SIZE_MAX / size) {
        return hig_error(HIG_EOVERFLOW, "Allocation size overflow.");
    }
    resized = realloc(*data, (size_t) grown * size);
    if (!resized) {
        return hig_error(HIG_ENOMEM, "Out of memory.");
    }
    *data = resized;
    *capacity = grown;
    return HIG_SUCCESS;
}

/* ---- Inputs ----------------------------------------------------------------- */

/* Any sum of payoff terms stays below DBL_MAX / 2^12, so no computation
 * of a payoff or of its error bound overflows (the rounding analysis assumes
 * none does). */
#define HIG_ARITHMETIC_BOUND (DBL_MAX / 4096.0)

HIG_SHARED hig_error_t hig_check_input(const hig_graph_t *graph, hig_options_t *options,
                                       bool cover) {
    const hig_int n = graph->vertex_count, m = graph->edge_count;
    const char *game = cover ? "The overlapping game" : "The hedonic game";
    double abs_edge_weight = 0.0, total_edge_weight = 0.0, total_node_weight = 0.0;
    double scale, cap;

    if (n < 0 || m < 0) {
        return hig_error(HIG_EINVAL, "Vertex and edge counts must be non-negative.");
    }
    if (n > HIG_ID_MAX - 1) {
        return hig_error(HIG_EOVERFLOW, "The core handles at most %d vertices.", HIG_ID_MAX - 1);
    }
    if (m > 0 && !graph->edges && !(graph->edge_first && graph->edge_second)) {
        return hig_error(HIG_EINVAL, "The edge list is missing.");
    }
    for (int64_t h = 0; h < 2 * m; h++) {
        const hig_int end = hig_edge_end(graph, h);
        if (end < 0 || end >= n) {
            return hig_error(HIG_EINVAL, "Edge %lld has an end outside [0, %lld).",
                             (long long) (h / 2), (long long) n);
        }
    }
    if (!options->rng || !options->rng->next32) {
        return hig_error(HIG_EINVAL, "A random-number source is required.");
    }
    if (!isfinite(options->resolution)) {
        return hig_error(HIG_EINVAL, "%s resolution must be finite.", game);
    }
    if (options->max_memberships < 1) {
        return hig_error(HIG_EINVAL, "max_memberships must be at least 1.");
    }
    if (options->max_total_communities <= 0) {
        options->max_total_communities = -1;
    }
    if (options->n_communities <= 0) {
        options->n_communities = -1;
    }
    if (hig_limit_set(options->max_total_communities) &&
        options->n_communities > options->max_total_communities) {
        return hig_error(HIG_EINVAL, "n_communities must not exceed max_total_communities.");
    }
    if (cover) {
        if (n < 1 || m < 1) {
            return hig_error(HIG_EINVAL, "%s requires a nonempty graph with at least one edge.",
                             game);
        }
        for (hig_int e = 0; e < m; e++) {
            if (hig_edge_end(graph, 2 * e) == hig_edge_end(graph, 2 * e + 1)) {
                return hig_error(HIG_EINVAL, "%s requires a loopless graph.", game);
            }
        }
        if (options->max_memberships > n) {
            return hig_error(HIG_EINVAL,
                             "%s max_memberships must not exceed the number of vertices.", game);
        }
        if (options->n_communities > n * options->max_memberships) {
            return hig_error(HIG_EINVAL, "n_communities exceeds n * max_memberships, the largest "
                                         "number of communities a cover can occupy.");
        }
    } else {
        if (options->max_memberships != 1) {
            return hig_error(HIG_EINVAL, "A partition has max_memberships = 1; use hig_cover.");
        }
        if (options->n_communities > n) {
            return hig_error(HIG_EINVAL,
                             "n_communities must not exceed the number of vertices for a partition.");
        }
    }
    for (hig_int e = 0; e < m; e++) {
        const double w = graph->edge_weights ? graph->edge_weights[e] : 1.0;
        if (!isfinite(w) || (cover && w < 0.0)) {
            return hig_error(HIG_EINVAL, cover ? "%s edge weights must be finite and non-negative."
                                               : "%s edge weights must be finite.", game);
        }
        abs_edge_weight += fabs(w);
        total_edge_weight += w;
    }
    if (!(abs_edge_weight <= HIG_ARITHMETIC_BOUND)) {
        return hig_error(HIG_EOVERFLOW, "%s total edge weight would overflow.", game);
    }
    if (cover && !(total_edge_weight > 0.0)) {
        return hig_error(HIG_EINVAL, "%s requires positive finite total edge weight.", game);
    }
    for (hig_int v = 0; v < n; v++) {
        const double w = graph->node_weights ? graph->node_weights[v] : 1.0;
        if (!isfinite(w) || w < 0.0) {
            return hig_error(HIG_EINVAL, "%s node weights must be finite and non-negative.", game);
        }
        total_node_weight += w;
    }
    /* |gamma| N^2 m bounds every crowding sum (m labels of mass <= N each). */
    scale = total_node_weight > 1.0 ? total_node_weight : 1.0;
    cap = HIG_ARITHMETIC_BOUND / scale / scale / (double) options->max_memberships;
    if (!isfinite(total_node_weight) || fabs(options->resolution) > cap) {
        return hig_error(HIG_EOVERFLOW, "%s crowding term would overflow.", game);
    }
    return HIG_SUCCESS;
}

/* ---- Adjacency -------------------------------------------------------------- */

HIG_SHARED void hig_adjacency_free(hig_adjacency_t *adjacency) {
    free(adjacency->start);
    free(adjacency->adj);
    free(adjacency->weight);
    adjacency->start = NULL;
    adjacency->adj = NULL;
    adjacency->weight = NULL;
}

/* Counting sort of the half-edges by owner. Edges are placed from the last
 * to the first at the end of each block, so a block lists v's edges in
 * increasing edge id and no cursor array is needed. */
HIG_SHARED hig_error_t hig_adjacency_build(hig_adjacency_t *adjacency, const hig_graph_t *graph) {
    const hig_id n = (hig_id) graph->vertex_count;
    const hig_int m = graph->edge_count;
    bool unit = true;
    hig_error_t status = HIG_SUCCESS;
    int64_t *start, half_edges;

    adjacency->n = n;
    adjacency->node_weight = NULL;
    adjacency->total_node_weight = 0.0;
    adjacency->max_degree = 0;
    for (hig_int e = 0; graph->edge_weights && e < m; e++) {
        if (graph->edge_weights[e] != 1.0) {
            unit = false;
            break;
        }
    }
    for (hig_id v = 0; graph->node_weights && v < n; v++) {
        if (graph->node_weights[v] != 1.0) {
            adjacency->node_weight = graph->node_weights;
            break;
        }
    }
    for (hig_id v = 0; v < n; v++) {
        adjacency->total_node_weight += adjacency->node_weight ? adjacency->node_weight[v] : 1.0;
    }

    HIG_TRY(hig_alloc(&adjacency->start, (int64_t) n + 1, sizeof(int64_t), true));
    start = adjacency->start;
    for (hig_int e = 0; e < m; e++) {
        const hig_int a = hig_edge_end(graph, 2 * e), b = hig_edge_end(graph, 2 * e + 1);
        if (a != b) {
            start[a + 1]++;
            start[b + 1]++;
        }
    }
    for (hig_id v = 0; v < n; v++) {
        if (start[v + 1] > adjacency->max_degree) {
            adjacency->max_degree = start[v + 1];
        }
        start[v + 1] += start[v];
    }
    half_edges = start[n];
    HIG_TRY(hig_alloc(&adjacency->adj, start[n], sizeof(hig_id), false));
    if (!unit) {
        HIG_TRY(hig_alloc(&adjacency->weight, start[n], sizeof(double), false));
    }
    /* start[v + 1] is the end of v's block; filling backwards moves it to the
     * block's beginning, so afterwards start[v + 1] = begin(v). */
    for (hig_int e = m - 1; e >= 0; e--) {
        const hig_int a = hig_edge_end(graph, 2 * e), b = hig_edge_end(graph, 2 * e + 1);
        if (a != b) {
            const int64_t ia = --start[a + 1], ib = --start[b + 1];
            adjacency->adj[ia] = (hig_id) b;
            adjacency->adj[ib] = (hig_id) a;
            if (!unit) {
                adjacency->weight[ia] = graph->edge_weights[e];
                adjacency->weight[ib] = graph->edge_weights[e];
            }
        }
    }
    memmove(start, start + 1, (size_t) n * sizeof(int64_t));
    start[n] = half_edges;
done:
    if (status != HIG_SUCCESS) {
        hig_adjacency_free(adjacency);
    }
    return status;
}

/* ---- Labels of a start ------------------------------------------------------ */

static int hig_compare_int(const void *a, const void *b) {
    const hig_int x = *(const hig_int *) a, y = *(const hig_int *) b;
    return (x > y) - (x < y);
}

HIG_SHARED hig_error_t hig_compact_labels(const hig_int *in, int64_t count, hig_id *out,
                                          int64_t *distinct) {
    hig_int largest = -1;
    hig_error_t status = HIG_SUCCESS;
    hig_id *rank = NULL;
    hig_int *sorted = NULL;
    int64_t next = 0;

    for (int64_t i = 0; i < count; i++) {
        if (in[i] > largest) {
            largest = in[i];
        }
    }
    if (largest < 4 * count + 64) {
        /* Dense: a rank per possible label. */
        HIG_TRY(hig_alloc(&rank, largest + 1, sizeof(hig_id), true));
        for (int64_t i = 0; i < count; i++) {
            rank[in[i]] = 1;
        }
        for (hig_int c = 0; c <= largest; c++) {
            if (rank[c]) {
                if (next >= HIG_ID_MAX) {
                    HIG_TRY(hig_error(HIG_EOVERFLOW, "Too many distinct labels."));
                }
                rank[c] = (hig_id) next++;
            } else {
                rank[c] = -1;
            }
        }
        for (int64_t i = 0; i < count; i++) {
            out[i] = rank[in[i]];
        }
    } else {
        HIG_TRY(hig_alloc(&sorted, count, sizeof(hig_int), false));
        memcpy(sorted, in, (size_t) count * sizeof(hig_int));
        qsort(sorted, (size_t) count, sizeof(hig_int), hig_compare_int);
        for (int64_t i = 0; i < count; i++) {
            if (i == 0 || sorted[i] != sorted[next - 1]) {
                sorted[next++] = sorted[i];
            }
        }
        if (next > HIG_ID_MAX) {
            HIG_TRY(hig_error(HIG_EOVERFLOW, "Too many distinct labels."));
        }
        for (int64_t i = 0; i < count; i++) {
            int64_t lo = 0, hi = next - 1;
            while (lo < hi) {
                const int64_t mid = lo + (hi - lo) / 2;
                if (sorted[mid] < in[i]) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out[i] = (hig_id) lo;
        }
    }
    *distinct = next;
done:
    free(rank);
    free(sorted);
    return status;
}
