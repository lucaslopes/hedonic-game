/*
   hedonic -- the start of a cover and the setup of its engine.
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

#include "cover.h"

#include <stdlib.h>
#include <string.h>

/* ---- Start and setup -------------------------------------------------------- */

HIG_SHARED void hig_sort_ids(hig_id *a, int64_t count) {
    for (int64_t i = 1; i < count; i++) {
        const hig_id x = a[i];
        int64_t j = i;
        while (j > 0 && a[j - 1] > x) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = x;
    }
}

static int hig_compare_ids(const void *a, const void *b) {
    const hig_id x = *(const hig_id *) a, y = *(const hig_id *) b;
    return (x > y) - (x < y);
}

/* The start rows (the caller's or the default) in the pool: labels
 * compacted, rows sorted and duplicate-free, count limits checked. */
static hig_error_t hig_cover_start(hig_cover_state_t *s, const hig_options_t *options, hig_id n,
                                   const hig_cover_t *start) {
    hig_int *labels = NULL;
    hig_id *ids = NULL;
    int64_t *offset = NULL, total = 0, distinct = 0;
    hig_error_t status = HIG_SUCCESS;

    HIG_TRY(hig_alloc(&offset, (int64_t) n + 1, sizeof(int64_t), false));
    if (start) {
        if (start->vertex_count != n || !start->offset || (!start->labels && n > 0)) {
            HIG_TRY(hig_error(HIG_EINVAL,
                              "Initial membership list length does not equal the number of vertices."));
        }
        offset[0] = 0;
        for (hig_id v = 0; v < n; v++) {
            const hig_int size = start->offset[v + 1] - start->offset[v];
            if (size < 1) {
                HIG_TRY(hig_error(HIG_EINVAL,
                                  "Initial overlapping membership vectors must be non-empty."));
            }
            offset[v + 1] = offset[v] + size;
        }
        total = offset[n];
        HIG_TRY(hig_alloc(&labels, total, sizeof(hig_int), false));
        for (hig_id v = 0; v < n; v++) {
            for (int64_t i = 0; i < offset[v + 1] - offset[v]; i++) {
                labels[offset[v] + i] = start->labels[start->offset[v] + i];
                if (labels[offset[v] + i] < 0) {
                    HIG_TRY(hig_error(HIG_EINVAL,
                                      "Initial overlapping membership indices must be non-negative."));
                }
            }
        }
    } else {
        /* Singletons; under a count limit, K labels round-robin (with more
         * labels than vertices, the extra ones are spread over the rows). */
        const hig_int target = hig_limit_set(options->n_communities) ? options->n_communities
            : hig_limit_set(options->max_total_communities) &&
                      options->max_total_communities < n ? options->max_total_communities
            : n;
        const hig_int extra = target > n ? target - n : 0;
        offset[0] = 0;
        for (hig_id v = 0; v < n; v++) {
            offset[v + 1] = offset[v] + 1 + extra / n + (v < extra % n);
        }
        total = offset[n];
        HIG_TRY(hig_alloc(&labels, total, sizeof(hig_int), false));
        for (hig_id v = 0; v < n; v++) {
            int64_t i = offset[v];
            labels[i++] = target < n ? v % target : v;
            for (hig_int c = n + v; c < target; c += n) {
                labels[i++] = c;
            }
        }
    }
    HIG_TRY(hig_alloc(&ids, total, sizeof(hig_id), false));
    HIG_TRY(hig_compact_labels(labels, total, ids, &distinct));

    /* Rows into the pool, each sorted and duplicate-free. */
    HIG_TRY(hig_alloc(&s->row, n, sizeof(hig_row_ref_t), false));
    s->pool_capacity = 1024;
    for (hig_id v = 0; v < n; v++) {
        hig_id *row = ids + offset[v];
        const int64_t size = offset[v + 1] - offset[v];
        int64_t kept = 1;
        if (size > 16) {
            qsort(row, (size_t) size, sizeof(hig_id), hig_compare_ids);
        } else {
            hig_sort_ids(row, size);
        }
        for (int64_t i = 1; i < size; i++) {
            if (row[i] != row[kept - 1]) {
                row[kept++] = row[i];
            }
        }
        if (kept > s->cap) {
            HIG_TRY(hig_error(HIG_EINVAL,
                              "Initial overlapping membership vector exceeds max_memberships."));
        }
        s->row[v].size = (hig_id) kept;
        s->row[v].capacity = hig_row_capacity(s, (hig_id) kept);
        s->row[v].start = s->pool_live;
        s->pool_live += s->row[v].capacity;
    }
    if (hig_limit_set(options->max_total_communities) &&
        distinct > options->max_total_communities) {
        HIG_TRY(hig_error(HIG_EINVAL, "Initial cover exceeds max_total_communities."));
    }
    if (hig_limit_set(options->n_communities) && distinct != options->n_communities) {
        HIG_TRY(hig_error(HIG_EINVAL,
                          "Initial cover does not contain exactly n_communities labels."));
    }
    s->pool_capacity = s->pool_live + s->pool_live / 2 + 1024;
    HIG_TRY(hig_alloc(&s->pool, s->pool_capacity, sizeof(hig_id), false));
    for (hig_id v = 0; v < n; v++) {
        memcpy(s->pool + s->row[v].start, ids + offset[v], (size_t) s->row[v].size * sizeof(hig_id));
    }
    s->pool_size = s->pool_live;
    s->label_count = (hig_id) distinct;
done:
    free(labels);
    free(ids);
    free(offset);
    return status;
}

HIG_SHARED hig_error_t hig_cover_init(hig_cover_state_t *s, const hig_graph_t *graph,
                                      const hig_options_t *options, const hig_cover_t *start) {
    const hig_id n = (hig_id) graph->vertex_count;
    hig_error_t status = HIG_SUCCESS;

    s->gamma = options->resolution;
    s->cap = (hig_id) options->max_memberships;
    s->isolation = options->allow_isolation != 0;
    s->exact = hig_limit_set(options->n_communities);
    s->max_total = options->max_total_communities;
    s->sqrt_cap_up = hig_up(sqrt((double) s->cap));
    s->use_heap = s->gamma < 0.0 || !s->isolation || hig_limit_set(s->max_total) || s->exact;
    s->heap.descending = s->gamma < 0.0;

    HIG_TRY(hig_cover_start(s, options, n, start));
    HIG_TRY(hig_adjacency_build(&s->graph, graph));
    HIG_TRY(hig_alloc(&s->inv_sqrt, (int64_t) s->cap + 1, sizeof(double), false));
    s->inv_sqrt[0] = 0.0;
    for (hig_id k = 1; k <= s->cap; k++) {
        s->inv_sqrt[k] = 1.0 / sqrt((double) k);
    }
    /* Can a term n_u f_u or w f_u fall below DBL_MIN (and lose its relative
     * accuracy)? f_u >= 1 / sqrt(cap) > 1 / (2 sqrt(cap)). */
    {
        const double smallest = 2.0 * DBL_MIN * 2.0 * s->sqrt_cap_up;
        for (hig_id v = 0; s->graph.node_weight && v < n; v++) {
            const double w = s->graph.node_weight[v];
            if (w > 0.0 && w < smallest) {
                s->tiny_mass = (double) n * HIG_ETA;
                break;
            }
        }
        for (int64_t i = 0; s->graph.weight && i < s->graph.start[n]; i++) {
            const double w = s->graph.weight[i];
            if (w > 0.0 && w < smallest) {
                s->tiny_support = true;
                break;
            }
        }
    }
    HIG_TRY(hig_alloc(&s->chosen, (int64_t) s->cap + 1, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&s->own_support, (int64_t) s->cap + 1, sizeof(double), false));
    HIG_TRY(hig_cover_reserve_labels(s, (int64_t) s->label_count + n + 16));
    for (hig_id v = 0; v < n; v++) {
        const hig_row_ref_t r = s->row[v];
        for (hig_id i = 0; i < r.size; i++) {
            s->group[s->pool[r.start + i]].members++;
        }
    }
    s->occupied = s->label_count;
    hig_cover_rebuild(s);   /* the masses, theta and the heap */
    HIG_TRY(hig_cover_reserve_candidates(s, 256));
    HIG_TRY(hig_queue_init(&s->queue, n, options->rng, options->interrupt));
done:
    return status;
}
