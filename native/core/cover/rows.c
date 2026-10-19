/*
   hedonic -- the labels and the rows of a cover.
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

HIG_SHARED hig_id hig_row_capacity(const hig_cover_state_t *s, hig_id size) {
    hig_id capacity = 2;
    while (capacity < size) {
        capacity *= 2;
    }
    return capacity < s->cap ? capacity : s->cap;
}

/* ---- Labels ----------------------------------------------------------------- */

HIG_SHARED hig_error_t hig_cover_reserve_labels(hig_cover_state_t *s, int64_t ids) {
    hig_error_t status = HIG_SUCCESS;
    int64_t capacity = s->label_capacity, tail_capacity = s->label_capacity;

    if (ids <= s->label_capacity) {
        return HIG_SUCCESS;
    }
    HIG_TRY(hig_grow(&s->group, &capacity, ids, sizeof(hig_group_t)));
    HIG_TRY(hig_grow(&s->tail, &tail_capacity, capacity, sizeof(hig_tail_t)));
    for (int64_t c = s->label_capacity; c < capacity; c++) {
        s->group[c] = (hig_group_t) { 0.0, 0, -1 };
        s->tail[c].lo = s->tail[c].err = 0.0;
    }
    if (s->heap_ready) {
        HIG_TRY(hig_heap_reserve(&s->heap, capacity));
    }
    s->label_capacity = capacity;
done:
    return status;
}

HIG_SHARED hig_error_t hig_cover_fresh_label(hig_cover_state_t *s, hig_id *label) {
    hig_error_t status = HIG_SUCCESS;

    if (s->free_count > 0) {
        *label = s->free_ids[--s->free_count];
        return HIG_SUCCESS;
    }
    if (s->label_count >= HIG_ID_MAX - 1) {
        return hig_error(HIG_EOVERFLOW, "The overlapping game ran out of label ids.");
    }
    HIG_TRY(hig_cover_reserve_labels(s, (int64_t) s->label_count + 1));
    *label = s->label_count++;
done:
    return status;
}

/* Keeps theta above c's error ratio, or asks for a rebuild. The division is
 * skipped when error <= fl(theta_below hi), a normal double: then
 * hig_ratio_up(error, hi) <= theta (1 - 16u)(1 + u)^4 (1 + 4u) + eta <=
 * theta, and error < 16u hi, so neither theta nor the rebuild flag would
 * change. */
static inline void hig_cover_track(hig_cover_state_t *s, hig_id c) {
    const double error = hig_add_up(s->tail[c].err, fabs(s->tail[c].lo));

    if (error != 0.0) {
        const double hi = s->group[c].mass;
        const double below = s->theta_below * hi;
        if (below >= DBL_MIN && error <= below) {
            return;
        }
        if (!(error <= HIG_REBUILD_RATIO * hi)) {
            s->rebuild = true;
        } else {
            const double ratio = hig_ratio_up(error, hi);
            if (ratio > s->theta) {
                s->theta = ratio;
                s->theta_below = ratio >= DBL_MIN ? ratio * (1.0 - 16.0 * HIG_U) : 0.0;
            }
        }
    }
}

/* Every mass summed afresh from the rows (and the heap reordered). */
HIG_SHARED void hig_cover_rebuild(hig_cover_state_t *s) {
    for (hig_id c = 0; c < s->label_count; c++) {
        s->group[c].mass = s->tail[c].lo = s->tail[c].err = 0.0;
    }
    for (hig_id v = 0; v < s->graph.n; v++) {
        const hig_row_ref_t r = s->row[v];
        const double term = (s->graph.node_weight ? s->graph.node_weight[v] : 1.0) *
                            s->inv_sqrt[r.size];
        for (hig_id i = 0; i < r.size; i++) {
            const hig_id c = s->pool[r.start + i];
            hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err, term);
        }
    }
    s->theta = s->theta_below = 0.0;
    for (hig_id c = 0; c < s->label_count; c++) {
        if (s->group[c].members > 0) {
            hig_cover_track(s, c);
        }
    }
    s->rebuild = false;   /* a fresh sum is as good as it gets */
    if (s->heap_ready) {
        hig_heap_clear(&s->heap);
        for (hig_id c = 0; c < s->label_count; c++) {
            if (s->group[c].members > 0) {
                hig_heap_insert(&s->heap, s->group, c);
            }
        }
    }
}

HIG_SHARED hig_error_t hig_cover_leave(hig_cover_state_t *s, hig_id c, double term) {
    hig_group_t *group = &s->group[c];
    hig_error_t status = HIG_SUCCESS;

    if (--group->members == 0) {
        group->mass = 0.0;
        s->tail[c].lo = s->tail[c].err = 0.0;
        HIG_TRY(hig_grow(&s->free_ids, &s->free_capacity, s->free_count + 1, sizeof(hig_id)));
        s->free_ids[s->free_count++] = c;
        s->occupied--;
        if (s->heap_ready) {
            hig_heap_remove(&s->heap, s->group, c);
        }
    } else {
        hig_mass_add(&group->mass, &s->tail[c].lo, &s->tail[c].err, -term);
        hig_cover_track(s, c);
        if (s->heap_ready) {
            hig_heap_update(&s->heap, s->group, c);
        }
    }
done:
    return status;
}

HIG_SHARED void hig_cover_join(hig_cover_state_t *s, hig_id c, double term) {
    hig_group_t *group = &s->group[c];

    if (group->members++ == 0) {
        group->mass = term;
        s->tail[c].lo = s->tail[c].err = 0.0;
        s->occupied++;
        if (s->heap_ready) {
            hig_heap_insert(&s->heap, s->group, c);
        }
    } else {
        hig_mass_add(&group->mass, &s->tail[c].lo, &s->tail[c].err, term);
        hig_cover_track(s, c);
        if (s->heap_ready) {
            hig_heap_update(&s->heap, s->group, c);
        }
    }
}

HIG_SHARED void hig_cover_rescale(hig_cover_state_t *s, hig_id c, double old_term, double new_term) {
    hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err, -old_term);
    hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err, new_term);
    hig_cover_track(s, c);
    if (s->heap_ready) {
        hig_heap_update(&s->heap, s->group, c);
    }
}

/* Build only when a visit first needs labels without edge support. The
 * mass/id ordering is the same as if every preceding update maintained it. */
HIG_SHARED hig_error_t hig_cover_prepare_heap(hig_cover_state_t *s) {
    hig_error_t status = hig_heap_reserve(&s->heap, s->label_capacity);
    if (status != HIG_SUCCESS) {
        return status;
    }
    for (hig_id c = 0; c < s->label_count; c++) {
        if (s->group[c].members > 0) {
            hig_heap_insert(&s->heap, s->group, c);
        }
    }
    s->heap_ready = true;
    return HIG_SUCCESS;
}

/* ---- Rows ------------------------------------------------------------------- */

/* Rewrites the pool with every row at the front of its own capacity. */
static hig_error_t hig_cover_compact_pool(hig_cover_state_t *s) {
    const hig_id n = s->graph.n;
    hig_id *pool;
    int64_t used = 0;
    hig_error_t status = HIG_SUCCESS;

    HIG_TRY(hig_alloc(&pool, s->pool_live + 1024, sizeof(hig_id), false));
    for (hig_id v = 0; v < n; v++) {
        memcpy(pool + used, s->pool + s->row[v].start, (size_t) s->row[v].size * sizeof(hig_id));
        s->row[v].start = used;
        used += s->row[v].capacity;
    }
    free(s->pool);
    s->pool = pool;
    s->pool_size = used;
    s->pool_capacity = s->pool_live + 1024;
done:
    return status;
}

HIG_SHARED hig_error_t hig_cover_write_row(hig_cover_state_t *s, hig_id v, const hig_id *labels,
                                           hig_id size) {
    hig_row_ref_t *r = &s->row[v];
    hig_error_t status = HIG_SUCCESS;

    if (size > r->capacity) {
        const hig_id capacity = hig_row_capacity(s, size);
        HIG_TRY(hig_grow(&s->pool, &s->pool_capacity, s->pool_size + capacity, sizeof(hig_id)));
        r = &s->row[v];
        s->pool_live += capacity - r->capacity;
        r->start = s->pool_size;
        r->capacity = capacity;
        s->pool_size += capacity;
    }
    memcpy(s->pool + r->start, labels, (size_t) size * sizeof(hig_id));
    r->size = size;
    if (s->pool_size > 2 * s->pool_live + 4096) {
        HIG_TRY(hig_cover_compact_pool(s));
    }
done:
    return status;
}
