/*
   hedonic -- the heap of communities by mass.
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

#include <stdlib.h>

/* ---- The heap of communities by mass ---------------------------------------- */

static inline bool hig_heap_before(const hig_heap_t *heap, const hig_group_t *groups, hig_id a,
                                   hig_id b) {
    const double ma = groups[a].mass, mb = groups[b].mass;
    if (ma != mb) {
        return heap->descending ? ma > mb : ma < mb;
    }
    return a < b;
}

static void hig_heap_place(hig_heap_t *heap, int64_t i, hig_id c) {
    heap->heap[i] = c;
    heap->pos[c] = (hig_id) i;
}

static void hig_heap_sift_up(hig_heap_t *heap, const hig_group_t *groups, int64_t i) {
    const hig_id c = heap->heap[i];
    while (i > 0) {
        const int64_t parent = (i - 1) / 2;
        if (!hig_heap_before(heap, groups, c, heap->heap[parent])) {
            break;
        }
        hig_heap_place(heap, i, heap->heap[parent]);
        i = parent;
    }
    hig_heap_place(heap, i, c);
}

static void hig_heap_sift_down(hig_heap_t *heap, const hig_group_t *groups, int64_t i) {
    const hig_id c = heap->heap[i];
    for (;;) {
        int64_t child = 2 * i + 1;
        if (child >= heap->size) {
            break;
        }
        if (child + 1 < heap->size &&
            hig_heap_before(heap, groups, heap->heap[child + 1], heap->heap[child])) {
            child++;
        }
        if (!hig_heap_before(heap, groups, heap->heap[child], c)) {
            break;
        }
        hig_heap_place(heap, i, heap->heap[child]);
        i = child;
    }
    hig_heap_place(heap, i, c);
}

HIG_SHARED hig_error_t hig_heap_reserve(hig_heap_t *heap, int64_t ids) {
    int64_t capacity = heap->capacity;
    hig_error_t status = HIG_SUCCESS;

    if (ids <= capacity) {
        return HIG_SUCCESS;
    }
    HIG_TRY(hig_grow(&heap->pos, &capacity, ids, sizeof(hig_id)));
    for (int64_t c = heap->capacity; c < capacity; c++) {
        heap->pos[c] = -1;
    }
    {
        int64_t heap_capacity = heap->capacity;
        HIG_TRY(hig_grow(&heap->heap, &heap_capacity, capacity, sizeof(hig_id)));
    }
    heap->capacity = capacity;
done:
    return status;
}

HIG_SHARED void hig_heap_free(hig_heap_t *heap) {
    free(heap->heap);
    free(heap->pos);
    free(heap->frontier);
    heap->heap = heap->pos = NULL;
    heap->frontier = NULL;
    heap->size = heap->capacity = heap->frontier_size = heap->frontier_capacity = 0;
}

HIG_SHARED void hig_heap_insert(hig_heap_t *heap, const hig_group_t *groups, hig_id c) {
    hig_heap_place(heap, heap->size++, c);
    hig_heap_sift_up(heap, groups, heap->size - 1);
}

HIG_SHARED void hig_heap_remove(hig_heap_t *heap, const hig_group_t *groups, hig_id c) {
    const int64_t i = heap->pos[c];
    const hig_id last = heap->heap[--heap->size];

    heap->pos[c] = -1;
    if (i < heap->size) {
        hig_heap_place(heap, i, last);
        hig_heap_sift_up(heap, groups, i);
        hig_heap_sift_down(heap, groups, heap->pos[last]);
    }
}

HIG_SHARED void hig_heap_clear(hig_heap_t *heap) {
    for (int64_t i = 0; i < heap->size; i++) {
        heap->pos[heap->heap[i]] = -1;
    }
    heap->size = 0;
}

HIG_SHARED void hig_heap_update(hig_heap_t *heap, const hig_group_t *groups, hig_id c) {
    hig_heap_sift_up(heap, groups, heap->pos[c]);
    hig_heap_sift_down(heap, groups, heap->pos[c]);
}

/* The walk keeps the heap indices it may visit next in a small heap of its
 * own: each step pops the best and offers its two children, so k steps cost
 * O(k log k) whatever the heap's size. */
static inline bool hig_frontier_before(const hig_heap_t *heap, const hig_group_t *groups,
                                       int64_t a, int64_t b) {
    return hig_heap_before(heap, groups, heap->heap[a], heap->heap[b]);
}

static hig_error_t hig_frontier_push(hig_heap_t *heap, const hig_group_t *groups, int64_t index) {
    hig_error_t status = HIG_SUCCESS;
    int64_t i;

    HIG_TRY(hig_grow(&heap->frontier, &heap->frontier_capacity, heap->frontier_size + 1,
                     sizeof(int64_t)));
    i = heap->frontier_size++;
    while (i > 0 && hig_frontier_before(heap, groups, index, heap->frontier[(i - 1) / 2])) {
        heap->frontier[i] = heap->frontier[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    heap->frontier[i] = index;
done:
    return status;
}

static int64_t hig_frontier_pop(hig_heap_t *heap, const hig_group_t *groups) {
    const int64_t top = heap->frontier[0];
    const int64_t last = heap->frontier[--heap->frontier_size];
    int64_t i = 0;

    for (;;) {
        int64_t child = 2 * i + 1;
        if (child >= heap->frontier_size) {
            break;
        }
        if (child + 1 < heap->frontier_size &&
            hig_frontier_before(heap, groups, heap->frontier[child + 1], heap->frontier[child])) {
            child++;
        }
        if (!hig_frontier_before(heap, groups, heap->frontier[child], last)) {
            break;
        }
        heap->frontier[i] = heap->frontier[child];
        i = child;
    }
    if (heap->frontier_size > 0) {
        heap->frontier[i] = last;
    }
    return top;
}

HIG_SHARED void hig_heap_walk_begin(hig_heap_t *heap) {
    heap->frontier_size = -1;   /* the root is offered by the first step */
}

HIG_SHARED hig_error_t hig_heap_walk_next(hig_heap_t *heap, const hig_group_t *groups,
                                          hig_id *c) {
    hig_error_t status = HIG_SUCCESS;
    int64_t i;

    if (heap->frontier_size < 0) {
        heap->frontier_size = 0;
        if (heap->size == 0) {
            *c = -1;
            return HIG_SUCCESS;
        }
        i = 0;
    } else {
        if (heap->frontier_size == 0) {
            *c = -1;
            return HIG_SUCCESS;
        }
        i = hig_frontier_pop(heap, groups);
    }
    if (2 * i + 1 < heap->size) {
        HIG_TRY(hig_frontier_push(heap, groups, 2 * i + 1));
    }
    if (2 * i + 2 < heap->size) {
        HIG_TRY(hig_frontier_push(heap, groups, 2 * i + 2));
    }
    *c = heap->heap[i];
done:
    return status;
}
