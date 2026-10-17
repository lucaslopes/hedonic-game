/*
   hedonic -- what the engine uses: errors, inputs, adjacency, the visiting
   queue, the heap of communities by mass, random numbers.
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

HIG_SHARED hig_error_t hig_check_input(const hig_graph_t *graph, hig_options_t *options) {
    const hig_int n = graph->vertex_count, m = graph->edge_count;
    const char *game = "The hedonic game";
    double abs_edge_weight = 0.0, total_node_weight = 0.0;
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
    if (options->n_communities > n) {
        return hig_error(HIG_EINVAL,
                         "n_communities must not exceed the number of vertices for a partition.");
    }
    for (hig_int e = 0; e < m; e++) {
        const double w = graph->edge_weights ? graph->edge_weights[e] : 1.0;
        if (!isfinite(w)) {
            return hig_error(HIG_EINVAL, "%s edge weights must be finite.", game);
        }
        abs_edge_weight += fabs(w);
    }
    if (!(abs_edge_weight <= HIG_ARITHMETIC_BOUND)) {
        return hig_error(HIG_EOVERFLOW, "%s total edge weight would overflow.", game);
    }
    for (hig_int v = 0; v < n; v++) {
        const double w = graph->node_weights ? graph->node_weights[v] : 1.0;
        if (!isfinite(w) || w < 0.0) {
            return hig_error(HIG_EINVAL, "%s node weights must be finite and non-negative.", game);
        }
        total_node_weight += w;
    }
    /* |gamma| N^2 bounds every crowding sum. */
    scale = total_node_weight > 1.0 ? total_node_weight : 1.0;
    cap = HIG_ARITHMETIC_BOUND / scale / scale;
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

/* ---- Random numbers --------------------------------------------------------- */

HIG_SHARED hig_error_t hig_random_below(const hig_rng_t *rng, uint32_t range, uint32_t *value) {
    const uint32_t threshold = (0u - range) % range;
    uint64_t product;

    for (;;) {
        int failed = 0;
        const uint32_t word = rng->next32(rng->state, &failed);
        if (failed) {
            return hig_error(HIG_ECALLBACK, "The random number generator failed.");
        }
        product = (uint64_t) word * range;
        if ((uint32_t) product >= threshold) {
            break;
        }
    }
    *value = (uint32_t) (product >> 32);
    return HIG_SUCCESS;
}

/* PCG32 (XSH-RR, 64-bit state), O'Neill 2014. */
void hig_pcg32_seed(hig_pcg32_t *pcg, uint64_t seed, uint64_t stream) {
    int unused = 0;

    pcg->state = 0;
    pcg->increment = (stream << 1) | 1u;
    hig_pcg32_next32(pcg, &unused);
    pcg->state += seed;
    hig_pcg32_next32(pcg, &unused);
}

uint32_t hig_pcg32_next32(void *state, int *failed) {
    hig_pcg32_t *pcg = (hig_pcg32_t *) state;
    const uint64_t old = pcg->state;
    const uint32_t xorshifted = (uint32_t) (((old >> 18) ^ old) >> 27);
    const uint32_t rotation = (uint32_t) (old >> 59);

    (void) failed;
    pcg->state = old * 6364136223846793005ULL + pcg->increment;
    return (xorshifted >> rotation) | (xorshifted << ((0u - rotation) & 31));
}

/* ---- The fused multiply-add (hedonic_check.h) -------------------------------- */

/* A dispatch target defines these in dispatch.c. */
#if !HIG_DISPATCH
const char *hig_fma_path(void) {
    return HIG_FMA_INSTRUCTION ? "instruction" : "soft";
}

int hig_fma_cpu(void) {
    return HIG_FMA_INSTRUCTION;
}

int hig_fma_select(int instruction) {
    (void) instruction;
    return 0;
}
#endif

/* ---- The visiting queue ----------------------------------------------------- */

HIG_SHARED void hig_queue_free(hig_queue_t *queue) {
    free(queue->ring);
    free(queue->queued);
    free(queue->order);
    free(queue->settled_at);
    queue->ring = queue->order = NULL;
    queue->queued = NULL;
    queue->settled_at = NULL;
}

HIG_SHARED hig_error_t hig_queue_init(hig_queue_t *queue, hig_id n, const hig_rng_t *rng,
                                      const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;

    queue->n = n;
    queue->head = queue->count = queue->moves = queue->visits = 0;
    queue->interrupt = interrupt;
    HIG_TRY(hig_alloc(&queue->ring, n, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&queue->queued, n, sizeof(uint8_t), true));
    HIG_TRY(hig_alloc(&queue->order, n, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&queue->settled_at, n, sizeof(int64_t), false));
    for (hig_id v = 0; v < n; v++) {
        queue->order[v] = v;
        queue->settled_at[v] = -1;
    }
    /* Fisher-Yates. */
    for (hig_id i = n - 1; i > 0; i--) {
        uint32_t j = 0;
        hig_id swap;
        HIG_TRY(hig_random_below(rng, (uint32_t) i + 1, &j));
        swap = queue->order[i];
        queue->order[i] = queue->order[j];
        queue->order[j] = swap;
    }
    for (hig_id i = 0; i < n; i++) {
        hig_queue_push(queue, queue->order[i]);
    }
done:
    return status;
}

HIG_SHARED hig_error_t hig_queue_next(hig_queue_t *queue, hig_id *v, bool *done) {
    if (queue->count == 0) {
        for (hig_id i = 0; i < queue->n; i++) {
            const hig_id u = queue->order[i];
            if (queue->settled_at[u] != queue->moves) {
                hig_queue_push(queue, u);
            }
        }
        if (queue->count == 0) {
            *done = true;
            return HIG_SUCCESS;
        }
    }
    *done = false;
    *v = queue->ring[queue->head];
    queue->head = queue->head + 1 == queue->n ? 0 : queue->head + 1;
    queue->count--;
    queue->queued[*v] = 0;
    if (++queue->visits % HIG_VISITS_PER_INTERRUPTION_CHECK == 0 && queue->interrupt &&
        queue->interrupt->check && queue->interrupt->check(queue->interrupt->state)) {
        return hig_error(HIG_INTERRUPTED, "Interrupted.");
    }
    return HIG_SUCCESS;
}

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
