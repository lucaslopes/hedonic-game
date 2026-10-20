/* Fixed-cover stability, using the paper's arithmetic, not certified moves.
 * Copyright (C) 2026 Lucas Lopes Felipe. SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Every sum has a specified order; do not use hig_fma or reassociation,
 * reciprocal multiplication for utilities, or a positive-prefix shortcut.
 */
#include "internal.h"
#include <stdlib.h>
#include <string.h>

struct hig_audit_state {
    hig_adjacency_t adj;
    hig_int *offset;
    hig_id *labels, *lightest;
    double *intensity, *mass;
    hig_int label_count, max_row;
    double utility_bound;
};

typedef struct { double mass; hig_id label; } audit_mass_t;
typedef struct { double a, b, score; hig_id label; int own; } audit_entry_t;
typedef struct {
    double *support, *cuts, *sort_buffer, *roots;
    uint8_t *seen;
    hig_id *touched;
    audit_entry_t *entries;
    hig_int *order, *scratch;
    double *values, *margin;  /* windowed search: sorted a or b, per-prefix rounding margins */
    int64_t cut_capacity, sort_capacity;
    hig_int touched_count, count;
    double cur_a, cur_b, norm;
} audit_work_t;

static hig_error_t audit_poll(const hig_interrupt_t *interrupt) {
    if (interrupt && interrupt->check && interrupt->check(interrupt->state)) {
        return hig_error(HIG_INTERRUPTED, "Stability audit interrupted.");
    }
    return HIG_SUCCESS;
}

static int audit_label_compare(const void *a, const void *b) {
    const hig_id x = *(const hig_id *) a, y = *(const hig_id *) b;
    return (x > y) - (x < y);
}
static int audit_mass_compare(const void *a, const void *b) {
    const audit_mass_t *x = a, *y = b;
    if (x->mass != y->mass) return (x->mass > y->mass) - (x->mass < y->mass);
    return (x->label > y->label) - (x->label < y->label);
}

void hig_audit_free(hig_audit_state_t *s) {
    if (!s) return;
    hig_adjacency_free(&s->adj);
    free(s->offset); free(s->labels); free(s->lightest);
    free(s->intensity); free(s->mass); free(s);
}

hig_error_t hig_audit_prepare(const hig_graph_t *g, const hig_cover_t *cover,
                              const hig_interrupt_t *interrupt, hig_audit_state_t **out) {
    hig_audit_state_t *s = NULL;
    audit_mass_t *ranked = NULL;
    hig_error_t status = HIG_SUCCESS;
    hig_int n, m, total, used = 0, largest = -1;
    double weight_sum = 0.0;
    if (!out) return hig_error(HIG_EINVAL, "Missing audit state output.");
    *out = NULL;
    if (!g || !cover || g->vertex_count < 0 || g->edge_count < 0)
        return hig_error(HIG_EINVAL, "Invalid audit graph or cover.");
    n = g->vertex_count; m = g->edge_count;
    if (n > HIG_ID_MAX - 1 || m > INT64_MAX / 2)
        return hig_error(HIG_EOVERFLOW, "Audit graph count overflow.");
    if (cover->vertex_count != n || !cover->offset || cover->offset[0] != 0 ||
        (n && !cover->labels) || (m && !g->edges && !(g->edge_first && g->edge_second)))
        return hig_error(HIG_EINVAL, "Missing or inconsistent audit input arrays.");
    for (hig_int v = 0; v < n; v++) {
        if ((v & 4095) == 0) HIG_TRY(audit_poll(interrupt));
        if (cover->offset[v + 1] <= cover->offset[v])
            HIG_TRY(hig_error(HIG_EINVAL, "Audit rows must be non-empty with increasing offsets."));
        if (g->node_weights && g->node_weights[v] != 1.0)
            HIG_TRY(hig_error(HIG_EINVAL, "Stability audits support unit node weights only."));
    }
    total = cover->offset[n];
    if ((uint64_t) total > SIZE_MAX / sizeof(hig_int))
        HIG_TRY(hig_error(HIG_EOVERFLOW, "Audit incidence count overflow."));
    for (hig_int e = 0; e < m; e++) {
        const hig_int a = hig_edge_end(g, 2 * e), b = hig_edge_end(g, 2 * e + 1);
        const double w = g->edge_weights ? g->edge_weights[e] : 1.0;
        if ((e & 4095) == 0) HIG_TRY(audit_poll(interrupt));
        if (a < 0 || a >= n || b < 0 || b >= n || a == b)
            HIG_TRY(hig_error(HIG_EINVAL, "Audit edges must have distinct ends in the vertex range."));
        if (!isfinite(w) || w < 0.0)
            HIG_TRY(hig_error(HIG_EINVAL, "Audit edge weights must be finite and non-negative."));
        weight_sum += w;
    }
    /* Bound every support/prefix sum, including repeated label contributions. */
    if (!(weight_sum <= (DBL_MAX / 4096.0) / (double) (total > 0 ? total : 1)))
        HIG_TRY(hig_error(HIG_EOVERFLOW, "Audit payoff sums would overflow."));
    HIG_TRY(hig_alloc(&s, 1, sizeof(*s), true));
    HIG_TRY(hig_alloc(&s->offset, n + 1, sizeof(hig_int), true));
    HIG_TRY(hig_alloc(&s->labels, total, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&s->intensity, n, sizeof(double), false));
    s->max_row = 1;
    for (hig_int v = 0; v < n; v++) {
        const hig_int begin = used, length = cover->offset[v + 1] - cover->offset[v];
        HIG_TRY(audit_poll(interrupt));
        for (hig_int j = 0; j < length; j++) {
            const hig_int c = cover->labels[cover->offset[v] + j];
            if (c < 0 || c >= total)
                HIG_TRY(hig_error(HIG_EINVAL, "Audit labels must be contiguous non-negative integers."));
            if (c >= HIG_ID_MAX)
                HIG_TRY(hig_error(HIG_EOVERFLOW, "Too many audit labels."));
            s->labels[begin + j] = (hig_id) c;
            if (c > largest) largest = c;
        }
        qsort(s->labels + begin, (size_t) length, sizeof(hig_id), audit_label_compare);
        for (hig_int j = 0; j < length; j++) {
            const hig_id c = s->labels[begin + j];
            if (j == 0 || c != s->labels[used - 1]) s->labels[used++] = c;
        }
        s->offset[v + 1] = used;
        if (used - begin > s->max_row) s->max_row = used - begin;
        s->intensity[v] = 1.0 / sqrt((double) (used - begin));
    }
    s->utility_bound = weight_sum * (double) (total > 0 ? total : 1);
    if ((double) n * (double) s->max_row > s->utility_bound)
        s->utility_bound = (double) n * (double) s->max_row;
    if (s->utility_bound < 1.0) s->utility_bound = 1.0;
    s->label_count = largest + 1;
    HIG_TRY(hig_alloc(&s->mass, s->label_count, sizeof(double), true));
    HIG_TRY(hig_alloc(&s->lightest, s->label_count, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&ranked, s->label_count, sizeof(*ranked), false));
    for (hig_int v = 0; v < n; v++) {
        if ((v & 4095) == 0) HIG_TRY(audit_poll(interrupt));
        for (hig_int j = s->offset[v]; j < s->offset[v + 1]; j++)
            s->mass[s->labels[j]] += s->intensity[v];
    }
    for (hig_int c = 0; c < s->label_count; c++) {
        if (s->mass[c] == 0.0)
            HIG_TRY(hig_error(HIG_EINVAL, "Audit labels must be contiguous from zero."));
        ranked[c] = (audit_mass_t) { s->mass[c], (hig_id) c };
    }
    qsort(ranked, (size_t) s->label_count, sizeof(*ranked), audit_mass_compare);
    for (hig_int c = 0; c < s->label_count; c++) s->lightest[c] = ranked[c].label;
    HIG_TRY(audit_poll(interrupt));
    HIG_TRY(hig_adjacency_build(&s->adj, g));
    HIG_TRY(audit_poll(interrupt));
    *out = s; s = NULL;
done:
    free(ranked); hig_audit_free(s);
    return status;
}

static void audit_work_free(audit_work_t *w) {
    free(w->support); free(w->seen); free(w->touched); free(w->entries);
    free(w->order); free(w->cuts); free(w->sort_buffer); free(w->roots);
    free(w->scratch); free(w->values); free(w->margin);
}
static hig_error_t audit_work_init(audit_work_t *w, const hig_audit_state_t *s, hig_int cap) {
    hig_error_t status = HIG_SUCCESS;
    const hig_int k = s->label_count;
    const hig_int top = cap < k + 1 ? cap : k + 1;
    HIG_TRY(hig_alloc(&w->support, k, sizeof(double), false));
    HIG_TRY(hig_alloc(&w->seen, k, sizeof(uint8_t), true));
    HIG_TRY(hig_alloc(&w->touched, k, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&w->entries, k + 1, sizeof(audit_entry_t), false));
    HIG_TRY(hig_alloc(&w->order, k + 1, sizeof(hig_int), false));
    HIG_TRY(hig_alloc(&w->roots, top + 1, sizeof(double), false));
    HIG_TRY(hig_alloc(&w->scratch, k + 1, sizeof(hig_int), false));
    HIG_TRY(hig_alloc(&w->values, k + 1, sizeof(double), false));
    HIG_TRY(hig_alloc(&w->margin, top + 1, sizeof(double), false));
    for (hig_int j = 1; j <= top; j++) w->roots[j] = sqrt((double) j);
done:
    return status;
}

static void audit_touch(audit_work_t *w, hig_id c) {
    if (!w->seen[c]) {
        w->seen[c] = 1;
        w->support[c] = 0.0;
        w->touched[w->touched_count++] = c;
    }
}

static hig_error_t audit_vertex(const hig_audit_state_t *s, hig_int v, hig_int cap,
                                const hig_audit_options_t *o, audit_work_t *w) {
    hig_error_t status = HIG_SUCCESS;
    hig_int kept;
    double correction_a = 0.0, correction_b = 0.0;
    for (hig_int j = 0; j < w->touched_count; j++) w->seen[w->touched[j]] = 0;
    w->touched_count = 0;
    for (hig_int j = s->offset[v]; j < s->offset[v + 1]; j++) {
        const hig_id c = s->labels[j];
        audit_touch(w, c); w->seen[c] = 2;
    }
    /* Loopless adjacency is in edge-id order. For a fixed vertex the two
     * contribution directions of the reference cannot interleave in one edge. */
    for (hig_int e = s->adj.start[v]; e < s->adj.start[v + 1]; e++) {
        const hig_id u = s->adj.adj[e];
        const double weight = s->adj.weight ? s->adj.weight[e] : 1.0;
        double term;
        if ((e & 4095) == 0) HIG_TRY(audit_poll(o->interrupt));
        if (weight == 0.0) continue;
        term = weight * s->intensity[u];
        for (hig_int j = s->offset[u]; j < s->offset[u + 1]; j++) {
            const hig_id c = s->labels[j];
            audit_touch(w, c); w->support[c] += term;
        }
    }
    /* seen: own=2, supported=1. Mark candidates with bit 4. Keep touched
     * zero-underflow entries for the unpruned reference's support keys. */
    kept = 0;
    for (hig_int j = 0; j < w->touched_count; j++) {
        const hig_id c = w->touched[j];
        if (!o->prune || w->seen[c] == 2 || w->support[c] > 0.0) {
            w->seen[c] |= 4; kept++;
        }
    }
    if (o->prune) {
        for (hig_int j = 0; j < s->label_count; j++) {
            const hig_id c = s->lightest[j];
            if (!(w->seen[c] & 4)) { audit_touch(w, c); w->seen[c] |= 4; break; }
        }
    } else {
        /* Avoid overflow in 8*cap and cap+kept+1. */
        const int dense = s->label_count <= 256 || cap >= (s->label_count + 7) / 8;
        const hig_int take = dense || cap >= s->label_count - kept - 1
                              ? s->label_count : cap + kept + 1;
        for (hig_int j = 0; j < take; j++) {
            const hig_id c = s->lightest[j];
            audit_touch(w, c); w->seen[c] |= 4;
        }
    }
    qsort(w->touched, (size_t) w->touched_count, sizeof(hig_id), audit_label_compare);
    w->count = 0;
    if (o->allow_isolation) w->entries[w->count++] = (audit_entry_t) {0, 0, 0, -1, 0};
    w->cur_a = w->cur_b = 0.0;
    w->norm = sqrt((double) (s->offset[v + 1] - s->offset[v]));
    for (hig_int j = 0; j < w->touched_count; j++) {
        const hig_id c = w->touched[j];
        const int own = (w->seen[c] & 3) == 2;
        const double a = w->support[c], b = s->mass[c] - (own ? s->intensity[v] : 0.0);
        if (!(w->seen[c] & 4)) continue;
        w->entries[w->count++] = (audit_entry_t) {a, b, 0.0, c, own};
        if (own) {
            /* Python 3.12 sum(), used ONLY by vertex_affine's two current
             * coefficients. State, cell prefixes and regrets use +=.
             * Both inputs are non-negative, finite, and bounded in prepare. */
            double t = w->cur_a + a;
            correction_a += fabs(w->cur_a) >= fabs(a) ? (w->cur_a - t) + a : (a - t) + w->cur_a;
            w->cur_a = t;
            t = w->cur_b + b;
            correction_b += fabs(w->cur_b) >= fabs(b) ? (w->cur_b - t) + b : (b - t) + w->cur_b;
            w->cur_b = t;
        }
    }
    w->cur_a = (w->cur_a + correction_a) / w->norm;
    w->cur_b = (w->cur_b + correction_b) / w->norm;
    for (hig_int j = 0; j < w->count; j++) w->order[j] = j;
done:
    return status;
}

static hig_error_t audit_options(const hig_audit_state_t *s, const hig_audit_options_t *o,
                                 hig_audit_selection_t sel, hig_int *cap) {
    if (!s || !o) return hig_error(HIG_EINVAL, "Missing audit state or options.");
    *cap = o->max_memberships == 0 ? s->max_row : o->max_memberships;
    if (*cap < s->max_row)
        return hig_error(HIG_EINVAL, "max_memberships must be at least the largest audit row.");
    if (!isfinite(o->atol) || !isfinite(o->rtol) || o->atol < 0.0 || o->rtol < 0.0)
        return hig_error(HIG_EINVAL, "Audit tolerances must be finite and non-negative.");
    /* Mass prefix terms are <= n * cap; keep the tolerance coefficients finite. */
    if (o->atol > DBL_MAX / 4096.0 ||
        o->rtol > (DBL_MAX / 4096.0) / s->utility_bound)
        return hig_error(HIG_EOVERFLOW, "Audit tolerance coefficients would overflow.");
    if (sel.begin < 0 || sel.end < sel.begin || (!sel.vertices && sel.end > s->adj.n))
        return hig_error(HIG_EINVAL, "Invalid audit vertex range.");
    if (sel.vertices) {
        for (hig_int j = sel.begin; j < sel.end; j++) {
            if ((j & 4095) == 0) {
                hig_error_t error = audit_poll(o->interrupt);
                if (error != HIG_SUCCESS) return error;
            }
            if (sel.vertices[j] < 0 || sel.vertices[j] >= s->adj.n)
                return hig_error(HIG_EINVAL, "Audit vertex outside the graph.");
        }
    }
    return audit_poll(o->interrupt);
}

/* Stable insertion sort, carried from one cell to the next. */
static hig_error_t audit_order(audit_work_t *w, double gamma, const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;
    uint64_t moves = 0;
    for (hig_int i = 0; i < w->count; i++)
        w->entries[i].score = w->entries[i].a - gamma * w->entries[i].b;
    for (hig_int i = 1; i < w->count; i++) {
        const hig_int x = w->order[i];
        if ((i & 4095) == 0) HIG_TRY(audit_poll(interrupt));
        const double sx = w->entries[x].score;
        hig_int j = i;
        while (j > 0) {
            const hig_int y = w->order[j - 1];
            const double sy = w->entries[y].score;
            if (!(sy < sx || (sy == sx && y > x))) break;
            w->order[j--] = y;
            if ((++moves & 4095) == 0) HIG_TRY(audit_poll(interrupt));
        }
        w->order[j] = x;
    }
done:
    return status;
}

/* A cancellable mergesort for the potentially quadratic breakpoint array. */
static hig_error_t audit_sort_cuts(audit_work_t *w, hig_int n, const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;
    double *a = w->cuts, *b;
    HIG_TRY(hig_grow(&w->sort_buffer, &w->sort_capacity, n, sizeof(double)));
    b = w->sort_buffer;
    for (hig_int width = 1; width < n;) {
        for (hig_int start = 0; start < n;) {
            const hig_int middle = n - start < width ? n : start + width;
            const hig_int end = n - middle < width ? n : middle + width;
            hig_int i = start, j = middle;
            for (hig_int k = start; k < end; k++) {
                if ((k & 4095) == 0) HIG_TRY(audit_poll(interrupt));
                b[k] = i < middle && (j == end || a[i] <= a[j]) ? a[i++] : a[j++];
            }
            start = end;
        }
        { double *tmp = a; a = b; b = tmp; }
        if (width > n / 2) break;
        width *= 2;
    }
    if (a != w->cuts) memcpy(w->cuts, a, (size_t) n * sizeof(double));
done:
    return status;
}

static hig_error_t audit_cuts(audit_work_t *w, const hig_audit_options_t *o, hig_int *count) {
    hig_error_t status = HIG_SUCCESS;
    hig_int used = 2;
    HIG_TRY(hig_grow(&w->cuts, &w->cut_capacity, 4, sizeof(double)));
    w->cuts[0] = 0.0; w->cuts[1] = 1.0;
    for (hig_int i = 0; i < w->count; i++) {
        HIG_TRY(audit_poll(o->interrupt));
        for (hig_int j = i + 1; j < w->count; j++) {
            const double db = w->entries[i].b - w->entries[j].b;
            if ((j & 4095) == 0) HIG_TRY(audit_poll(o->interrupt));
            if (db != 0.0) {
                const double gamma = (w->entries[i].a - w->entries[j].a) / db;
                if (gamma > 0.0 && gamma < 1.0) {
                    HIG_TRY(hig_grow(&w->cuts, &w->cut_capacity, used + 1, sizeof(double)));
                    w->cuts[used++] = gamma;
                }
            }
        }
    }
    HIG_TRY(hig_grow(&w->cuts, &w->cut_capacity, used + 2, sizeof(double)));
    if (w->cur_b != 0.0) {
        for (int level = 0; level < 2; level++) {
            const double gamma = (w->cur_a - (level == 0 ? 1.0 : -1.0)) / w->cur_b;
            if (gamma > 0.0 && gamma < 1.0) w->cuts[used++] = gamma;
        }
    }
    HIG_TRY(audit_sort_cuts(w, used, o->interrupt));
    *count = used;
done:
    return status;
}

void hig_audit_intervals_free(hig_audit_intervals_t *r) {
    if (!r) return;
    free(r->offset); free(r->intervals); memset(r, 0, sizeof(*r));
}

/* ---- Cells ---------------------------------------------------------------
 *
 * One cell [p, q] between consecutive cuts: the reference's prefix
 * inequalities in the score order at its midpoint, and its stable piece.
 * Both the exhaustive and the windowed search run exactly this. */

typedef struct {
    hig_interval_t *intervals;
    int64_t capacity;
    hig_int count, row_begin;
} audit_out_t;

static hig_error_t audit_emit(audit_out_t *r, double lo, double hi) {
    hig_error_t status = HIG_SUCCESS;
    if (r->count > r->row_begin && lo <= r->intervals[r->count - 1].hi + 1e-15) {
        if (hi > r->intervals[r->count - 1].hi) r->intervals[r->count - 1].hi = hi;
    } else {
        if (r->count == INT64_MAX) HIG_TRY(hig_error(HIG_EOVERFLOW, "Too many intervals."));
        HIG_TRY(hig_grow(&r->intervals, &r->capacity, r->count + 1, sizeof(hig_interval_t)));
        r->intervals[r->count++] = (hig_interval_t) {lo, hi};
    }
done:
    return status;
}

/* w->order must hold the score order at the cell's midpoint (audit_order). */
static hig_error_t audit_cell(audit_work_t *w, const hig_audit_options_t *o, hig_int cap,
                              double p, double q, audit_out_t *r) {
    const double mid = 0.5 * (p + q);
    double lo = p, hi = q, sum_a = 0.0, sum_b = 0.0, tau_a, tau_b;
    const double u_mid = w->cur_a - mid * w->cur_b;
    const hig_int top = cap < w->count ? cap : w->count;
    if (fabs(u_mid) <= 1.0) { tau_a = o->atol + o->rtol; tau_b = 0.0; }
    else {
        const double sign = u_mid > 0.0 ? 1.0 : -1.0;
        tau_a = o->atol + o->rtol * sign * w->cur_a;
        tau_b = -o->rtol * sign * w->cur_b;
    }
    for (hig_int k = 1; k <= top; k++) {
        const audit_entry_t *entry = &w->entries[w->order[k - 1]];
        double c0, c1;
        sum_a += entry->a; sum_b += entry->b;
        c0 = sum_a / w->roots[k] - w->cur_a - tau_a;
        c1 = -sum_b / w->roots[k] + w->cur_b - tau_b;
        if (c1 > 0.0) { const double x = -c0 / c1; if (x < hi) hi = x; }
        else if (c1 < 0.0) { const double x = -c0 / c1; if (x > lo) lo = x; }
        else if (c0 > 0.0) return HIG_SUCCESS;
        if (lo > hi) return HIG_SUCCESS;
    }
    return lo > hi ? HIG_SUCCESS : audit_emit(r, lo, hi);
}

/* The order audit_order's carried insertion sort ends in (scores descending,
 * ties by ascending index), by a mergesort: insertion sort from a distant
 * order can be quadratic. */
static int audit_before(const audit_entry_t *e, hig_int x, hig_int y) {
    return e[x].score > e[y].score || (e[x].score == e[y].score && x < y);
}
static hig_error_t audit_sort_order(audit_work_t *w, double gamma, const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;
    hig_int *a = w->order, *b = w->scratch, n = w->count;
    uint64_t steps = 0;
    for (hig_int i = 0; i < n; i++) {
        w->entries[i].score = w->entries[i].a - gamma * w->entries[i].b;
        a[i] = i;
    }
    for (hig_int width = 1; width < n; width *= 2) {
        for (hig_int start = 0; start < n; start += 2 * width) {
            const hig_int middle = n - start < width ? n : start + width;
            const hig_int end = n - middle < width ? n : middle + width;
            hig_int i = start, j = middle;
            for (hig_int k = start; k < end; k++) {
                if ((++steps & 65535) == 0) HIG_TRY(audit_poll(interrupt));
                b[k] = i < middle && (j == end || !audit_before(w->entries, a[j], a[i])) ? a[i++] : a[j++];
            }
        }
        { hig_int *tmp = a; a = b; b = tmp; }
        if (width > n / 2) break;
    }
    if (a != w->order) memcpy(w->order, a, (size_t) n * sizeof(hig_int));
done:
    return status;
}

/* The same order from whatever w->order holds: insertion sort while it
 * stays cheap (a nearby point or cell), else the mergesort. */
static hig_error_t audit_resort(audit_work_t *w, double gamma, const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;
    uint64_t moves = 0, budget = 64;
    for (hig_int n = w->count; n > 1; n /= 2) budget += (uint64_t) w->count;
    for (hig_int i = 0; i < w->count; i++)
        w->entries[i].score = w->entries[i].a - gamma * w->entries[i].b;
    for (hig_int i = 1; i < w->count; i++) {
        const hig_int x = w->order[i];
        hig_int j = i;
        while (j > 0 && audit_before(w->entries, x, w->order[j - 1])) {
            w->order[j] = w->order[j - 1]; j--;
            if (++moves > budget) { w->order[j] = x; return audit_sort_order(w, gamma, interrupt); }
        }
        w->order[j] = x;
        if ((i & 4095) == 0) HIG_TRY(audit_poll(interrupt));
    }
done:
    return status;
}

/* ---- Exhaustive search: every cell of every pairwise crossing ------------ */

static hig_error_t audit_vertex_exhaustive(audit_work_t *w, const hig_audit_options_t *o,
                                           hig_int cap, audit_out_t *r) {
    hig_error_t status = HIG_SUCCESS;
    hig_int cuts = 0;
    HIG_TRY(audit_cuts(w, o, &cuts));
    for (hig_int cell = 0; cell + 1 < cuts; cell++) {
        const double p = w->cuts[cell], q = w->cuts[cell + 1];
        if ((cell & 255) == 0) HIG_TRY(audit_poll(o->interrupt));
        if (!(q > p)) continue;
        HIG_TRY(audit_order(w, 0.5 * (p + q), o->interrupt));
        HIG_TRY(audit_cell(w, o, cap, p, q, r));
    }
done:
    return status;
}

/* ---- Windowed search -------------------------------------------------------
 *
 * Exact arithmetic first. Every candidate set S of size k has an affine
 * utility U_S(x) = (a_S - x b_S) / sqrt(k); the current utility u(x) is
 * affine. The best response best(x) = max_S U_S(x) is convex, and so is
 * best_k(x), the best over sets of size k. Hence:
 *
 *  - Exclusion. For any S, g(x) = best(x) - u(x) >= U_S(x) - u(x), a line.
 *    The reference's tolerance never exceeds T = atol + rtol * max(1, |u(0)|,
 *    |u(1)|). A cell whose stable piece contains x needs g(x) <= T up to the
 *    rounding of its prefix sums and of its midpoint order, and M (below)
 *    is at least four times every such bound. So x where some line exceeds
 *    T + M lies in no piece. Newton steps along the best set's line at the
 *    current endpoint shrink [0, 1] to a window [wl, wr] outside which
 *    every x is excluded; most never-stable vertices end there, with no cut.
 *
 *  - Certification. Inside a range of cells [A, B] whose cells all take one
 *    tolerance line tol(x) (the reference picks it from u at the midpoint;
 *    atol alone is below every choice), best_k(x) - u(x) - tol(x) is convex.
 *    If it is at most -margin_k at A and at B, for every k, it is so on all
 *    of [A, B], and every prefix inequality of every cell there, whatever
 *    its midpoint order, holds with room for its rounding: no inequality
 *    moves lo or hi off the cell, and each cell's piece is the whole cell.
 *    The pieces then merge into [A, B], which is emitted in one step.
 *
 * Everything else, the cells near the ends of the stable set, runs the
 * reference's cell code (audit_cell) on the reference's own cut values: the
 * cuts in the window, plus the nearest cut on either side of it. The cells
 * are those of the exhaustive search, in its order; the skipped ones emit
 * nothing or a whole-cell piece, so the output is the same, bit for bit. */

typedef struct {
    double x;            /* the point (a cut) */
    double alpha, beta;  /* the best set's deviation line: g(y) >= alpha - beta * y */
    double peak;         /* max over prefixes k of utility_k + margin_k, at x */
} audit_point_t;

/* Best-prefix line and certification peak at x (order: scratch use of w->order). */
static hig_error_t audit_point(audit_work_t *w, const hig_audit_options_t *o, hig_int top,
                               double x, audit_point_t *pt) {
    hig_error_t status = HIG_SUCCESS;
    double sum_a = 0.0, sum_b = 0.0, best = -INFINITY, peak = -INFINITY;
    HIG_TRY(audit_resort(w, x, o->interrupt));
    pt->x = x; pt->alpha = pt->beta = 0.0;
    for (hig_int k = 1; k <= top; k++) {
        const audit_entry_t *entry = &w->entries[w->order[k - 1]];
        double utility;
        sum_a += entry->a; sum_b += entry->b;
        utility = (sum_a - x * sum_b) / w->roots[k];
        if (utility > best) { best = utility; pt->alpha = sum_a / w->roots[k]; pt->beta = sum_b / w->roots[k]; }
        if (utility + w->margin[k] > peak) peak = utility + w->margin[k];
    }
    pt->alpha -= w->cur_a; pt->beta -= w->cur_b; pt->peak = peak;
done:
    return status;
}

static int audit_desc(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x < y) - (x > y);
}

typedef struct {
    double limit;      /* T + M: a deviation line above it excludes */
    double atol, rtol, band;
    hig_int cap, top;
} audit_window_t;

/* Margins: margin_k >= 4x the rounding of a prefix-k inequality (sums of k
 * non-negative terms of at most the top-k a and b, sqrt, divisions, current
 * and tolerance terms), and of this file's own evaluation of it. */
static void audit_margins(audit_work_t *w, const hig_audit_options_t *o, audit_window_t *win) {
    double top_a = 0.0, top_b = 0.0, all_a = 0.0, all_b = 0.0, t;
    const double fixed = fabs(w->cur_a) + fabs(w->cur_b) + o->atol + o->rtol * (fabs(w->cur_a) + fabs(w->cur_b));
    for (hig_int i = 0; i < w->count; i++) { w->values[i] = w->entries[i].a; all_a += w->entries[i].a; }
    qsort(w->values, (size_t) w->count, sizeof(double), audit_desc);
    for (hig_int k = 1; k <= win->top; k++) w->margin[k] = (top_a += w->values[k - 1]);
    for (hig_int i = 0; i < w->count; i++) { w->values[i] = w->entries[i].b; all_b += w->entries[i].b; }
    qsort(w->values, (size_t) w->count, sizeof(double), audit_desc);
    for (hig_int k = 1; k <= win->top; k++) {
        top_b += w->values[k - 1];
        w->margin[k] = 8.0 * (double) (k + 8) * DBL_EPSILON * ((w->margin[k] + top_b) / w->roots[k] + fixed);
    }
    t = fabs(w->cur_a) > fabs(w->cur_a - w->cur_b) ? fabs(w->cur_a) : fabs(w->cur_a - w->cur_b);
    t = o->atol + o->rtol * (t > 1.0 ? t : 1.0);
    win->limit = t + 8.0 * (double) (w->count + 8) * DBL_EPSILON * (all_a + all_b + fixed + t);
    /* u at a cell's midpoint is within 4 eps (|cur_a| + |cur_b|) of exact. */
    win->band = 1.0 + 1e-6 + 16.0 * DBL_EPSILON * (fabs(w->cur_a) + fabs(w->cur_b));
}

/* Newton steps from both ends of [0, 1] along best-set lines. */
static hig_error_t audit_window(audit_work_t *w, const hig_audit_options_t *o, audit_window_t *win,
                                double *wl, double *wr, int *empty) {
    hig_error_t status = HIG_SUCCESS;
    audit_point_t pt;
    *wl = 0.0; *wr = 1.0; *empty = 0;
    for (int side = 0; side < 2; side++) {
        for (int step = 0; step < 64; step++) {
            const double x = side == 0 ? *wl : *wr;
            double next;
            HIG_TRY(audit_point(w, o, win->top, x, &pt));
            if (pt.alpha - pt.beta * x <= win->limit) break;
            if (side == 0 ? !(pt.beta > 0.0) : !(pt.beta < 0.0)) { *empty = 1; goto done; }
            next = (pt.alpha - win->limit) / pt.beta;
            if (side == 0 ? !(next > *wl) : !(next < *wr)) break;
            if (side == 0) *wl = next; else *wr = next;
            if (*wl > *wr) { *empty = 1; goto done; }
        }
    }
done:
    return status;
}

/* Every cut of the exhaustive search in [wl, wr], sorted, with the nearest
 * cut below wl in front and the nearest above wr at the end. With a core
 * [A, B] (core_lo <= core_hi), cuts inside it are not kept: only the first
 * and the last, at *core and *core + 1 (*core = -1: no cut in the core). */
static hig_error_t audit_window_cuts(audit_work_t *w, const hig_audit_options_t *o, double wl, double wr,
                                     double core_lo, double core_hi, hig_int *count, hig_int *core) {
    hig_error_t status = HIG_SUCCESS;
    double below = -INFINITY, above = INFINITY, first = INFINITY, last = -INFINITY;
    /* Multipliers for the cheap rejections (see the loop). */
    double floor = 0.0, ceiling = INFINITY, inner_lo = INFINITY, inner_hi = -INFINITY;
    hig_int used = 0, split;
#define AUDIT_KEEP(value) do { const double cut_ = (value); \
        if (cut_ < wl) { if (cut_ > below) { below = cut_; if (below > 0.0) floor = below * (1.0 - 1e-12); } } \
        else if (cut_ > wr) { if (cut_ < above) { above = cut_; ceiling = above * (1.0 + 1e-12); } } \
        else if (cut_ >= core_lo && cut_ <= core_hi) { \
            if (cut_ < first) first = cut_; \
            if (cut_ > last) last = cut_; \
            inner_lo = first * (1.0 + 1e-12); inner_hi = last * (1.0 - 1e-12); \
        } else { HIG_TRY(hig_grow(&w->cuts, &w->cut_capacity, used + 5, sizeof(double))); w->cuts[1 + used++] = cut_; } \
    } while (0)
    HIG_TRY(hig_grow(&w->cuts, &w->cut_capacity, 6, sizeof(double)));
    AUDIT_KEEP(0.0); AUDIT_KEEP(1.0);
    if (w->cur_b != 0.0) {
        for (int level = 0; level < 2; level++) {
            const double gamma = (w->cur_a - (level == 0 ? 1.0 : -1.0)) / w->cur_b;
            if (gamma > 0.0 && gamma < 1.0) AUDIT_KEEP(gamma);
        }
    }
    for (hig_int i = 0; i < w->count; i++) {
        const double ai = w->entries[i].a, bi = w->entries[i].b;
        HIG_TRY(audit_poll(o->interrupt));
        for (hig_int j = i + 1; j < w->count; j++) {
            double da = ai - w->entries[j].a, db = bi - w->entries[j].b;
            if ((j & 4095) == 0) HIG_TRY(audit_poll(o->interrupt));
            /* The exhaustive search's cut is da / db; negating both leaves it
             * unchanged. It lies in (0, 1) only if 0 < da < db (db > 0). A
             * product test that holds with room for its rounding puts it at
             * or below `below`, at or above `above`, or in [first, last]:
             * a repeat or a cut that changes nothing kept here. */
            if (db < 0.0) { da = -da; db = -db; }
            if (!(da > 0.0) || !(da < db) || da < floor * db || da > ceiling * db ||
                (da > inner_lo * db && da < inner_hi * db)) continue;
            {
                const double gamma = da / db;
                if (gamma > 0.0 && gamma < 1.0) AUDIT_KEEP(gamma);
            }
        }
    }
#undef AUDIT_KEEP
    /* Sort the kept cuts in place at w->cuts + 1. */
    {
        double *saved = w->cuts;
        w->cuts = saved + 1;
        status = audit_sort_cuts(w, used, o->interrupt);
        w->cuts = saved;
        HIG_TRY(status);
    }
    *core = -1;
    if (first <= last) {  /* the core's first and last cut, between the kept ones below and above it */
        for (split = 0; split < used && w->cuts[1 + split] < core_lo; split++) {}
        memmove(w->cuts + 3 + split, w->cuts + 1 + split, (size_t) (used - split) * sizeof(double));
        w->cuts[1 + split] = first; w->cuts[2 + split] = last;
        *core = split; used += 2;
    }
    if (below > -INFINITY) { w->cuts[0] = below; used++; if (*core >= 0) ++*core; }
    else memmove(w->cuts, w->cuts + 1, (size_t) used * sizeof(double));
    if (above < INFINITY) w->cuts[used++] = above;
    *count = used;
done:
    return status;
}

/* Cells first .. last - 1 one by one, as the exhaustive search runs them. */
static hig_error_t audit_cells(audit_work_t *w, const hig_audit_options_t *o, audit_window_t *win,
                               hig_int first, hig_int last, audit_out_t *r) {
    hig_error_t status = HIG_SUCCESS;
    for (hig_int cell = first; cell < last; cell++) {
        const double p = w->cuts[cell], q = w->cuts[cell + 1];
        if (!(q > p)) continue;
        HIG_TRY(audit_resort(w, 0.5 * (p + q), o->interrupt));
        HIG_TRY(audit_cell(w, o, win->cap, p, q, r));
    }
done:
    return status;
}

/* Is every cell of [A, B] a whole-cell piece? (Certification, above.) */
static int audit_certified(const audit_work_t *w, const audit_window_t *win,
                           const audit_point_t *a, const audit_point_t *b) {
    const double ua = w->cur_a - a->x * w->cur_b, ub = w->cur_a - b->x * w->cur_b;
    double ta = win->atol, tb = 0.0;  /* tolerance line ta + x * tb */
    if (ua >= win->band && ub >= win->band) {
        ta = win->atol + win->rtol * w->cur_a; tb = -win->rtol * w->cur_b;
    } else if (ua <= -win->band && ub <= -win->band) {
        ta = win->atol - win->rtol * w->cur_a; tb = win->rtol * w->cur_b;
    }
    return a->peak - ua - (ta + a->x * tb) <= 0.0 && b->peak - ub - (ta + b->x * tb) <= 0.0;
}
static int audit_excluded(const audit_window_t *win, const audit_point_t *a, const audit_point_t *b) {
    return (a->alpha - a->beta * a->x > win->limit && a->alpha - a->beta * b->x > win->limit) ||
           (b->alpha - b->beta * a->x > win->limit && b->alpha - b->beta * b->x > win->limit);
}

/* Cells first .. last - 1 (cuts first .. last), left to right. */
static hig_error_t audit_range(audit_work_t *w, const hig_audit_options_t *o, audit_window_t *win,
                               hig_int first, hig_int last, const audit_point_t *a,
                               const audit_point_t *b, audit_out_t *r) {
    hig_error_t status = HIG_SUCCESS;
    audit_point_t middle;
    hig_int half;
    if (last - first <= 8) return audit_cells(w, o, win, first, last, r);
    HIG_TRY(audit_poll(o->interrupt));
    if (audit_excluded(win, a, b)) goto done;
    if (audit_certified(w, win, a, b)) {
        if (w->cuts[last] > w->cuts[first]) HIG_TRY(audit_emit(r, w->cuts[first], w->cuts[last]));
        goto done;
    }
    half = first + (last - first) / 2;
    HIG_TRY(audit_point(w, o, win->top, w->cuts[half], &middle));
    HIG_TRY(audit_range(w, o, win, first, half, a, &middle, r));
    HIG_TRY(audit_range(w, o, win, half, last, &middle, b, r));
done:
    return status;
}

/* A certified core [A, B] of the window, for vertices with many cuts: a
 * certified midpoint, then bisection towards both ends of the window. */
static hig_error_t audit_core(audit_work_t *w, const hig_audit_options_t *o, const audit_window_t *win,
                              double wl, double wr, double *core_lo, double *core_hi) {
    hig_error_t status = HIG_SUCCESS;
    audit_point_t centre, ends[2], probe;
    *core_lo = INFINITY; *core_hi = -INFINITY;
    HIG_TRY(audit_point(w, o, win->top, 0.5 * (wl + wr), &centre));
    if (!audit_certified(w, win, &centre, &centre)) goto done;
    for (int side = 0; side < 2; side++) {
        double bad = side == 0 ? wl : wr;
        ends[side] = centre;
        for (int step = 0; step < 48; step++) {
            const double x = 0.5 * (bad + ends[side].x);
            /* Narrow enough when the band left holds few cuts (at most about
             * count^2 / 2 cross [0, 1]), or at the resolution of the window. */
            if (!(fabs(ends[side].x - bad) > 1e-9 * (wr - wl)) ||
                fabs(ends[side].x - bad) * (double) w->count * (double) w->count <= 256.0 ||
                x == bad || x == ends[side].x) break;
            HIG_TRY(audit_point(w, o, win->top, x, &probe));
            if (side == 0 ? audit_certified(w, win, &probe, &centre) : audit_certified(w, win, &centre, &probe))
                ends[side] = probe;
            else bad = x;
        }
    }
    if (audit_certified(w, win, &ends[0], &ends[1])) { *core_lo = ends[0].x; *core_hi = ends[1].x; }
done:
    return status;
}

static hig_error_t audit_vertex_windowed(audit_work_t *w, const hig_audit_options_t *o,
                                         hig_int cap, audit_out_t *r) {
    hig_error_t status = HIG_SUCCESS;
    audit_window_t win = {0};
    audit_point_t a, b, c, d;
    double wl, wr, core_lo = INFINITY, core_hi = -INFINITY;
    hig_int cuts, core;
    int empty;
    win.atol = o->atol; win.rtol = o->rtol; win.cap = cap;
    win.top = cap < w->count ? cap : w->count;
    audit_margins(w, o, &win);
    HIG_TRY(audit_window(w, o, &win, &wl, &wr, &empty));
    if (empty) goto done;
    if (w->count >= 128) HIG_TRY(audit_core(w, o, &win, wl, wr, &core_lo, &core_hi));
    HIG_TRY(audit_window_cuts(w, o, wl, wr, core_lo, core_hi, &cuts, &core));
    if (cuts < 2) goto done;
    HIG_TRY(audit_point(w, o, win.top, w->cuts[0], &a));
    HIG_TRY(audit_point(w, o, win.top, w->cuts[cuts - 1], &b));
    if (core < 0) {
        HIG_TRY(audit_range(w, o, &win, 0, cuts - 1, &a, &b, r));
        goto done;
    }
    /* Left of the core, the core as one piece (its cells are certified), right of it. */
    HIG_TRY(audit_point(w, o, win.top, w->cuts[core], &c));
    HIG_TRY(audit_point(w, o, win.top, w->cuts[core + 1], &d));
    HIG_TRY(audit_range(w, o, &win, 0, core, &a, &c, r));
    if (w->cuts[core + 1] > w->cuts[core]) HIG_TRY(audit_emit(r, w->cuts[core], w->cuts[core + 1]));
    HIG_TRY(audit_range(w, o, &win, core + 1, cuts - 1, &d, &b, r));
done:
    return status;
}

static hig_error_t audit_intervals(const hig_audit_state_t *s, const hig_audit_options_t *o,
                                   hig_audit_selection_t sel, hig_audit_intervals_t *out, int exhaustive) {
    audit_work_t w = {0};
    audit_out_t r = {0};
    hig_error_t status = HIG_SUCCESS;
    hig_int cap;
    if (!out) return hig_error(HIG_EINVAL, "Missing audit interval output.");
    memset(out, 0, sizeof(*out));
    HIG_TRY(audit_options(s, o, sel, &cap));
    if (sel.end - sel.begin == INT64_MAX)
        HIG_TRY(hig_error(HIG_EOVERFLOW, "Audit result size overflow."));
    out->vertex_count = sel.end - sel.begin;
    HIG_TRY(hig_alloc(&out->offset, out->vertex_count + 1, sizeof(hig_int), true));
    HIG_TRY(audit_work_init(&w, s, cap));
    for (hig_int v = sel.begin; v < sel.end; v++) {
        HIG_TRY(audit_poll(o->interrupt));
        HIG_TRY(audit_vertex(s, sel.vertices ? sel.vertices[v] : v, cap, o, &w));
        r.row_begin = r.count;
        if (exhaustive) HIG_TRY(audit_vertex_exhaustive(&w, o, cap, &r));
        else HIG_TRY(audit_vertex_windowed(&w, o, cap, &r));
        out->offset[v - sel.begin + 1] = r.count;
    }
    out->interval_count = r.count;
    out->intervals = r.intervals; r.intervals = NULL;
done:
    audit_work_free(&w);
    free(r.intervals);
    if (status != HIG_SUCCESS) hig_audit_intervals_free(out);
    return status;
}

hig_error_t hig_audit_intervals(const hig_audit_state_t *s, const hig_audit_options_t *o,
                                hig_audit_selection_t sel, hig_audit_intervals_t *out) {
    return audit_intervals(s, o, sel, out, 0);
}

hig_error_t hig_audit_intervals_exhaustive(const hig_audit_state_t *s, const hig_audit_options_t *o,
                                           hig_audit_selection_t sel, hig_audit_intervals_t *out) {
    return audit_intervals(s, o, sel, out, 1);
}

hig_error_t hig_audit_regrets(const hig_audit_state_t *s, const hig_audit_options_t *o,
                              hig_audit_selection_t sel, const double *gammas, hig_int ng,
                              double *regret, double *tolerance) {
    audit_work_t w = {0};
    hig_error_t status = HIG_SUCCESS;
    hig_int cap;
    HIG_TRY(audit_options(s, o, sel, &cap));
    if (ng < 0 || (ng && !gammas))
        HIG_TRY(hig_error(HIG_EINVAL, "Invalid audit resolution array."));
    if (ng && (uint64_t) (sel.end - sel.begin) > SIZE_MAX / sizeof(double) / (uint64_t) ng)
        HIG_TRY(hig_error(HIG_EOVERFLOW, "Audit regret output size overflow."));
    if (ng && sel.end > sel.begin && (!regret || !tolerance))
        HIG_TRY(hig_error(HIG_EINVAL, "Missing audit regret output arrays."));
    for (hig_int j = 0; j < ng; j++) {
        if (!isfinite(gammas[j]) || gammas[j] < 0.0 || gammas[j] > 1.0)
            HIG_TRY(hig_error(HIG_EINVAL, "Audit resolutions must be finite and in [0, 1]."));
    }
    if (ng == 0) goto done;
    HIG_TRY(audit_work_init(&w, s, cap));
    for (hig_int v = sel.begin; v < sel.end; v++) {
        HIG_TRY(audit_poll(o->interrupt));
        HIG_TRY(audit_vertex(s, sel.vertices ? sel.vertices[v] : v, cap, o, &w));
        for (hig_int j = 0; j < ng; j++) {
            double current = 0.0, prefix = 0.0, best = -INFINITY, diff;
            const hig_int top = cap < w.count ? cap : w.count;
            const hig_int out = (v - sel.begin) * ng + j;
            HIG_TRY(audit_poll(o->interrupt));
            HIG_TRY(audit_order(&w, gammas[j], o->interrupt));
            for (hig_int i = 0; i < w.count; i++)
                if (w.entries[i].own) current += w.entries[i].score;
            current /= w.norm;
            for (hig_int k = 1; k <= top; k++) {
                double utility;
                prefix += w.entries[w.order[k - 1]].score;
                utility = prefix / w.roots[k];
                if (utility > best) best = utility;
            }
            diff = best - current;
            regret[out] = diff > 1e-15 ? diff : 0.0;
            tolerance[out] = o->atol + o->rtol * (fabs(current) > 1.0 ? fabs(current) : 1.0);
        }
    }
done:
    audit_work_free(&w);
    return status;
}
