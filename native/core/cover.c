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

#include "internal.h"

#include <stdlib.h>
#include <string.h>

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

static hig_id hig_row_capacity(const hig_cover_state_t *s, hig_id size) {
    hig_id capacity = 2;
    while (capacity < size) {
        capacity *= 2;
    }
    return capacity < s->cap ? capacity : s->cap;
}

static void hig_sort_ids(hig_id *a, int64_t count);

/* ---- Labels ----------------------------------------------------------------- */

static hig_error_t hig_cover_reserve_labels(hig_cover_state_t *s, int64_t ids) {
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

static hig_error_t hig_cover_fresh_label(hig_cover_state_t *s, hig_id *label) {
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
static void hig_cover_rebuild(hig_cover_state_t *s) {
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

static hig_error_t hig_cover_leave(hig_cover_state_t *s, hig_id c, double term) {
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

static void hig_cover_join(hig_cover_state_t *s, hig_id c, double term) {
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

static void hig_cover_rescale(hig_cover_state_t *s, hig_id c, double old_term, double new_term) {
    hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err, -old_term);
    hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err, new_term);
    hig_cover_track(s, c);
    if (s->heap_ready) {
        hig_heap_update(&s->heap, s->group, c);
    }
}

/* Build only when a visit first needs labels without edge support. The
 * mass/id ordering is the same as if every preceding update maintained it. */
static hig_error_t hig_cover_prepare_heap(hig_cover_state_t *s) {
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

static hig_error_t hig_cover_write_row(hig_cover_state_t *s, hig_id v, const hig_id *labels,
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

/* ---- A visit ---------------------------------------------------------------- */

static hig_error_t hig_cover_reserve_candidates(hig_cover_state_t *s, int64_t needed) {
    hig_error_t status = HIG_SUCCESS;
    int64_t capacity = s->candidate_capacity, value_capacity = s->candidate_capacity;

    if (needed <= s->candidate_capacity) {
        return HIG_SUCCESS;
    }
    HIG_TRY(hig_grow(&s->candidate, &capacity, needed, sizeof(hig_id)));
    HIG_TRY(hig_grow(&s->value, &value_capacity, capacity, sizeof(double)));
    s->candidate_capacity = capacity;
done:
    return status;
}

static inline bool hig_scored_before(double gain_a, hig_id label_a, double gain_b,
                                     hig_id label_b) {
    return gain_a > gain_b || (gain_a == gain_b && label_a < label_b);
}

static int hig_scored_compare(const void *a, const void *b) {
    const hig_scored_t *x = (const hig_scored_t *) a, *y = (const hig_scored_t *) b;
    if (hig_scored_before(x->gain, x->label, y->gain, y->label)) {
        return -1;
    }
    return hig_scored_before(y->gain, y->label, x->gain, x->label) ? 1 : 0;
}

/* The `limit` best positive gains, best first; returns how many. */
static hig_error_t hig_cover_top_positive(hig_cover_state_t *s, int64_t count, hig_id limit,
                                          int64_t *kept_out) {
    hig_error_t status = HIG_SUCCESS;
    int64_t kept = 0;

    if (limit <= 0) {
        *kept_out = 0;
        return HIG_SUCCESS;
    }
    if (limit <= HIG_SHORT_SELECTION) {
        HIG_TRY(hig_grow(&s->top, &s->top_capacity, limit, sizeof(hig_scored_t)));
        for (int64_t i = 0; i < count; i++) {
            const double gain = s->value[i];
            const hig_id label = s->candidate[i];
            int64_t slot;
            if (!(gain > 0.0)) {
                continue;
            }
            if (kept < limit) {
                slot = kept++;
            } else if (hig_scored_before(gain, label, s->top[kept - 1].gain,
                                         s->top[kept - 1].label)) {
                slot = kept - 1;
            } else {
                continue;
            }
            while (slot > 0 && hig_scored_before(gain, label, s->top[slot - 1].gain,
                                                 s->top[slot - 1].label)) {
                s->top[slot] = s->top[slot - 1];
                slot--;
            }
            s->top[slot] = (hig_scored_t) { gain, label, (hig_id) i };
        }
    } else {
        HIG_TRY(hig_grow(&s->top, &s->top_capacity, count, sizeof(hig_scored_t)));
        for (int64_t i = 0; i < count; i++) {
            if (s->value[i] > 0.0) {
                s->top[kept++] = (hig_scored_t) { s->value[i], s->candidate[i], (hig_id) i };
            }
        }
        qsort(s->top, (size_t) kept, sizeof(hig_scored_t), hig_scored_compare);
        if (kept > limit) {
            kept = limit;
        }
    }
    *kept_out = kept;
done:
    return status;
}

/* KAPPA bounds the relative error of a mass term n_u f_u as stored (the
 * rounding of 1 / sqrt(k), of the division and of the product), so a mass
 * that sums terms of weight M errs by at most KAPPA M from the exact one. */
#define HIG_KAPPA (5.0 * HIG_U)

/* A bound of |exact gain - r|, r = fl(s - gn m) the computed gain of a
 * label: s its computed support (error <= es), m its computed mass without v
 * (error <= em), gn = fl(gamma n_v) (error <= eg). */
static inline double hig_cover_bound(double r, double es, double gn, double m, double em,
                                     double eg) {
    return hig_add_up(hig_add_up(hig_add_up(hig_mul_up(HIG_U, fabs(r)), HIG_ETA), es),
                      hig_add_up(hig_mul_up(fabs(gn), em), hig_mul_up(eg, hig_add_up(m, em))));
}

/* >= 1 / sqrt(k) given f = fl(1 / fl(sqrt(k))). */
static inline double hig_inv_sqrt_up(double f) {
    return hig_fma(f, 8.0 * HIG_U, f);
}

/* What a visit computed, for its decision and for the bounds. */
typedef struct {
    hig_id k, mandatory, single, j_best;
    int64_t count, kept;
    bool uses_fresh;
    double gn, old_term, eg, es_coef, es_floor;
    double current, abs_current, best_score, e_pre;
    double max_support, max_mass, min_other, max_other;   /* over the other candidates */
} hig_cover_visit_t;

/* The rigorous bounds of a visit's comparison (native/README.md): `move` for
 * the improvement of the best set over v's row (e_best + e_cur), `gain` for
 * the gain of any set v could hold (e_pre + sqrt(cap) bmax + e_cur). v's own
 * labels are candidates 0..k-1, their supports in s->own_support. */
static HIG_NOINLINE void hig_cover_bounds(const hig_cover_state_t *s, const hig_cover_visit_t *x,
                                          double *move, double *gain) {
    const hig_group_t *group = s->group;
    const hig_id k = x->k;
    double own_bounds = 0.0, own_max = 0.0, bmax, e_cur, e_best;

    for (hig_id i = 0; i < k; i++) {
        const hig_id c = s->candidate[i];
        const hig_group_t *g = &group[c];
        if (g->members != 1) {
            /* c's mass without v: (hi - old_term) + lo, two exact residuals,
             * c's accumulated error, the terms' own error. */
            double m1, t1, mass, t2, em, bound;
            hig_two_sum(g->mass, -x->old_term, &m1, &t1);
            hig_two_sum(m1, s->tail[c].lo, &mass, &t2);
            em = hig_add_up(hig_add_up(fabs(t1), fabs(t2)), s->tail[c].err);
            em = hig_add_up(hig_add_up(em, hig_mul_up(HIG_KAPPA, hig_add_up(fabs(mass), em))),
                            s->tiny_mass);
            bound = hig_cover_bound(s->value[i], hig_add_up(hig_mul_up(x->es_coef,
                                                                       s->own_support[i]),
                                                            x->es_floor),
                                    x->gn, fabs(mass), em, x->eg);
            own_bounds = hig_add_up(own_bounds, bound);
            if (bound > own_max) {
                own_max = bound;
            }
        }
    }
    /* bmax bounds the error of every candidate's gain (the other labels'
     * together, from the largest support, mass and |gain| among them); e_cur
     * that of the payoff of v's row. */
    {
        const double theta_c = hig_add_up(s->theta, hig_mul_up(HIG_KAPPA,
                                                               hig_add_up(1.0, s->theta)));
        const double em = hig_add_up(hig_mul_up(theta_c, x->max_mass), s->tiny_mass);
        const double abs_gain = x->max_other > -x->min_other ? x->max_other : -x->min_other;
        bmax = x->count > k
                   ? hig_cover_bound(abs_gain, hig_add_up(hig_mul_up(x->es_coef, x->max_support),
                                                          x->es_floor),
                                     x->gn, x->max_mass, em, x->eg)
                   : 0.0;
        if (own_max > bmax) {
            bmax = own_max;
        }
        e_cur = hig_add_up(hig_mul_up(hig_inv_sqrt_up(s->inv_sqrt[k]),
                                      hig_add_up(own_bounds, hig_mul_up(hig_gamma((double) k),
                                                                        x->abs_current))),
                           hig_add_up(hig_mul_up(6.0 * HIG_U, fabs(x->current)), 2.0 * HIG_ETA));
    }
    if (x->uses_fresh || (x->single < 0 && x->j_best == 0)) {
        e_best = 0.0;     /* a fresh label, or the mandatory labels alone: exactly 0 */
    } else if (x->single >= 0) {
        e_best = bmax;
    } else {
        e_best = hig_add_up(hig_mul_up(hig_inv_sqrt_up(s->inv_sqrt[x->mandatory + x->j_best]),
                                       hig_mul_up((double) x->j_best, bmax)),
                            x->e_pre);
    }
    *move = hig_add_up(e_best, e_cur);
    /* Every set sigma v could hold is worth at most the best prefix score of
     * the computed gains plus its rounding (e_pre) plus sqrt(|sigma|) bmax
     * (native/README.md, "The best set is a prefix"). */
    *gain = hig_add_up(hig_add_up(x->e_pre, hig_mul_up(s->sqrt_cap_up, bmax)), e_cur);
}

/* A visit of v. With `probe`, v does not move: *probe receives the bound of
 * the exact gain of v's best deviation (what a visit that does not move
 * records, computed in full), and the state is left as it was. */
HIG_ALWAYS_INLINE hig_error_t hig_cover_visit(hig_cover_state_t *s, hig_id v, const bool weighted,
                                             double *probe) {
    const hig_row_ref_t r = s->row[v];
    const hig_id k = r.size;
    const int64_t begin = s->graph.start[v], end = s->graph.start[v + 1];
    const double nv = s->graph.node_weight ? s->graph.node_weight[v] : 1.0;
    const double gn = s->gamma * nv;
    const double old_term = nv * s->inv_sqrt[k];
    const bool fresh_allowed = s->isolation && !s->exact &&
                               (s->max_total < 0 || s->occupied < s->max_total);
    hig_group_t *group = s->group;
    int64_t count = 0, slotted = 0, kept = 0;
    hig_id mandatory = 0, chosen_size = 0, single = -1, j_best = 0;
    double current = 0.0, max_gain = -INFINITY, best_score;
    double min_other = INFINITY, max_other = -INFINITY, max_support = 0.0, max_mass = 0.0;
    double abs_current = 0.0, own_hi = 0.0, own_support = 0.0, bound;
    double eg = 0.0, es_coef = 0.0, es_floor = 0.0, e_pre = 0.0, d, e, coarse;
    bool uses_fresh = false, same;
    hig_cover_visit_t x;
    hig_error_t status = HIG_SUCCESS;

    /* Candidates: v's labels first, then its neighbours' labels. */
    HIG_TRY(hig_cover_reserve_candidates(s, (int64_t) k + 1));
    {
        const hig_id *own = s->pool + r.start;
        for (hig_id i = 0; i < k; i++) {
            group[own[i]].slot = i;
            s->candidate[i] = own[i];
            s->value[i] = 0.0;
        }
    }
    count = slotted = k;
    for (int64_t i = begin; i < end; i++) {
        const hig_row_ref_t ru = s->row[s->graph.adj[i]];
        const double w = weighted ? s->graph.weight[i] : 1.0;
        const double support = w * s->inv_sqrt[ru.size];
        const hig_id *labels = s->pool + ru.start;
        if (count + ru.size > s->candidate_capacity) {
            status = hig_cover_reserve_candidates(s, count + ru.size);
            if (status != HIG_SUCCESS) {
                break;
            }
        }
        for (hig_id j = 0; j < ru.size; j++) {
            const hig_id c = labels[j];
            const hig_id slot = group[c].slot;
            if (slot < 0) {
                group[c].slot = (hig_id) count;
                s->candidate[count] = c;
                s->value[count] = support;
                count++;
            } else {
                s->value[slot] += support;
            }
        }
    }
    slotted = count;
    if (status != HIG_SUCCESS) {
        goto reset;
    }

    /* The error bounds' ingredients: a support sums at most d_v terms w f_u
     * (each off by 4u or less), gn = fl(gamma n_v). */
    eg = s->graph.node_weight ? hig_product_error(s->gamma, nv, gn) : 0.0;
    es_coef = hig_gamma((double) (end - begin) + 8.0);
    es_floor = s->tiny_support ? (double) (end - begin) * HIG_ETA : 0.0;

    /* Gains, and the payoff of v's row. Its labels' supports are kept for
     * their bounds, which only the rare full comparison computes. */
    for (int64_t i = 0; i < k; i++) {
        const hig_id c = s->candidate[i];
        const hig_group_t *g = &group[c];
        double gain;
        s->own_support[i] = s->value[i];
        if (g->members == 1) {
            gain = 0.0;   /* v alone holds it: no support, no crowding (exactly) */
            mandatory += s->exact;
        } else {
            /* c's mass without v (the same value as hig_cover_bounds's). */
            gain = hig_fma(-gn, (g->mass - old_term) + s->tail[c].lo, s->value[i]);
            own_hi = g->mass > own_hi ? g->mass : own_hi;
            own_support = s->value[i] > own_support ? s->value[i] : own_support;
        }
        current += gain;
        abs_current += fabs(gain);
        s->value[i] = gain;
        if (gain > max_gain) {
            max_gain = gain;
        }
    }
    /* The other labels: their bounds are taken together, from the largest
     * support, mass and |gain| among them. */
    for (int64_t i = k; i < count; i++) {
        const double mass = group[s->candidate[i]].mass;
        const double gain = hig_fma(-gn, mass, s->value[i]);
        if (s->value[i] > max_support) {
            max_support = s->value[i];
        }
        if (mass > max_mass) {
            max_mass = mass;
        }
        if (gain < min_other) {
            min_other = gain;
        }
        if (gain > max_other) {
            max_other = gain;
        }
        s->value[i] = gain;
    }
    if (max_other > max_gain) {
        max_gain = max_other;
    }
    current *= s->inv_sqrt[k];
    if (fresh_allowed && max_gain < 0.0) {
        max_gain = 0.0;
    }

    /* Labels v has no support from: the walk's labels stand for all of them
     * (their masses are within theta of the walked ones'). */
    if (s->use_heap && (gn < 0.0 || max_gain < 0.0)) {
        const hig_id want = gn < 0.0 ? s->cap : 1;
        hig_id taken = 0;
        if (!s->heap_ready) {
            status = hig_cover_prepare_heap(s);
            if (status != HIG_SUCCESS) {
                goto reset;
            }
        }
        hig_heap_walk_begin(&s->heap);
        while (taken < want) {
            hig_id c;
            double gain;
            status = hig_heap_walk_next(&s->heap, group, &c);
            if (status != HIG_SUCCESS || c < 0) {
                break;
            }
            if (group[c].slot >= 0) {
                continue;
            }
            status = hig_cover_reserve_candidates(s, count + 1);
            if (status != HIG_SUCCESS) {
                break;
            }
            gain = -gn * group[c].mass;
            s->candidate[count] = c;
            s->value[count] = gain;
            if (group[c].mass > max_mass) {
                max_mass = group[c].mass;
            }
            if (gain < min_other) {
                min_other = gain;
            }
            if (gain > max_other) {
                max_other = gain;
            }
            count++;
            taken++;
        }
    }
reset:
    for (int64_t i = 0; i < slotted; i++) {
        group[s->candidate[i]].slot = -1;
    }
    if (status != HIG_SUCCESS) {
        goto done;
    }

    /* The best set: the mandatory labels and the best prefix of the
     * positive gains, or the single best label. `same`: it is v's row (the
     * mandatory labels are v's, and so is each label with index < k). */
    HIG_TRY(hig_cover_top_positive(s, count, s->cap - mandatory, &kept));
    if (mandatory == 0 && kept == 0) {
        double best_gain = -INFINITY;
        for (int64_t i = 0; i < count; i++) {
            if (single < 0 || hig_scored_before(s->value[i], s->candidate[i], best_gain,
                                                s->candidate[single])) {
                single = (hig_id) i;
                best_gain = s->value[i];
            }
        }
        if (fresh_allowed && (single < 0 || best_gain < 0.0)) {
            uses_fresh = true;
            best_gain = 0.0;
        }
        best_score = best_gain;
        same = !uses_fresh && k == 1 && single == 0;
    } else {
        double prefix = 0.0;
        best_score = 0.0;   /* j = 0: the mandatory labels alone, worth 0 */
        j_best = 0;
        for (int64_t j = 1; j <= kept; j++) {
            const double score = (prefix += s->top[j - 1].gain) * s->inv_sqrt[mandatory + j];
            if ((mandatory == 0 && j == 1) || score > best_score) {
                best_score = score;
                j_best = (hig_id) j;
            }
        }
        if (best_score > 0.0) {
            /* Each computed prefix score is positive, at most best_score, and
             * within (j + 4) roundings of its exact value. */
            e_pre = hig_add_up(hig_mul_up(hig_gamma((double) s->cap + 6.0), best_score),
                               2.0 * HIG_ETA);
        }
        same = mandatory + j_best == k;
        for (hig_id j = 0; same && j < j_best; j++) {
            same = s->top[j].index < k;
        }
    }

    x = (hig_cover_visit_t) { .k = k, .mandatory = mandatory, .single = single,
                              .j_best = j_best, .count = count, .kept = kept,
                              .uses_fresh = uses_fresh, .gn = gn, .old_term = old_term,
                              .eg = eg, .es_coef = es_coef, .es_floor = es_floor,
                              .current = current, .abs_current = abs_current,
                              .best_score = best_score, .e_pre = e_pre,
                              .max_support = max_support, .max_mass = max_mass,
                              .min_other = min_other, .max_other = max_other };
    if (probe) {
        double move, gain;
        hig_cover_bounds(s, &x, &move, &gain);
        hig_difference(best_score, current, gain, &d, &e);
        *probe = hig_gain_up(d, e);
        goto done;
    }

    /* `coarse` bounds both bounds hig_cover_bounds computes, and the
     * rounding of the difference, from the largest |gain| (A), support, mass
     * and mass error over all candidates: every label's gain errs by at most
     * b, v's payoff by sqrt(k) (b + 1.125 u A) + 6u |current| + 2 eta, so both
     * bounds are at most e_pre + sqrt(cap) b + that. An own label's mass
     * without v is at most (1 + theta) hi + n_v f_v (hi the largest own
     * mass), its error (two residuals of u each, and err <= theta hi) at
     * most own_em. Doubling the sum covers the upward roundings of the full
     * bounds and the roundings and underflows of computing it plainly (every
     * term is non-negative). */
    d = best_score - current;
    {
        const double other = max_other > -min_other ? max_other : -min_other;
        const double abs_gain = other > abs_current ? other : abs_current;
        const double own_mass = (1.0 + s->theta) * own_hi + old_term;
        const double own_em = s->theta * own_hi + 3.0 * HIG_U * own_mass;
        const double mass = own_mass > max_mass ? own_mass : max_mass;
        const double support = own_support > max_support ? own_support : max_support;
        const double em = own_em + (s->theta + HIG_KAPPA * (1.0 + s->theta) + 4.0 * HIG_U) *
                                       (mass + own_em) + s->tiny_mass + 2.0 * HIG_ETA;
        const double b = HIG_U * abs_gain + 8.0 * HIG_ETA + es_coef * support + es_floor +
                         fabs(gn) * em + eg * (mass + em);
        const double root_k = s->inv_sqrt[k] * (double) k;
        coarse = 2.0 * (e_pre + s->sqrt_cap_up * b + root_k * (b + 1.125 * HIG_U * abs_gain) +
                        6.0 * HIG_U * fabs(current) + HIG_U * fabs(d) + 32.0 * HIG_ETA);
    }
    /* The bound recorded with a visit that does not move v: the full one
     * (>= 0), or minus a coarse one (< 0) that hig_cover_slack replaces by
     * the full one if it matters. */
    if (same) {
        /* v's row is its best set: no move. */
        const double t = d + coarse;
        bound = t > 0.0 ? -hig_up(t) : 0.0;
        goto settled;
    }
    if (d > coarse && !s->trace) {
        e = 0.0;   /* certainly an improvement: d > coarse >= the full bound */
    } else if (!(d + coarse > 0.0)) {
        bound = 0.0;   /* certainly no improvement, nor any gain */
        goto settled;
    } else {
        double move, gain;
        hig_cover_bounds(s, &x, &move, &gain);
        hig_difference(best_score, current, move, &d, &e);
        if (!(d > e)) {
            hig_difference(best_score, current, gain, &d, &e);
            bound = hig_gain_up(d, e);
            goto settled;
        }
    }

    /* The new row, sorted. */
    if (mandatory > 0) {
        const hig_id *own = s->pool + r.start;
        for (hig_id i = 0; i < k; i++) {
            if (group[own[i]].members == 1) {
                s->chosen[chosen_size++] = own[i];
            }
        }
    }
    if (mandatory > 0 || kept > 0) {
        for (hig_id j = 0; j < j_best; j++) {
            s->chosen[chosen_size++] = s->top[j].label;
        }
    } else if (!uses_fresh) {
        s->chosen[chosen_size++] = s->candidate[single];
    }
    hig_sort_ids(s->chosen, chosen_size);
    if (uses_fresh) {
        hig_id fresh = -1;
        HIG_TRY(hig_cover_fresh_label(s, &fresh));
        s->chosen[chosen_size++] = fresh;   /* the only label: still sorted */
    }

    /* Masses: leave, rescale, join (rows are sorted, merge them). */
    {
        const double new_term = nv * s->inv_sqrt[chosen_size];
        const hig_id *own = s->pool + r.start;
        hig_id a = 0, b = 0;
        while (a < k || b < chosen_size) {
            if (b >= chosen_size || (a < k && own[a] < s->chosen[b])) {
                HIG_TRY(hig_cover_leave(s, own[a++], old_term));
            } else if (a >= k || s->chosen[b] < own[a]) {
                hig_cover_join(s, s->chosen[b++], new_term);
            } else {
                if (chosen_size != k) {
                    hig_cover_rescale(s, own[a], old_term, new_term);
                }
                a++;
                b++;
            }
        }
    }
    HIG_TRY(hig_cover_write_row(s, v, s->chosen, chosen_size));
    hig_queue_moved(&s->queue, v);
    if (s->rebuild) {
        hig_cover_rebuild(s);
    }
    if (s->trace) {
        if (s->trace->move) {
            s->trace->move(s->trace->state, v, d, e);
        }
        if (s->trace->row) {
            s->trace->row(s->trace->state, v, (const int32_t *) s->chosen, chosen_size);
        }
    }
    for (int64_t i = begin; i < end; i++) {
        hig_queue_push(&s->queue, s->graph.adj[i]);
    }
    goto done;

settled:
    hig_queue_settled_bound(&s->queue, v, bound);
done:
    return status;
}

static hig_error_t hig_cover_run(hig_cover_state_t *s) {
    hig_error_t status = HIG_SUCCESS;

    for (;;) {
        hig_id v;
        bool done;
        HIG_TRY(hig_queue_next(&s->queue, &v, &done));
        if (done) {
            break;
        }
        HIG_TRY(s->graph.weight ? hig_cover_visit(s, v, true, NULL)
                                : hig_cover_visit(s, v, false, NULL));
    }
done:
    return status;
}

static int hig_compare_bounds_descending(const void *a, const void *b) {
    const double x = ((const hig_scored_t *) a)->gain, y = ((const hig_scored_t *) b)->gain;
    return x > y ? -1 : x < y ? 1 : 0;
}

/* The slack: the largest gain bound of the last visits, all made in the
 * final state (the run ends only when every vertex was last visited after
 * the last move). Coarse bounds are replaced by full ones, largest first,
 * until the next coarse bound cannot raise the maximum: the same value as
 * recording every full bound. */
static hig_error_t hig_cover_slack(hig_cover_state_t *s) {
    const hig_id n = s->graph.n;
    const hig_settled_t *settled = s->queue.settled;
    hig_scored_t *open = NULL;   /* not s->top: the visits below use it */
    int64_t count = 0;
    hig_error_t status = HIG_SUCCESS;

    s->slack = 0.0;
    for (hig_id v = 0; v < n; v++) {
        if (settled[v].bound > s->slack) {
            s->slack = settled[v].bound;
        }
    }
    for (hig_id v = 0; v < n; v++) {
        count += -settled[v].bound > s->slack;
    }
    if (count == 0) {
        return HIG_SUCCESS;
    }
    HIG_TRY(hig_alloc(&open, count, sizeof(hig_scored_t), false));
    count = 0;
    for (hig_id v = 0; v < n; v++) {
        if (-settled[v].bound > s->slack) {
            open[count++] = (hig_scored_t) { -settled[v].bound, v, 0 };
        }
    }
    qsort(open, (size_t) count, sizeof(hig_scored_t), hig_compare_bounds_descending);
    for (int64_t i = 0; i < count && open[i].gain > s->slack; i++) {
        const hig_id v = open[i].label;
        double full = 0.0;
        HIG_TRY(s->graph.weight ? hig_cover_visit(s, v, true, &full)
                                : hig_cover_visit(s, v, false, &full));
        if (full > s->slack) {
            s->slack = full;
        }
    }
done:
    free(open);
    return status;
}

/* ---- Start and setup -------------------------------------------------------- */

static void hig_sort_ids(hig_id *a, int64_t count) {
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

static hig_error_t hig_cover_init(hig_cover_state_t *s, const hig_graph_t *graph,
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
