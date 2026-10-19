/*
   hedonic -- a visit of a cover's vertex: candidates, bounds, the best set;
   the run and the equilibrium slack.
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

/* ---- A visit ---------------------------------------------------------------- */

HIG_SHARED hig_error_t hig_cover_reserve_candidates(hig_cover_state_t *s, int64_t needed) {
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

HIG_SHARED hig_error_t hig_cover_run(hig_cover_state_t *s) {
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
HIG_SHARED hig_error_t hig_cover_slack(hig_cover_state_t *s) {
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
