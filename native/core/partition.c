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

#include "internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    hig_adjacency_t graph;
    double gamma;
    bool isolation, exact;
    hig_int max_total;          /* -1: no limit */
    hig_id *membership;
    hig_group_t *group;         /* n ids */
    hig_tail_t *tail;           /* the masses' low halves and error bounds */
    double theta;               /* >= (err + |lo|) / mass of every community */
    double crowd;               /* 4 total_up (theta + 5u): coarse's |gamma n_v| term */
    bool rebuild;               /* a mass lost too much of itself to cancellation */
    double total_up;            /* an upper bound of every community's mass, with room */
    hig_id *free_ids;           /* empty community ids, a stack */
    hig_id free_count;
    hig_id occupied;
    bool use_heap, heap_ready;
    hig_heap_t heap;
    hig_id *candidate;          /* the communities of v's neighbours */
    double *candidate_weight;   /* w(v, c) for each */
    hig_queue_t queue;
    double slack;               /* the largest gain bound found since the last move */
    const hig_trace_t *trace;
} hig_partition_state_t;

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

/* ---- Masses ----------------------------------------------------------------- */

static void hig_partition_set_theta(hig_partition_state_t *s, double theta) {
    s->theta = theta;
    s->crowd = hig_mul_up(4.0 * s->total_up, hig_add_up(theta, 5.0 * HIG_U));
}

/* Keeps theta above c's error ratio, or asks for a rebuild. */
HIG_ALWAYS_INLINE void hig_partition_track(hig_partition_state_t *s, hig_id c) {
    if (s->tail[c].err != 0.0 || s->tail[c].lo != 0.0) {
        const double error = hig_add_up(s->tail[c].err, fabs(s->tail[c].lo));
        const double hi = s->group[c].mass;
        if (!(error <= HIG_REBUILD_RATIO * hi)) {
            s->rebuild = true;
        } else {
            const double ratio = hig_ratio_up(error, hi);
            if (ratio > s->theta) {
                hig_partition_set_theta(s, ratio);
            }
        }
    }
}

/* Every mass summed afresh from the members (and the heap reordered). */
static void hig_partition_rebuild(hig_partition_state_t *s) {
    const hig_id n = s->graph.n;

    for (hig_id c = 0; c < n; c++) {
        s->group[c].mass = s->tail[c].lo = s->tail[c].err = 0.0;
    }
    for (hig_id v = 0; v < n; v++) {
        const hig_id c = s->membership[v];
        hig_mass_add(&s->group[c].mass, &s->tail[c].lo, &s->tail[c].err,
                     s->graph.node_weight ? s->graph.node_weight[v] : 1.0);
    }
    hig_partition_set_theta(s, 0.0);
    s->rebuild = false;
    for (hig_id c = 0; c < n; c++) {
        if (s->group[c].members > 0) {
            hig_partition_track(s, c);
        }
    }
    s->rebuild = false;   /* a fresh sum is as good as it gets */
    if (s->heap_ready) {
        hig_heap_clear(&s->heap);
        for (hig_id c = 0; c < n; c++) {
            if (s->group[c].members > 0) {
                hig_heap_insert(&s->heap, s->group, c);
            }
        }
    }
}

HIG_ALWAYS_INLINE void hig_partition_leave(hig_partition_state_t *s, hig_id c, double weight) {
    hig_group_t *group = &s->group[c];

    if (--group->members == 0) {
        group->mass = 0.0;
        s->tail[c].lo = s->tail[c].err = 0.0;
        s->free_ids[s->free_count++] = c;
        s->occupied--;
        if (s->heap_ready) {
            hig_heap_remove(&s->heap, s->group, c);
        }
    } else {
        hig_mass_add(&group->mass, &s->tail[c].lo, &s->tail[c].err, -weight);
        hig_partition_track(s, c);
        if (s->heap_ready) {
            hig_heap_update(&s->heap, s->group, c);
        }
    }
}

HIG_ALWAYS_INLINE void hig_partition_join(hig_partition_state_t *s, hig_id c, double weight) {
    hig_group_t *group = &s->group[c];

    if (group->members++ == 0) {
        group->mass = weight;
        s->tail[c].lo = s->tail[c].err = 0.0;
        s->occupied++;
        if (s->heap_ready) {
            hig_heap_insert(&s->heap, s->group, c);
        }
    } else {
        hig_mass_add(&group->mass, &s->tail[c].lo, &s->tail[c].err, weight);
        hig_partition_track(s, c);
        if (s->heap_ready) {
            hig_heap_update(&s->heap, s->group, c);
        }
    }
}

/* ---- Payoff bounds ---------------------------------------------------------- */

/* Most visits never need a community without edge support. Until the first
 * walk, its heap can be absent: a heap built from the current masses has
 * exactly the same mass/id order as one maintained since initialization. */
static hig_error_t hig_partition_prepare_heap(hig_partition_state_t *s) {
    hig_error_t status = hig_heap_reserve(&s->heap, s->graph.n);
    if (status != HIG_SUCCESS) {
        return status;
    }
    for (hig_id c = 0; c < s->graph.n; c++) {
        if (s->group[c].members > 0) {
            hig_heap_insert(&s->heap, s->group, c);
        }
    }
    s->heap_ready = true;
    return HIG_SUCCESS;
}

/* A bound of |exact payoff - r|, r = fl(w - gn m) the computed payoff of an
 * action: w the computed edge weight to it (error <= ew), m the computed mass
 * of its members other than v (error <= em), gn = fl(gamma n_v) (error <=
 * eg). When gn m is not tiny, the exact value of w - gn m is known as a sum
 * of doubles, so a payoff computed exactly gets the bound 0. */
static inline double hig_partition_bound(double r, double w, double gn, double m, double ew,
                                         double em, double eg) {
    double rounding = hig_add_up(hig_mul_up(HIG_U, fabs(r)), HIG_ETA);
    const double p = gn * m;

    if (fabs(p) >= HIG_EXACT_PRODUCT || gn == 0.0 || m == 0.0) {
        const double pe = hig_fma(gn, m, -p);   /* gn m = p + pe exactly */
        double sum, te, refined;
        hig_two_sum(w, -p, &sum, &te);      /* w - p = sum + te exactly */
        refined = hig_add_up(hig_add_up(hig_up(fabs(sum - r)), fabs(te)), fabs(pe));
        if (refined < rounding) {
            rounding = refined;
        }
    }
    return hig_add_up(hig_add_up(rounding, ew),
                      hig_add_up(hig_mul_up(fabs(gn), em), hig_mul_up(eg, hig_add_up(fabs(m), em))));
}

/* ---- A visit ---------------------------------------------------------------- */

/* What a visit found, for the rare paths that bound its payoffs. */
typedef struct {
    hig_id v, c0, walked, count;    /* count: the candidates, still in s->candidate */
    bool alone, isolation, walk;
    int64_t begin, end;             /* v's neighbours */
    double nv, gn, eg, ew;          /* v's weight, gamma n_v, the errors of gamma n_v and of each w(v, c) */
    double stay, own_weight;        /* the payoff of staying, w(v, c0 \ v) */
    double walked_gain;
    double stay_bound;              /* < 0: not computed yet */
} hig_partition_visit_t;

/* Recomputes what the visit of v computed, with the same expressions in the
 * same order (so the same values): the rare paths get it from here rather
 * than the visit keeping it all live. */
static void hig_partition_recall(const hig_partition_state_t *s, hig_id v, hig_id count,
                                 hig_id walked, bool walk, hig_partition_visit_t *x) {
    const hig_group_t *group = s->group;
    double abs_weight = 0.0;

    x->v = v;
    x->c0 = s->membership[v];
    x->walked = walked;
    x->count = count;
    x->alone = group[x->c0].members == 1;
    x->walk = walk;
    x->begin = s->graph.start[v];
    x->end = s->graph.start[v + 1];
    x->nv = s->graph.node_weight ? s->graph.node_weight[v] : 1.0;
    x->gn = s->gamma * x->nv;
    x->own_weight = 0.0;
    for (hig_id k = 0; k < count; k++) {
        if (s->candidate[k] == x->c0) {
            x->own_weight = s->candidate_weight[k];
        }
    }
    if (s->graph.weight) {
        for (int64_t i = x->begin; i < x->end; i++) {
            abs_weight += fabs(s->graph.weight[i]);
        }
        x->ew = hig_mul_up(hig_gamma((double) (x->end - x->begin) + 2.0), abs_weight);
    } else {
        x->ew = 0.0;
    }
    x->eg = s->graph.node_weight ? hig_product_error(s->gamma, x->nv, x->gn) : 0.0;
    x->stay = hig_fma(-x->gn, x->alone ? 0.0 : (group[x->c0].mass - x->nv) + s->tail[x->c0].lo,
                      x->own_weight);
    x->isolation = s->isolation && !s->exact && !x->alone &&
                   (s->max_total < 0 || s->occupied < s->max_total);
    x->walked_gain = walked >= 0 ? -x->gn * group[walked].mass : 0.0;
    x->stay_bound = -1.0;
}

/* c0's mass without v is (hi - n_v) + lo rounded twice: its error is the
 * two exact residuals and c0's own error. */
static double hig_partition_stay_bound(const hig_partition_state_t *s, hig_partition_visit_t *x) {
    if (x->stay_bound < 0.0) {
        const double hi = x->alone ? 0.0 : s->group[x->c0].mass;
        const double lo = x->alone ? 0.0 : s->tail[x->c0].lo;
        const double err = x->alone ? 0.0 : s->tail[x->c0].err;
        double m1, t1, mass = 0.0, t2, error = 0.0;
        if (hi != 0.0 || lo != 0.0 || err != 0.0) {
            hig_two_sum(hi, -x->nv, &m1, &t1);
            hig_two_sum(m1, lo, &mass, &t2);
            error = hig_add_up(hig_add_up(fabs(t1), fabs(t2)), err);
        }
        x->stay_bound = hig_partition_bound(x->stay, x->own_weight, x->gn, mass, x->ew, error,
                                            x->eg);
    }
    return x->stay_bound;
}

/* The bound of a community's payoff -gn M (+ w): mass error from the
 * community's own record, or, for the community the heap walk found (which
 * stands for every community v has no edge to), theta M. */
static double hig_partition_community_bound(const hig_partition_state_t *s,
                                            const hig_partition_visit_t *x, hig_id c, double gain,
                                            double w, bool proxy) {
    const double hi = s->group[c].mass;
    const double em = proxy ? hig_mul_up(s->theta, hi)
                            : hig_add_up(s->tail[c].err, fabs(s->tail[c].lo));
    return hig_partition_bound(gain, w, x->gn, hi, proxy ? 0.0 : x->ew, em, x->eg);
}

/* Moves v from c0 to best (-1: an empty community) and requeues its
 * neighbours. Returns the community v joined. */
HIG_ALWAYS_INLINE hig_id hig_partition_move(hig_partition_state_t *s, hig_id v, hig_id c0,
                                           hig_id best, double nv, int64_t begin, int64_t end) {
    if (best < 0) {
        best = s->free_ids[--s->free_count];   /* c0 keeps v's company: never empty */
    }
    hig_partition_leave(s, c0, nv);
    hig_partition_join(s, best, nv);
    s->membership[v] = best;
    hig_queue_moved(&s->queue, v);
    s->slack = 0.0;
    if (s->rebuild) {
        hig_partition_rebuild(s);
    }
    for (int64_t i = begin; i < end; i++) {
        const hig_id u = s->graph.adj[i];
        if (s->membership[u] != best) {   /* native/README.md, "Requeueing" */
            hig_queue_push(&s->queue, u);
        }
    }
    return best;
}

/* The visit's best action beats staying: moves v to best (-1: an empty
 * community; alt: its index among the candidates) when the improvement is
 * certain. `certain`: the coarse bound already shows it (the visit's
 * comment); otherwise, and for a trace, the improvement gets its own bound. */
static HIG_NOINLINE bool hig_partition_try_move(hig_partition_state_t *s, hig_id v, hig_id count,
                                                hig_id walked, bool walk, hig_id best, hig_id alt,
                                                bool certain) {
    hig_partition_visit_t x;
    double d = 0.0, e = 0.0;

    hig_partition_recall(s, v, count, walked, walk, &x);
    if (!certain || s->trace) {
        double bound, best_gain;
        if (best < 0) {
            best_gain = 0.0;
            bound = 0.0;
        } else if (best == walked) {
            best_gain = x.walked_gain;
            bound = hig_partition_community_bound(s, &x, best, best_gain, 0.0, true);
        } else {
            best_gain = hig_fma(-x.gn, s->group[best].mass, s->candidate_weight[alt]);
            bound = hig_partition_community_bound(s, &x, best, best_gain,
                                                  s->candidate_weight[alt], false);
        }
        hig_difference(best_gain, x.stay, hig_add_up(bound, hig_partition_stay_bound(s, &x)),
                       &d, &e);
        if (!certain && !(d > e)) {
            return false;
        }
    }
    best = hig_partition_move(s, v, x.c0, best, x.nv, x.begin, x.end);
    if (s->trace) {
        const int32_t label = best;
        if (s->trace->move) {
            s->trace->move(s->trace->state, v, d, e);
        }
        if (s->trace->row) {
            s->trace->row(s->trace->state, v, &label, 1);
        }
    }
    return true;
}

/* v stays: raises the slack to the largest upper bound of the exact gain of
 * a deviation that the coarse bound did not discard (payoff above cutoff).
 * The deviations: every community of a neighbour (scanned when `near`: one
 * is above cutoff), an empty community when allowed, and the communities v
 * has no edge to, all dominated by the one the walk found or, when it did
 * not walk (gamma n_v >= 0), by payoff 0. */
static HIG_NOINLINE void hig_partition_raise_slack(hig_partition_state_t *s, hig_id v,
                                                   hig_id count, hig_id walked, bool walk,
                                                   double cutoff, bool near) {
    hig_partition_visit_t visit, *x = &visit;
    double slack = 0.0, d, e, gain_up;

    hig_partition_recall(s, v, count, walked, walk, x);

    for (hig_id k = 0; near && k < x->count; k++) {
        const hig_id c = s->candidate[k];
        if (c != x->c0) {
            const double gain = hig_fma(-x->gn, s->group[c].mass, s->candidate_weight[k]);
            if (gain > cutoff) {
                const double bound = hig_partition_community_bound(s, x, c, gain,
                                                                   s->candidate_weight[k], false);
                hig_difference(gain, x->stay, hig_add_up(bound, hig_partition_stay_bound(s, x)),
                               &d, &e);
                gain_up = hig_gain_up(d, e);
                if (gain_up > slack) {
                    slack = gain_up;
                }
            }
        }
    }
    if (x->walked >= 0 && x->walked_gain > cutoff) {
        const double bound = hig_partition_community_bound(s, x, x->walked, x->walked_gain, 0.0,
                                                           true);
        hig_difference(x->walked_gain, x->stay, hig_add_up(bound, hig_partition_stay_bound(s, x)),
                       &d, &e);
        gain_up = hig_gain_up(d, e);
        if (gain_up > slack) {
            slack = gain_up;
        }
    }
    if ((x->isolation || !x->walk) && 0.0 > cutoff) {
        hig_difference(0.0, x->stay, hig_partition_stay_bound(s, x), &d, &e);
        gain_up = hig_gain_up(d, e);
        if (gain_up > slack) {
            slack = gain_up;
        }
    }
    if (slack > s->slack) {
        s->slack = slack;
    }
}

/* `weighted` is a constant at each call, so the compiler builds a unit-weight
 * and a weighted version of the scan. */
HIG_ALWAYS_INLINE hig_error_t hig_partition_visit(hig_partition_state_t *s, hig_id v,
                                              const bool weighted) {
    hig_group_t *group = s->group;
    const hig_id c0 = s->membership[v];
    const bool alone = group[c0].members == 1;
    const int64_t begin = s->graph.start[v], end = s->graph.start[v + 1];
    const double nv = s->graph.node_weight ? s->graph.node_weight[v] : 1.0;
    const double gn = s->gamma * nv;
    hig_id count = 0, best = c0, alt = -1, walked = -1;
    double abs_weight = 0.0, own_weight = 0.0, stay, best_gain, other_gain = -INFINITY;
    double walked_gain = 0.0, eg, ew, coarse, cutoff;
    bool isolation, walk;
    hig_error_t status = HIG_SUCCESS;

    if (alone && s->exact) {
        /* Leaving would empty c0 and break the exact count: no deviation. */
        hig_queue_settled(&s->queue, v);
        return HIG_SUCCESS;
    }
    for (int64_t i = begin; i < end; i++) {
        const hig_id c = s->membership[s->graph.adj[i]];
        /* Own support needs no sparse lookup. Keep the same unit additions
         * in CSR order; 0 + 1 is exactly the old first assignment of 1. */
        if (!weighted && c == c0) {
            own_weight += 1.0;
            continue;
        }
        const double w = weighted ? s->graph.weight[i] : 1.0;
        const hig_id slot = group[c].slot;
        if (weighted) {
            abs_weight += fabs(w);
        }
        if (slot < 0) {
            group[c].slot = count;
            s->candidate[count] = c;
            s->candidate_weight[count] = w;
            count++;
        } else {
            s->candidate_weight[slot] += w;
        }
    }
    if (!weighted) {
        abs_weight = (double) (end - begin);
        if (own_weight > 0.0) {
            /* The rare bound paths recall it from the candidate array.
             * c0 is skipped by scoring and the heap walk, so appending it
             * preserves the encounter order and ties of every alternative. */
            s->candidate[count] = c0;
            s->candidate_weight[count++] = own_weight;
        }
    }
    if (weighted && group[c0].slot >= 0) {
        own_weight = s->candidate_weight[group[c0].slot];
    }
    stay = hig_fma(-gn, alone ? 0.0 : (group[c0].mass - nv) + s->tail[c0].lo, own_weight);
    /* The best action so far starts at staying, so the branch is rarely
     * taken; other_gain, the best payoff of another candidate, is a select. */
    best_gain = stay;
    for (hig_id k = 0; k < count; k++) {
        const hig_id c = s->candidate[k];
        if (c != c0) {
            const double gain = hig_fma(-gn, group[c].mass, s->candidate_weight[k]);
            other_gain = gain > other_gain ? gain : other_gain;
            if (gain > best_gain) {
                best = c;
                alt = k;
                best_gain = gain;
            }
        }
    }
    isolation = s->isolation && !s->exact && !alone &&
                (s->max_total < 0 || s->occupied < s->max_total);
    if (isolation && best_gain < 0.0) {
        best = -1;   /* an empty community: payoff 0 */
        best_gain = 0.0;
    }
    walk = s->use_heap && (gn < 0.0 || best_gain < 0.0);
    if (walk) {
        if (!s->heap_ready) {
            status = hig_partition_prepare_heap(s);
            if (status != HIG_SUCCESS) {
                goto reset;
            }
        }
        hig_heap_walk_begin(&s->heap);
        for (;;) {
            hig_id c;
            status = hig_heap_walk_next(&s->heap, group, &c);
            if (status != HIG_SUCCESS || c < 0) {
                break;
            }
            if (c != c0 && group[c].slot < 0) {
                walked = c;
                walked_gain = -gn * group[c].mass;
                if (walked_gain > best_gain) {
                    best = c;
                    best_gain = walked_gain;
                }
                break;
            }
        }
    }
reset:
    for (hig_id k = 0; k < count; k++) {
        group[s->candidate[k]].slot = -1;
    }
    if (status != HIG_SUCCESS) {
        return status;
    }

    /* The errors of gamma n_v and of each w(v, c) (internal.h). */
    eg = s->graph.node_weight ? hig_product_error(s->gamma, nv, gn) : 0.0;
    ew = weighted ? hig_mul_up(hig_gamma((double) (end - begin) + 2.0), abs_weight) : 0.0;
    /* coarse is at least four times a bound of every payoff's error here
     * (u |payoff| + eta + e_w + |gamma n_v| theta M + e_{gamma n_v} M, M <=
     * total_up; staying adds two roundings of c0's mass, 2 u M), so two
     * payoffs whose computed values differ by more than 3 coarse (fl(stay +-
     * 3 coarse) is within coarse / 4 of it, since coarse >= 4 u |stay|)
     * certainly compare the same way exactly: a move by more is certain, a
     * deviation by less than -3 coarse certainly loses. */
    coarse = 4.0 * (HIG_U * abs_weight + HIG_ETA + ew + eg * s->total_up) + fabs(gn) * s->crowd;
    if (best != c0) {
        const bool certain = best_gain > stay + 3.0 * coarse;
        if (certain && !s->trace) {
            hig_partition_move(s, v, c0, best, nv, begin, end);
            return HIG_SUCCESS;
        } else if (hig_partition_try_move(s, v, count, walked, walk, best, alt, certain)) {
            return HIG_SUCCESS;
        }
    }

    /* v stays. The deviations the coarse bound does not discard get their
     * own bounds. */
    cutoff = stay - 3.0 * coarse;
    /* Payoff 0 against staying alone (stay = 0, both exact) adds nothing
     * unless the weights' error bound e_w is positive: skipped. */
    if (other_gain > cutoff || (walked >= 0 && walked_gain > cutoff) ||
        ((isolation || !walk) && 0.0 > cutoff && !(alone && ew == 0.0))) {
        hig_partition_raise_slack(s, v, count, walked, walk, cutoff, other_gain > cutoff);
    }
    hig_queue_settled(&s->queue, v);
    return HIG_SUCCESS;
}

static hig_error_t hig_partition_run(hig_partition_state_t *s) {
    hig_error_t status = HIG_SUCCESS;

    for (;;) {
        hig_id v;
        bool done;
        HIG_TRY(hig_queue_next(&s->queue, &v, &done));
        if (done) {
            break;
        }
        HIG_TRY(s->graph.weight ? hig_partition_visit(s, v, true)
                                : hig_partition_visit(s, v, false));
    }
done:
    return status;
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
