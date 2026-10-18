/*
   hedonic -- the independent verifier of a partition or a cover.
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
 * hig_verify_partition and hig_verify_cover (hedonic_check.h) take a result
 * and score every vertex's feasible deviations with no queue, no settlement
 * record, no heap and no accumulator shared with the engines: the masses are
 * summed afresh, each candidate's error is bounded on its own (the engines
 * bound some of them together), and the communities a vertex has no edge to
 * are read off one sorted array. For each deviation the report keeps the
 * certified lower bound of its exact gain (when it is certainly positive) and
 * an upper bound, so a result whose engine claimed slack s is wrong if any
 * certified gain exceeds s.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

#define HIG_KAPPA_CHECK (5.0 * HIG_U)   /* as HIG_KAPPA of cover/visit.c */

/* A lower bound of d - e > 0 (0 otherwise). */
static double hig_certified(double d, double e) {
    const double t = d - e;
    return t > 0.0 ? hig_fma(t, -4.0 * HIG_U, t) : 0.0;
}

static uint32_t hig_unused_next32(void *state, int *failed) {
    (void) state;
    (void) failed;
    return 0;
}

typedef struct {
    double mass;
    hig_id id;
} hig_ranked_t;

static int hig_ranked_ascending(const void *a, const void *b) {
    const hig_ranked_t *x = (const hig_ranked_t *) a, *y = (const hig_ranked_t *) b;
    if (x->mass != y->mass) {
        return x->mass < y->mass ? -1 : 1;
    }
    return (x->id > y->id) - (x->id < y->id);
}

static int hig_ranked_descending(const void *a, const void *b) {
    const hig_ranked_t *x = (const hig_ranked_t *) a, *y = (const hig_ranked_t *) b;
    if (x->mass != y->mass) {
        return x->mass > y->mass ? -1 : 1;
    }
    return (x->id > y->id) - (x->id < y->id);
}

static void hig_note(hig_verify_t *report, double d, double e, bool *improvable) {
    const double lower = hig_certified(d, e), upper = hig_gain_up(d, e);
    if (lower > report->max_certified) {
        report->max_certified = lower;
    }
    if (upper > report->max_bound) {
        report->max_bound = upper;
    }
    if (d > e) {
        *improvable = true;
    }
}

/* ---- Partitions -------------------------------------------------------------- */

/* |exact w - gn m - r| for r = fl(w - gn m): one rounding, inputs' errors. */
static double hig_check_partition_bound(double r, double gn, double m, double ew, double em,
                                        double eg) {
    return hig_add_up(hig_add_up(hig_add_up(hig_mul_up(HIG_U, fabs(r)), HIG_ETA), ew),
                      hig_add_up(hig_mul_up(fabs(gn), em), hig_mul_up(eg, hig_add_up(fabs(m), em))));
}

hig_error_t hig_verify_partition(const hig_graph_t *graph, const hig_options_t *options_in,
                                 const hig_int *membership, hig_verify_t *report) {
    hig_options_t options = *options_in;
    hig_rng_t unused = { hig_unused_next32, NULL };
    hig_adjacency_t adj;
    hig_id *label = NULL, *members = NULL, *touched = NULL;
    double *hi = NULL, *lo = NULL, *err = NULL, *to = NULL, theta = 0.0;
    uint8_t *seen = NULL;
    hig_ranked_t *ranked = NULL;
    int64_t k = 0;
    hig_id n;
    bool exact;
    hig_error_t status = HIG_SUCCESS;

    memset(&adj, 0, sizeof(adj));
    memset(report, 0, sizeof(*report));
    if (!options.rng) {
        options.rng = &unused;
    }
    HIG_TRY(hig_check_input(graph, &options, false));
    n = (hig_id) graph->vertex_count;
    exact = hig_limit_set(options.n_communities);
    HIG_TRY(hig_adjacency_build(&adj, graph));
    for (hig_id v = 0; v < n; v++) {
        if (membership[v] < 0) {
            HIG_TRY(hig_error(HIG_EINVAL, "Membership labels must be non-negative."));
        }
    }
    HIG_TRY(hig_alloc(&label, n, sizeof(hig_id), false));
    HIG_TRY(hig_compact_labels(membership, n, label, &k));
    HIG_TRY(hig_alloc(&members, k, sizeof(hig_id), true));
    HIG_TRY(hig_alloc(&hi, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&lo, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&err, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&to, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&seen, k, sizeof(uint8_t), true));
    HIG_TRY(hig_alloc(&touched, adj.max_degree + 1, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&ranked, k, sizeof(hig_ranked_t), false));
    for (hig_id v = 0; v < n; v++) {
        hig_mass_add(&hi[label[v]], &lo[label[v]], &err[label[v]],
                     adj.node_weight ? adj.node_weight[v] : 1.0);
        members[label[v]]++;
    }
    for (int64_t c = 0; c < k; c++) {
        const double e = hig_add_up(err[c], fabs(lo[c]));
        if (e != 0.0) {
            const double ratio = hi[c] > 0.0 ? hig_ratio_up(e, hi[c]) : INFINITY;
            if (ratio > theta) {
                theta = ratio;
            }
        }
        ranked[c] = (hig_ranked_t) { hi[c], (hig_id) c };
    }
    qsort(ranked, (size_t) k, sizeof(hig_ranked_t),
          options.resolution < 0.0 ? hig_ranked_descending : hig_ranked_ascending);

    for (hig_id v = 0; v < n; v++) {
        const hig_id c0 = label[v];
        const bool alone = members[c0] == 1;
        const double nv = adj.node_weight ? adj.node_weight[v] : 1.0;
        const double gn = options.resolution * nv;
        const double eg = adj.node_weight ? hig_product_error(options.resolution, nv, gn) : 0.0;
        double abs_weight = 0.0, ew, stay, stay_mass = 0.0, stay_error = 0.0, stay_bound, d, e;
        int64_t count = 0;
        bool improvable = false;

        if (alone && exact) {
            continue;
        }
        report->vertices++;
        for (int64_t i = adj.start[v]; i < adj.start[v + 1]; i++) {
            const hig_id c = label[adj.adj[i]];
            const double w = adj.weight ? adj.weight[i] : 1.0;
            abs_weight += fabs(w);
            if (!seen[c]) {
                seen[c] = 1;
                to[c] = 0.0;
                touched[count++] = c;
            }
            to[c] += w;
        }
        ew = adj.weight ? hig_mul_up(hig_gamma((double) (adj.start[v + 1] - adj.start[v]) + 2.0),
                                     abs_weight)
                        : 0.0;
        if (!alone) {
            double m1, t1, t2;
            hig_two_sum(hi[c0], -nv, &m1, &t1);
            hig_two_sum(m1, lo[c0], &stay_mass, &t2);
            stay_error = hig_add_up(hig_add_up(fabs(t1), fabs(t2)), err[c0]);
        }
        stay = hig_fma(-gn, stay_mass, seen[c0] ? to[c0] : 0.0);
        stay_bound = hig_check_partition_bound(stay, gn, stay_mass, ew, stay_error, eg);

        /* Every community of a neighbour. */
        for (int64_t i = 0; i < count; i++) {
            const hig_id c = touched[i];
            if (c != c0) {
                const double r = hig_fma(-gn, hi[c], to[c]);
                const double bound = hig_check_partition_bound(
                    r, gn, hi[c], ew, hig_add_up(err[c], fabs(lo[c])), eg);
                hig_difference(r, stay, hig_add_up(bound, stay_bound), &d, &e);
                hig_note(report, d, e, &improvable);
            }
        }
        /* An empty community. */
        if (options.allow_isolation && !exact && !alone &&
            (!hig_limit_set(options.max_total_communities) ||
             k < options.max_total_communities)) {
            hig_difference(0.0, stay, stay_bound, &d, &e);
            hig_note(report, d, e, &improvable);
        }
        /* The best community v has no edge to stands for all of them. */
        for (int64_t i = 0; i < k; i++) {
            const hig_id c = ranked[i].id;
            if (c != c0 && !seen[c]) {
                const double r = -gn * hi[c];
                const double bound = hig_check_partition_bound(r, gn, hi[c], 0.0,
                                                                hig_mul_up(theta, hi[c]), eg);
                hig_difference(r, stay, hig_add_up(bound, stay_bound), &d, &e);
                hig_note(report, d, e, &improvable);
                break;
            }
        }
        for (int64_t i = 0; i < count; i++) {
            seen[touched[i]] = 0;
        }
        report->improvable += improvable;
    }
done:
    hig_adjacency_free(&adj);
    free(label);
    free(members);
    free(hi);
    free(lo);
    free(err);
    free(to);
    free(seen);
    free(touched);
    free(ranked);
    return status;
}

/* ---- Covers ------------------------------------------------------------------ */

typedef struct {
    double gain, bound;
    hig_id label;
} hig_check_scored_t;

static int hig_check_scored_compare(const void *a, const void *b) {
    const hig_check_scored_t *x = (const hig_check_scored_t *) a,
                             *y = (const hig_check_scored_t *) b;
    if (x->gain != y->gain) {
        return x->gain > y->gain ? -1 : 1;
    }
    return (x->label > y->label) - (x->label < y->label);
}

hig_error_t hig_verify_cover(const hig_graph_t *graph, const hig_options_t *options_in,
                             const hig_cover_t *cover, hig_verify_t *report) {
    hig_options_t options = *options_in;
    hig_rng_t unused = { hig_unused_next32, NULL };
    hig_adjacency_t adj;
    hig_id *label = NULL, *members = NULL, *touched = NULL, n, cap;
    hig_int *flat = NULL;
    int64_t k = 0, total;
    double *hi = NULL, *lo = NULL, *err = NULL, *support = NULL, *inv_sqrt = NULL;
    double theta = 0.0, theta_c, tiny_mass = 0.0, sqrt_cap_up;
    uint8_t *seen = NULL;
    hig_ranked_t *ranked = NULL;
    hig_check_scored_t *scored = NULL;
    int64_t scored_capacity = 0;
    bool exact, tiny_support = false;
    hig_error_t status = HIG_SUCCESS;

    memset(&adj, 0, sizeof(adj));
    memset(report, 0, sizeof(*report));
    if (!options.rng) {
        options.rng = &unused;
    }
    HIG_TRY(hig_check_input(graph, &options, true));
    n = (hig_id) graph->vertex_count;
    cap = (hig_id) options.max_memberships;
    exact = hig_limit_set(options.n_communities);
    if (!cover || cover->vertex_count != n || !cover->offset || !cover->labels) {
        HIG_TRY(hig_error(HIG_EINVAL, "The cover does not match the graph."));
    }
    total = cover->offset[n];
    for (hig_id v = 0; v < n; v++) {
        const hig_int size = cover->offset[v + 1] - cover->offset[v];
        if (size < 1 || size > cap) {
            HIG_TRY(hig_error(HIG_EINVAL, "Row %d has %lld labels.", (int) v, (long long) size));
        }
        for (hig_int i = cover->offset[v]; i < cover->offset[v + 1]; i++) {
            if (cover->labels[i] < 0 ||
                (i > cover->offset[v] && cover->labels[i] <= cover->labels[i - 1])) {
                HIG_TRY(hig_error(HIG_EINVAL, "Row %d is not sorted and distinct.", (int) v));
            }
        }
    }
    HIG_TRY(hig_adjacency_build(&adj, graph));
    HIG_TRY(hig_alloc(&flat, total, sizeof(hig_int), false));
    memcpy(flat, cover->labels, (size_t) total * sizeof(hig_int));
    HIG_TRY(hig_alloc(&label, total, sizeof(hig_id), false));
    HIG_TRY(hig_compact_labels(flat, total, label, &k));
    HIG_TRY(hig_alloc(&members, k, sizeof(hig_id), true));
    HIG_TRY(hig_alloc(&hi, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&lo, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&err, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&support, k, sizeof(double), true));
    HIG_TRY(hig_alloc(&seen, k, sizeof(uint8_t), true));
    HIG_TRY(hig_alloc(&touched, k, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&ranked, k, sizeof(hig_ranked_t), false));
    HIG_TRY(hig_alloc(&inv_sqrt, (int64_t) cap + 1, sizeof(double), false));
    inv_sqrt[0] = 0.0;
    for (hig_id j = 1; j <= cap; j++) {
        inv_sqrt[j] = 1.0 / sqrt((double) j);
    }
    sqrt_cap_up = hig_up(sqrt((double) cap));
    {
        const double smallest = 4.0 * DBL_MIN * sqrt_cap_up;
        for (hig_id v = 0; adj.node_weight && v < n; v++) {
            if (adj.node_weight[v] > 0.0 && adj.node_weight[v] < smallest) {
                tiny_mass = (double) n * HIG_ETA;
            }
        }
        for (int64_t i = 0; adj.weight && i < adj.start[n]; i++) {
            if (adj.weight[i] > 0.0 && adj.weight[i] < smallest) {
                tiny_support = true;
            }
        }
    }
    for (hig_id v = 0; v < n; v++) {
        const hig_int size = cover->offset[v + 1] - cover->offset[v];
        const double term = (adj.node_weight ? adj.node_weight[v] : 1.0) * inv_sqrt[size];
        for (hig_int i = cover->offset[v]; i < cover->offset[v + 1]; i++) {
            const hig_id c = label[i];
            hig_mass_add(&hi[c], &lo[c], &err[c], term);
            members[c]++;
        }
    }
    for (int64_t c = 0; c < k; c++) {
        const double e = hig_add_up(err[c], fabs(lo[c]));
        if (e != 0.0) {
            const double ratio = hi[c] > 0.0 ? hig_ratio_up(e, hi[c]) : INFINITY;
            if (ratio > theta) {
                theta = ratio;
            }
        }
        ranked[c] = (hig_ranked_t) { hi[c], (hig_id) c };
    }
    theta_c = hig_add_up(theta, hig_mul_up(HIG_KAPPA_CHECK, hig_add_up(1.0, theta)));
    qsort(ranked, (size_t) k, sizeof(hig_ranked_t),
          options.resolution < 0.0 ? hig_ranked_descending : hig_ranked_ascending);

    for (hig_id v = 0; v < n; v++) {
        const hig_int row_begin = cover->offset[v], row_end = cover->offset[v + 1];
        const hig_id size = (hig_id) (row_end - row_begin);
        const double nv = adj.node_weight ? adj.node_weight[v] : 1.0;
        const double gn = options.resolution * nv;
        const double eg = adj.node_weight ? hig_product_error(options.resolution, nv, gn) : 0.0;
        const double old_term = nv * inv_sqrt[size];
        const int64_t degree = adj.start[v + 1] - adj.start[v];
        const double es_coef = hig_gamma((double) degree + 8.0);
        const double es_floor = tiny_support ? (double) degree * HIG_ETA : 0.0;
        const bool fresh = options.allow_isolation && !exact &&
                           (!hig_limit_set(options.max_total_communities) ||
                            k < options.max_total_communities);
        int64_t count = 0, used = 0, positive = 0;
        hig_id mandatory = 0;
        double current = 0.0, abs_current = 0.0, own_bounds = 0.0, e_cur, bmax = 0.0;
        double best, e_pre = 0.0, e_best = 0.0, d, e;
        bool improvable = false;

        /* Supports: v's labels first, then its neighbours'. */
        for (hig_int i = row_begin; i < row_end; i++) {
            seen[label[i]] = 2;
            support[label[i]] = 0.0;
            touched[count++] = label[i];
        }
        for (int64_t i = adj.start[v]; i < adj.start[v + 1]; i++) {
            const hig_id u = adj.adj[i];
            const hig_int ku = cover->offset[u + 1] - cover->offset[u];
            const double s = (adj.weight ? adj.weight[i] : 1.0) * inv_sqrt[ku];
            for (hig_int j = cover->offset[u]; j < cover->offset[u + 1]; j++) {
                const hig_id c = label[j];
                if (!seen[c]) {
                    seen[c] = 1;
                    support[c] = 0.0;
                    touched[count++] = c;
                }
                support[c] += s;
            }
        }
        HIG_TRY(hig_grow(&scored, &scored_capacity, count + cap + 1, sizeof(hig_check_scored_t)));

        /* Gains with their bounds. */
        for (int64_t i = 0; i < count; i++) {
            const hig_id c = touched[i];
            const double es = hig_add_up(hig_mul_up(es_coef, support[c]), es_floor);
            double gain, bound;
            if (seen[c] == 2) {
                if (members[c] == 1) {
                    gain = bound = 0.0;
                    mandatory += exact;
                } else {
                    double m1, t1, mass, t2, em;
                    hig_two_sum(hi[c], -old_term, &m1, &t1);
                    hig_two_sum(m1, lo[c], &mass, &t2);
                    gain = hig_fma(-gn, mass, support[c]);
                    em = hig_add_up(hig_add_up(fabs(t1), fabs(t2)), err[c]);
                    em = hig_add_up(hig_add_up(em, hig_mul_up(HIG_KAPPA_CHECK,
                                                              hig_add_up(fabs(mass), em))),
                                    tiny_mass);
                    bound = hig_check_partition_bound(gain, gn, fabs(mass), es, em, eg);
                }
                current += gain;
                abs_current += fabs(gain);
                own_bounds = hig_add_up(own_bounds, bound);
                if (exact && members[c] == 1) {
                    continue;   /* a mandatory label is in every set: not scored */
                }
            } else {
                const double em = hig_add_up(
                    hig_add_up(hig_add_up(err[c], fabs(lo[c])),
                               hig_mul_up(HIG_KAPPA_CHECK,
                                          hig_add_up(hi[c], hig_add_up(err[c], fabs(lo[c]))))),
                    tiny_mass);
                gain = hig_fma(-gn, hi[c], support[c]);
                bound = hig_check_partition_bound(gain, gn, hi[c], es, em, eg);
            }
            scored[used++] = (hig_check_scored_t) { gain, bound, c };
        }
        /* The labels v has no support from: the cap heaviest (gamma n_v < 0)
         * or the lightest stand for all of them. */
        {
            const hig_id want = gn < 0.0 ? cap : 1;
            hig_id taken = 0;
            for (int64_t i = 0; i < k && taken < want; i++) {
                const hig_id c = ranked[i].id;
                if (!seen[c]) {
                    const double em = hig_add_up(hig_mul_up(theta_c, hi[c]), tiny_mass);
                    const double gain = -gn * hi[c];
                    scored[used++] = (hig_check_scored_t) {
                        gain, hig_check_partition_bound(gain, gn, hi[c], 0.0, em, eg), c };
                    taken++;
                }
            }
        }
        for (int64_t i = 0; i < count; i++) {
            seen[touched[i]] = 0;
        }
        for (int64_t i = 0; i < used; i++) {
            if (scored[i].bound > bmax) {
                bmax = scored[i].bound;
            }
        }
        current *= inv_sqrt[size];
        e_cur = hig_add_up(hig_mul_up(hig_fma(inv_sqrt[size], 8.0 * HIG_U, inv_sqrt[size]),
                                      hig_add_up(own_bounds, hig_mul_up(hig_gamma((double) size),
                                                                        abs_current))),
                           hig_add_up(hig_mul_up(6.0 * HIG_U, fabs(current)), 2.0 * HIG_ETA));

        /* The best set: a prefix of the positive gains (with the mandatory
         * labels), or the single best label, or a fresh one. */
        qsort(scored, (size_t) used, sizeof(hig_check_scored_t), hig_check_scored_compare);
        while (positive < used && positive < cap - mandatory && scored[positive].gain > 0.0) {
            positive++;
        }
        if (mandatory == 0 && positive == 0) {
            best = used > 0 ? scored[0].gain : -INFINITY;
            e_best = used > 0 ? scored[0].bound : 0.0;
            if (fresh && !(best >= 0.0)) {
                best = 0.0;
                e_best = 0.0;
            }
        } else {
            double prefix = 0.0, bounds = 0.0;
            int64_t j_best = 0;
            best = 0.0;
            for (int64_t j = 1; j <= positive; j++) {
                const double score = (prefix += scored[j - 1].gain) * inv_sqrt[mandatory + j];
                bounds = hig_add_up(bounds, scored[j - 1].bound);
                if ((mandatory == 0 && j == 1) || score > best) {
                    best = score;
                    j_best = j;
                    e_best = hig_mul_up(hig_fma(inv_sqrt[mandatory + j], 8.0 * HIG_U,
                                                inv_sqrt[mandatory + j]),
                                        bounds);
                }
            }
            if (best > 0.0) {
                e_pre = hig_add_up(hig_mul_up(hig_gamma((double) cap + 6.0), best), 2.0 * HIG_ETA);
            }
            if (j_best > 0) {
                e_best = hig_add_up(e_best, e_pre);
            }
        }
        report->vertices++;
        /* The computed best set's certified gain, and an upper bound of every
         * set's gain. */
        hig_difference(best, current, hig_add_up(e_best, e_cur), &d, &e);
        if (d > e) {
            improvable = true;
            if (hig_certified(d, e) > report->max_certified) {
                report->max_certified = hig_certified(d, e);
            }
        }
        hig_difference(best, current,
                       hig_add_up(hig_add_up(e_pre, hig_mul_up(sqrt_cap_up, bmax)), e_cur), &d,
                       &e);
        if (hig_gain_up(d, e) > report->max_bound) {
            report->max_bound = hig_gain_up(d, e);
        }
        report->improvable += improvable;
    }
done:
    hig_adjacency_free(&adj);
    free(flat);
    free(label);
    free(members);
    free(hi);
    free(lo);
    free(err);
    free(support);
    free(seen);
    free(touched);
    free(ranked);
    free(inv_sqrt);
    free(scored);
    return status;
}
