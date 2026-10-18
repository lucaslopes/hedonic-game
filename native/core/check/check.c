/*
   hedonic -- the independent verifier of a partition.
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
 * hig_verify_partition (hedonic_check.h) takes a result
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
