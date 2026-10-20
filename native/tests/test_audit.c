/* Public fixed-cover audit API, hand cases and failure cleanup.
 * Copyright (C) 2026 Lucas Lopes Felipe. SPDX-License-Identifier: GPL-3.0-or-later */
#include "hedonic_core.h"
#include "check/hedonic_check.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #expr, hig_error_message()); exit(1); } } while (0)
static int stop(void *p) { (void) p; return 1; }


/* The windowed search against the exhaustive one, bit for bit, on random
 * covers: integer weights (ties, repeated cuts) and real ones, every option. */
static uint64_t fuzz_state = 0x9e3779b97f4a7c15u;
static uint64_t fuzz_next(void) {
    fuzz_state ^= fuzz_state << 13; fuzz_state ^= fuzz_state >> 7; fuzz_state ^= fuzz_state << 17;
    return fuzz_state;
}
static hig_int fuzz_below(hig_int n) { return (hig_int) (fuzz_next() % (uint64_t) n); }

static void fuzz_windowed(int games) {
    static const double tolerances[][2] = {{1e-10, 1e-9}, {0.0, 0.0}, {1e-3, 1e-2}, {0.05, 0.0}, {0.0, 0.1}};
    hig_int compared = 0, intervals = 0;
    for (int game = 0; game < games; game++) {
        /* Small games, then larger ones, then a few whose vertices have
         * hundreds of candidates (the windowed search's certified core). */
        const hig_int n = 3 + fuzz_below(game < games / 2 ? 14 : game % 16 ? 60 : 240);
        const hig_int labels = 1 + fuzz_below(n);
        const int integer = (int) fuzz_below(2);
        hig_int m = 0, *edges = malloc(sizeof(hig_int) * (size_t) (n * (n - 1))), offset[256],
                *rows = malloc(sizeof(hig_int) * (size_t) (4 * n)), total = 0;
        double *weights = malloc(sizeof(double) * (size_t) (n * (n - 1) / 2));
        CHECK(edges && rows && weights);
        hig_graph_t graph = {0};
        hig_cover_t cover;
        hig_audit_state_t *state = NULL;
        const double p = 0.1 + 0.6 * (double) fuzz_below(100) / 100.0;
        for (hig_int a = 0; a < n; a++)
            for (hig_int b = a + 1; b < n; b++)
                if ((double) fuzz_below(1000) / 1000.0 < p) {
                    edges[2 * m] = a; edges[2 * m + 1] = b;
                    weights[m++] = integer ? (double) (1 + fuzz_below(3)) : (double) (1 + fuzz_below(1000)) / 997.0;
                }
        offset[0] = 0;
        for (hig_int v = 0; v < n; v++) {
            const hig_int size = 1 + fuzz_below(fuzz_below(3) == 0 ? 4 : 2);
            for (hig_int j = 0; j < size; j++) rows[total++] = fuzz_below(labels);
            offset[v + 1] = total;
        }
        /* Contiguous labels: renumber the ones used. */
        {
            hig_int map[256];
            hig_int next = 0;
            for (hig_int c = 0; c < 256; c++) map[c] = -1;
            for (hig_int j = 0; j < total; j++) {
                if (map[rows[j]] < 0) map[rows[j]] = next++;
                rows[j] = map[rows[j]];
            }
        }
        graph.vertex_count = n; graph.edge_count = m; graph.edges = edges;
        graph.edge_weights = fuzz_below(4) ? weights : NULL;
        cover = (hig_cover_t) {n, offset, rows};
        CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_SUCCESS);
        for (int variant = 0; variant < 8; variant++) {
            hig_audit_options_t o = HIG_AUDIT_OPTIONS_INIT;
            hig_audit_selection_t selection = {0, n, NULL};
            hig_audit_intervals_t fast = {0}, full = {0};
            const int t = (int) fuzz_below(5);
            o.atol = tolerances[t][0]; o.rtol = tolerances[t][1];
            o.allow_isolation = (int) fuzz_below(2);
            o.prune = (int) fuzz_below(4) != 0;
            o.max_memberships = fuzz_below(2) ? 0 : 4 + fuzz_below(60);
            CHECK(hig_audit_intervals(state, &o, selection, &fast) == HIG_SUCCESS);
            CHECK(hig_audit_intervals_exhaustive(state, &o, selection, &full) == HIG_SUCCESS);
            CHECK(fast.interval_count == full.interval_count);
            CHECK(!memcmp(fast.offset, full.offset, (size_t) (n + 1) * sizeof(hig_int)));
            CHECK(!full.interval_count ||
                  !memcmp(fast.intervals, full.intervals, (size_t) full.interval_count * sizeof(hig_interval_t)));
            compared += n; intervals += full.interval_count;
            hig_audit_intervals_free(&fast); hig_audit_intervals_free(&full);
        }
        hig_audit_free(state);
        free(edges); free(rows); free(weights);
    }
    printf("windowed audit: %lld vertex audits, %lld intervals, identical to the exhaustive search\n",
           (long long) compared, (long long) intervals);
}

int main(void) {
    hig_int edges[] = {0, 1}, offsets[] = {0, 1, 2}, labels[] = {0, 1};
    double weights[] = {0.5}, gamma[] = {0, 0.5, 1}, regret[6], tolerance[6];
    hig_graph_t graph = {.vertex_count=2, .edge_count=1, .edges=edges, .edge_weights=weights};
    hig_cover_t cover = {2, offsets, labels};
    hig_audit_state_t *state = NULL;
    hig_audit_options_t o = HIG_AUDIT_OPTIONS_INIT;
    hig_audit_selection_t selection = {0, 2, NULL};
    hig_audit_intervals_t result = {0};
    hig_interrupt_t interrupt = {stop, NULL};
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_SUCCESS);
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_SUCCESS);
    CHECK(result.vertex_count == 2 && result.interval_count == 2);
    CHECK(result.offset[0] == 0 && result.offset[1] == 1 && result.offset[2] == 2);
    CHECK(result.intervals[0].lo == 0.5 - (o.atol + o.rtol));
    CHECK(result.intervals[0].hi == 1.0);
    CHECK(hig_audit_regrets(state, &o, selection, gamma, 3, regret, tolerance) == HIG_SUCCESS);
    CHECK(regret[0] == .5 && regret[1] == 0.0 && regret[2] == 0.0);
    for (int j = 0; j < 6; j++) CHECK(tolerance[j] == o.atol + o.rtol);
    hig_audit_intervals_free(&result);
    o.interrupt = &interrupt;
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_INTERRUPTED);
    CHECK(!result.offset && !result.intervals);
    o.interrupt = NULL;
    o.max_memberships = -1;
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_EINVAL);
    o.max_memberships = 0;
    o.atol = NAN;
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_EINVAL);
    o.atol = 1e-10;
    gamma[0] = -0.1;
    CHECK(hig_audit_regrets(state, &o, selection, gamma, 3, regret, tolerance) == HIG_EINVAL);
    selection.end = 3;
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_EINVAL);
    hig_audit_free(state); state = NULL;

    /* Repeated labels count once; a private label has exactly zero crowding. */
    {
        hig_int rows[] = {0, 0, 1, 0, 1}, starts[] = {0, 3, 5};
        cover.offset = starts; cover.labels = rows;
        CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_SUCCESS);
        selection = (hig_audit_selection_t) {0, 2, NULL};
        o.max_memberships = 1;
        CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_EINVAL);
        o.max_memberships = 2;
        CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_SUCCESS);
        CHECK(rows[0] == 0 && rows[1] == 0 && rows[2] == 1 && starts[2] == 5);
        hig_audit_intervals_free(&result);
        hig_audit_free(state); state = NULL;
    }
    cover.offset = offsets; cover.labels = labels;
    CHECK(hig_audit_prepare(&graph, &cover, &interrupt, &state) == HIG_INTERRUPTED && !state);
    weights[0] = -1;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
    weights[0] = INFINITY;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
    weights[0] = 0;
    edges[1] = 0;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
    edges[1] = 1;
    labels[1] = 2;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
    labels[1] = 1;
    offsets[1] = 0;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
    offsets[1] = 1;
    {
        double nodes[] = {1, 2}; graph.node_weights = nodes;
        CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EINVAL && !state);
        nodes[1] = 1;
        CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_SUCCESS);
        hig_audit_free(state); state = NULL;
    }
    graph.node_weights = NULL;
    graph.edge_count = INT64_MAX;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_EOVERFLOW && !state);
    graph.edge_count = 0;
    CHECK(hig_audit_prepare(&graph, &cover, NULL, &state) == HIG_SUCCESS);
    o.allow_isolation = 1;
    CHECK(hig_audit_intervals(state, &o, selection, &result) == HIG_SUCCESS);
    CHECK(result.interval_count == 2 && result.intervals[0].lo == 0 && result.intervals[0].hi == 1);
    hig_audit_intervals_free(&result); hig_audit_free(state);
    fuzz_windowed(getenv("HIG_AUDIT_FUZZ") ? atoi(getenv("HIG_AUDIT_FUZZ")) : 400);
    puts("fixed-cover audit: hand cases, validation and interruption passed");
    return 0;
}
