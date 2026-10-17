/* Public fixed-cover audit API, hand cases and failure cleanup.
 * Copyright (C) 2026 Lucas Lopes Felipe. SPDX-License-Identifier: GPL-3.0-or-later */
#include "hedonic_core.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #expr, hig_error_message()); exit(1); } } while (0)
static int stop(void *p) { (void) p; return 1; }

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
    puts("fixed-cover audit: hand cases, validation and interruption passed");
    return 0;
}
