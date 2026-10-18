/*
   hedonic -- detect(): an igraph.Graph in, result lists out.
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

#include "binding.h"

PyObject *py_detect(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = { "graph", "rng", "resolution", "max_memberships", "start",
                              "allow_isolation", "edge_weights", "node_weights",
                              "max_total_communities", "n_communities", NULL };
    PyObject *graph_obj, *rng, *start_obj = Py_None, *edge_weights = Py_None;
    PyObject *node_weights = Py_None, *max_total_obj = Py_None, *n_communities_obj = Py_None;
    double resolution;
    long long max_memberships = 1;
    int allow_isolation = 0;
    Py_ssize_t n = 0, m = 0;
    graph_edges_t edges = { NULL, NULL, NULL, NULL };
    buffers_t buffers;
    hig_graph_t graph;
    py_rng_state_t rng_state = { 0 };
    hig_rng_t hig_rng;
    hig_interrupt_t interrupt = { py_check_signals, NULL };
    hig_options_t options;
    start_rows_t start = { NULL, NULL, -1 };
    int have_start = 0;
    hig_int *membership = NULL, nb_clusters = 0;
    double quality = 0.0, slack = 0.0;
    hig_error_t status;
    PyObject *result = NULL;

    (void) self;
    memset(&buffers, 0, sizeof(buffers));
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOd|$LOpOOOO", kwlist, &graph_obj, &rng,
                                     &resolution, &max_memberships, &start_obj, &allow_isolation, &edge_weights, &node_weights,
                                     &max_total_obj, &n_communities_obj)) {
        return NULL;
    }
    if (max_memberships != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "hedonic 1 detects partitions only (max_memberships=1); "
                        "overlapping covers are hedonic 2");
        return NULL;
    }
    options = (hig_options_t) {
        .resolution = resolution,
        .max_memberships = max_memberships,
        .allow_isolation = allow_isolation,
        .interrupt = &interrupt
    };
    if (optional_limit(max_total_obj, &options.max_total_communities, "max_total_communities") < 0 ||
        optional_limit(n_communities_obj, &options.n_communities, "n_communities") < 0) {
        return NULL;
    }
    if (graph_edges(graph_obj, &n, &m, &edges) < 0) {
        return NULL;
    }
    if (edge_weights != Py_None) {
        if (get_buffer(edge_weights, &buffers.edge_weights, 'd', m, "edge_weights") < 0) {
            goto done;
        }
        buffers.have_edge_weights = 1;
    }
    if (node_weights != Py_None) {
        if (get_buffer(node_weights, &buffers.node_weights, 'd', n, "node_weights") < 0) {
            goto done;
        }
        buffers.have_node_weights = 1;
    }
    graph = (hig_graph_t) {
        .vertex_count = n,
        .edge_count = m,
        .edges = edges.owned,
        .edge_weights = buffers.have_edge_weights ? (const double *) buffers.edge_weights.buf : NULL,
        .node_weights = buffers.have_node_weights ? (const double *) buffers.node_weights.buf : NULL,
        .edge_first = edges.first,
        .edge_second = edges.second
    };
    if (start_obj != Py_None) {
        have_start = read_start_rows(start_obj, n, max_memberships, &start);
        if (have_start < 0) {
            goto done;
        }
    }
    if (make_rng(rng, &rng_state) < 0) {
        goto done;
    }
    hig_rng = rng_callbacks(&rng_state);
    options.rng = &hig_rng;

    {
        PyObject *list;
        membership = PyMem_Calloc(n > 0 ? (size_t) n : 1, sizeof(hig_int));
        if (!membership) {
            PyErr_NoMemory();
            goto done;
        }
        if (have_start) {
            for (Py_ssize_t v = 0; v < n; v++) {
                membership[v] = start.labels[start.offset[v]];
            }
            start_rows_release(&start);
        }
        status = hig_partition(&graph, &options, have_start, membership, &nb_clusters, &quality,
                               &slack);
        if (status != HIG_SUCCESS) {
            raise_error(status);
            goto done;
        }
        graph_edges_release(&edges);
        list = int_list(membership, n);
        if (list) {
            result = Py_BuildValue("(NLdd)", list, (long long) nb_clusters, quality, slack);
        }
    }
done:
    PyMem_Free(membership);
    start_rows_release(&start);
    if (rng_release(&rng_state) < 0) {
        Py_CLEAR(result);
    }
    buffers_release(&buffers);
    graph_edges_release(&edges);
    return result;
}
