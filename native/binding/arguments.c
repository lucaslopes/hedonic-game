/*
   hedonic -- plain-data arguments: buffers, the graph, limits.
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
 * Arguments of partition() and cover(): buffers of 64-bit integers and
 * doubles checked for kind and length, the hig_graph_t over them, and the
 * optional count limits.
 */

#include "binding.h"

void buffers_release(buffers_t *b) {
    if (b->have_edges) {
        PyBuffer_Release(&b->edges);
    }
    if (b->have_edge_weights) {
        PyBuffer_Release(&b->edge_weights);
    }
    if (b->have_node_weights) {
        PyBuffer_Release(&b->node_weights);
    }
}

int get_buffer(PyObject *obj, Py_buffer *view, char kind, Py_ssize_t expected,
               const char *name) {
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        return -1;
    }
    if (view->itemsize != 8 || !view->format ||
        (kind == 'i' ? !(strcmp(view->format, "q") == 0 || strcmp(view->format, "l") == 0 ||
                         strcmp(view->format, "<q") == 0 || strcmp(view->format, "=q") == 0)
                     : !(strcmp(view->format, "d") == 0 || strcmp(view->format, "<d") == 0 ||
                         strcmp(view->format, "=d") == 0))) {
        PyErr_Format(PyExc_TypeError, "%s must be a contiguous buffer of %s", name,
                     kind == 'i' ? "64-bit integers (array('q'))" : "doubles (array('d'))");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->len / 8 != expected) {
        PyErr_Format(PyExc_ValueError, "%s has %zd entries, expected %zd", name, view->len / 8,
                     expected);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

/* Fills the graph from n, the edge buffer and the weight buffers (None for
 * unit weights). */
int make_graph(Py_ssize_t n, PyObject *edges, PyObject *edge_weights,
               PyObject *node_weights, hig_graph_t *graph, buffers_t *b) {
    Py_ssize_t m;

    memset(b, 0, sizeof(*b));
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "n must be non-negative");
        return -1;
    }
    if (PyObject_GetBuffer(edges, &b->edges, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        return -1;
    }
    m = b->edges.len / 16;
    PyBuffer_Release(&b->edges);
    if (get_buffer(edges, &b->edges, 'i', 2 * m, "edges") < 0) {
        return -1;
    }
    b->have_edges = 1;
    if (edge_weights != Py_None) {
        if (get_buffer(edge_weights, &b->edge_weights, 'd', m, "edge_weights") < 0) {
            buffers_release(b);
            return -1;
        }
        b->have_edge_weights = 1;
    }
    if (node_weights != Py_None) {
        if (get_buffer(node_weights, &b->node_weights, 'd', n, "node_weights") < 0) {
            buffers_release(b);
            return -1;
        }
        b->have_node_weights = 1;
    }
    *graph = (hig_graph_t) {
        .vertex_count = n,
        .edge_count = m,
        .edges = (const hig_int *) b->edges.buf,
        .edge_weights = b->have_edge_weights ? (const double *) b->edge_weights.buf : NULL,
        .node_weights = b->have_node_weights ? (const double *) b->node_weights.buf : NULL
    };
    return 0;
}

int optional_limit(PyObject *obj, hig_int *limit, const char *name) {
    if (obj == Py_None) {
        *limit = -1;
        return 0;
    }
    *limit = PyLong_AsLongLong(obj);
    if (*limit == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (*limit < 1) {
        PyErr_Format(PyExc_ValueError, "%s must be a positive integer or None", name);
        return -1;
    }
    return 0;
}
