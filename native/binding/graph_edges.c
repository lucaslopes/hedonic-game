/*
   hedonic -- the edges of an igraph.Graph, borrowed when possible.
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

/* The head of igraph's igraph_t (include/igraph_datatype.h, igraph 0.10 and
 * 1.0), whose `from` and `to` vectors the public IGRAPH_FROM/IGRAPH_TO macros
 * read. python-igraph hands C code this struct through
 * Graph.__graph_as_capsule(); everything read from it is checked against the
 * Python API below, and any mismatch falls back to get_edgelist(). */
typedef struct {
    int64_t *stor_begin;
    int64_t *stor_end;
    int64_t *end;
} ig_vector_int_t;

typedef struct {
    int64_t n;
    unsigned char directed;   /* igraph_bool_t; padding follows */
    ig_vector_int_t from;
    ig_vector_int_t to;
} ig_graph_head_t;

static long long call_count(PyObject *graph, const char *method) {
    PyObject *value = PyObject_CallMethod(graph, method, NULL);
    long long count;

    if (!value) {
        return -1;
    }
    count = PyLong_AsLongLong(value);
    Py_DECREF(value);
    return count;
}

/* Reads edge e as python-igraph reports it (graph.es[e].tuple) and compares
 * it with (u, v); -1 on a Python error. */
static int edge_matches(PyObject *graph, Py_ssize_t e, int64_t u, int64_t v) {
    PyObject *es = PyObject_GetAttrString(graph, "es"), *edge = NULL, *pair = NULL;
    int same = -1;
    long long a, b;

    if (!es) {
        return -1;
    }
    edge = PySequence_GetItem(es, e);
    if (edge) {
        pair = PyObject_GetAttrString(edge, "tuple");
    }
    if (pair && PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2) {
        a = PyLong_AsLongLong(PyTuple_GET_ITEM(pair, 0));
        b = PyLong_AsLongLong(PyTuple_GET_ITEM(pair, 1));
        if (!PyErr_Occurred()) {
            same = a == u && b == v;
        }
    }
    Py_XDECREF(pair);
    Py_XDECREF(edge);
    Py_DECREF(es);
    return same;
}

void graph_edges_release(graph_edges_t *edges) {
    Py_CLEAR(edges->capsule);
    PyMem_Free(edges->owned);
    edges->owned = NULL;
    edges->first = edges->second = NULL;
}

/* Borrows the edges of the igraph_t behind the capsule, in get_edgelist()'s
 * order. Returns 1 on success, 0 when the struct does not look like the
 * documented one (no exception set), -1 on a Python error. */
static int edges_from_capsule(PyObject *graph, Py_ssize_t n, Py_ssize_t m, graph_edges_t *edges) {
    PyObject *capsule = PyObject_CallMethod(graph, "__graph_as_capsule", NULL);
    const ig_graph_head_t *head;
    const Py_ssize_t probes[3] = { 0, m / 2, m - 1 };

    if (!capsule) {
        PyErr_Clear();
        return 0;
    }
    head = (const ig_graph_head_t *) PyCapsule_GetPointer(capsule, NULL);
    if (!head) {
        PyErr_Clear();
        Py_DECREF(capsule);
        return 0;
    }
    if (head->n != n || head->directed != 0 ||
        !head->from.stor_begin || !head->to.stor_begin ||
        head->from.end - head->from.stor_begin != m || head->to.end - head->to.stor_begin != m ||
        head->from.end > head->from.stor_end || head->to.end > head->to.stor_end) {
        Py_DECREF(capsule);
        return 0;
    }
    /* Undirected edges come out as (to, from): igraph_edge(). */
    for (Py_ssize_t e = 0; e < m; e++) {
        const int64_t u = head->to.stor_begin[e], v = head->from.stor_begin[e];
        if (u < 0 || u >= n || v < 0 || v >= n) {
            Py_DECREF(capsule);
            return 0;
        }
    }
    for (int i = 0; i < 3 && m > 0; i++) {
        const int same = edge_matches(graph, probes[i], head->to.stor_begin[probes[i]],
                                      head->from.stor_begin[probes[i]]);
        if (same <= 0) {
            Py_DECREF(capsule);
            return same;
        }
    }
    edges->capsule = capsule;
    edges->first = head->to.stor_begin;
    edges->second = head->from.stor_begin;
    return 1;
}

static int edges_from_edgelist(PyObject *graph, Py_ssize_t m, hig_int *edges) {
    PyObject *list = PyObject_CallMethod(graph, "get_edgelist", NULL), *seq;

    if (!list) {
        return -1;
    }
    seq = PySequence_Fast(list, "get_edgelist() must return a sequence");
    Py_DECREF(list);
    if (!seq) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != m) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_RuntimeError, "get_edgelist() disagrees with ecount()");
        return -1;
    }
    for (Py_ssize_t e = 0; e < m; e++) {
        PyObject *pair = PySequence_Fast_GET_ITEM(seq, e);
        if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_TypeError, "get_edgelist() must return pairs");
            return -1;
        }
        edges[2 * e] = PyLong_AsLongLong(PyTuple_GET_ITEM(pair, 0));
        edges[2 * e + 1] = PyLong_AsLongLong(PyTuple_GET_ITEM(pair, 1));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}

/* The edges of an undirected igraph.Graph: borrowed when the capsule can be
 * read, else copied from get_edgelist(). Returns 0, or -1 with an exception. */
int graph_edges(PyObject *graph, Py_ssize_t *n_out, Py_ssize_t *m_out,
                graph_edges_t *edges) {
    const long long n = call_count(graph, "vcount"), m = call_count(graph, "ecount");
    int got;

    if (n < 0 || m < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid vertex or edge count");
        }
        return -1;
    }
    got = edges_from_capsule(graph, (Py_ssize_t) n, (Py_ssize_t) m, edges);
    if (got == 0) {
        if ((unsigned long long) m > PY_SSIZE_T_MAX / (2 * sizeof(hig_int))) {
            PyErr_NoMemory();
            return -1;
        }
        edges->owned = PyMem_Malloc(m > 0 ? (size_t) (2 * m) * sizeof(hig_int) : 1);
        if (!edges->owned) {
            PyErr_NoMemory();
            return -1;
        }
        got = edges_from_edgelist(graph, (Py_ssize_t) m, edges->owned) == 0 ? 1 : -1;
    }
    if (got < 0) {
        graph_edges_release(edges);
        return -1;
    }
    *n_out = (Py_ssize_t) n;
    *m_out = (Py_ssize_t) m;
    return 0;
}
