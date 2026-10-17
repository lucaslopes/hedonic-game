/*
   hedonic -- partition() over plain Python data.
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
#include "hedonic_check.h"

#include <string.h>

/* ---- Traces (tests only) ----------------------------------------------------- */

/* A traced run's record: the start rows, then (v, row, improvement, bound)
 * for every accepted move. */
typedef struct {
    PyObject *start;
    PyObject *moves;
    int moving;
    hig_int moved;
    double improvement, bound;
    int failed;
} tracer_t;

static void tracer_row(void *state, hig_int v, const int32_t *labels, hig_int size) {
    tracer_t *t = (tracer_t *) state;
    PyObject *row, *item;

    if (t->failed) {
        return;
    }
    row = PyList_New(size);
    if (!row) {
        t->failed = 1;
        return;
    }
    for (hig_int i = 0; i < size; i++) {
        PyObject *label = PyLong_FromLong(labels[i]);
        if (!label) {
            Py_DECREF(row);
            t->failed = 1;
            return;
        }
        PyList_SET_ITEM(row, i, label);
    }
    if (t->moving) {
        t->moving = 0;
        item = Py_BuildValue("(LNdd)", (long long) v, row, t->improvement, t->bound);
        if (!item || PyList_Append(t->moves, item) < 0) {
            t->failed = 1;
        }
        Py_XDECREF(item);
    } else {
        if (PyList_Append(t->start, row) < 0) {
            t->failed = 1;
        }
        Py_DECREF(row);
    }
}

static void tracer_move(void *state, hig_int v, double improvement, double bound) {
    tracer_t *t = (tracer_t *) state;
    t->moving = 1;
    t->moved = v;
    t->improvement = improvement;
    t->bound = bound;
}

static int tracer_init(tracer_t *t, hig_trace_t *trace) {
    memset(t, 0, sizeof(*t));
    t->start = PyList_New(0);
    t->moves = PyList_New(0);
    if (!t->start || !t->moves) {
        Py_CLEAR(t->start);
        Py_CLEAR(t->moves);
        return -1;
    }
    trace->row = tracer_row;
    trace->move = tracer_move;
    trace->state = t;
    return 0;
}

/* The fourth item of a traced result: {"start", "moves", "raw_slack"}. */
static PyObject *tracer_result(tracer_t *t, double raw_slack) {
    PyObject *record;

    if (t->failed) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "the trace could not be recorded");
        }
        return NULL;
    }
    record = Py_BuildValue("{sOsOsd}", "start", t->start, "moves", t->moves, "raw_slack",
                           raw_slack);
    return record;
}

static void tracer_release(tracer_t *t) {
    Py_CLEAR(t->start);
    Py_CLEAR(t->moves);
}

/* ---- partition() ----------------------------------------------------------- */

static PyObject *run_partition(PyObject *args, PyObject *kwds, int traced) {
    static char *kwlist[] = { "n", "edges", "rng", "edge_weights", "node_weights", "membership",
                              "resolution", "max_total_communities", "n_communities",
                              "allow_isolation", NULL };
    Py_ssize_t n;
    PyObject *edges, *rng, *edge_weights = Py_None, *node_weights = Py_None;
    PyObject *membership_obj = Py_None, *max_total_obj = Py_None, *n_communities_obj = Py_None;
    double resolution = 1.0;
    int allow_isolation = 1;
    hig_graph_t graph;
    buffers_t buffers;
    py_rng_state_t rng_state = { 0 };
    hig_rng_t hig_rng;
    hig_interrupt_t interrupt = { py_check_signals, NULL };
    hig_options_t options;
    hig_int *membership = NULL, nb_clusters = 0;
    double quality = 0.0;
    hig_error_t status;
    PyObject *result = NULL, *list = NULL;
    tracer_t tracer;
    hig_trace_t trace;
    double raw_slack = 0.0;

    memset(&tracer, 0, sizeof(tracer));
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "nOO|OOOdOOp", kwlist, &n, &edges, &rng,
                                     &edge_weights, &node_weights, &membership_obj, &resolution,
                                     &max_total_obj, &n_communities_obj, &allow_isolation)) {
        return NULL;
    }
    options = (hig_options_t) {
        .resolution = resolution,
        .allow_isolation = allow_isolation,
        .interrupt = &interrupt
    };
    if (optional_limit(max_total_obj, &options.max_total_communities, "max_total_communities") < 0 ||
        optional_limit(n_communities_obj, &options.n_communities, "n_communities") < 0) {
        return NULL;
    }
    if (make_graph(n, edges, edge_weights, node_weights, &graph, &buffers) < 0) {
        return NULL;
    }
    if (make_rng(rng, &rng_state) < 0) {
        goto done;
    }
    hig_rng = rng_callbacks(&rng_state);
    options.rng = &hig_rng;

    membership = PyMem_Calloc(n > 0 ? (size_t) n : 1, sizeof(hig_int));
    if (!membership) {
        PyErr_NoMemory();
        goto done;
    }
    if (membership_obj != Py_None) {
        PyObject *seq = PySequence_Fast(membership_obj, "membership must be a sequence");
        if (!seq) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(seq) != n) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError,
                            "Membership vector length does not equal the number of vertices.");
            goto done;
        }
        for (Py_ssize_t v = 0; v < n; v++) {
            membership[v] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, v));
            if (membership[v] == -1 && PyErr_Occurred()) {
                Py_DECREF(seq);
                goto done;
            }
        }
        Py_DECREF(seq);
    }

    if (traced) {
        if (tracer_init(&tracer, &trace) < 0) {
            goto done;
        }
        status = hig_partition_traced(&graph, &options, membership_obj != Py_None, membership,
                                      &nb_clusters, &quality, &raw_slack, &trace);
    } else {
        status = hig_partition(&graph, &options, membership_obj != Py_None, membership,
                               &nb_clusters, &quality, NULL);
    }
    if (status != HIG_SUCCESS) {
        raise_error(status);
        goto done;
    }
    list = PyList_New(n);
    if (!list) {
        goto done;
    }
    for (Py_ssize_t v = 0; v < n; v++) {
        PyObject *label = PyLong_FromLongLong(membership[v]);
        if (!label) {
            Py_CLEAR(list);
            goto done;
        }
        PyList_SET_ITEM(list, v, label);
    }
    if (traced) {
        PyObject *record = tracer_result(&tracer, raw_slack);
        if (!record) {
            Py_CLEAR(list);
            goto done;
        }
        result = Py_BuildValue("(NLdN)", list, (long long) nb_clusters, quality, record);
    } else {
        result = Py_BuildValue("(NLd)", list, (long long) nb_clusters, quality);
    }
done:
    tracer_release(&tracer);
    PyMem_Free(membership);
    if (rng_release(&rng_state) < 0) {
        Py_CLEAR(result);
    }
    buffers_release(&buffers);
    return result;
}

PyObject *py_partition(PyObject *self, PyObject *args, PyObject *kwds) {
    (void) self;
    return run_partition(args, kwds, 0);
}

PyObject *py_trace_partition(PyObject *self, PyObject *args, PyObject *kwds) {
    (void) self;
    return run_partition(args, kwds, 1);
}
