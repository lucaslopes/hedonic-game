/*
   hedonic -- _verify_partition(), for tests.
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
 * The independent verifier of the core (native/core/check.c) over plain
 * Python data: a vertex count, a flat edge buffer, the result to check and
 * the options it was computed under. Returns {"vertices", "improvable",
 * "max_certified", "max_bound"} (raw gains, not divided by |W|).
 */

#include "binding.h"
#include "hedonic_check.h"

static PyObject *report_dict(const hig_verify_t *report) {
    return Py_BuildValue("{sLsLsdsd}", "vertices", (long long) report->vertices, "improvable",
                         (long long) report->improvable, "max_certified", report->max_certified,
                         "max_bound", report->max_bound);
}

/* Reads a sequence of integers into a new PyMem array of `count` entries. */
static hig_int *read_ints(PyObject *obj, Py_ssize_t count, const char *what) {
    PyObject *seq = PySequence_Fast(obj, what);
    hig_int *out;

    if (!seq) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(seq) != count) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s has the wrong length", what);
        return NULL;
    }
    out = PyMem_Malloc(count > 0 ? (size_t) count * sizeof(hig_int) : 1);
    if (!out) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        out[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (out[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(seq);
            PyMem_Free(out);
            return NULL;
        }
    }
    Py_DECREF(seq);
    return out;
}

PyObject *py_verify_partition(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = { "n", "edges", "membership", "edge_weights", "node_weights",
                              "resolution", "max_total_communities", "n_communities",
                              "allow_isolation", NULL };
    Py_ssize_t n;
    PyObject *edges, *membership_obj, *edge_weights = Py_None, *node_weights = Py_None;
    PyObject *max_total_obj = Py_None, *n_communities_obj = Py_None, *result = NULL;
    double resolution = 1.0;
    int allow_isolation = 1;
    hig_graph_t graph;
    buffers_t buffers;
    hig_options_t options;
    hig_verify_t report;
    hig_int *membership = NULL;
    hig_error_t status;

    (void) self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "nOO|OOdOOp", kwlist, &n, &edges, &membership_obj,
                                     &edge_weights, &node_weights, &resolution, &max_total_obj,
                                     &n_communities_obj, &allow_isolation)) {
        return NULL;
    }
    options = (hig_options_t) { .resolution = resolution, .allow_isolation = allow_isolation };
    if (optional_limit(max_total_obj, &options.max_total_communities, "max_total_communities") < 0 ||
        optional_limit(n_communities_obj, &options.n_communities, "n_communities") < 0) {
        return NULL;
    }
    if (make_graph(n, edges, edge_weights, node_weights, &graph, &buffers) < 0) {
        return NULL;
    }
    membership = read_ints(membership_obj, n, "membership");
    if (!membership) {
        goto done;
    }
    status = hig_verify_partition(&graph, &options, membership, &report);
    if (status != HIG_SUCCESS) {
        raise_error(status);
        goto done;
    }
    result = report_dict(&report);
done:
    PyMem_Free(membership);
    buffers_release(&buffers);
    return result;
}
