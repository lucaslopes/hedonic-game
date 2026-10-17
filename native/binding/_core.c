/*
   hedonic -- the CPython binding of the C core.
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
 * The module hedonic._core (see binding.h for its files). Every call runs
 * best-response local moving to an equilibrium (native/include/hedonic_core.h). detect() takes an
 * igraph.Graph, and is what Game.community_hedonic calls; partition() and
 * cover() take plain Python data: a vertex count, a flat edge list (any
 * buffer of 64-bit integers, e.g. array('q')), optional weight buffers of
 * doubles and an optional start state. Random numbers are those of the
 * caller's generator (32-bit words of getrandbits), used once per call to
 * shuffle the visiting order.
 */

#include "binding.h"

int py_check_signals(void *state) {
    (void) state;
    return PyErr_CheckSignals() != 0;
}

/* Sets the Python exception of a failed call, unless a callback already set
 * one (an exception from getrandbits, or KeyboardInterrupt). */
PyObject *raise_error(hig_error_t code) {
    PyObject *type;

    if (PyErr_Occurred()) {
        return NULL;
    }
    switch (code) {
    case HIG_ENOMEM:
        return PyErr_NoMemory();
    case HIG_EOVERFLOW:
        type = PyExc_OverflowError;
        break;
    case HIG_INTERRUPTED:
        type = PyExc_KeyboardInterrupt;
        break;
    case HIG_EINVAL:
        type = PyExc_ValueError;
        break;
    default:
        type = PyExc_RuntimeError;
        break;
    }
    PyErr_SetString(type, hig_error_message());
    return NULL;
}

/* ---- Module ---------------------------------------------------------------- */

static PyMethodDef core_methods[] = {
    { "detect", (PyCFunction) (void (*)(void)) py_detect, METH_VARARGS | METH_KEYWORDS,
      "detect(graph, rng, resolution, *, max_memberships=1, start=None, "
      "allow_isolation=False, edge_weights=None, node_weights=None, "
      "max_total_communities=None, n_communities=None)\n\n"
      "Best-response local moving on an undirected igraph.Graph, run until no vertex\n"
      "can improve. Returns (membership, nb_clusters, quality, equilibrium_slack)\n"
      "for max_memberships=1, else (rows, clusters, nb_clusters, quality,\n"
      "equilibrium_slack).\n"
      "`start` is flat labels or per-vertex rows, checked as hedonic._native documents;\n"
      "weights are buffers of doubles or None." },
    { "partition", (PyCFunction) (void (*)(void)) py_partition, METH_VARARGS | METH_KEYWORDS,
      "partition(n, edges, rng, edge_weights=None, node_weights=None, membership=None, "
      "resolution=1.0, max_total_communities=None, n_communities=None, "
      "allow_isolation=True) -> (membership, nb_clusters, quality)\n\n"
      "Local moving of a partition: the CPM hedonic game. `edges` is a flat buffer of\n"
      "64-bit integers (u0, v0, u1, v1, ...); weights are buffers of doubles or None;\n"
      "`rng` is an object with getrandbits(). A given membership is the start." },
    { "cover", (PyCFunction) (void (*)(void)) py_cover, METH_VARARGS | METH_KEYWORDS,
      "cover(n, edges, rng, edge_weights=None, node_weights=None, memberships=None, "
      "max_memberships=2, resolution=1.0, max_total_communities=None, n_communities=None, "
      "allow_isolation=True) -> (memberships, nb_clusters, quality)\n\n"
      "Local moving of a cover: the unit-l2 CPM hedonic game, at most max_memberships\n"
      "labels per vertex. Given memberships (one row per vertex) are the start." },
    { "_trace_partition", (PyCFunction) (void (*)(void)) py_trace_partition,
      METH_VARARGS | METH_KEYWORDS,
      "partition() that also returns {'start', 'moves', 'raw_slack'}: the start rows in\n"
      "the core's internal labels, (v, new row, improvement, bound) for every move, and\n"
      "the slack before division by |W|. For tests." },
    { "_trace_cover", (PyCFunction) (void (*)(void)) py_trace_cover,
      METH_VARARGS | METH_KEYWORDS, "cover() with the record of _trace_partition(). For tests." },
    { "_verify_partition", (PyCFunction) (void (*)(void)) py_verify_partition,
      METH_VARARGS | METH_KEYWORDS,
      "_verify_partition(n, edges, membership, edge_weights=None, node_weights=None,\n"
      "resolution=1.0, max_total_communities=None, n_communities=None,\n"
      "allow_isolation=True) -> {'vertices', 'improvable', 'max_certified', 'max_bound'}\n\n"
      "The independent verifier (native/core/check/check.c): every deviation of every vertex\n"
      "scored afresh with certified bounds. For tests." },
    { "_verify_cover", (PyCFunction) (void (*)(void)) py_verify_cover,
      METH_VARARGS | METH_KEYWORDS,
      "_verify_cover(n, edges, memberships, max_memberships, ...): the same for a cover\n"
      "(sorted rows of distinct labels). For tests." },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT, "hedonic._core",
    "The C core of hedonic: best-response local moving for the CPM hedonic game.", -1, core_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject *module = PyModule_Create(&core_module);

    if (!module) {
        return NULL;
    }
    {
        PyObject *numbers = PyImport_ImportModule("numbers");
        if (!numbers) {
            Py_DECREF(module);
            return NULL;
        }
        integral_type = PyObject_GetAttrString(numbers, "Integral");
        Py_DECREF(numbers);
        if (!integral_type) {
            Py_DECREF(module);
            return NULL;
        }
    }
    {
        PyObject *random_module = PyImport_ImportModule("random");
        if (!random_module) {
            Py_DECREF(module);
            return NULL;
        }
        random_type = PyObject_GetAttrString(random_module, "Random");
        Py_DECREF(random_module);
        if (!random_type) {
            Py_DECREF(module);
            return NULL;
        }
    }
    if (PyModule_AddStringConstant(module, "core_version", HIG_VERSION_STRING) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
