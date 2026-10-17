/*
   hedonic -- internal declarations of the CPython binding.
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
 * The binding hedonic._core, one file per step:
 *
 *   _core.c         the module, its exception and the error mapping
 *   random.c        the caller's random.Random as the core's generator
 *   arguments.c     buffers, the graph and limits of partition()
 *   plain.c         partition() over plain Python data (and its traced form
 *                   _trace_partition(), for tests)
 *   graph_edges.c   the edges of an igraph.Graph, borrowed when possible
 *   start_rows.c    the start state of detect(), checked
 *   results.c       the result list of detect()
 *   detect.c        detect(), what Game.community_hedonic calls
 *   checking.c      _verify_partition(), for tests
 *
 * The GIL is held throughout: interruption checks and, for a generator that
 * is not an exact random.Random, every random number are Python calls.
 */

#ifndef HIG_BINDING_H
#define HIG_BINDING_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "hedonic_core.h"

/* ---- _core.c --------------------------------------------------------------- */

/* The core's interruption callback: KeyboardInterrupt and other signals. */
int py_check_signals(void *state);
/* Sets the Python exception of a failed call, unless a callback already set
 * one; returns NULL. */
PyObject *raise_error(hig_error_t code);

/* ---- random.c -------------------------------------------------------------- */

#define MT_N 624

/* CPython's Mersenne Twister state (Modules/_randommodule.c). */
typedef struct {
    uint32_t state[MT_N];
    int index;
} mt_state_t;

typedef struct {
    PyObject *getrandbits;
    PyObject *bits;
    PyObject *rng;          /* the generator whose state `mt` copies, or NULL */
    PyObject *gauss_next;   /* its getstate()[2], written back unchanged */
    mt_state_t mt;
} py_rng_state_t;

extern PyObject *random_type;   /* random.Random, set by the module */

/* Prepares `state` for the generator `rng`: 0, or -1 with a Python error. */
int make_rng(PyObject *rng, py_rng_state_t *state);
hig_rng_t rng_callbacks(py_rng_state_t *state);
/* Writes a copied state back to its generator and releases `state`: 0, or -1
 * with a Python error. */
int rng_release(py_rng_state_t *state);

/* ---- arguments.c ----------------------------------------------------------- */

typedef struct {
    Py_buffer edges;
    Py_buffer edge_weights;
    Py_buffer node_weights;
    int have_edges, have_edge_weights, have_node_weights;
} buffers_t;

void buffers_release(buffers_t *b);
int get_buffer(PyObject *obj, Py_buffer *view, char kind, Py_ssize_t expected,
               const char *name);
int make_graph(Py_ssize_t n, PyObject *edges, PyObject *edge_weights,
               PyObject *node_weights, hig_graph_t *graph, buffers_t *b);
int optional_limit(PyObject *obj, hig_int *limit, const char *name);

/* ---- plain.c --------------------------------------------------------------- */

PyObject *py_partition(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *py_trace_partition(PyObject *self, PyObject *args, PyObject *kwds);

/* ---- checking.c ------------------------------------------------------------ */

PyObject *py_verify_partition(PyObject *self, PyObject *args, PyObject *kwds);

/* ---- graph_edges.c --------------------------------------------------------- */

/* The edges of the graph for one call: borrowed from the igraph_t behind the
 * capsule (no copy, as python-igraph's own binding reads them), or an owned
 * copy from get_edgelist(). */
typedef struct {
    PyObject *capsule;        /* keeps the borrowed view's capsule alive */
    const hig_int *first;     /* borrowed: edge e is (first[e], second[e]) */
    const hig_int *second;
    hig_int *owned;           /* fallback: edge e is (owned[2e], owned[2e + 1]) */
} graph_edges_t;

void graph_edges_release(graph_edges_t *edges);
int graph_edges(PyObject *graph, Py_ssize_t *n_out, Py_ssize_t *m_out, graph_edges_t *edges);

/* ---- start_rows.c ---------------------------------------------------------- */

typedef struct {
    hig_int *labels;    /* distinct labels of every row, rows in order */
    hig_int *offset;    /* row v is labels[offset[v] .. offset[v + 1] - 1] */
    hig_int max_label;
} start_rows_t;

extern PyObject *integral_type;   /* numbers.Integral, set by the module */

void start_rows_release(start_rows_t *rows);
int read_start_rows(PyObject *obj, Py_ssize_t n, start_rows_t *rows);

/* ---- results.c ------------------------------------------------------------- */

PyObject *int_list(const hig_int *values, hig_int count);

/* ---- detect.c -------------------------------------------------------------- */

PyObject *py_detect(PyObject *self, PyObject *args, PyObject *kwds);

#endif
