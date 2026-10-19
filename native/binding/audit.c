/* Fixed-cover audit binding. Copyright (C) 2026 Lucas Lopes Felipe.
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "binding.h"

#define AUDIT_CAPSULE "hedonic.audit_state"
static void audit_capsule_free(PyObject *capsule) {
    hig_audit_free(PyCapsule_GetPointer(capsule, AUDIT_CAPSULE));
}

/* Variable-length typed buffer. The caller releases only on success. */
static int audit_buffer(PyObject *obj, Py_buffer *view, char kind, const char *name) {
    Py_buffer probe;
    Py_ssize_t count;
    if (PyObject_GetBuffer(obj, &probe, PyBUF_SIMPLE) < 0) return -1;
    count = probe.len / 8;
    PyBuffer_Release(&probe);
    return get_buffer(obj, view, kind, count, name);
}

PyObject *py_audit_prepare(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *names[] = {"graph", "offsets", "labels", "edge_weights", "node_weights", NULL};
    PyObject *graph_obj, *offset_obj, *label_obj, *weights = Py_None, *nodes = Py_None;
    PyObject *result = NULL;
    Py_ssize_t n = 0, m = 0;
    graph_edges_t edges = {0};
    Py_buffer offset = {0}, labels = {0};
    buffers_t buffers = {0};
    hig_graph_t graph;
    hig_cover_t cover;
    hig_audit_state_t *state = NULL;
    hig_interrupt_t interrupt = {py_check_signals, NULL};
    hig_error_t status;
    (void) self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|OO", names,
                                    &graph_obj, &offset_obj, &label_obj, &weights, &nodes)) return NULL;
    if (graph_edges(graph_obj, &n, &m, &edges) < 0) return NULL;
    if (n == PY_SSIZE_T_MAX) { PyErr_SetString(PyExc_OverflowError, "Too many audit vertices"); goto done; }
    if (get_buffer(offset_obj, &offset, 'i', n + 1, "offsets") < 0) { offset.obj = NULL; goto done; }
    if (audit_buffer(label_obj, &labels, 'i', "labels") < 0) { labels.obj = NULL; goto done; }
    /* Check every offset against the actual buffer, before the C API reads it. */
    for (Py_ssize_t v = 0; v <= n; v++) {
        hig_int x = ((hig_int *) offset.buf)[v];
        if (x < 0 || x > labels.len / 8) {
            PyErr_SetString(PyExc_ValueError, "Audit row offset outside labels buffer"); goto done;
        }
    }
    if (weights != Py_None) {
        if (get_buffer(weights, &buffers.edge_weights, 'd', m, "edge_weights") < 0) goto done;
        buffers.have_edge_weights = 1;
    }
    if (nodes != Py_None) {
        if (get_buffer(nodes, &buffers.node_weights, 'd', n, "node_weights") < 0) goto done;
        buffers.have_node_weights = 1;
    }
    graph = (hig_graph_t) {.vertex_count=n, .edge_count=m, .edges=edges.owned,
        .edge_first=edges.first, .edge_second=edges.second,
        .edge_weights=buffers.have_edge_weights ? buffers.edge_weights.buf : NULL,
        .node_weights=buffers.have_node_weights ? buffers.node_weights.buf : NULL};
    cover = (hig_cover_t) {n, offset.buf, labels.buf};
    status = hig_audit_prepare(&graph, &cover, &interrupt, &state);
    if (status != HIG_SUCCESS) { raise_error(status); goto done; }
    result = PyCapsule_New(state, AUDIT_CAPSULE, audit_capsule_free);
    if (result) state = NULL;
done:
    hig_audit_free(state);
    if (offset.obj) PyBuffer_Release(&offset);
    if (labels.obj) PyBuffer_Release(&labels);
    buffers_release(&buffers); graph_edges_release(&edges);
    return result;
}

PyObject *py_audit_intervals(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *names[] = {"state", "vertices", "max_memberships", "allow_isolation", "atol", "rtol", "prune", NULL};
    PyObject *capsule, *vertices, *result = NULL;
    Py_buffer selected = {0};
    hig_audit_state_t *state;
    hig_audit_options_t o = HIG_AUDIT_OPTIONS_INIT;
    hig_interrupt_t interrupt = {py_check_signals, NULL};
    hig_audit_intervals_t intervals = {0};
    hig_error_t status;
    long long cap = 0;
    (void) self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Lpddp", names, &capsule, &vertices,
                                    &cap, &o.allow_isolation, &o.atol, &o.rtol, &o.prune)) return NULL;
    state = PyCapsule_GetPointer(capsule, AUDIT_CAPSULE);
    if (!state) return NULL;
    if (audit_buffer(vertices, &selected, 'i', "vertices") < 0) return NULL;
    o.max_memberships = cap; o.interrupt = &interrupt;
    status = hig_audit_intervals(state, &o, (hig_audit_selection_t) {0, selected.len / 8, selected.buf}, &intervals);
    if (status != HIG_SUCCESS) { raise_error(status); goto done; }
    result = PyList_New((Py_ssize_t) intervals.vertex_count);
    if (!result) goto done;
    for (hig_int v = 0; v < intervals.vertex_count; v++) {
        const hig_int start = intervals.offset[v], end = intervals.offset[v + 1];
        PyObject *row = PyList_New((Py_ssize_t) (end - start));
        if (!row) { Py_CLEAR(result); goto done; }
        PyList_SET_ITEM(result, (Py_ssize_t) v, row);
        for (hig_int j = start; j < end; j++) {
            PyObject *pair = Py_BuildValue("(dd)", intervals.intervals[j].lo, intervals.intervals[j].hi);
            if (!pair) { Py_CLEAR(result); goto done; }
            PyList_SET_ITEM(row, (Py_ssize_t) (j - start), pair);
        }
        if ((v & 4095) == 0 && PyErr_CheckSignals() != 0) { Py_CLEAR(result); goto done; }
    }
done:
    hig_audit_intervals_free(&intervals); PyBuffer_Release(&selected);
    return result;
}

PyObject *py_audit_regrets(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *names[] = {"state", "vertices", "resolutions", "regret", "tolerance",
        "max_memberships", "allow_isolation", "atol", "rtol", "prune", NULL};
    PyObject *capsule, *vertices, *gammas, *regret, *tolerance, *result = NULL;
    Py_buffer selected = {0}, gamma = {0}, r = {0}, t = {0};
    hig_audit_state_t *state;
    hig_audit_options_t o = HIG_AUDIT_OPTIONS_INIT;
    hig_interrupt_t interrupt = {py_check_signals, NULL};
    hig_error_t status;
    long long cap = 0;
    Py_ssize_t nv, ng, count;
    (void) self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOO|Lpddp", names,
        &capsule, &vertices, &gammas, &regret, &tolerance, &cap,
        &o.allow_isolation, &o.atol, &o.rtol, &o.prune)) return NULL;
    state = PyCapsule_GetPointer(capsule, AUDIT_CAPSULE);
    if (!state) return NULL;
    if (audit_buffer(vertices, &selected, 'i', "vertices") < 0) return NULL;
    if (audit_buffer(gammas, &gamma, 'd', "resolutions") < 0) { gamma.obj = NULL; goto done; }
    nv = selected.len / 8; ng = gamma.len / 8;
    if (ng && nv > PY_SSIZE_T_MAX / 8 / ng) {
        PyErr_SetString(PyExc_OverflowError, "Audit regret output size overflow"); goto done;
    }
    count = nv * ng;
    if (get_buffer(regret, &r, 'd', count, "regret") < 0) { r.obj = NULL; goto done; }
    if (get_buffer(tolerance, &t, 'd', count, "tolerance") < 0) { t.obj = NULL; goto done; }
    if (r.readonly || t.readonly) { PyErr_SetString(PyExc_ValueError, "Audit outputs must be writable"); goto done; }
    o.max_memberships = cap; o.interrupt = &interrupt;
    status = hig_audit_regrets(state, &o, (hig_audit_selection_t) {0, nv, selected.buf},
                               gamma.buf, ng, r.buf, t.buf);
    if (status != HIG_SUCCESS) { raise_error(status); goto done; }
    result = Py_NewRef(Py_None);
done:
    PyBuffer_Release(&selected);
    if (gamma.obj) PyBuffer_Release(&gamma);
    if (r.obj) PyBuffer_Release(&r);
    if (t.obj) PyBuffer_Release(&t);
    return result;
}
