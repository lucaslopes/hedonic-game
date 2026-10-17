/*
   hedonic -- result lists of detect().
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

/* [i0, i1, ...] as a list of ints. */
PyObject *int_list(const hig_int *values, hig_int count) {
    PyObject *list = PyList_New((Py_ssize_t) count);

    if (!list) {
        return NULL;
    }
    for (hig_int i = 0; i < count; i++) {
        PyObject *value = PyLong_FromLongLong(values[i]);
        if (!value) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, (Py_ssize_t) i, value);
    }
    return list;
}

/* The rows of a cover (one list of labels per vertex) and its clusters (one
 * list of vertices per label, vertices increasing). Each label and vertex is
 * one int object shared by both. */
int cover_lists(const hig_cover_t *cover, hig_int nb_clusters, PyObject **rows_out,
                PyObject **clusters_out) {
    const hig_int n = cover->vertex_count;
    PyObject *rows = NULL, *clusters = NULL, *label_objects = NULL;
    Py_ssize_t *fill = NULL;
    int status = -1;

    rows = PyList_New((Py_ssize_t) n);
    clusters = PyList_New((Py_ssize_t) nb_clusters);
    label_objects = PyList_New((Py_ssize_t) nb_clusters);
    fill = PyMem_Calloc(nb_clusters > 0 ? (size_t) nb_clusters : 1, sizeof(Py_ssize_t));
    if (!rows || !clusters || !label_objects || !fill) {
        if (!PyErr_Occurred()) {
            PyErr_NoMemory();
        }
        goto done;
    }
    for (hig_int c = 0; c < nb_clusters; c++) {
        PyObject *label = PyLong_FromLongLong(c);
        if (!label) {
            goto done;
        }
        PyList_SET_ITEM(label_objects, (Py_ssize_t) c, label);
    }
    for (hig_int v = 0; v < n; v++) {
        for (hig_int i = cover->offset[v]; i < cover->offset[v + 1]; i++) {
            fill[cover->labels[i]]++;
        }
    }
    for (hig_int c = 0; c < nb_clusters; c++) {
        PyObject *cluster = PyList_New(fill[c]);
        if (!cluster) {
            goto done;
        }
        PyList_SET_ITEM(clusters, (Py_ssize_t) c, cluster);
        fill[c] = 0;
    }
    for (hig_int v = 0; v < n; v++) {
        const hig_int *row = cover->labels + cover->offset[v];
        const hig_int size = cover->offset[v + 1] - cover->offset[v];
        PyObject *vertex = PyLong_FromLongLong(v), *labels;
        if (!vertex) {
            goto done;
        }
        labels = PyList_New((Py_ssize_t) size);
        if (!labels) {
            Py_DECREF(vertex);
            goto done;
        }
        for (hig_int i = 0; i < size; i++) {
            const hig_int c = row[i];
            PyObject *label = PyList_GET_ITEM(label_objects, (Py_ssize_t) c);
            Py_INCREF(label);
            PyList_SET_ITEM(labels, (Py_ssize_t) i, label);
            Py_INCREF(vertex);
            PyList_SET_ITEM(PyList_GET_ITEM(clusters, (Py_ssize_t) c), fill[c]++, vertex);
        }
        Py_DECREF(vertex);
        PyList_SET_ITEM(rows, (Py_ssize_t) v, labels);
    }
    *rows_out = rows;
    *clusters_out = clusters;
    rows = clusters = NULL;
    status = 0;
done:
    PyMem_Free(fill);
    Py_XDECREF(label_objects);
    Py_XDECREF(rows);
    Py_XDECREF(clusters);
    return status;
}
