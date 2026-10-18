/*
   hedonic -- the start state of detect(), checked as hedonic._native documents.
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

/* A label too large for int64: such a start can never be contiguous. */
#define HIG_LABEL_TOO_LARGE INT64_MAX

PyObject *integral_type;   /* numbers.Integral */

static int is_integer_label(PyObject *label) {
    if (PyBool_Check(label)) {
        return 0;
    }
    if (PyLong_Check(label)) {
        return 1;
    }
    return PyObject_IsInstance(label, integral_type);
}

static int compare_labels(const void *a, const void *b) {
    const hig_int x = *(const hig_int *) a, y = *(const hig_int *) b;
    return (x > y) - (x < y);
}

void start_rows_release(start_rows_t *rows) {
    PyMem_Free(rows->labels);
    PyMem_Free(rows->offset);
    rows->labels = rows->offset = NULL;
}

/* Reads one label: a non-negative integer (bool excluded). */
static int read_label(PyObject *obj, hig_int *label) {
    int integral = is_integer_label(obj);
    long long value;
    int overflow = 0;

    if (integral < 0) {
        return -1;
    }
    if (!integral) {
        PyErr_SetString(PyExc_ValueError, "membership labels must be integers");
        return -1;
    }
    if (PyLong_Check(obj)) {
        value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    } else {
        PyObject *index = PyNumber_Index(obj);
        if (!index) {
            return -1;
        }
        value = PyLong_AsLongLongAndOverflow(index, &overflow);
        Py_DECREF(index);
    }
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (overflow < 0 || (!overflow && value < 0)) {
        PyErr_SetString(PyExc_ValueError, "membership labels must be non-negative");
        return -1;
    }
    /* A label beyond int64 cannot be contiguous from 0 with n rows. */
    *label = overflow > 0 ? HIG_LABEL_TOO_LARGE : (hig_int) value;
    return 0;
}

/* The checks of hedonic._native._start_rows, in the same order and with the
 * same messages: one row per vertex; flat and nested rows not mixed; rows
 * non-empty; labels non-negative integers; a repeated label counts once;
 * at most max_memberships distinct labels per row; labels 0 .. K - 1 all
 * used. Returns 1 with rows filled, 0 for an empty start (no vertices), -1 on
 * error. */
int read_start_rows(PyObject *obj, Py_ssize_t n, hig_int max_memberships,
                    start_rows_t *rows) {
    PyObject *seq = PySequence_Fast(obj, "initial_membership must be a sequence");
    Py_ssize_t size, used = 0, capacity;
    hig_int *scratch = NULL;
    Py_ssize_t scratch_capacity = 0;
    unsigned char *present = NULL;
    int nested, result = -1;

    memset(rows, 0, sizeof(*rows));
    if (!seq) {
        return -1;
    }
    size = PySequence_Fast_GET_SIZE(seq);
    if (size != n) {
        PyErr_Format(PyExc_ValueError, "memberships must contain one row per vertex (%zd)", n);
        goto done;
    }
    if (size == 0) {
        result = 0;
        goto done;
    }
    nested = PyList_Check(PySequence_Fast_GET_ITEM(seq, 0)) ||
             PyTuple_Check(PySequence_Fast_GET_ITEM(seq, 0));
    capacity = size;
    rows->labels = PyMem_Malloc((size_t) capacity * sizeof(hig_int));
    rows->offset = PyMem_Malloc((size_t) (size + 1) * sizeof(hig_int));
    if (!rows->labels || !rows->offset) {
        PyErr_NoMemory();
        goto done;
    }
    rows->offset[0] = 0;
    rows->max_label = -1;
    for (Py_ssize_t v = 0; v < size; v++) {
        PyObject *entry = PySequence_Fast_GET_ITEM(seq, v);
        const int entry_nested = PyList_Check(entry) || PyTuple_Check(entry);
        Py_ssize_t count, kept;
        hig_int *row;
        if (entry_nested != nested) {
            PyErr_SetString(PyExc_ValueError, "membership cannot mix flat and nested rows");
            goto done;
        }
        count = nested ? PySequence_Fast_GET_SIZE(entry) : 1;
        if (count == 0) {
            PyErr_SetString(PyExc_ValueError, "each membership row must be non-empty");
            goto done;
        }
        if (used + count > capacity) {
            Py_ssize_t grown = capacity;
            hig_int *resized;
            while (grown < used + count) {
                grown = grown > PY_SSIZE_T_MAX / 2 ? used + count : 2 * grown;
            }
            resized = PyMem_Realloc(rows->labels, (size_t) grown * sizeof(hig_int));
            if (!resized) {
                PyErr_NoMemory();
                goto done;
            }
            rows->labels = resized;
            capacity = grown;
        }
        row = rows->labels + used;
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = nested ? (PyList_Check(entry) ? PyList_GET_ITEM(entry, i)
                                                           : PyTuple_GET_ITEM(entry, i))
                                    : entry;
            if (read_label(item, &row[i]) < 0) {
                goto done;
            }
        }
        /* Distinct labels in order of first appearance. */
        if (count <= 16) {
            kept = 0;
            for (Py_ssize_t i = 0; i < count; i++) {
                Py_ssize_t j = 0;
                while (j < kept && row[j] != row[i]) {
                    j++;
                }
                if (j == kept) {
                    row[kept++] = row[i];
                }
            }
        } else {
            if (count > scratch_capacity) {
                hig_int *resized = PyMem_Realloc(scratch, (size_t) count * sizeof(hig_int));
                if (!resized) {
                    PyErr_NoMemory();
                    goto done;
                }
                scratch = resized;
                scratch_capacity = count;
            }
            memcpy(scratch, row, (size_t) count * sizeof(hig_int));
            qsort(scratch, (size_t) count, sizeof(hig_int), compare_labels);
            kept = 1;
            for (Py_ssize_t i = 1; i < count; i++) {
                if (scratch[i] != scratch[kept - 1]) {
                    scratch[kept++] = scratch[i];
                }
            }
            /* The row order does not matter to the core (it sorts rows). */
            memcpy(row, scratch, (size_t) kept * sizeof(hig_int));
        }
        if (kept > max_memberships) {
            PyErr_Format(PyExc_ValueError, "a vertex exceeds max_memberships (%zd > %lld)", kept,
                         (long long) max_memberships);
            goto done;
        }
        for (Py_ssize_t i = 0; i < kept; i++) {
            if (row[i] > rows->max_label) {
                rows->max_label = row[i];
            }
        }
        used += kept;
        rows->offset[v + 1] = used;
    }
    /* Contiguous from 0: every label below the largest is used, which needs
     * at least max_label + 1 incidences. */
    if (rows->max_label >= used) {
        PyErr_SetString(PyExc_ValueError, "membership labels must be contiguous from 0");
        goto done;
    }
    present = PyMem_Calloc((size_t) rows->max_label + 1, 1);
    if (!present) {
        PyErr_NoMemory();
        goto done;
    }
    {
        hig_int distinct = 0;
        for (Py_ssize_t i = 0; i < used; i++) {
            if (!present[rows->labels[i]]) {
                present[rows->labels[i]] = 1;
                distinct++;
            }
        }
        if (distinct != rows->max_label + 1) {
            PyErr_SetString(PyExc_ValueError, "membership labels must be contiguous from 0");
            goto done;
        }
    }
    result = 1;
done:
    PyMem_Free(present);
    PyMem_Free(scratch);
    Py_DECREF(seq);
    if (result < 0) {
        start_rows_release(rows);
    }
    return result;
}
