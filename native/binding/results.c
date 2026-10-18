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
