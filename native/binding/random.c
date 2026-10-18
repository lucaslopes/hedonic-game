/*
   hedonic -- the caller's random.Random as the core's generator.
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
 * Random numbers for the core. An exact random.Random (methods not overridden
 * on the instance) is run here: CPython's Mersenne Twister on a copy of its
 * state, written back after the call, so every draw is that of
 * rng.getrandbits(32) without a Python call per draw. Any other generator is
 * called through getrandbits(32).
 */

#include "binding.h"

/* ---- CPython's Mersenne Twister (Modules/_randommodule.c) ----------------- */
/*
 * The generator step below follows MT19937 (genrand_int32 of mt19937ar.c) as
 * CPython's Modules/_randommodule.c runs it. Its original notice:
 *
 *   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 *     3. The names of its contributors may not be used to endorse or promote
 *        products derived from this software without specific prior written
 *        permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define MT_M 397

static uint32_t mt_next(mt_state_t *mt) {
    static const uint32_t mag01[2] = { 0x0U, 0x9908b0dfU };
    uint32_t *s = mt->state, y;

    if (mt->index >= MT_N) {
        int k;
        for (k = 0; k < MT_N - MT_M; k++) {
            y = (s[k] & 0x80000000U) | (s[k + 1] & 0x7fffffffU);
            s[k] = s[k + MT_M] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; k < MT_N - 1; k++) {
            y = (s[k] & 0x80000000U) | (s[k + 1] & 0x7fffffffU);
            s[k] = s[k + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (s[MT_N - 1] & 0x80000000U) | (s[0] & 0x7fffffffU);
        s[MT_N - 1] = s[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
        mt->index = 0;
    }
    y = s[mt->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static uint32_t py_next32(void *state, int *failed) {
    py_rng_state_t *rng = (py_rng_state_t *) state;
    PyObject *result = PyObject_CallFunctionObjArgs(rng->getrandbits, rng->bits, NULL);
    unsigned long value;

    if (!result) {
        *failed = 1;
        return 0;
    }
    value = PyLong_AsUnsignedLong(result);
    Py_DECREF(result);
    if (value == (unsigned long) -1 && PyErr_Occurred()) {
        *failed = 1;
        return 0;
    }
    return (uint32_t) value;
}

static uint32_t mt_next32(void *state, int *failed) {
    (void) failed;
    return mt_next(&((py_rng_state_t *) state)->mt);
}

PyObject *random_type;   /* random.Random */

/* Copies the state of an exact random.Random whose methods are the class's
 * (none set on the instance): 1 when copied, 0 when the generator is to be
 * called instead, -1 on a Python error. */
static int mt_take(PyObject *rng, py_rng_state_t *state) {
    PyObject *dict, *got, *words;
    long version;

    if ((PyObject *) Py_TYPE(rng) != random_type) {
        return 0;
    }
    dict = PyObject_GenericGetDict(rng, NULL);
    if (!dict) {
        PyErr_Clear();
        return 0;
    }
    if (PyDict_GetItemString(dict, "getrandbits") || PyDict_GetItemString(dict, "random") ||
        PyDict_GetItemString(dict, "getstate") || PyDict_GetItemString(dict, "setstate")) {
        Py_DECREF(dict);
        return 0;
    }
    Py_DECREF(dict);
    got = PyObject_CallMethod(rng, "getstate", NULL);
    if (!got) {
        return -1;
    }
    if (!PyTuple_Check(got) || PyTuple_GET_SIZE(got) != 3) {
        Py_DECREF(got);
        return 0;
    }
    version = PyLong_AsLong(PyTuple_GET_ITEM(got, 0));
    words = PyTuple_GET_ITEM(got, 1);
    if (version != 3 || !PyTuple_Check(words) || PyTuple_GET_SIZE(words) != MT_N + 1) {
        PyErr_Clear();
        Py_DECREF(got);
        return 0;
    }
    for (Py_ssize_t i = 0; i <= MT_N; i++) {
        const unsigned long value = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(words, i));
        if (PyErr_Occurred() || (i < MT_N ? value > 0xffffffffUL : value > MT_N)) {
            PyErr_Clear();
            Py_DECREF(got);
            return 0;
        }
        if (i < MT_N) {
            state->mt.state[i] = (uint32_t) value;
        } else {
            state->mt.index = (int) value;
        }
    }
    state->gauss_next = Py_NewRef(PyTuple_GET_ITEM(got, 2));
    state->rng = Py_NewRef(rng);
    Py_DECREF(got);
    return 1;
}

/* Writes the advanced state back, so the generator ends where the same draws
 * through its methods would have left it. */
static int mt_give_back(py_rng_state_t *state) {
    PyObject *words = PyTuple_New(MT_N + 1), *full, *done;

    if (!words) {
        return -1;
    }
    for (Py_ssize_t i = 0; i <= MT_N; i++) {
        PyObject *value = PyLong_FromUnsignedLong(i < MT_N ? state->mt.state[i]
                                                           : (unsigned long) state->mt.index);
        if (!value) {
            Py_DECREF(words);
            return -1;
        }
        PyTuple_SET_ITEM(words, i, value);
    }
    full = Py_BuildValue("(iNO)", 3, words, state->gauss_next);
    if (!full) {
        return -1;
    }
    done = PyObject_CallMethod(state->rng, "setstate", "(N)", full);
    if (!done) {
        return -1;
    }
    Py_DECREF(done);
    return 0;
}

int make_rng(PyObject *rng, py_rng_state_t *state) {
    const int native = mt_take(rng, state);

    if (native != 0) {
        return native < 0 ? -1 : 0;
    }
    state->getrandbits = PyObject_GetAttrString(rng, "getrandbits");
    if (!state->getrandbits) {
        return -1;
    }
    state->bits = PyLong_FromLong(32);
    if (!state->bits) {
        Py_CLEAR(state->getrandbits);
        return -1;
    }
    return 0;
}

/* The core's view of a Python generator: getrandbits(32). */
hig_rng_t rng_callbacks(py_rng_state_t *state) {
    if (state->rng) {
        return (hig_rng_t) { mt_next32, state };
    }
    return (hig_rng_t) { py_next32, state };
}

/* Ends the call's use of the generator: a copied state goes back to it,
 * also after an error (whose exception is kept). Returns -1 when that write
 * itself fails. */
int rng_release(py_rng_state_t *state) {
    int status = 0;

    if (state->rng) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        status = mt_give_back(state);
        if (type) {
            if (status < 0) {
                PyErr_Clear();
            }
            PyErr_Restore(type, value, traceback);
        }
    }
    Py_CLEAR(state->getrandbits);
    Py_CLEAR(state->bits);
    Py_CLEAR(state->rng);
    Py_CLEAR(state->gauss_next);
    return status;
}
