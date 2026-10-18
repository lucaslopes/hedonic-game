/*
   hedonic -- the visiting queue.
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

#include "internal.h"

#include <stdlib.h>

/* ---- The visiting queue ----------------------------------------------------- */

HIG_SHARED void hig_queue_free(hig_queue_t *queue) {
    free(queue->ring);
    free(queue->queued);
    free(queue->order);
    free(queue->settled);
    queue->ring = queue->order = NULL;
    queue->queued = NULL;
    queue->settled = NULL;
}

HIG_SHARED hig_error_t hig_queue_init(hig_queue_t *queue, hig_id n, const hig_rng_t *rng,
                                      const hig_interrupt_t *interrupt) {
    hig_error_t status = HIG_SUCCESS;

    queue->n = n;
    queue->head = queue->count = queue->moves = queue->visits = 0;
    queue->rng = rng;
    queue->interrupt = interrupt;
    HIG_TRY(hig_alloc(&queue->ring, n, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&queue->queued, n, sizeof(uint8_t), true));
    HIG_TRY(hig_alloc(&queue->order, n, sizeof(hig_id), false));
    HIG_TRY(hig_alloc(&queue->settled, n, sizeof(hig_settled_t), false));
    for (hig_id v = 0; v < n; v++) {
        queue->order[v] = v;
        queue->settled[v].at = -1;
        queue->settled[v].bound = 0.0;
    }
    /* Fisher-Yates. */
    for (hig_id i = n - 1; i > 0; i--) {
        uint32_t j = 0;
        hig_id swap;
        HIG_TRY(hig_random_below(rng, (uint32_t) i + 1, &j));
        swap = queue->order[i];
        queue->order[i] = queue->order[j];
        queue->order[j] = swap;
    }
    for (hig_id i = 0; i < n; i++) {
        hig_queue_push(queue, queue->order[i]);
    }
done:
    return status;
}

HIG_SHARED hig_error_t hig_queue_next(hig_queue_t *queue, hig_id *v, bool *done) {
    if (queue->count == 0) {
        /* The unsettled vertices, in a fresh random order (Fisher-Yates
         * over them only; native/README.md, "The equilibrium certificate
         * and its slack", on why one fixed order is slow). */
        queue->head = 0;
        for (hig_id i = 0; i < queue->n; i++) {
            const hig_id u = queue->order[i];
            if (queue->settled[u].at != queue->moves) {
                hig_queue_push(queue, u);
            }
        }
        if (queue->count == 0) {
            *done = true;
            return HIG_SUCCESS;
        }
        for (int64_t i = queue->count - 1; i > 0; i--) {
            uint32_t j = 0;
            hig_id swap;
            hig_error_t status = hig_random_below(queue->rng, (uint32_t) i + 1, &j);
            if (status != HIG_SUCCESS) {
                return status;
            }
            swap = queue->ring[i];
            queue->ring[i] = queue->ring[j];
            queue->ring[j] = swap;
        }
    }
    *done = false;
    *v = queue->ring[queue->head];
    queue->head = queue->head + 1 == queue->n ? 0 : queue->head + 1;
    queue->count--;
    queue->queued[*v] = 0;
    if (++queue->visits % HIG_VISITS_PER_INTERRUPTION_CHECK == 0 && queue->interrupt &&
        queue->interrupt->check && queue->interrupt->check(queue->interrupt->state)) {
        return hig_error(HIG_INTERRUPTED, "Interrupted.");
    }
    return HIG_SUCCESS;
}
