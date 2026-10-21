/*
 * The C core alone on a perflog input (native/rust/bench/perflog.py writes it and
 * builds this file): the input check, the adjacency build and the whole call
 * timed apart, and the core's own heap (live peak, bytes and count of
 * allocations) counted by wrappers around malloc, calloc, realloc and free
 * that only this program's copy of the core sees. The Rust counterpart is
 * native/rust/examples/perflog_core.rs; both print one JSON line. Benchmarks only.
 *
 *     cc -std=c11 -O3 -DNDEBUG -ffp-contract=off -fno-wrapv -Inative/c/include \
 *        -Inative/c/core native/rust/bench/perflog_core.c -o perflog_core_c
 *
 * It needs hedonic 2.0's C core in native/c (the paper branch).
 *     perflog_core_c INPUT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <libproc.h>
#include <unistd.h>
#endif

/* Instructions retired by this process so far (proc_pid_rusage, macOS; 0 elsewhere). */
static double instructions(void) {
#ifdef __APPLE__
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *) &ri) == 0) {
        return (double) ri.ri_instructions;
    }
#endif
    return 0;
}

/* ---- Counted allocations: a 16-byte header holds the requested size. ---- */
static size_t live_bytes, peak_bytes, total_bytes, alloc_count;

static void grew(size_t by) {
    live_bytes += by;
    if (live_bytes > peak_bytes) {
        peak_bytes = live_bytes;
    }
    total_bytes += by;
    alloc_count++;
}

static void *pl_malloc(size_t size) {
    char *p = malloc(size + 16);
    if (!p) {
        return NULL;
    }
    *(size_t *) p = size;
    grew(size);
    return p + 16;
}

static void *pl_calloc(size_t count, size_t size) {
    char *p;
    if (size && count > ((size_t) -1 - 16) / size) {
        return NULL;
    }
    p = calloc(1, count * size + 16);
    if (!p) {
        return NULL;
    }
    *(size_t *) p = count * size;
    grew(count * size);
    return p + 16;
}

static void pl_free(void *q) {
    char *p;
    if (!q) {
        return;
    }
    p = (char *) q - 16;
    live_bytes -= *(size_t *) p;
    free(p);
}

static void *pl_realloc(void *q, size_t size) {
    char *p, *r;
    size_t old;
    if (!q) {
        return pl_malloc(size);
    }
    p = (char *) q - 16;
    old = *(size_t *) p;
    r = realloc(p, size + 16);
    if (!r) {
        return NULL;
    }
    *(size_t *) r = size;
    live_bytes -= old;
    grew(size);
    return r + 16;
}

#define malloc pl_malloc
#define calloc pl_calloc
#define realloc pl_realloc
#define free pl_free
#include "../../c/unity.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

/* ---- CPython's Mersenne Twister (genrand_uint32 of _randommodule.c). ---- */
typedef struct {
    uint32_t s[624];
    int index;
} mt_t;

static uint32_t mt_next(void *state, int *failed) {
    static const uint32_t mag01[2] = { 0x0U, 0x9908b0dfU };
    mt_t *mt = state;
    uint32_t *s = mt->s, y;
    int k;
    (void) failed;
    if (mt->index >= 624) {
        for (k = 0; k < 624 - 397; k++) {
            y = (s[k] & 0x80000000U) | (s[k + 1] & 0x7fffffffU);
            s[k] = s[k + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; k < 623; k++) {
            y = (s[k] & 0x80000000U) | (s[k + 1] & 0x7fffffffU);
            s[k] = s[k + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (s[623] & 0x80000000U) | (s[0] & 0x7fffffffU);
        s[623] = s[396] ^ (y >> 1) ^ mag01[y & 0x1U];
        mt->index = 0;
    }
    y = s[mt->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

static uint64_t fnv(uint64_t h, const int64_t *values, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        uint64_t v = (uint64_t) values[i];
        for (int b = 0; b < 8; b++) {
            h = (h ^ ((v >> (8 * b)) & 0xff)) * 0x100000001b3ULL;
        }
    }
    return h;
}

static void print_list(const char *name, const double *xs, int64_t k) {
    printf("\"%s\": [", name);
    for (int64_t i = 0; i < k; i++) {
        printf("%s%.9f", i ? ", " : "", xs[i]);
    }
    printf("], ");
}

static void *slurp(FILE *f, size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p || fread(p, 1, bytes, f) != bytes) {
        fprintf(stderr, "short input\n");
        exit(2);
    }
    return p;
}

/* Counts from here on: the live peak above the live bytes now. */
static size_t window_live, window_total, window_count;

static void window_open(void) {
    window_live = live_bytes;
    peak_bytes = live_bytes;
    window_total = total_bytes;
    window_count = alloc_count;
}

static void window_json(char *out, size_t size) {
    snprintf(out, size, "{\"peak_live_bytes\": %zu, \"total_bytes\": %zu, \"count\": %zu}",
             peak_bytes - window_live, total_bytes - window_total, alloc_count - window_count);
}

int main(int argc, char **argv) {
    FILE *f;
    char magic[8], alloc_adjacency[160] = "null", alloc_call[160] = "null", result[256] = "null";
    int64_t h[16], words[625], *edges, *start = NULL, *offset = NULL, *labels = NULL;
    double fl[4], *ew = NULL, *nw = NULL;
    double *check_s, *adjacency_s, *prepare_s, *call_s, *call_instructions;
    mt_t mt0;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) {
        fprintf(stderr, "perflog_core_c INPUT\n");
        return 2;
    }
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "HPLOG001", 8) || fread(h, 8, 16, f) != 16
            || fread(fl, 8, 4, f) != 4 || fread(words, 8, 625, f) != 625) {
        fprintf(stderr, "bad input\n");
        return 2;
    }
    const int64_t mode = h[0], n = h[1], m = h[2], cap = h[3], start_kind = h[6], reps = h[10];
    for (int i = 0; i < 624; i++) {
        mt0.s[i] = (uint32_t) words[i];
    }
    mt0.index = (int) words[624];
    edges = slurp(f, (size_t) (16 * m));
    if (h[4]) {
        ew = slurp(f, (size_t) (8 * m));
    }
    if (h[5]) {
        nw = slurp(f, (size_t) (8 * n));
    }
    if (start_kind == 1) {
        start = slurp(f, (size_t) (8 * n));
    } else if (start_kind == 2) {
        offset = slurp(f, (size_t) (8 * (n + 1)));
        labels = slurp(f, (size_t) (8 * h[12]));
    }
    fclose(f);
    check_s = malloc(sizeof(double) * (size_t) reps);
    adjacency_s = malloc(sizeof(double) * (size_t) reps);
    prepare_s = malloc(sizeof(double) * (size_t) reps);
    call_s = malloc(sizeof(double) * (size_t) reps);
    call_instructions = malloc(sizeof(double) * (size_t) reps);
    hig_graph_t g = { n, m, edges, NULL, NULL, ew, nw };
    hig_options_t o0 = { fl[0], cap, h[8], h[9], (int) h[7], NULL, NULL };
    int64_t checks = 0, preps = 0;
    for (int64_t rep = 0; rep < reps; rep++) {
        mt_t mt = mt0;
        hig_rng_t rng = { mt_next, &mt };
        hig_options_t o = o0;
        double t;
        o.rng = &rng;
        if (mode != 2) {
            hig_options_t oc = o;
            hig_adjacency_t a;
            memset(&a, 0, sizeof a);
            t = now();
            if (hig_check_input(&g, &oc, mode == 1)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            check_s[checks] = now() - t;
            window_open();
            t = now();
            if (hig_adjacency_build(&a, &g)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            adjacency_s[checks++] = now() - t;
            if (rep == 0) {
                window_json(alloc_adjacency, sizeof alloc_adjacency);
            }
            hig_adjacency_free(&a);
        }
        window_open();
        double i0 = instructions();
        t = now();
        if (mode == 0) {
            int64_t *membership = malloc(sizeof(int64_t) * (size_t) n), count;
            double quality, slack;
            if (start) {
                memcpy(membership, start, sizeof(int64_t) * (size_t) n);
            }
            if (hig_partition(&g, &o, start != NULL, membership, &count, &quality, &slack)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            call_s[rep] = now() - t;
            if (rep == 0) {
                window_json(alloc_call, sizeof alloc_call);
                snprintf(result, sizeof result,
                         "{\"count\": %lld, \"quality\": %.17g, \"slack\": %.17g, \"hash\": \"%016llx\"}",
                         (long long) count, quality, slack,
                         (unsigned long long) fnv(0xcbf29ce484222325ULL, membership, n));
            }
            free(membership);
        } else if (mode == 1) {
            hig_cover_t in = { n, offset, labels }, out;
            int64_t count;
            double quality, slack;
            memset(&out, 0, sizeof out);
            if (hig_cover(&g, &o, offset ? &in : NULL, &out, &count, &quality, &slack)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            call_s[rep] = now() - t;
            if (rep == 0) {
                uint64_t hash = fnv(0xcbf29ce484222325ULL, out.offset, n + 1);
                window_json(alloc_call, sizeof alloc_call);
                hash = fnv(hash, out.labels, out.offset[n]);
                snprintf(result, sizeof result,
                         "{\"count\": %lld, \"quality\": %.17g, \"slack\": %.17g, \"hash\": \"%016llx\"}",
                         (long long) count, quality, slack, (unsigned long long) hash);
            }
            hig_cover_free(&out);
        } else {
            hig_cover_t in = { n, offset, labels };
            hig_audit_state_t *state = NULL;
            hig_audit_options_t ao = HIG_AUDIT_OPTIONS_INIT;
            hig_audit_intervals_t out;
            int64_t *selected = malloc(sizeof(int64_t) * (size_t) (n ? n : 1));
            hig_audit_selection_t sel = { 0, n, selected };
            for (int64_t v = 0; v < n; v++) {
                selected[v] = v;
            }
            if (hig_audit_prepare(&g, &in, NULL, &state)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            prepare_s[preps++] = now() - t;
            ao.max_memberships = cap;
            ao.allow_isolation = (int) h[7];
            ao.prune = (int) h[11];
            ao.atol = fl[1];
            ao.rtol = fl[2];
            memset(&out, 0, sizeof out);
            t = now();
            if (hig_audit_intervals(state, &ao, sel, &out)) {
                fprintf(stderr, "%s\n", hig_error_message());
                return 1;
            }
            call_s[rep] = now() - t;
            if (rep == 0) {
                window_json(alloc_call, sizeof alloc_call);
                snprintf(result, sizeof result, "{\"count\": %lld, \"hash\": \"%016llx\"}",
                         (long long) out.interval_count,
                         (unsigned long long) fnv(0xcbf29ce484222325ULL, (const int64_t *) out.intervals,
                                                  2 * out.interval_count));
            }
            hig_audit_intervals_free(&out);
            hig_audit_free(state);
            free(selected);
        }
        call_instructions[rep] = instructions() - i0;
    }
    printf("{\"core\": \"c\", \"mode\": %lld, ", (long long) mode);
    print_list("check_s", check_s, checks);
    print_list("adjacency_s", adjacency_s, checks);
    print_list("prepare_s", prepare_s, preps);
    print_list("call_s", call_s, reps);
    printf("\"call_instructions\": [");
    for (int64_t i = 0; i < reps; i++) {
        printf("%s%.0f", i ? ", " : "", call_instructions[i]);
    }
    printf("], ");
    printf("\"alloc_adjacency\": %s, \"alloc_call\": %s, \"result\": %s}\n", alloc_adjacency, alloc_call, result);
    return 0;
}
