/*
   hedonic -- tests of the C core.
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
 * The C core alone (no Python): every returned partition of a small game is
 * checked against an independent brute-force best response computed from the
 * definition of the payoffs (every community), together with the count
 * limits, the structure of the result and the returned quality. Then fixed points, determinism, callbacks and error
 * paths, and first the soft fma (core/softfma.h) against the FMA instruction.
 * Meant to run under ASan/UBSan (make test) and at -O3 (make test-release);
 * an argument multiplies the number of random games, HIG_FMA_CASES sets the
 * number of soft fma cases (default 10^6, whose results are pinned by a
 * hash), and HIG_TEST_FMA=soft|instruction selects the build of an x86_64
 * dispatch target (core/dispatch.h).
 */

#include "hedonic_core.h"
#include "hedonic_check.h"   /* the traced entry points and the verifier */
#include "softfma.h"         /* the core's soft fma, compared with the instruction */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        checks++;                                                            \
        if (!(cond)) {                                                       \
            failures++;                                                      \
            fprintf(stderr, "%s:%d: CHECK(%s) failed: ", __FILE__, __LINE__, \
                    #cond);                                                  \
            fprintf(stderr, __VA_ARGS__);                                    \
            fprintf(stderr, "\n");                                           \
        }                                                                    \
    } while (0)

/* Zachary's karate club, 34 vertices, 78 edges. */
static const hig_int ZACHARY[] = {
    0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 10, 0, 11, 0, 12, 0, 13,
    0, 17, 0, 19, 0, 21, 0, 31, 1, 2, 1, 3, 1, 7, 1, 13, 1, 17, 1, 19, 1, 21,
    1, 30, 2, 3, 2, 7, 2, 27, 2, 28, 2, 32, 2, 9, 2, 8, 2, 13, 3, 7, 3, 12,
    3, 13, 4, 6, 4, 10, 5, 6, 5, 10, 5, 16, 6, 16, 8, 30, 8, 32, 8, 33, 9, 33,
    13, 33, 14, 32, 14, 33, 15, 32, 15, 33, 18, 32, 18, 33, 19, 33, 20, 32,
    20, 33, 22, 32, 22, 33, 23, 25, 23, 27, 23, 32, 23, 33, 23, 29, 24, 25,
    24, 27, 24, 31, 25, 31, 26, 29, 26, 33, 27, 33, 28, 31, 28, 33, 29, 32,
    29, 33, 30, 32, 30, 33, 31, 32, 31, 33, 32, 33};

/* ---- Random numbers for the tests ------------------------------------------- */

static uint64_t test_state = 0x9e3779b97f4a7c15ULL;

static uint64_t next64(void) {
    uint64_t z = (test_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double uniform(void) {
    return (double) (next64() >> 11) / 9007199254740992.0;   /* 2^53 */
}

static hig_int below(hig_int n) {
    return (hig_int) (next64() % (uint64_t) n);
}

static hig_options_t options(double resolution, int allow_isolation, const hig_rng_t *rng) {
    hig_options_t o;
    memset(&o, 0, sizeof o);
    o.resolution = resolution;
    o.max_total_communities = -1;
    o.n_communities = -1;
    o.allow_isolation = allow_isolation;
    o.rng = rng;
    return o;
}

typedef struct {
    hig_pcg32_t pcg;
    hig_rng_t rng;
} test_rng_t;

static void seed_rng(test_rng_t *r, uint64_t seed) {
    hig_pcg32_seed(&r->pcg, seed, 54);
    r->rng.next32 = hig_pcg32_next32;
    r->rng.state = &r->pcg;
}

/* CPython's random.Random(seed) for a seed below 2^32: MT19937 seeded by
 * init_by_array({seed}) (mt19937ar.c, as Modules/_randommodule.c seeds an
 * int), drawn 32 bits at a time like getrandbits(32). The core reads its
 * random words only through next32, so these runs are the ones
 * Game.community_hedonic(seed=...) makes, and the pins below are the
 * Python pins of tests/test_native_*.py. */
typedef struct {
    uint32_t state[624];
    int index;
    hig_rng_t rng;
} test_mt_t;

static uint32_t mt_next32(void *state, int *failed) {
    test_mt_t *mt = state;
    uint32_t *s = mt->state, y;
    (void) failed;
    if (mt->index >= 624) {
        for (int k = 0; k < 624; k++) {
            y = (s[k] & 0x80000000U) | (s[(k + 1) % 624] & 0x7fffffffU);
            s[k] = s[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        }
        mt->index = 0;
    }
    y = s[mt->index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    return y ^ (y >> 18);
}

static void seed_mt(test_mt_t *mt, uint32_t seed) {
    uint32_t *s = mt->state;
    int i = 1, j = 0;
    s[0] = 19650218U;
    for (int k = 1; k < 624; k++) {
        s[k] = 1812433253U * (s[k - 1] ^ (s[k - 1] >> 30)) + (uint32_t) k;
    }
    for (int k = 624; k > 0; k--) {   /* init_by_array with the one-word key {seed} */
        s[i] = (s[i] ^ ((s[i - 1] ^ (s[i - 1] >> 30)) * 1664525U)) + seed + (uint32_t) j;
        i++;
        j = 0;
        if (i >= 624) {
            s[0] = s[623];
            i = 1;
        }
    }
    for (int k = 623; k > 0; k--) {
        s[i] = (s[i] ^ ((s[i - 1] ^ (s[i - 1] >> 30)) * 1566083941U)) - (uint32_t) i;
        i++;
        if (i >= 624) {
            s[0] = s[623];
            i = 1;
        }
    }
    s[0] = 0x80000000U;
    mt->index = 624;
    mt->rng.next32 = mt_next32;
    mt->rng.state = mt;
}

/* ---- Brute-force best responses --------------------------------------------- */

static double node_weight(const hig_graph_t *g, hig_int v) {
    return g->node_weights ? g->node_weights[v] : 1.0;
}

static double edge_weight(const hig_graph_t *g, hig_int e) {
    return g->edge_weights ? g->edge_weights[e] : 1.0;
}

/* A slack far above the core's tolerance and far below any real gain. */
static double slack(const hig_graph_t *g, hig_int v, double gamma) {
    double a = 0.0, total = 0.0;
    for (hig_int e = 0; e < g->edge_count; e++) {
        if (g->edges[2 * e] == v || g->edges[2 * e + 1] == v) {
            a += fabs(edge_weight(g, e));
        }
    }
    for (hig_int u = 0; u < g->vertex_count; u++) {
        total += node_weight(g, u);
    }
    return 1e-9 * (1.0 + a + fabs(gamma) * node_weight(g, v) * total);
}

/* The largest gain of a feasible unilateral move in a partition, from the
 * definition u_v(C) = w(v, C \ v) - gamma n_v N(C \ v), in units of the
 * vertex's slack: at most 1 at an equilibrium. */
static double partition_regret(const hig_graph_t *g, const hig_int *membership, hig_int k,
                               const hig_options_t *o) {
    const hig_int n = g->vertex_count;
    double *to = calloc((size_t) k, sizeof *to), *mass = calloc((size_t) k, sizeof *mass);
    hig_int *size = calloc((size_t) k, sizeof *size);
    const int exact = o->n_communities > 0, limited = o->max_total_communities > 0;
    double worst = 0.0;

    for (hig_int v = 0; v < n; v++) {
        mass[membership[v]] += node_weight(g, v);
        size[membership[v]]++;
    }
    for (hig_int v = 0; v < n; v++) {
        const double nv = node_weight(g, v);
        const hig_int own = membership[v];
        const int alone = size[own] == 1;
        double current, best;
        if (exact && alone) {
            continue;   /* leaving would empty a community */
        }
        memset(to, 0, (size_t) k * sizeof *to);
        for (hig_int e = 0; e < g->edge_count; e++) {
            const hig_int a = g->edges[2 * e], b = g->edges[2 * e + 1];
            if (a == v && b != v) to[membership[b]] += edge_weight(g, e);
            if (b == v && a != v) to[membership[a]] += edge_weight(g, e);
        }
        current = to[own] - o->resolution * nv * (mass[own] - nv);
        best = current;
        for (hig_int c = 0; c < k; c++) {
            if (c != own) {
                const double u = to[c] - o->resolution * nv * mass[c];
                if (u > best) best = u;
            }
        }
        if (o->allow_isolation && !exact && !alone &&
            (!limited || k < o->max_total_communities) && 0.0 > best) {
            best = 0.0;
        }
        if ((best - current) / slack(g, v, o->resolution) > worst) {
            worst = (best - current) / slack(g, v, o->resolution);
        }
    }
    free(to);
    free(mass);
    free(size);
    return worst;
}

/* Potential / W from the definitions. */
static double partition_potential(const hig_graph_t *g, const hig_int *membership, hig_int k,
                                  double gamma) {
    double q = 0.0, total = 0.0, *mass = calloc((size_t) k, sizeof *mass);
    for (hig_int e = 0; e < g->edge_count; e++) {
        total += edge_weight(g, e);
        if (membership[g->edges[2 * e]] == membership[g->edges[2 * e + 1]]) {
            q += 2.0 * edge_weight(g, e);
        }
    }
    for (hig_int v = 0; v < g->vertex_count; v++) mass[membership[v]] += node_weight(g, v);
    for (hig_int c = 0; c < k; c++) q -= gamma * mass[c] * mass[c];
    free(mass);
    return q / (2.0 * total);
}

static int close_to(double a, double b) {
    return fabs(a - b) <= 1e-10 * (1.0 + fabs(a) + fabs(b));
}

/* Labels 0 .. k-1 in order of first appearance, all used. */
static int partition_well_formed(const hig_int *membership, hig_int n, hig_int k) {
    hig_int next = 0;
    for (hig_int v = 0; v < n; v++) {
        if (membership[v] < 0 || membership[v] > next) return 0;
        if (membership[v] == next) next++;
    }
    return next == k;
}

/* ---- Random small games ------------------------------------------------------ */

#define SMALL_N 10
#define SMALL_M 64

typedef struct {
    hig_int n, m;
    hig_int edges[2 * SMALL_M];
    double edge_weights[SMALL_M];
    double node_weights[SMALL_N];
    hig_graph_t graph;
} small_graph_t;

/* A random graph of 2 .. largest vertices. */
static void random_graph(small_graph_t *s, hig_int largest) {
    const double p = 0.15 + 0.6 * uniform();
    const int signed_weights = below(4) == 0;
    const int weighted = signed_weights || below(3) == 0, node_weighted = below(3) == 0;
    s->n = 2 + below(largest - 1);
    do {
        s->m = 0;
        for (hig_int a = 0; a < s->n; a++) {
            /* The odd self-loop. */
            for (hig_int b = a + (below(8) != 0); b < s->n; b++) {
                if (uniform() < p && s->m < SMALL_M - 1) {
                    s->edges[2 * s->m] = a;
                    s->edges[2 * s->m + 1] = b;
                    s->edge_weights[s->m] = signed_weights ? uniform() * 3.0 - 1.0
                                            : below(5) == 0 ? (double) (1 + below(3))
                                                            : 0.1 + 2.0 * uniform();
                    s->m++;
                }
            }
        }
        if (s->m > 0 && below(5) == 0) {   /* a multi-edge */
            s->edges[2 * s->m] = s->edges[0];
            s->edges[2 * s->m + 1] = s->edges[1];
            s->edge_weights[s->m] = s->edge_weights[0];
            s->m++;
        }
    } while (s->m == 0);
    for (hig_int v = 0; v < s->n; v++) {
        s->node_weights[v] = below(6) == 0 ? 0.0 : 0.2 + 2.0 * uniform();
    }
    memset(&s->graph, 0, sizeof s->graph);
    s->graph.vertex_count = s->n;
    s->graph.edge_count = s->m;
    s->graph.edges = s->edges;
    s->graph.edge_weights = weighted ? s->edge_weights : NULL;
    s->graph.node_weights = node_weighted ? s->node_weights : NULL;
}

static double total_edge_weight(const hig_graph_t *g) {
    double w = 0.0;
    for (hig_int e = 0; e < g->edge_count; e++) w += edge_weight(g, e);
    return w;
}

static const double RESOLUTIONS[] = {-0.7, -0.05, 0.0, 0.02, 0.1, 0.35, 0.8, 2.0};

static void random_limits(hig_options_t *o, hig_int n) {
    switch (below(4)) {
    case 0:
        o->max_total_communities = 1 + below(n);
        break;
    case 1:
        o->n_communities = 1 + below(n);
        break;
    default:
        break;
    }
}

static hig_int start_labels(const hig_options_t *o, hig_int n) {
    return o->n_communities > 0 ? o->n_communities
           : o->max_total_communities > 0 ? o->max_total_communities
           : n;
}

/* The independent verifier (check.c) on a result: no deviation it can
 * certify gains more than the raw slack the engine reported. */
static hig_int verified = 0, improvable = 0;

static void verify(const hig_graph_t *g, const hig_options_t *o, const hig_int *membership,
                   double raw_slack, int i) {
    hig_verify_t report;
    const hig_error_t status = hig_verify_partition(g, o, membership, &report);
    CHECK(status == HIG_SUCCESS, "case %d: verifier: %s", i, hig_error_message());
    if (status != HIG_SUCCESS) return;
    CHECK(report.max_certified <= raw_slack, "case %d: the verifier certifies a gain %a above "
          "the slack %a", i, report.max_certified, raw_slack);
    verified += report.vertices;
    improvable += report.improvable;
}

/* Unit weights at a resolution whose products are exact: the slack is 0. */
static int exact_arithmetic(const hig_graph_t *g, double gamma) {
    return !g->edge_weights && !g->node_weights && (gamma == 0.0 || gamma == 2.0);
}

static void test_random_partitions(int cases) {
    int run = 0, not_equilibria = 0, exact = 0;
    hig_int membership[SMALL_N];

    for (int i = 0; i < cases; i++) {
        small_graph_t s;
        test_rng_t r;
        hig_options_t o;
        hig_int k = 0;
        double quality = 0.0, raw_slack = -1.0, regret;
        const int start = below(3) == 0;
        hig_error_t status;
        random_graph(&s, SMALL_N);
        seed_rng(&r, (uint64_t) i);
        o = options(RESOLUTIONS[below(8)], (int) below(2), &r.rng);
        random_limits(&o, s.n);
        if (!(total_edge_weight(&s.graph) > 0.0)) continue;
        if (start) {
            const hig_int labels = start_labels(&o, s.n);
            for (hig_int v = 0; v < s.n; v++) membership[v] = 3 * (v < labels ? v : below(labels));
        }
        status = hig_partition_traced(&s.graph, &o, start, membership, &k, &quality, &raw_slack,
                                      NULL);
        CHECK(status == HIG_SUCCESS, "case %d: %s", i, hig_error_message());
        if (status != HIG_SUCCESS) continue;
        run++;
        CHECK(partition_well_formed(membership, s.n, k), "case %d", i);
        CHECK(o.n_communities <= 0 || k == o.n_communities, "case %d: k=%lld", i, (long long) k);
        CHECK(o.max_total_communities <= 0 || k <= o.max_total_communities, "case %d", i);
        CHECK(close_to(quality, partition_potential(&s.graph, membership, k, o.resolution)),
              "case %d: quality %.17g vs %.17g", i, quality,
              partition_potential(&s.graph, membership, k, o.resolution));
        regret = partition_regret(&s.graph, membership, k, &o);
        not_equilibria += regret > 1.0;
        CHECK(regret <= 1.0, "case %d: partition regret %g slacks (gamma %g, isolation %d)", i,
              regret, o.resolution, o.allow_isolation);
        CHECK(raw_slack >= 0.0, "case %d: slack %g", i, raw_slack);
        if (exact_arithmetic(&s.graph, o.resolution)) {
            exact++;
            CHECK(raw_slack == 0.0, "case %d: slack %a in exact arithmetic", i, raw_slack);
        }
        verify(&s.graph, &o, membership, raw_slack, i);
    }
    printf("random partitions: %d games checked by brute force, %d not at equilibrium, "
           "%d in exact arithmetic (slack 0)\n", run, not_equilibria, exact);
}

/* ---- Fixed games -------------------------------------------------------------- */

static hig_graph_t zachary(void) {
    hig_graph_t g;
    memset(&g, 0, sizeof g);
    g.vertex_count = 34;
    g.edge_count = 78;
    g.edges = ZACHARY;
    return g;
}

static void test_zachary(void) {
    const hig_graph_t g = zachary();
    const double density = 78.0 / (34.0 * 33.0 / 2.0);
    hig_int membership[34], again[34], k = 0, k2 = 0;
    double quality = 0.0, quality2 = 0.0;
    test_rng_t r;
    hig_options_t o;

    for (int isolation = 0; isolation <= 1; isolation++) {
        seed_rng(&r, 0);
        o = options(density, isolation, &r.rng);
        CHECK(hig_partition(&g, &o, 0, membership, &k, &quality, NULL) == HIG_SUCCESS, "%s",
              hig_error_message());
        CHECK(partition_regret(&g, membership, k, &o) <= 1.0, "isolation %d", isolation);
        CHECK(k >= 2 && k <= 8, "k = %lld", (long long) k);
        CHECK(quality > 0.3, "quality %g", quality);
        /* The same random words give the same result. */
        seed_rng(&r, 0);
        CHECK(hig_partition(&g, &o, 0, again, &k2, &quality2, NULL) == HIG_SUCCESS, "%s",
              hig_error_message());
        CHECK(k2 == k && quality2 == quality && memcmp(again, membership, sizeof again) == 0,
              "not deterministic");
        /* An equilibrium is a fixed point: started there, nothing moves. */
        memcpy(again, membership, sizeof again);
        seed_rng(&r, 99);
        CHECK(hig_partition(&g, &o, 1, again, &k2, &quality2, NULL) == HIG_SUCCESS, "%s",
              hig_error_message());
        CHECK(memcmp(again, membership, sizeof again) == 0 && quality2 == quality,
              "an equilibrium start moved");
    }
}

/* ---- Bit pins ------------------------------------------------------------------ */
/*
 * Results recorded on macOS arm64 (Apple clang -O3) that every platform must
 * reproduce bit for bit, from the seeded order of random.Random(seed) (the
 * same values are pinned in tests/test_native_core.py and
 * tests/test_native_backend.py). A partition is pinned as its labels in
 * order of first appearance, with the hex of the quality and of the slack.
 * HIG_PRINT_PINS=1 prints the values instead of checking them.
 */

typedef struct {
    const char *name;
    uint32_t seed;
    double resolution;        /* DENSITY: the graph density */
    int isolation, weighted, node_weighted;
    hig_int max_total, exact;
    hig_int k;
    double quality, slack;    /* the slack is the raw (unnormalized) one */
    hig_int membership[34];
} pin_t;

#define DENSITY 1e300

static const pin_t PINS[] = {
    {"zachary partition seed 0", 0, DENSITY, 0, 0, 0, -1, -1, 5, 0x1.b8f407b8f407dp-2, 0x0p+0,
     {0, 0, 0, 0, 1, 2, 2, 0, 3, 0, 1, 1, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 4, 4, 3, 4, 4, 3, 3, 4, 3, 3}},
    {"weighted partition gamma 0.2 isolation seed 1", 1, 0x1.999999999999ap-3, 1, 1, 0, -1, -1, 10, 0x1.96f439aec498ep-2, 0x1.0c0000000000ap-50,
     {0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 1, 4, 0, 0, 5, 5, 1, 0, 6, 0, 6, 7, 5, 8, 9, 9, 8, 6, 6, 8, 2, 6, 5, 6}},
    {"exactly 3 communities gamma 0.1 seed 3", 3, 0x1.999999999999ap-4, 0, 0, 0, -1, 3, 3, 0x1.01f81f81f81f8p-1, 0x0p+0,
     {0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2}},
    {"exact arithmetic partition gamma 0.5 isolation seed 7", 7, 0x1p-1, 1, 0, 0, -1, -1, 19, 0x1.20d20d20d20d2p-4, 0x0p+0,
     {0, 0, 0, 0, 1, 2, 2, 0, 3, 4, 1, 5, 6, 0, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 15, 18, 17, 3, 16, 3, 3}},
};

/* The pins' edge and node weights: dyadic, so the inputs are the same
 * everywhere. */
static double pin_edge_weight(hig_int e) {
    return 0.5 + 0.25 * (double) (e % 7);
}

static double pin_node_weight(hig_int v) {
    return 0.5 * (double) (1 + v % 4);
}

static void test_pins(void) {
    const int print = getenv("HIG_PRINT_PINS") != NULL;
    double ew[78], nw[34];

    for (hig_int e = 0; e < 78; e++) ew[e] = pin_edge_weight(e);
    for (hig_int v = 0; v < 34; v++) nw[v] = pin_node_weight(v);
    for (size_t p = 0; p < sizeof PINS / sizeof PINS[0]; p++) {
        const pin_t *pin = &PINS[p];
        hig_graph_t g = zachary();
        hig_options_t o;
        test_mt_t mt;
        hig_int k = 0, membership[34];
        hig_int names[34], next = 0;
        double quality = 0.0, slack = -1.0;
        int same = 1;

        g.edge_weights = pin->weighted ? ew : NULL;
        g.node_weights = pin->node_weighted ? nw : NULL;
        seed_mt(&mt, pin->seed);
        o = options(pin->resolution == DENSITY ? 78.0 / 561.0 : pin->resolution, pin->isolation,
                    &mt.rng);
        o.max_total_communities = pin->max_total;
        o.n_communities = pin->exact;
        CHECK(hig_partition_traced(&g, &o, 0, membership, &k, &quality, &slack, NULL) ==
              HIG_SUCCESS,
              "%s: %s", pin->name, hig_error_message());
        for (hig_int c = 0; c < 34; c++) names[c] = -1;
        for (hig_int v = 0; v < 34; v++) {
            if (names[membership[v]] < 0) names[membership[v]] = next++;
            membership[v] = names[membership[v]];
            same &= membership[v] == pin->membership[v];
        }
        if (print) {
            printf("    {\"%s\", %u, %a, %d, %d, %d, %lld, %lld, %lld, %a, %a,\n     {",
                   pin->name, pin->seed, pin->resolution, pin->isolation, pin->weighted,
                   pin->node_weighted, (long long) pin->max_total, (long long) pin->exact,
                   (long long) k, quality, slack);
            for (hig_int v = 0; v < 34; v++) printf("%s%lld", v ? ", " : "", (long long) membership[v]);
            printf("}},\n");
            continue;
        }
        CHECK(k == pin->k && quality == pin->quality && slack == pin->slack && same,
              "%s: k %lld quality %a slack %a", pin->name, (long long) k, quality, slack);
    }
}

/* The two edge layouts give the same game. */
static void test_edge_layouts(void) {
    hig_graph_t g = zachary(), split = zachary();
    hig_int first[78], second[78], a[34], b[34], ka = 0, kb = 0;
    test_rng_t r;
    hig_options_t o;
    double qa = 0.0, qb = 1.0;

    for (int e = 0; e < 78; e++) {
        first[e] = ZACHARY[2 * e];
        second[e] = ZACHARY[2 * e + 1];
    }
    split.edges = NULL;
    split.edge_first = first;
    split.edge_second = second;
    seed_rng(&r, 3);
    o = options(0.1, 0, &r.rng);
    CHECK(hig_partition(&g, &o, 0, a, &ka, &qa, NULL) == HIG_SUCCESS, "%s", hig_error_message());
    seed_rng(&r, 3);
    CHECK(hig_partition(&split, &o, 0, b, &kb, &qb, NULL) == HIG_SUCCESS, "%s", hig_error_message());
    CHECK(ka == kb && qa == qb && memcmp(a, b, sizeof a) == 0, "partition layouts differ");
}

/* A game on which the earlier core cycled between equal-payoff moves: it
 * must stop, at an equilibrium, from every order. */
static void test_former_tie_cycle(void) {
    static const hig_int edges[] = {9, 20, 20, 19, 9, 3, 8, 11, 4, 10, 10, 17, 16, 7, 0, 17,
                                    20, 5, 3, 15, 2, 8, 15, 13, 17, 13, 21, 20, 8, 4};
    static const double nw[] = {0.5, 1.3, 1.7, 0.7, 1.9, 1.3, 1.3, 0.7, 0.9, 1.7, 2.3,
                                1.7, 0.9, 0.7, 1.1, 1.9, 2.1, 0.7, 1.5, 1.9, 0.9, 1.7};
    hig_graph_t g;
    hig_int membership[22], k = 0;
    test_rng_t r;
    hig_options_t o;

    memset(&g, 0, sizeof g);
    g.vertex_count = 22;
    g.edge_count = 15;
    g.edges = edges;
    g.node_weights = nw;
    for (uint64_t seed = 0; seed < 50; seed++) {
        for (int isolation = 0; isolation <= 1; isolation++) {
            seed_rng(&r, seed);
            o = options(0.342, isolation, &r.rng);
            CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_SUCCESS, "%s",
                  hig_error_message());
            CHECK(partition_regret(&g, membership, k, &o) <= 1.0, "seed %llu",
                  (unsigned long long) seed);
        }
    }
}

/* Count limits on a ring, from the default starts. */
static void test_counts(void) {
    hig_int edges[2 * 12], membership[12], k = 0;
    hig_graph_t g;
    test_rng_t r;
    hig_options_t o;

    for (hig_int v = 0; v < 12; v++) {
        edges[2 * v] = v;
        edges[2 * v + 1] = (v + 1) % 12;
    }
    memset(&g, 0, sizeof g);
    g.vertex_count = 12;
    g.edge_count = 12;
    g.edges = edges;
    for (hig_int target = 1; target <= 12; target++) {
        seed_rng(&r, (uint64_t) target);
        o = options(0.3, 1, &r.rng);
        o.n_communities = target;
        CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_SUCCESS, "%s",
              hig_error_message());
        CHECK(k == target, "target %lld gave %lld", (long long) target, (long long) k);
        CHECK(partition_regret(&g, membership, k, &o) <= 1.0, "target %lld", (long long) target);
    }
}

/* A larger planted partition, mainly for the sanitizers. */
static void test_planted(void) {
    const hig_int n = 1500, blocks = 15;
    hig_int *edges = NULL, m = 0, capacity = 0, k = 0;
    hig_int *membership = malloc((size_t) n * sizeof(hig_int));
    double *weights = NULL, quality = 0.0;
    hig_graph_t g;
    test_rng_t r;
    hig_options_t o;

    for (hig_int a = 0; a < n; a++) {
        for (hig_int b = a + 1; b < n; b++) {
            if (uniform() < (a % blocks == b % blocks ? 0.12 : 0.004)) {
                if (m == capacity) {
                    capacity = capacity ? 2 * capacity : 4096;
                    edges = realloc(edges, (size_t) capacity * 2 * sizeof(hig_int));
                    weights = realloc(weights, (size_t) capacity * sizeof(double));
                }
                edges[2 * m] = a;
                edges[2 * m + 1] = b;
                weights[m] = 0.5 + uniform();
                m++;
            }
        }
    }
    memset(&g, 0, sizeof g);
    g.vertex_count = n;
    g.edge_count = m;
    g.edges = edges;
    for (int weighted = 0; weighted <= 1; weighted++) {
        g.edge_weights = weighted ? weights : NULL;
        for (int isolation = 0; isolation <= 1; isolation++) {
            seed_rng(&r, 11);
            o = options(0.02, isolation, &r.rng);
            CHECK(hig_partition(&g, &o, 0, membership, &k, &quality, NULL) == HIG_SUCCESS, "%s",
                  hig_error_message());
            CHECK(partition_well_formed(membership, n, k), "planted");
            CHECK(close_to(quality, partition_potential(&g, membership, k, 0.02)), "quality");
            CHECK(partition_regret(&g, membership, k, &o) <= 1.0, "planted regret");
            CHECK(k >= blocks / 2 && k <= 4 * blocks, "planted k = %lld", (long long) k);
        }
    }
    free(edges);
    free(weights);
    free(membership);
}

/* ---- Callbacks and errors ---------------------------------------------------- */

static uint32_t failing_next32(void *state, int *failed) {
    (void) state;
    *failed = 1;
    return 0;
}

static int interrupt_now(void *state) {
    (*(int *) state)++;
    return 1;
}

static void test_callbacks(void) {
    const hig_int n = 40000;
    hig_int *edges = malloc((size_t) n * 2 * sizeof(hig_int));
    hig_int *membership = malloc((size_t) n * sizeof(hig_int)), k = 0;
    hig_graph_t g;
    hig_rng_t failing = {failing_next32, NULL};
    hig_interrupt_t interrupt;
    int calls = 0;
    test_rng_t r;
    hig_options_t o;

    for (hig_int v = 0; v < n; v++) {
        edges[2 * v] = v;
        edges[2 * v + 1] = (v + 1) % n;
    }
    memset(&g, 0, sizeof g);
    g.vertex_count = n;
    g.edge_count = n;
    g.edges = edges;
    o = options(0.1, 0, &failing);
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_ECALLBACK, "rng failure");
    interrupt.check = interrupt_now;
    interrupt.state = &calls;
    seed_rng(&r, 1);
    o = options(0.1, 0, &r.rng);
    o.interrupt = &interrupt;
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_INTERRUPTED, "interrupt");
    CHECK(calls == 1, "interrupt called %d times", calls);
    free(edges);
    free(membership);
}

static void test_errors(void) {
    static const hig_int loop[] = {0, 1, 1, 1};
    static const hig_int outside[] = {0, 5};
    static const hig_int double_edge[] = {0, 1, 0, 1};
    static const double negative[] = {1.0, -1.0};
    const double nan_weight[] = {NAN, 1.0};
    hig_graph_t g = zachary(), bad;
    hig_int membership[34], k = 0;
    test_rng_t r;
    hig_options_t o;

    seed_rng(&r, 0);
    o = options(NAN, 0, &r.rng);
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "NaN resolution");
    CHECK(strstr(hig_error_message(), "resolution must be finite") != NULL, "%s",
          hig_error_message());
    o = options(0.1, 0, NULL);
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "no rng");
    o = options(0.1, 0, &r.rng);
    o.n_communities = 35;
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "K above n");
    o.n_communities = 5;
    o.max_total_communities = 4;
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "K above max");
    o = options(1e305, 0, &r.rng);
    CHECK(hig_partition(&g, &o, 0, membership, &k, NULL, NULL) == HIG_EOVERFLOW,
          "overflowing resolution");

    bad = g;
    bad.vertex_count = 2;
    bad.edge_count = 2;
    bad.edges = loop;
    o = options(0.1, 0, &r.rng);
    CHECK(hig_partition(&bad, &o, 0, membership, &k, NULL, NULL) == HIG_SUCCESS,
          "a loop is fine: %s", hig_error_message());
    bad.edge_count = 1;
    bad.edges = outside;
    CHECK(hig_partition(&bad, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "end outside");
    bad.edge_count = 2;
    bad.edges = double_edge;
    bad.edge_weights = negative;
    CHECK(hig_partition(&bad, &o, 0, membership, &k, NULL, NULL) == HIG_SUCCESS,
          "signed weights: %s", hig_error_message());
    bad.edge_weights = nan_weight;
    CHECK(hig_partition(&bad, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "NaN weight");
    bad.edge_weights = NULL;
    bad.node_weights = negative;
    CHECK(hig_partition(&bad, &o, 0, membership, &k, NULL, NULL) == HIG_EINVAL, "negative node weight");

    /* Starts. */
    o = options(0.1, 0, &r.rng);
    for (int v = 0; v < 34; v++) membership[v] = v % 3;
    membership[5] = -1;
    CHECK(hig_partition(&g, &o, 1, membership, &k, NULL, NULL) == HIG_EINVAL, "negative label");
    for (int v = 0; v < 34; v++) membership[v] = v % 3;
    o.max_total_communities = 2;
    CHECK(hig_partition(&g, &o, 1, membership, &k, NULL, NULL) == HIG_EINVAL, "start above max");
    for (int v = 0; v < 34; v++) membership[v] = v % 3;
    o.max_total_communities = -1;
    o.n_communities = 4;
    CHECK(hig_partition(&g, &o, 1, membership, &k, NULL, NULL) == HIG_EINVAL, "start count");
}

/* ---- The soft fma ------------------------------------------------------------- */

/* The FMA instruction, where the test can reach it: always on arm64, on
 * x86_64 when the CPU has it (hig_fma_cpu). */
#if (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
#define TEST_FMA_INSTRUCTION 2      /* always there */
static double instruction_fma(double x, double y, double z) {
    return __builtin_fma(x, y, z);
}
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define TEST_FMA_INSTRUCTION 1      /* if the CPU has it */
__attribute__((target("fma"))) static double instruction_fma(double x, double y, double z) {
    return __builtin_fma(x, y, z);
}
#elif defined(_MSC_VER) && defined(_M_X64) && !defined(_M_ARM64EC)
#include <immintrin.h>
#define TEST_FMA_INSTRUCTION 1
static double instruction_fma(double x, double y, double z) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(x), _mm_set_sd(y), _mm_set_sd(z)));
}
#else
#define TEST_FMA_INSTRUCTION 0
static double instruction_fma(double x, double y, double z) {
    return fma(x, y, z);
}
#endif

static uint64_t fma_state = 0x243f6a8885a308d3ULL;

static uint64_t fma_next(void) {
    uint64_t z = (fma_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double from_bits(uint64_t b) {
    double x;
    memcpy(&x, &b, sizeof x);
    return x;
}

static uint64_t to_bits(double x) {
    uint64_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

static int fma_range(int lo, int hi) {
    return lo + (int) (fma_next() % (uint64_t) (hi - lo + 1));
}

/* A random double with unbiased exponent in [lo, hi] (clamped to the
 * subnormals and the largest finite exponent), random sign and significand. */
static double random_double(int lo, int hi) {
    int e = fma_range(lo, hi) + 1023;
    const uint64_t r = fma_next();
    if (e < 0) e = 0;
    if (e > 2046) e = 2046;
    return from_bits((r & 0x800fffffffffffffULL) | ((uint64_t) e << 52));
}

static double random_sign(double x) {
    return fma_next() & 1 ? -x : x;
}

static double special_value(void) {
    static const double values[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, DBL_MIN, -DBL_MIN,
                                    4.9406564584124654e-324, -4.9406564584124654e-324,
                                    DBL_MAX, -DBL_MAX, 1.0, -1.0, 0x1p-1022 - 0x1p-1074};
    const uint64_t i = fma_next() % 16;
    return i < 14 ? values[i] : random_double(-1080, 1030);
}

#define FMA_CLASSES 12
static const char *const FMA_CLASS_NAMES[FMA_CLASSES] = {
    "random bits", "random doubles", "moderate exponents", "cancellation",
    "exact zero", "signed zeros", "subnormal", "overflow", "specials",
    "halfway (integers)", "exponent gaps", "near half an ulp"};

/* One adversarial case of class cls. */
static void fma_case(int cls, double *x, double *y, double *z) {
    switch (cls) {
    case 0:
        *x = from_bits(fma_next()); *y = from_bits(fma_next()); *z = from_bits(fma_next());
        break;
    case 1:
        *x = random_double(-1080, 1030); *y = random_double(-1080, 1030);
        *z = random_double(-1080, 1030);
        break;
    case 2: {
        int ex = fma_range(-40, 40), ey = fma_range(-40, 40);
        *x = random_double(ex, ex); *y = random_double(ey, ey);
        *z = random_double(ex + ey - 60, ex + ey + 60);
        break;
    }
    case 3: {   /* z = -fl(x y) moved by a few ulps: the result is the product's error */
        int ex = fma_range(-300, 300), ey = fma_range(-300, 300);
        *x = random_double(ex, ex); *y = random_double(ey, ey);
        *z = from_bits(to_bits(-(*x * *y)) + (uint64_t) (int64_t) fma_range(-3, 3));
        break;
    }
    case 4: {   /* short significands: x y exact, z = -x y */
        const double a = (double) (fma_next() >> 38), b = (double) (fma_next() >> 38);
        const int s = fma_range(-500, 500);
        *x = random_sign(ldexp(a, s)); *y = random_sign(ldexp(b, fma_range(-500, 500) - s));
        *z = -(*x * *y);
        break;
    }
    case 5: {   /* zeros, and products that underflow to a signed zero */
        const uint64_t r = fma_next() % 4;
        *x = random_double(-700, -400); *y = random_double(-700, -400);
        *z = random_double(-1080, 100);
        if (r == 0) *x = fma_next() & 1 ? 0.0 : -0.0;
        if (r == 1) *y = fma_next() & 1 ? 0.0 : -0.0;
        if (r >= 2) *z = fma_next() & 1 ? 0.0 : -0.0;
        break;
    }
    case 6: {   /* subnormal products and results */
        const int s = fma_range(-1140, -990), ex = fma_range(-600, -400);
        *x = random_double(ex, ex); *y = random_double(s - ex, s - ex);
        *z = random_sign(random_double(-1080, -1015));
        if (fma_next() & 1) *z = -(*x * *y) + random_sign(random_double(-1080, -1070));
        break;
    }
    case 7: {   /* products near and above the overflow threshold */
        const int s = fma_range(990, 1060), ex = fma_range(500, 520);
        *x = random_double(ex, ex); *y = random_double(s - ex, s - ex);
        *z = random_sign(random_double(1000, 1023));
        break;
    }
    case 8:
        *x = special_value(); *y = special_value(); *z = special_value();
        break;
    case 9: {   /* a b + c in integers, scaled: up to 55 bits, often exactly halfway */
        const double a = (double) ((fma_next() >> 37) | (1ULL << 26));
        const double b = (double) ((fma_next() >> 37) | (1ULL << 26));
        const int wide = (int) (fma_next() % 3);
        const double c = (double) ((int64_t) (fma_next() >> (wide == 2 ? 13 : 41)) -
                                   (wide == 2 ? (int64_t) 1 << 50 : (int64_t) 1 << 22));
        const uint64_t band = fma_next() % 4;   /* anywhere, subnormal results, overflow */
        const int s = band == 2 ? fma_range(-1135, -1060)
                    : band == 3 ? fma_range(940, 990) : fma_range(-1180, 990);
        const int s1 = s / 2;
        *x = random_sign(ldexp(a, s1)); *y = random_sign(ldexp(b, s - s1));
        *z = ldexp(c, s);
        break;
    }
    case 10: {  /* exponent gaps from 54 to 1100 between x y and z */
        const int ex = fma_range(-30, 30), ey = fma_range(-30, 30);
        const int gap = fma_range(54, 1100) * (fma_next() & 1 ? 1 : -1);
        *x = random_double(ex, ex); *y = random_double(ey, ey);
        *z = random_sign(ldexp(random_double(0, 0), ex + ey + gap));
        break;
    }
    default: {  /* z a small multiple of a quarter ulp of fl(x y) */
        int e;
        const int ex = fma_range(-200, 200), ey = fma_range(-200, 200);
        *x = random_double(ex, ex); *y = random_double(ey, ey);
        (void) frexp(*x * *y, &e);
        *z = random_sign(ldexp((double) fma_range(1, 8), e - 53 - fma_range(0, 3)));
        break;
    }
    }
}

/* hig_soft_fma against the instruction (where the test has it) on `cases`
 * adversarial cases, and, for the default count, the hash of its results
 * against the one pinned on the instruction (so a target without the
 * instruction checks the same results). */
#define FMA_CASES 1000000L
#define FMA_HASH 0x6dd6a440fccd0f72ULL

static void test_soft_fma(long cases) {
    const int instruction = TEST_FMA_INSTRUCTION == 2 ||
                            (TEST_FMA_INSTRUCTION == 1 && hig_fma_cpu());
    long counts[FMA_CLASSES] = {0}, mismatches = 0;
    uint64_t hash = 0xcbf29ce484222325ULL;

    fma_state = 0x243f6a8885a308d3ULL;
    for (long i = 0; i < cases; i++) {
        const int cls = (int) (fma_next() % FMA_CLASSES);
        double x, y, z, soft;
        uint64_t bits;

        fma_case(cls, &x, &y, &z);
        soft = hig_soft_fma(x, y, z);
        bits = soft != soft ? 0x7ff8000000000000ULL : to_bits(soft);
        hash = (hash ^ bits) * 0x100000001b3ULL;
        counts[cls]++;
        if (instruction) {
            const double hard = instruction_fma(x, y, z);
            if (!(soft != soft && hard != hard) && to_bits(soft) != to_bits(hard)) {
                if (mismatches++ < 10) {
                    fprintf(stderr, "soft fma(%a, %a, %a) = %a, the instruction %a (%s)\n",
                            x, y, z, soft, hard, FMA_CLASS_NAMES[cls]);
                }
            }
        }
    }
    printf("soft fma: %ld cases (", cases);
    for (int c = 0; c < FMA_CLASSES; c++) {
        printf("%s%s %ld", c ? ", " : "", FMA_CLASS_NAMES[c], counts[c]);
    }
    if (instruction) {
        printf("), %ld differ from the instruction\n", mismatches);
    } else {
        printf("), no instruction to compare with\n");
    }
    CHECK(mismatches == 0, "%ld soft fma results differ from the instruction", mismatches);
    if (cases == FMA_CASES) {
        CHECK(hash == FMA_HASH, "soft fma results: hash %#llx, pinned %#llx",
              (unsigned long long) hash, (unsigned long long) FMA_HASH);
    }
}

/* HIG_TEST_FMA=soft|instruction sends the core's calls to that build on an
 * x86_64 dispatch target. Elsewhere, or for the instruction on a CPU without
 * it, the run is skipped (exit status 77); 0 means go on. */
#define SKIPPED 77

static int select_fma_build(void) {
    const char *want = getenv("HIG_TEST_FMA");

    if (want && *want) {
        const int instruction = strcmp(want, "instruction") == 0;
        if (!instruction && strcmp(want, "soft") != 0) {
            fprintf(stderr, "HIG_TEST_FMA must be soft or instruction, not %s\n", want);
            return 2;
        }
        if (!hig_fma_select(instruction)) {
            printf("HIG_TEST_FMA=%s: skipped (fma: %s, CPU FMA instruction: %s)\n", want,
                   hig_fma_path(), hig_fma_cpu() ? "yes" : "no");
            return SKIPPED;
        }
    }
    printf("fma: %s (CPU FMA instruction: %s)\n", hig_fma_path(), hig_fma_cpu() ? "yes" : "no");
    return 0;
}

int main(int argc, char **argv) {
    const int scale = argc > 1 ? atoi(argv[1]) : 1;
    const char *fma_cases = getenv("HIG_FMA_CASES");
    const int selected = select_fma_build();

    if (selected) {
        return selected;
    }
    test_soft_fma(fma_cases && *fma_cases ? atol(fma_cases) : FMA_CASES);
    test_pins();
    test_zachary();
    test_edge_layouts();
    test_former_tie_cycle();
    test_counts();
    test_planted();
    test_callbacks();
    test_errors();
    test_random_partitions(1500 * scale);
    printf("verifier: %lld vertices of the random games, %lld with a certainly improving "
           "deviation, none above the slack\n", (long long) verified, (long long) improvable);
    CHECK(improvable == 0, "the verifier found %lld improvable vertices", (long long) improvable);
    printf("%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
