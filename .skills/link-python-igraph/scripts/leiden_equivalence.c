/*
 * Build-to-build equivalence harness for igraph's Leiden implementation.
 *
 * It generates a deterministic battery of small instances with its own
 * PRNG (so instance parameters never depend on the algorithm's random
 * number consumption), seeds igraph's default RNG per instance, calls the
 * public Leiden entry points, and prints one line per instance with the
 * error code or an exact fingerprint of the result: an FNV-1a hash of the
 * membership (or cover, or diagnostic traces) and the quality as a hex
 * float. Running the same binary against two libraries and diffing the
 * outputs checks that a refactor is behaviour-preserving bit for bit.
 *
 * It covers what the Python differential does not reach: directed graphs,
 * signed edge weights, vertex in-weights, self-loops, the simple interface
 * with all three objectives, global community-count limits for partitions
 * and covers, the diagnostic entry point, and error paths.
 *
 * POSIX only (fork, alarm). Build and run (macOS; use LD_LIBRARY_PATH on
 * Linux):
 *
 *   P=~/Code/community/opus-worktrees/_prefix/igraph-1.0.0.5
 *   cc -O1 -I$P/include/igraph leiden_equivalence.c -L$P/lib -ligraph \
 *      -Wl,-rpath,$P/lib -o leiden_equivalence
 *   ./leiden_equivalence 20000 > new.txt
 *   DYLD_LIBRARY_PATH=/path/to/reference/lib ./leiden_equivalence 20000 > ref.txt
 *   cmp ref.txt new.txt
 *
 * API generations. The source compiles against three interfaces:
 *
 *   - 1.0.0.5 final (IGRAPH_LEIDEN_TRACE_SCHEMA_VERSION defined): fork-base
 *     igraph_community_leiden(), diagnostics with count limits and counters.
 *     The diagnostic fingerprint hashes the first 12 move-trace columns (the
 *     columns of earlier builds), so its output compares line by line with
 *     a binary built against the earlier interface. Build one binary per
 *     prefix, each against its own headers.
 *   - 1.0.0.4 / 1.0.0.5 review builds: 15-argument igraph_community_leiden()
 *     and overlapping-only diagnostics.
 *   - -DLEIDEN_EQ_BASE_API: only the igraph 1.0.0 entry points
 *     (igraph_community_leiden with 11 arguments and the simple interface),
 *     for comparing a fork build with upstream igraph 1.0.0 through one
 *     binary.
 *
 * With LEIDEN_EQ_SELFCHECK=1 (1.0.0.5 final only) every instance is re-run
 * through the other entry points with the same random state: the diagnostic
 * entry point (full and counters-only) must return the main result, and a
 * disjoint call with the igraph 1.0.0 defaults and a non-negative resolution
 * must match the fork-base symbol. Mismatches are printed to stderr as "SELFCHECK-MISMATCH"; every
 * completed check prints "SELFCHECK ok <index> <kind>" to stderr.
 */

#include <igraph.h>

#if defined(IGRAPH_LEIDEN_TRACE_SCHEMA_VERSION) && !defined(LEIDEN_EQ_BASE_API)
#define LEIDEN_EQ_FINAL_API 1
#endif

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* Every instance runs in a forked child with a wall-clock limit, so a
 * non-terminating call is recorded as "timeout" instead of blocking the
 * battery. The parent replays the same instance generation without calling
 * igraph ("dry run") to keep its PRNG in step with the child. */
static igraph_bool_t dry_run = false;

/* ---- independent instance PRNG (splitmix64) ---------------------------- */

static uint64_t prng_state;

static uint64_t prng_next(void) {
    uint64_t z = (prng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static igraph_int_t prng_int(igraph_int_t lo, igraph_int_t hi) { /* inclusive */
    return lo + (igraph_int_t) (prng_next() % (uint64_t) (hi - lo + 1));
}

static double prng_unit(void) {
    return (double) (prng_next() >> 11) / 9007199254740992.0;
}

static igraph_bool_t prng_chance(double p) {
    return prng_unit() < p;
}

/* ---- fingerprints ------------------------------------------------------- */

static uint64_t fnv(uint64_t h, const void *data, size_t size) {
    const unsigned char *p = (const unsigned char *) data;
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv_int(uint64_t h, igraph_int_t x) {
    int64_t v = (int64_t) x;
    return fnv(h, &v, sizeof v);
}

static uint64_t fnv_real(uint64_t h, igraph_real_t x) {
    return fnv(h, &x, sizeof x);
}

static uint64_t hash_membership(const igraph_vector_int_t *m) {
    uint64_t h = 1469598103934665603ULL;
    for (igraph_int_t i = 0; i < igraph_vector_int_size(m); i++) {
        h = fnv_int(h, VECTOR(*m)[i]);
    }
    return h;
}

static uint64_t hash_cover(const igraph_vector_int_list_t *c) {
    uint64_t h = 1469598103934665603ULL;
    for (igraph_int_t v = 0; v < igraph_vector_int_list_size(c); v++) {
        const igraph_vector_int_t *row = igraph_vector_int_list_get_ptr(c, v);
        h = fnv_int(h, -1 - v);
        for (igraph_int_t i = 0; i < igraph_vector_int_size(row); i++) {
            h = fnv_int(h, VECTOR(*row)[i]);
        }
    }
    return h;
}

static uint64_t hash_matrix(uint64_t h, const igraph_matrix_t *m, igraph_int_t max_columns) {
    /* Opt-in comparison across the 19 -> 21 projection-trace extension.
     * Cover, quality, moves, and all legacy projection columns remain in
     * the fingerprint. New label counts need their separate schema tests.
     * The default continues to hash the entire native matrix, except that
     * 1.0.0.5 final builds pass max_columns = 12 for the move trace. */
    igraph_int_t columns = igraph_matrix_ncol(m);
    if (columns == 21 && getenv("LEIDEN_EQ_LEGACY_TRACE") != NULL) {
        columns = 19;
    }
    if (max_columns > 0 && columns > max_columns) {
        columns = max_columns;
    }
    h = fnv_int(h, igraph_matrix_nrow(m));
    h = fnv_int(h, columns);
    for (igraph_int_t i = 0; i < igraph_matrix_nrow(m); i++) {
        for (igraph_int_t j = 0; j < columns; j++) {
            h = fnv_real(h, MATRIX(*m, i, j));
        }
    }
    return h;
}

/* ---- instance generation ------------------------------------------------ */

static void random_graph(igraph_t *graph, igraph_int_t n, igraph_bool_t directed,
                         igraph_bool_t loops) {
    igraph_vector_int_t edges;
    const double p = (double[]) {0.08, 0.15, 0.3, 0.6}[prng_int(0, 3)];
    igraph_vector_int_init(&edges, 0);
    for (igraph_int_t u = 0; u < n; u++) {
        for (igraph_int_t v = directed ? 0 : u; v < n; v++) {
            if (u == v && !(loops && prng_chance(0.1))) {
                continue;
            }
            if (u != v && !prng_chance(p)) {
                continue;
            }
            igraph_vector_int_push_back(&edges, u);
            igraph_vector_int_push_back(&edges, v);
        }
    }
    if (igraph_vector_int_size(&edges) == 0 && n >= 2) {
        igraph_vector_int_push_back(&edges, 0);
        igraph_vector_int_push_back(&edges, 1);
    }
    igraph_create(graph, &edges, n, directed);
    igraph_vector_int_destroy(&edges);
}

/* kind: 0 none, 1 positive integers, 2 positive reals, 3 signed reals,
 * 4 non-negative reals including zeros. */
static igraph_vector_t *random_weights(igraph_vector_t *store, igraph_int_t size, int kind) {
    if (kind == 0) {
        return NULL;
    }
    igraph_vector_init(store, size);
    for (igraph_int_t i = 0; i < size; i++) {
        switch (kind) {
        case 1: VECTOR(*store)[i] = (double) prng_int(1, 3); break;
        case 2: VECTOR(*store)[i] = 0.1 + 1.9 * prng_unit(); break;
        case 3: VECTOR(*store)[i] = -1.0 + 3.0 * prng_unit(); break;
        default: VECTOR(*store)[i] = prng_chance(0.2) ? 0.0 : 2.0 * prng_unit(); break;
        }
    }
    return store;
}

static void free_weights(igraph_vector_t *w) {
    if (w) {
        igraph_vector_destroy(w);
    }
}

static void random_count_limits(igraph_int_t upper, igraph_int_t *max_total,
                                igraph_int_t *exact) {
    *max_total = -1;
    *exact = -1;
    if (prng_chance(0.2)) {
        *max_total = prng_int(1, upper);
    } else if (prng_chance(0.25)) {
        *exact = prng_int(1, upper);
    }
    if (prng_chance(0.02)) {
        *exact = 0; /* invalid on purpose */
    }
}

static const igraph_real_t RESOLUTIONS[] = {-0.3, -0.05, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0};
static const igraph_real_t BETAS[] = {0.0, 0.001, 0.01, 0.1, 1.0};
static const igraph_int_t BUDGETS[] = {-1, -1, 0, 1, 2, 3};

static igraph_real_t pick_resolution(void) { return RESOLUTIONS[prng_int(0, 7)]; }
static igraph_real_t pick_beta(void) { return BETAS[prng_int(0, 4)]; }
static igraph_int_t pick_budget(void) { return BUDGETS[prng_int(0, 5)]; }

/* ---- the four instance families ---------------------------------------- */

#ifdef LEIDEN_EQ_FINAL_API
static igraph_bool_t selfcheck_enabled(void) {
    return getenv("LEIDEN_EQ_SELFCHECK") != NULL;
}

static void selfcheck_report(igraph_int_t index, const char *kind, igraph_bool_t ok,
                             const char *what) {
    if (ok) {
        fprintf(stderr, "SELFCHECK ok %" IGRAPH_PRId " %s\n", index, kind);
    } else {
        fprintf(stderr, "SELFCHECK-MISMATCH %" IGRAPH_PRId " %s %s\n", index, kind, what);
    }
}

static igraph_bool_t same_outcome(igraph_error_t code_a, uint64_t h_a, igraph_int_t nb_a,
                                  igraph_real_t q_a, igraph_error_t code_b, uint64_t h_b,
                                  igraph_int_t nb_b, igraph_real_t q_b) {
    if (code_a != code_b) {
        return false;
    }
    if (code_a != IGRAPH_SUCCESS) {
        return true;
    }
    return h_a == h_b && nb_a == nb_b &&
           (q_a == q_b || (q_a != q_a && q_b != q_b));
}

/* The diagnostic entry point (full, then counters only) and, for the igraph
 * 1.0.0 defaults, the fork-base symbol must reproduce the main result. */
static void selfcheck_disjoint(igraph_int_t index, uint64_t seed, const igraph_t *graph,
                               const igraph_vector_t *ew, const igraph_vector_t *out,
                               const igraph_vector_t *in, igraph_real_t resolution,
                               igraph_real_t beta, igraph_int_t max_total, igraph_int_t exact,
                               igraph_bool_t start, igraph_int_t budget,
                               igraph_bool_t isolation, igraph_bool_t local,
                               igraph_bool_t use_list,
                               const igraph_vector_int_t *start_membership,
                               const igraph_vector_int_list_t *start_memberships,
                               igraph_error_t code, uint64_t h, igraph_int_t nb,
                               igraph_real_t quality) {
    for (int mode = 0; mode < 3; mode++) {
        igraph_vector_int_t membership;
        igraph_vector_int_list_t memberships;
        igraph_matrix_t moves, projections;
        igraph_vector_int_t counters;
        igraph_int_t nb2 = -7;
        igraph_real_t q2 = -7.0;
        igraph_error_t code2;
        uint64_t h2 = 0;
        const char *kind = mode == 0 ? "disjoint-diagnostics" :
                           mode == 1 ? "disjoint-counters" : "disjoint-base";

        /* The fork-base symbol keeps the igraph 1.0.0 candidate set; it
         * agrees with the extended defaults for a non-negative resolution
         * and non-negative vertex weights (the battery draws non-negative
         * vertex weights). */
        if (mode == 2 && !(max_total < 0 && exact < 0 && isolation && !local && !use_list &&
                           resolution >= 0.0)) {
            continue;
        }
        igraph_vector_int_init_copy(&membership, start_membership);
        igraph_vector_int_list_init_copy(&memberships, start_memberships);
        igraph_rng_seed(igraph_rng_default(), seed);
        if (mode == 2) {
            code2 = igraph_community_leiden(graph, ew, out, in, resolution, beta, start, budget,
                                            &membership, &nb2, &q2);
        } else {
            code2 = igraph_community_leiden_with_diagnostics(
                graph, ew, out, in, resolution, beta, 1, max_total, exact, start, budget,
                isolation, local, use_list ? NULL : &membership, &memberships, &nb2, &q2,
                mode == 0 ? &moves : NULL, mode == 0 ? &projections : NULL, &counters);
        }
        if (code2 == IGRAPH_SUCCESS) {
            h2 = use_list ? hash_cover(&memberships) : hash_membership(&membership);
            if (mode < 2) {
                igraph_bool_t ok = VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_VISITS] ==
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_ACCEPTED_MOVES] +
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_REJECTED_VISITS];
                if (mode == 0) {
                    ok = ok && igraph_matrix_nrow(&moves) ==
                         VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_ACCEPTED_MOVES] &&
                         igraph_matrix_nrow(&projections) == 0;
                    igraph_matrix_destroy(&projections);
                    igraph_matrix_destroy(&moves);
                }
                igraph_vector_int_destroy(&counters);
                if (!ok) {
                    selfcheck_report(index, kind, false, "counters");
                }
            }
        }
        selfcheck_report(index, kind,
                         same_outcome(code, h, nb, quality, code2, h2, nb2, q2), "result");
        igraph_vector_int_list_destroy(&memberships);
        igraph_vector_int_destroy(&membership);
    }
}
#endif

static void run_disjoint(igraph_int_t index, uint64_t seed) {
    const igraph_int_t n = prng_int(1, 26);
    const igraph_bool_t directed = prng_chance(0.3);
    igraph_t graph;
    igraph_vector_t ew_store, out_store, in_store;
    igraph_vector_t *ew, *out, *in = NULL;
    igraph_vector_int_t membership;
    igraph_vector_int_list_t memberships;
    igraph_int_t nb = -7, max_total, exact;
    igraph_real_t quality = -7.0;
    igraph_bool_t start = prng_chance(0.4), use_list = prng_chance(0.2);
    igraph_error_t code;

    random_graph(&graph, n, directed, prng_chance(0.3));
    ew = random_weights(&ew_store, igraph_ecount(&graph), prng_int(0, 3));
    out = random_weights(&out_store, n, prng_chance(0.5) ? 0 : 2);
    if (directed && prng_chance(0.4)) {
        in = random_weights(&in_store, n, 2);
    }
    random_count_limits(n, &max_total, &exact);
    igraph_vector_int_init(&membership, n);
    igraph_vector_int_list_init(&memberships, n);
    if (start) {
        const igraph_int_t k = prng_int(1, n);
        for (igraph_int_t v = 0; v < n; v++) {
            VECTOR(membership)[v] = prng_int(0, k - 1);
            igraph_vector_int_push_back(igraph_vector_int_list_get_ptr(&memberships, v),
                                        VECTOR(membership)[v]);
        }
    }
    const igraph_real_t resolution = pick_resolution(), beta = pick_beta();
    const igraph_int_t budget = pick_budget();
    const igraph_bool_t isolation = prng_chance(0.5), local = prng_chance(0.5);
#ifdef LEIDEN_EQ_FINAL_API
    igraph_vector_int_t start_membership;
    igraph_vector_int_list_t start_memberships;
    const igraph_bool_t selfcheck = !dry_run && selfcheck_enabled();
    if (selfcheck) {
        igraph_vector_int_init_copy(&start_membership, &membership);
        igraph_vector_int_list_init_copy(&start_memberships, &memberships);
    }
#endif
    igraph_rng_seed(igraph_rng_default(), seed);
#ifdef LEIDEN_EQ_BASE_API
    /* igraph 1.0.0 entry point: disjoint, isolation allowed, multilevel. */
    use_list = false;
    code = dry_run ? IGRAPH_SUCCESS : igraph_community_leiden(
        &graph, ew, out, in, resolution, beta, start, budget, &membership, &nb, &quality);
#else
    code = dry_run ? IGRAPH_SUCCESS : igraph_community_leiden_with_constraints(
        &graph, ew, out, in, resolution, beta, 1, max_total, exact, start, budget,
        isolation, local, use_list ? NULL : &membership, &memberships, &nb, &quality);
#endif
    if (dry_run) {
        /* generation only */
    } else if (code == IGRAPH_SUCCESS) {
        uint64_t h = use_list ? hash_cover(&memberships) : hash_membership(&membership);
        printf("%" IGRAPH_PRId " disjoint ok h=%016" PRIx64 " nb=%" IGRAPH_PRId " q=%a\n",
               index, h, nb, quality);
    } else {
        printf("%" IGRAPH_PRId " disjoint error=%d\n", index, (int) code);
    }
#ifdef LEIDEN_EQ_FINAL_API
    if (selfcheck) {
        const uint64_t h = code != IGRAPH_SUCCESS ? 0 :
                           use_list ? hash_cover(&memberships) : hash_membership(&membership);
        fflush(stdout);
        selfcheck_disjoint(index, seed, &graph, ew, out, in, resolution, beta, max_total, exact,
                           start, budget, isolation, local, use_list, &start_membership,
                           &start_memberships, code, h, nb, quality);
        igraph_vector_int_list_destroy(&start_memberships);
        igraph_vector_int_destroy(&start_membership);
    }
#else
    (void) max_total; (void) exact; (void) isolation; (void) local;
#endif
    igraph_vector_int_list_destroy(&memberships);
    igraph_vector_int_destroy(&membership);
    free_weights(in);
    free_weights(out);
    free_weights(ew);
    igraph_destroy(&graph);
}

static void run_simple(igraph_int_t index, uint64_t seed) {
    const igraph_int_t n = prng_int(1, 26);
    const igraph_bool_t directed = prng_chance(0.4);
    const igraph_leiden_objective_t objective =
        (igraph_leiden_objective_t[]) {IGRAPH_LEIDEN_OBJECTIVE_MODULARITY,
                                       IGRAPH_LEIDEN_OBJECTIVE_CPM,
                                       IGRAPH_LEIDEN_OBJECTIVE_ER}[prng_int(0, 2)];
    igraph_t graph;
    igraph_vector_t ew_store, *ew;
    igraph_vector_int_t membership;
    igraph_int_t nb = -7;
    igraph_real_t quality = -7.0;
    const igraph_bool_t start = prng_chance(0.3);
    igraph_error_t code;

    random_graph(&graph, n, directed, prng_chance(0.3));
    ew = random_weights(&ew_store, igraph_ecount(&graph), prng_int(0, 3));
    igraph_vector_int_init(&membership, n);
    if (start) {
        for (igraph_int_t v = 0; v < n; v++) {
            VECTOR(membership)[v] = prng_int(0, n - 1);
        }
    }
    const igraph_real_t resolution = 0.05 + 2.0 * prng_unit(), beta = pick_beta();
    const igraph_int_t budget = pick_budget();
    igraph_rng_seed(igraph_rng_default(), seed);
    if (getenv("LEIDEN_EQ_VERBOSE") && !dry_run) {
        fprintf(stderr, "simple n=%" IGRAPH_PRId " m=%" IGRAPH_PRId " directed=%d objective=%d "
                "weights=%s resolution=%.17g beta=%g start=%d budget=%" IGRAPH_PRId "\n",
                n, igraph_ecount(&graph), (int) directed, (int) objective,
                ew ? "yes" : "no", resolution, beta, (int) start, budget);
        for (igraph_int_t e = 0; e < igraph_ecount(&graph); e++) {
            fprintf(stderr, "  %" IGRAPH_PRId "-%" IGRAPH_PRId " w=%.17g\n", IGRAPH_FROM(&graph, e),
                    IGRAPH_TO(&graph, e), ew ? VECTOR(*ew)[e] : 1.0);
        }
        fprintf(stderr, "  start:");
        for (igraph_int_t v = 0; v < n; v++) {
            fprintf(stderr, " %" IGRAPH_PRId, VECTOR(membership)[v]);
        }
        fprintf(stderr, "\n  rng seed %" PRIu64 "\n", seed);
    }
    code = dry_run ? IGRAPH_SUCCESS :
           igraph_community_leiden_simple(&graph, ew, objective, resolution, beta, start,
                                          budget, &membership, &nb, &quality);
    if (dry_run) {
        /* generation only */
    } else if (code == IGRAPH_SUCCESS) {
        printf("%" IGRAPH_PRId " simple ok h=%016" PRIx64 " nb=%" IGRAPH_PRId " q=%a\n",
               index, hash_membership(&membership), nb, quality);
    } else {
        printf("%" IGRAPH_PRId " simple error=%d\n", index, (int) code);
    }
    igraph_vector_int_destroy(&membership);
    free_weights(ew);
    igraph_destroy(&graph);
}

static void random_cover(igraph_vector_int_list_t *cover, igraph_int_t n, igraph_int_t cap) {
    const igraph_int_t k = prng_int(1, n);
    for (igraph_int_t v = 0; v < n; v++) {
        igraph_vector_int_t *row = igraph_vector_int_list_get_ptr(cover, v);
        const igraph_int_t size = prng_int(1, cap < k ? cap : k);
        for (igraph_int_t i = 0; i < size; i++) {
            igraph_vector_int_push_back(row, prng_int(0, k - 1)); /* duplicates allowed */
        }
    }
}

#ifndef LEIDEN_EQ_BASE_API
#ifdef LEIDEN_EQ_FINAL_API
/* The other extended entry points must reproduce the main result: the
 * constrained one for a diagnostic call, the diagnostic one (full and
 * counters only) for a constrained call. */
static void selfcheck_overlapping(igraph_int_t index, uint64_t seed, const igraph_t *graph,
                                  const igraph_vector_t *ew, const igraph_vector_t *nw,
                                  igraph_real_t resolution, igraph_real_t beta,
                                  igraph_int_t max_memberships, igraph_int_t max_total,
                                  igraph_int_t exact, igraph_bool_t start, igraph_int_t budget,
                                  igraph_bool_t isolation, igraph_bool_t local,
                                  const igraph_vector_int_list_t *start_cover,
                                  igraph_error_t code, uint64_t cover_hash, igraph_int_t nb,
                                  igraph_real_t quality) {
    for (int mode = 0; mode < 3; mode++) {
        igraph_vector_int_list_t cover;
        igraph_matrix_t moves, projections;
        igraph_vector_int_t counters;
        igraph_int_t nb2 = -7;
        igraph_real_t q2 = -7.0;
        igraph_error_t code2;
        uint64_t h2 = 0;
        const char *kind = mode == 0 ? "overlapping-constraints" :
                           mode == 1 ? "overlapping-diagnostics" : "overlapping-counters";

        igraph_vector_int_list_init_copy(&cover, start_cover);
        igraph_rng_seed(igraph_rng_default(), seed);
        if (mode == 0) {
            code2 = igraph_community_leiden_with_constraints(
                graph, ew, nw, NULL, resolution, beta, max_memberships, max_total, exact,
                start, budget, isolation, local, NULL, &cover, &nb2, &q2);
        } else {
            code2 = igraph_community_leiden_with_diagnostics(
                graph, ew, nw, NULL, resolution, beta, max_memberships, max_total, exact,
                start, budget, isolation, local, NULL, &cover, &nb2, &q2,
                mode == 1 ? &moves : NULL, mode == 1 ? &projections : NULL, &counters);
        }
        if (code2 == IGRAPH_SUCCESS) {
            h2 = hash_cover(&cover);
            if (mode > 0) {
                igraph_bool_t ok =
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_VISITS] ==
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_ACCEPTED_MOVES] +
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_REJECTED_VISITS] &&
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_PROPOSALS] ==
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_PROPOSALS_IMPROVED] +
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_PROPOSALS_TIED] +
                    VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_PROPOSALS_REJECTED];
                if (mode == 1) {
                    ok = ok && igraph_matrix_nrow(&moves) ==
                         VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_ACCEPTED_MOVES] &&
                         igraph_matrix_nrow(&projections) ==
                         VECTOR(counters)[IGRAPH_LEIDEN_COUNTER_PROJECTION_ROWS];
                    igraph_matrix_destroy(&projections);
                    igraph_matrix_destroy(&moves);
                }
                igraph_vector_int_destroy(&counters);
                if (!ok) {
                    selfcheck_report(index, kind, false, "counters");
                }
            }
        }
        selfcheck_report(index, kind,
                         same_outcome(code, cover_hash, nb, quality, code2, h2, nb2, q2),
                         "result");
        igraph_vector_int_list_destroy(&cover);
    }
}
#endif

static void run_overlapping(igraph_int_t index, uint64_t seed, igraph_bool_t diagnostics) {
    const igraph_int_t n = prng_int(2, 22);
    igraph_t graph;
    igraph_vector_t ew_store, nw_store, *ew, *nw;
    igraph_vector_int_list_t cover;
    igraph_matrix_t moves, projections;
    igraph_int_t nb = -7, max_total = -1, exact = -1;
    igraph_real_t quality = -7.0;
    const igraph_int_t cap = (igraph_int_t[]) {2, 2, 3, 4, 0}[prng_int(0, 4)];
    const igraph_int_t max_memberships = cap == 0 ? n : (cap < n ? cap : n);
    const igraph_bool_t start = prng_chance(0.35);
    igraph_error_t code;
    uint64_t cover_hash = 0;

    random_graph(&graph, n, prng_chance(0.03), prng_chance(0.05));
    ew = random_weights(&ew_store, igraph_ecount(&graph), (int[]) {0, 0, 1, 2, 4}[prng_int(0, 4)]);
    nw = random_weights(&nw_store, n, prng_chance(0.6) ? 0 : 4);
    if (!diagnostics) {
        random_count_limits(n * max_memberships, &max_total, &exact);
    }
    igraph_vector_int_list_init(&cover, n);
    if (start) {
        random_cover(&cover, n, max_memberships);
    }
    const igraph_real_t resolution = pick_resolution(), beta = pick_beta();
    const igraph_int_t budget = pick_budget();
    const igraph_bool_t isolation = prng_chance(0.5), local = prng_chance(0.5);
#ifdef LEIDEN_EQ_FINAL_API
    igraph_vector_int_list_t start_cover;
    const igraph_bool_t selfcheck = !dry_run && selfcheck_enabled();
    if (selfcheck) {
        igraph_vector_int_list_init_copy(&start_cover, &cover);
    }
#endif
    igraph_rng_seed(igraph_rng_default(), seed);
    if (dry_run) {
        code = IGRAPH_EINVAL; /* nothing to print or free */
    } else if (diagnostics) {
#ifdef LEIDEN_EQ_FINAL_API
        code = igraph_community_leiden_with_diagnostics(
            &graph, ew, nw, NULL, resolution, beta, max_memberships, -1, -1, start, budget,
            isolation, local, NULL, &cover, &nb, &quality, &moves, &projections, NULL);
#else
        code = igraph_community_leiden_with_diagnostics(
            &graph, ew, nw, NULL, resolution, beta, max_memberships, start, budget,
            isolation, local, &cover, &nb, &quality, &moves, &projections);
#endif
    } else {
        code = igraph_community_leiden_with_constraints(
            &graph, ew, nw, NULL, resolution, beta, max_memberships, max_total, exact,
            start, budget, isolation, local, NULL, &cover, &nb, &quality);
    }
    if (code == IGRAPH_SUCCESS) {
        uint64_t h = cover_hash = hash_cover(&cover);
        if (diagnostics) {
#ifdef LEIDEN_EQ_FINAL_API
            h = hash_matrix(h, &moves, 12);
#else
            h = hash_matrix(h, &moves, 0);
#endif
            h = hash_matrix(h, &projections, 0);
            igraph_matrix_destroy(&projections);
            igraph_matrix_destroy(&moves);
        }
        printf("%" IGRAPH_PRId " %s ok h=%016" PRIx64 " nb=%" IGRAPH_PRId " q=%a\n", index,
               diagnostics ? "diagnostics" : "overlapping", h, nb, quality);
    } else if (!dry_run) {
        printf("%" IGRAPH_PRId " %s error=%d\n", index,
               diagnostics ? "diagnostics" : "overlapping", (int) code);
    }
#ifdef LEIDEN_EQ_FINAL_API
    if (selfcheck) {
        fflush(stdout);
        selfcheck_overlapping(index, seed, &graph, ew, nw, resolution, beta, max_memberships,
                              max_total, exact, start, budget, isolation, local, &start_cover,
                              code, cover_hash, nb, quality);
        igraph_vector_int_list_destroy(&start_cover);
    }
#else
    (void) cover_hash;
#endif
    igraph_vector_int_list_destroy(&cover);
    free_weights(nw);
    free_weights(ew);
    igraph_destroy(&graph);
}
#endif

static void run_instance(igraph_int_t i, uint64_t seed) {
#ifdef LEIDEN_EQ_BASE_API
    /* Only the igraph 1.0.0 entry points. */
    if (i % 5 == 1) {
        run_simple(i, seed);
    } else {
        run_disjoint(i, seed);
    }
#else
    switch (i % 5) {
    case 0: run_disjoint(i, seed); break;
    case 1: run_simple(i, seed); break;
    case 2: run_disjoint(i, seed); break;
    case 3: run_overlapping(i, seed, false); break;
    default: run_overlapping(i, seed, prng_chance(0.5)); break;
    }
#endif
}

/* Usage: leiden_equivalence [count] [prng seed] [timeout seconds] [only index] */
int main(int argc, char **argv) {
    const igraph_int_t count = argc > 1 ? atol(argv[1]) : 2000;
    const unsigned timeout = argc > 3 ? (unsigned) atoi(argv[3]) : 20;
    const igraph_int_t only = argc > 4 ? atol(argv[4]) : -1;
    prng_state = argc > 2 ? strtoull(argv[2], NULL, 10) : 20260927ULL;

    igraph_set_error_handler(igraph_error_handler_ignore);
    igraph_set_warning_handler(igraph_warning_handler_ignore);
    for (igraph_int_t i = 0; i < count; i++) {
        const uint64_t seed = prng_next() >> 33;
        pid_t child;
        int status;

        if (only >= 0 && i != only) {
            dry_run = true;
            run_instance(i, seed);
            dry_run = false;
            continue;
        }
        fflush(stdout);
        child = fork();
        if (child == 0) {
            alarm(timeout);
            run_instance(i, seed);
            fflush(stdout);
            _exit(0);
        }
        dry_run = true;
        run_instance(i, seed);
        dry_run = false;
        waitpid(child, &status, 0);
        if (WIFSIGNALED(status)) {
            printf("%" IGRAPH_PRId " %s\n", i,
                   WTERMSIG(status) == SIGALRM ? "timeout" : "crashed");
        }
    }
    return 0;
}
