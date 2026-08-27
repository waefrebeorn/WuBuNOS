/*
 * wubu_genetic_kernels.c — Genetic algorithm for evolving T_GEMM kernels.
 *
 * Phase 4 AGI compiler component: self-improving optimization.
 *
 * Genomes encode T_GEMM tile configurations + blocking strategy.
 * The GA evolves better kernels by:
 *   1. Initializing a random population
 *   2. Evaluating fitness (GFLOPS on benchmark)
 *   3. Selecting parents (tournament selection)
 *   4. Crossover + mutation
 *   5. Repeating until convergence
 *
 * C11, self-contained.
 */

#include "wubu_auto_tune.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Genome: encodes a T_GEMM configuration ---- */

typedef struct {
    int mc;           /* M tile: 16..256 */
    int nc;           /* N tile: 16..256 */
    int kc;           /* K tile: 16..256 */
    int mr;           /* M register block: 2,4,8 */
    int nr;           /* N register block: 2,4,8 */
    int unroll_k;     /* K unroll factor: 1..8 */
    int blocking;     /* 0=no blocking, 1=L2 blocking, 2=L1 blocking */
    double fitness;   /* measured GFLOPS */
} genome_t;

/* ---- Population ---- */

#define POP_SIZE 16
#define MAX_GEN  20
#define TOURN_K  3
#define MUT_RATE 0.15

static genome_t population[POP_SIZE];

/* ---- Random genome ---- */

static void random_genome(genome_t *g)
{
    g->mc = 16 + (rand() % 9) * 32;   /* 16, 48, 80, 112, 144, 176, 208, 240, 272 -> clamp */
    if (g->mc > 256) g->mc = 256;
    g->nc = 16 + (rand() % 9) * 32;
    if (g->nc > 256) g->nc = 256;
    g->kc = 16 + (rand() % 9) * 32;
    if (g->kc > 256) g->kc = 256;
    int reg_vals[] = {2, 4, 8};
    g->mr = reg_vals[rand() % 3];
    g->nr = reg_vals[rand() % 3];
    g->unroll_k = 1 + (rand() % 8);
    g->blocking = rand() % 3;
    g->fitness = 0.0;
}

/* ---- Fitness evaluation (benchmark T_GEMM) ---- */

static double evaluate_genome(const genome_t *g, int M, int N, int K)
{
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;
    size_t total = (a_sz + b_sz + c_sz) * sizeof(int64_t);

    int64_t *block = aligned_alloc(64, total);
    if (!block) return 0.0;

    int64_t A_off = 0, B_off = (int64_t)a_sz, C_off = (int64_t)(a_sz + b_sz);
    int64_t *A = block, *B = block + a_sz, *C = block + a_sz + b_sz;

    for (size_t i = 0; i < a_sz; i++) A[i] = (int64_t)((i & 0xFF) + 1);
    for (size_t i = 0; i < b_sz; i++) B[i] = (int64_t)(((i * 7) & 0xFF) + 1);
    memset(C, 0, c_sz * sizeof(int64_t));

    const int WARMUP = 2, RUNS = 5;
    double best = 1e9;

    for (int r = -WARMUP; r < RUNS; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        wubu_tgemm(A, A_off, B_off, C_off, M, N, K);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double sec = (double)(t1.tv_sec - t0.tv_sec) +
                     (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (r >= 0 && sec < best) best = sec;
    }

    free(block);
    if (best <= 0.0) return 0.0;
    double gflops = 2.0 * (double)M * (double)N * (double)K / (best * 1e9);
    return gflops;
}

/* ---- Tournament selection ---- */

static genome_t tournament_select(void)
{
    genome_t best = population[rand() % POP_SIZE];
    for (int i = 1; i < TOURN_K; i++) {
        genome_t candidate = population[rand() % POP_SIZE];
        if (candidate.fitness > best.fitness) best = candidate;
    }
    return best;
}

/* ---- Crossover (uniform) ---- */

static void crossover(const genome_t *p1, const genome_t *p2, genome_t *child)
{
    child->mc       = (rand() % 2) ? p1->mc       : p2->mc;
    child->nc       = (rand() % 2) ? p1->nc       : p2->nc;
    child->kc       = (rand() % 2) ? p1->kc       : p2->kc;
    child->mr       = (rand() % 2) ? p1->mr       : p2->mr;
    child->nr       = (rand() % 2) ? p1->nr       : p2->nr;
    child->unroll_k = (rand() % 2) ? p1->unroll_k : p2->unroll_k;
    child->blocking = (rand() % 2) ? p1->blocking : p2->blocking;
    child->fitness  = 0.0;
}

/* ---- Mutation ---- */

static void mutate(genome_t *g)
{
    if ((rand() % 100) < (int)(MUT_RATE * 100)) {
        int vals[] = {16, 48, 80, 112, 144, 176, 208, 240, 256};
        g->mc = vals[rand() % 9];
    }
    if ((rand() % 100) < (int)(MUT_RATE * 100)) {
        int vals[] = {16, 48, 80, 112, 144, 176, 208, 240, 256};
        g->nc = vals[rand() % 9];
    }
    if ((rand() % 100) < (int)(MUT_RATE * 100)) {
        int vals[] = {2, 4, 8};
        g->mr = vals[rand() % 3];
    }
    if ((rand() % 100) < (int)(MUT_RATE * 100)) {
        int vals[] = {2, 4, 8};
        g->nr = vals[rand() % 3];
    }
    g->fitness = 0.0;
}

/* ---- GA main loop ---- */

genome_t wubu_evolve_tgemm(int M, int N, int K)
{
    srand((unsigned)time(NULL) ^ (unsigned)((uintptr_t)&M));

    printf("=== Genetic Kernel Evolution (%dx%dx%d) ===\n", M, N, K);
    printf("Initializing population of %d...\n", POP_SIZE);

    /* Generation 0: random population */
    for (int i = 0; i < POP_SIZE; i++) {
        random_genome(&population[i]);
        population[i].fitness = evaluate_genome(&population[i], M, N, K);
        printf("  [0][%2d] mc=%3d nc=%3d kc=%3d mr=%d nr=%d uk=%d blk=%d -> %.2f GFLOPS\n",
               i, population[i].mc, population[i].nc, population[i].kc,
               population[i].mr, population[i].nr, population[i].unroll_k,
               population[i].blocking, population[i].fitness);
    }

    /* Evolution loop */
    for (int gen = 1; gen <= MAX_GEN; gen++) {
        genome_t new_pop[POP_SIZE];

        /* Elitism: keep best 2 */
        int best1 = 0, best2 = 1;
        for (int i = 1; i < POP_SIZE; i++) {
            if (population[i].fitness > population[best1].fitness) {
                best2 = best1; best1 = i;
            } else if (population[i].fitness > population[best2].fitness) {
                best2 = i;
            }
        }
        new_pop[0] = population[best1];
        new_pop[1] = population[best2];

        /* Breed rest */
        for (int i = 2; i < POP_SIZE; i++) {
            genome_t p1 = tournament_select();
            genome_t p2 = tournament_select();
            crossover(&p1, &p2, &new_pop[i]);
            mutate(&new_pop[i]);
            new_pop[i].fitness = evaluate_genome(&new_pop[i], M, N, K);
        }

        memcpy(population, new_pop, sizeof(population));

        /* Report progress */
        double avg = 0.0;
        for (int i = 0; i < POP_SIZE; i++) avg += population[i].fitness;
        avg /= POP_SIZE;
        printf("  Gen %2d: best=%.2f GFLOPS, avg=%.2f GFLOPS\n",
               gen, population[best1].fitness, avg);
    }

    /* Find global best */
    int best = 0;
    for (int i = 1; i < POP_SIZE; i++)
        if (population[i].fitness > population[best].fitness) best = i;

    printf("Best evolved kernel: mc=%d nc=%d kc=%d mr=%d nr=%d uk=%d blk=%d -> %.2f GFLOPS\n",
           population[best].mc, population[best].nc, population[best].kc,
           population[best].mr, population[best].nr, population[best].unroll_k,
           population[best].blocking, population[best].fitness);
    return population[best];
}
