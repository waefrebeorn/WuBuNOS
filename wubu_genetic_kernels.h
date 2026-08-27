/* wubu_genetic_kernels.h — genetic algorithm for kernel evolution */
#ifndef WUBU_GENETIC_KERNELS_H
#define WUBU_GENETIC_KERNELS_H

#include "wubu_genome.h"

/* Evolve optimal T_GEMM config via genetic algorithm for given dims */
genome_t wubu_evolve_tgemm(int M, int N, int K);

#endif /* WUBU_GENETIC_KERNELS_H */
