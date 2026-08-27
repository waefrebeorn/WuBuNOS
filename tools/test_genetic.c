#include "wubu_genetic_kernels.h"
#include <stdio.h>
int main() {
    printf("=== Genetic Kernel Evolution Test ===\n");
    genome_t best = wubu_evolve_tgemm(64, 64, 64);
    printf("Final best: mc=%d nc=%d kc=%d mr=%d nr=%d uk=%d blk=%d -> %.2f GFLOPS\n",
           best.mc, best.nc, best.kc, best.mr, best.nr,
           best.unroll_k, best.blocking, best.fitness);
    return (best.fitness > 0.0) ? 0 : 1;
}
