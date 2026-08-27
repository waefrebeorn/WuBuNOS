/* wubu_genome.h — shared genome type for GA/auto-tune/cost-model */
#ifndef WUBU_GENOME_H
#define WUBU_GENOME_H

#include <stdint.h>

typedef struct {
    int mc;           /* M tile: 16..256 */
    int nc;           /* N tile: 16..256 */
    int kc;           /* K tile: 16..256 */
    int mr;           /* M register block: 2,4,8 */
    int nr;           /* N register block: 2,4,8 */
    int unroll_k;     /* K unroll factor: 1..8 */
    int blocking;     /* 0=none, 1=L2, 2=L1 */
    double fitness;   /* measured GFLOPS */
} genome_t;

#endif /* WUBU_GENOME_H */
