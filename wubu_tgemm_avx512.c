/* wubu_tgemm_avx512.c — AVX-512DQ native int64 GEMM path.
 * Compiled ONLY with -mavx512dq -mavx512f -mavx512vl.
 * Uses native vpmullq (8 int64 products per 512-bit vector).
 * C18 pure, no external deps beyond immintrin.h. */
#include "wubu_tgemm.h"
#include <immintrin.h>

void tgemm_avx512(int64_t *mem, int64_t A, int64_t B,
                  int64_t C, int M, int N, int K)
{
    #pragma omp parallel for schedule(static) if(M >= 8)
    for (int i = 0; i < M; i++) {
        const int64_t *a = &mem[A + (int64_t)i * K];
        int64_t       *c = &mem[C + (int64_t)i * N];
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m512i acc = _mm512_loadu_si512(&c[j]);
            for (int k = 0; k < K; k++) {
                __m512i bv = _mm512_loadu_si512(&mem[B + (int64_t)k * N + j]);
                __m512i av = _mm512_set1_epi64(a[k]);
                __m512i prod = _mm512_mullo_epi64(av, bv);
                acc = _mm512_add_epi64(acc, prod);
            }
            _mm512_storeu_si512(&c[j], acc);
        }
        for (; j < N; j++) {
            int64_t s = c[j];
            const int64_t *bk = &mem[B] + j;
            for (int k = 0; k < K; k++)
                s += a[k] * bk[(size_t)k * N];
            c[j] = s;
        }
    }
}
