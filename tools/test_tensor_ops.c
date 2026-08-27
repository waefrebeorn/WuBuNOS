/* Test the new MIR tensor ops through the interpreter */
#include "wubu_mir.h"
#include "wubu_softfloat.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void store_f32(wubu_mir_prog_t *p, int64_t base, int idx, float val) {
    p->mem[base + idx] = (int64_t)wubu_sf_f32_from_host(val);
}
static float read_f32(const wubu_mir_prog_t *p, int64_t base, int idx) {
    return wubu_sf_f32_to_host((uint32_t)p->mem[base + idx]);
}

int main(void) {
    int errors = 0;

    /* ---- Test T_SIGMOID ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t base_in = 1, base_out = 5;
        p.total_mem = base_out + 3;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p, base_in, 0, 0.0f);
        store_f32(&p, base_in, 1, 1.0f);
        store_f32(&p, base_in, 2, -1.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, base_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, base_out);
        wubu_mir_tsigmoid(&p, vr_in, 0, vr_out, 3);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        float g0 = read_f32(&p, base_out, 0);
        float g1 = read_f32(&p, base_out, 1);
        float g2 = read_f32(&p, base_out, 2);
        printf("SIGMOID: [%.5f %.5f %.5f] want [0.50000 0.73106 0.26894] %s\n",
               g0, g1, g2,
               (fabsf(g0-0.5f)<0.001f && fabsf(g1-0.73106f)<0.001f && fabsf(g2-0.26894f)<0.001f)?"OK":"FAIL");
        if (fabsf(g0 - 0.5f) > 0.001f) errors++;
        if (fabsf(g1 - 0.73106f) > 0.001f) errors++;
        if (fabsf(g2 - 0.26894f) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_GELU ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_out = 2;
        p.total_mem = b_out + 1;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p, b_in, 0, 1.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_tgelu(&p, vr_in, 0, vr_out, 1);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        float got = read_f32(&p, b_out, 0);
        printf("GELU(1.0): %.5f want 0.84119 %s\n", got, fabsf(got - 0.84119f) < 0.01f ? "OK" : "FAIL");
        if (fabsf(got - 0.84119f) > 0.01f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_RELU ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_out = 5;
        p.total_mem = b_out + 3;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p, b_in, 0, -1.0f); store_f32(&p, b_in, 1, 0.0f); store_f32(&p, b_in, 2, 2.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_trelu(&p, vr_in, 0, vr_out, 3);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        printf("RELU: [%.1f %.1f %.1f] want [0.0 0.0 2.0] %s\n",
               read_f32(&p,b_out,0), read_f32(&p,b_out,1), read_f32(&p,b_out,2),
               (read_f32(&p,b_out,0)==0 && read_f32(&p,b_out,1)==0 && read_f32(&p,b_out,2)==2.0f)?"OK":"FAIL");
        if (read_f32(&p, b_out, 0) != 0.0f || read_f32(&p, b_out, 1) != 0.0f || read_f32(&p, b_out, 2) != 2.0f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_SUM: sum of [1,2,3,4] = 10 ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1;
        p.total_mem = b_in + 4;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        for (int i = 0; i < 4; i++) store_f32(&p, b_in, i, (float)(i+1));
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_mir_tsum(&p, vr_in, 0, 0, 4);
        wubu_mir_ret(&p, 0);  /* return vr0 (where T_SUM wrote the result) */
        int64_t result = wubu_mir_interp(&p);
        float got = wubu_sf_f32_to_host((uint32_t)result);
        printf("SUM: %.1f want 10.0 %s\n", got, fabsf(got - 10.0f) < 0.001f ? "OK" : "FAIL");
        if (fabsf(got - 10.0f) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_SOFTMAX ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_out = 5;
        p.total_mem = b_out + 4;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        for (int i = 0; i < 4; i++) store_f32(&p, b_in, i, (float)(i+1));
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_tsoftmax(&p, vr_in, 0, vr_out, 4);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += read_f32(&p, b_out, i);
        printf("SOFTMAX sum: %.5f want 1.0 %s\n", sum, fabsf(sum - 1.0f) < 0.001f ? "OK" : "FAIL");
        if (fabsf(sum - 1.0f) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_RMS_NORM ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_wt = 5, b_out = 9;
        p.total_mem = b_out + 4;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p,b_in,0,1.0f); store_f32(&p,b_in,1,2.0f); store_f32(&p,b_in,2,3.0f); store_f32(&p,b_in,3,4.0f);
        for (int i = 0; i < 4; i++) store_f32(&p, b_wt, i, 1.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_wt = wubu_mir_const(&p, b_wt);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_trms_norm(&p, vr_in, vr_wt, vr_out, 4);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        float ms = (1+4+9+16)/4.0f + 1e-6f;
        float inv = 1.0f / sqrtf(ms);
        float g0 = read_f32(&p, b_out, 0);
        printf("RMS_NORM[0]: %.5f want %.5f %s\n", g0, 1.0f*inv, fabsf(g0 - 1.0f*inv) < 0.001f ? "OK" : "FAIL");
        if (fabsf(g0 - 1.0f*inv) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_EXP ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_out = 5;
        p.total_mem = b_out + 3;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p, b_in, 0, 0.0f); store_f32(&p, b_in, 1, 1.0f); store_f32(&p, b_in, 2, -1.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_texp(&p, vr_in, 0, vr_out, 3);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        printf("EXP: [%.5f %.5f %.5f] want [1.00000 2.71828 0.36788] %s\n",
               read_f32(&p,b_out,0), read_f32(&p,b_out,1), read_f32(&p,b_out,2),
               (fabsf(read_f32(&p,b_out,0)-1.0f)<0.001f && fabsf(read_f32(&p,b_out,1)-2.71828f)<0.001f && fabsf(read_f32(&p,b_out,2)-0.36788f)<0.001f)?"OK":"FAIL");
        if (fabsf(read_f32(&p, b_out, 0) - 1.0f) > 0.001f) errors++;
        if (fabsf(read_f32(&p, b_out, 1) - 2.71828f) > 0.001f) errors++;
        if (fabsf(read_f32(&p, b_out, 2) - 0.36788f) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    /* ---- Test T_TANH ---- */
    {
        wubu_mir_prog_t p; wubu_mir_init(&p);
        int64_t b_in = 1, b_out = 2;
        p.total_mem = b_out + 1;
        p.mem = (int64_t*)calloc(p.total_mem + 1, sizeof(int64_t));
        store_f32(&p, b_in, 0, 1.0f);
        wubu_vr_t vr_in = wubu_mir_const(&p, b_in);
        wubu_vr_t vr_out = wubu_mir_const(&p, b_out);
        wubu_mir_ttanh(&p, vr_in, 0, vr_out, 1);
        wubu_mir_ret(&p, vr_out);
        wubu_mir_interp(&p);
        float got = read_f32(&p, b_out, 0);
        printf("TANH(1.0): %.5f want %.5f %s\n", got, tanhf(1.0f), fabsf(got - tanhf(1.0f)) < 0.001f ? "OK" : "FAIL");
        if (fabsf(got - tanhf(1.0f)) > 0.001f) errors++;
        wubu_mir_free(&p);
        free(p.mem);
    }

    if (errors == 0)
        printf("\n=== ALL %d TENSOR OP TESTS PASSED ===\n", 6);
    else
        printf("\n=== %d TENSOR OP TESTS FAILED ===\n", errors);

    return errors;
}
