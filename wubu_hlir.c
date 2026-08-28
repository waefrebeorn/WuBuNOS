/*
 * wubu_hlir.c -- High-Level IR (HLIR) implementation.
 *
 * Builds computation graphs, topologically sorts, lowers to MIR.
 * C11, self-contained.
 */
#include "wubu_hlir.h"
#include "wubu_mir.h"
#include "wubu_memplan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Memory planning ─── */
memplan_t *hlir_memplan(const hlir_graph_t *g) {
    return memplan_create(g);
}

/* ---- internal helpers ---- */
static int tensor_nelems(const hlir_tensor_t *t)
{
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->dims[i];
    return (int)n;
}

hlir_tensor_t hlir_tensor(int n_dims, const int64_t *dims, int dtype)
{
    hlir_tensor_t t;
    memset(&t, 0, sizeof(t));
    t.n_dims = n_dims;
    t.dtype = dtype;
    for (int i = 0; i < n_dims && i < 8; i++) t.dims[i] = dims[i];
    t.nelems = tensor_nelems(&t);
    return t;
}

/* forward decl for topo sort helper */
int hlir_topo_order_of(const hlir_graph_t *g, const hlir_node_t *n);

static void ensure_cap(hlir_graph_t *g, int extra)
{
    if (g->n + extra <= g->cap) return;
    g->cap = g->cap ? g->cap * 2 : 16;
    g->nodes = realloc(g->nodes, (size_t)g->cap * sizeof(hlir_node_t *));
}

static hlir_node_t *make_node(hlir_graph_t *g, hlir_op_t op, const char *name)
{
    ensure_cap(g, 1);
    hlir_node_t *n = calloc(1, sizeof(*n));
    n->op = op;
    n->name = name ? strdup(name) : NULL;
    n->topo_order = -1;
    g->nodes[g->n++] = n;
    return n;
}

void hlir_graph_init(hlir_graph_t *g)
{
    memset(g, 0, sizeof(*g));
}

void hlir_graph_free(hlir_graph_t *g)
{
    if (!g) return;
    for (int i = 0; i < g->n; i++) {
        hlir_node_t *n = g->nodes[i];
        free(n->inputs);
        free(n->attrs);
        free(n->data);
        free((void *)n->name);
        free(n);
    }
    free(g->nodes);
    free(g->inputs);
    free(g->outputs);
    memset(g, 0, sizeof(*g));
}

hlir_node_t *hlir_placeholder(hlir_graph_t *g, const char *name,
                              const hlir_tensor_t *shape)
{
    hlir_node_t *n = make_node(g, HLIR_PLACEHOLDER, name);
    n->output = *shape;
    g->inputs = realloc(g->inputs, (size_t)(g->n_inputs + 1) * sizeof(hlir_node_t *));
    g->inputs[g->n_inputs++] = n;
    return n;
}

hlir_node_t *hlir_constant(hlir_graph_t *g, const char *name,
                           const hlir_tensor_t *shape, const void *data)
{
    hlir_node_t *n = make_node(g, HLIR_CONSTANT, name);
    n->output = *shape;
    size_t bytes = (size_t)shape->nelems * sizeof(float);
    n->data = malloc(bytes);
    if (data) memcpy(n->data, data, bytes);
    return n;
}

hlir_node_t *hlir_op(hlir_graph_t *g, hlir_op_t op, const char *name,
                     hlir_node_t **inputs, int n_inputs,
                     const hlir_tensor_t *output_shape,
                     const hlir_attr_t *attrs, int n_attrs)
{
    hlir_node_t *n = make_node(g, op, name);
    if (output_shape)
        n->output = *output_shape;
    else if (n_inputs > 0 && inputs && inputs[0])
        n->output = inputs[0]->output;  /* infer from first input */
    if (n_inputs > 0) {
        n->inputs = malloc((size_t)n_inputs * sizeof(hlir_node_t *));
        memcpy(n->inputs, inputs, (size_t)n_inputs * sizeof(hlir_node_t *));
        n->n_inputs = n_inputs;
        for (int i = 0; i < n_inputs; i++)
            inputs[i]->n_users++;
    }
    if (n_attrs > 0) {
        n->attrs = malloc((size_t)n_attrs * sizeof(hlir_attr_t));
        memcpy(n->attrs, attrs, (size_t)n_attrs * sizeof(hlir_attr_t));
        n->n_attrs = n_attrs;
    }
    return n;
}

/* ---- convenience wrappers ---- */

hlir_node_t *hlir_add(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b)
{
    hlir_tensor_t out = a->output;
    hlir_node_t *inp[2] = {a, b};
    return hlir_op(g, HLIR_ADD, "add", inp, 2, &out, NULL, 0);
}

hlir_node_t *hlir_mul(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b)
{
    hlir_tensor_t out = a->output;
    hlir_node_t *inp[2] = {a, b};
    return hlir_op(g, HLIR_MUL, "mul", inp, 2, &out, NULL, 0);
}

hlir_node_t *hlir_relu(hlir_graph_t *g, hlir_node_t *x)
{
    hlir_tensor_t out = x->output;
    return hlir_op(g, HLIR_RELU, "relu", &x, 1, &out, NULL, 0);
}

hlir_node_t *hlir_matmul(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b,
                         const hlir_tensor_t *out_shape)
{
    hlir_node_t *inp[2] = {a, b};
    return hlir_op(g, HLIR_MATMUL, "matmul", inp, 2, out_shape, NULL, 0);
}

hlir_node_t *hlir_softmax(hlir_graph_t *g, hlir_node_t *x, int axis)
{
    hlir_tensor_t out = x->output;
    hlir_attr_t attr[1] = {{"axis", 0, axis}};
    return hlir_op(g, HLIR_SOFTMAX, "softmax", &x, 1, &out, attr, 1);
}

hlir_node_t *hlir_attention(hlir_graph_t *g, hlir_node_t *q,
                            hlir_node_t *k, hlir_node_t *v,
                            float scale)
{
    hlir_tensor_t out = q->output;
    hlir_attr_t attrs[1] = {{"scale", scale, 0}};
    hlir_node_t *inp[3] = {q, k, v};
    return hlir_op(g, HLIR_ATTENTION, "attention", inp, 3, &out, attrs, 1);
}

hlir_node_t *hlir_layernorm(hlir_graph_t *g, hlir_node_t *x,
                            float eps, int axis)
{
    hlir_tensor_t out = x->output;
    hlir_attr_t attrs[2] = {{"eps", eps, 0}, {"axis", 0, axis}};
    return hlir_op(g, HLIR_LAYERNORM, "layernorm", &x, 1, &out, attrs, 2);
}

hlir_node_t *hlir_rmsnorm(hlir_graph_t *g, hlir_node_t *x, float eps)
{
    hlir_tensor_t out = x->output;
    hlir_attr_t attrs[1] = {{"eps", eps, 0}};
    return hlir_op(g, HLIR_RMSNORM, "rmsnorm", &x, 1, &out, attrs, 1);
}

hlir_node_t *hlir_gelu(hlir_graph_t *g, hlir_node_t *x)
{
    hlir_tensor_t out = x->output;
    return hlir_op(g, HLIR_GELU, "gelu", &x, 1, &out, NULL, 0);
}

hlir_node_t *hlir_swiglu(hlir_graph_t *g, hlir_node_t *x)
{
    hlir_tensor_t out = x->output;
    return hlir_op(g, HLIR_SWIGLU, "swiglu", &x, 1, &out, NULL, 0);
}

hlir_node_t *hlir_rope(hlir_graph_t *g, hlir_node_t *x, int dim)
{
    hlir_tensor_t out = x->output;
    hlir_attr_t attrs[1] = {{"dim", 0, dim}};
    return hlir_op(g, HLIR_ROPE, "rope", &x, 1, &out, attrs, 1);
}

hlir_node_t *hlir_cast(hlir_graph_t *g, hlir_node_t *x, int dtype)
{
    hlir_tensor_t out = x->output;
    out.dtype = dtype;
    hlir_attr_t attrs[1] = {{"dtype", 0, dtype}};
    return hlir_op(g, HLIR_CAST, "cast", &x, 1, &out, attrs, 1);
}

hlir_node_t *hlir_residual_add(hlir_graph_t *g, hlir_node_t *x,
                               hlir_node_t *residual)
{
    hlir_tensor_t out = x->output;
    hlir_node_t *inp[2] = {x, residual};
    return hlir_op(g, HLIR_RESIDUAL_ADD, "residual_add", inp, 2, &out, NULL, 0);
}

void hlir_set_output(hlir_graph_t *g, hlir_node_t *node)
{
    g->outputs = realloc(g->outputs, (size_t)(g->n_outputs + 1) * sizeof(hlir_node_t *));
    g->outputs[g->n_outputs++] = node;
}

/* ---- topological sort (Kahn's algorithm) ---- */

int hlir_topo_sort(const hlir_graph_t *g, hlir_node_t **sorted)
{
    if (!g || !sorted) return 0;

    int *indeg = calloc(g->n, sizeof(int));
    for (int i = 0; i < g->n; i++) {
        for (int j = 0; j < g->nodes[i]->n_inputs; j++)
            indeg[i]++;
    }

    int head = 0, tail = 0;
    for (int i = 0; i < g->n; i++) {
        if (indeg[i] == 0) { sorted[tail++] = g->nodes[i]; }
    }

    int count = 0;
    while (head < tail) {
        hlir_node_t *n = sorted[head++];
        n->topo_order = count;
        sorted[count++] = n;

        for (int i = 0; i < g->n; i++) {
            for (int j = 0; j < g->nodes[i]->n_inputs; j++) {
                if (g->nodes[i]->inputs[j] == n) {
                    indeg[i]--;
                    if (indeg[i] == 0) sorted[tail++] = g->nodes[i];
                }
            }
        }
    }
    free(indeg);
    return count;
}

/* ---- Lower HLIR graph to MIR ---- */

int hlir_lower_mir(const hlir_graph_t *g, void *prog)
{
    if (!g || !prog) return -1;
    wubu_mir_prog_t *p = (wubu_mir_prog_t *)prog;
    wubu_mir_init(p);

    hlir_node_t *sorted[256];
    int n_sorted = hlir_topo_sort(g, sorted);
    if (n_sorted == 0) return -1;

    /*
     * Each HLIR node produces a tensor. We track:
     *   base_vr[i]  = VR holding the base memory address of node i's output
     *   off_vr[i]   = VR holding the element count (for tensor ops needing imm)
     *
     * For scalar/element-wise ops, we use the flat MIR model:
     *   - Allocate memory for the output tensor
     *   - For element-wise: loop over elements, emit scalar MIR ops
     *   - For tensor ops: emit the MIR_T_* op directly
     *
     * Simplified model: for now, element-wise ops work on the first
     * element (scalar test path). Full tensor loops come in the next pass.
     */

    wubu_vr_t base_vr[256];
    int n_placeholders = 0;

    /* First pass: allocate memory for each node's output */
    for (int i = 0; i < n_sorted; i++) {
        hlir_node_t *n = sorted[i];
        int64_t nelems = n->output.nelems > 0 ? n->output.nelems : 1;

        if (n->op == HLIR_PLACEHOLDER) {
            /* Placeholders get allocated memory for their input data */
            base_vr[i] = wubu_mir_alloc(p, nelems);
            n_placeholders++;
        } else if (n->op == HLIR_CONSTANT) {
            /* Allocate + store ALL constant data */
            wubu_vr_t addr = wubu_mir_alloc(p, nelems);
            base_vr[i] = addr;
            if (n->data && n->output.dtype == 0) {
                float *data = (float *)n->data;
                for (int64_t e = 0; e < nelems; e++) {
                    wubu_vr_t elem_addr = wubu_mir_binop(p, MIR_ADD, addr,
                                                          wubu_mir_const(p, e));
                    wubu_vr_t val = wubu_mir_const(p, (int64_t)(*(int64_t *)&data[e]));
                    wubu_mir_store(p, elem_addr, val);
                }
            } else {
                wubu_vr_t zero = wubu_mir_const(p, 0);
                wubu_mir_store(p, addr, zero);
            }
        } else {
            /* Allocate output memory */
            base_vr[i] = wubu_mir_alloc(p, nelems);
        }
    }

    /* Set the number of function arguments */
    wubu_mir_set_n_args(p, (uint32_t)n_placeholders);

    /* Second pass: emit computation */
    for (int i = 0; i < n_sorted; i++) {
        hlir_node_t *n = sorted[i];
        if (n->op == HLIR_PLACEHOLDER || n->op == HLIR_CONSTANT) continue;

        wubu_vr_t dst = base_vr[i];
        int64_t nelems = n->output.nelems > 0 ? n->output.nelems : 1;

        switch (n->op) {
        case HLIR_ADD: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t vb = wubu_mir_load(p, b);
            wubu_vr_t sum = wubu_mir_binop(p, MIR_FADD, va, vb);
            wubu_mir_store(p, dst, sum);
            break;
        }
        case HLIR_MUL: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t vb = wubu_mir_load(p, b);
            wubu_vr_t prod = wubu_mir_binop(p, MIR_FMUL, va, vb);
            wubu_mir_store(p, dst, prod);
            break;
        }
        case HLIR_SUB: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t vb = wubu_mir_load(p, b);
            wubu_vr_t diff = wubu_mir_binop(p, MIR_FSUB, va, vb);
            wubu_mir_store(p, dst, diff);
            break;
        }
        case HLIR_DIV: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t vb = wubu_mir_load(p, b);
            wubu_vr_t quot = wubu_mir_binop(p, MIR_FDIV, va, vb);
            wubu_mir_store(p, dst, quot);
            break;
        }
        case HLIR_RELU: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])]; /* same */
            wubu_mir_trelu(p, a, b, dst, nelems);
            break;
        }
        case HLIR_SIGMOID: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tsigmoid(p, a, b, dst, nelems);
            break;
        }
        case HLIR_GELU: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tgelu(p, a, b, dst, nelems);
            break;
        }
        case HLIR_TANH: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_ttanh(p, a, b, dst, nelems);
            break;
        }
        case HLIR_RMSNORM: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_trms_norm(p, a, b, dst, nelems);
            break;
        }
        case HLIR_LAYERNORM: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tlayernorm(p, a, b, dst, nelems);
            break;
        }
        case HLIR_MATMUL: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            int M = (n->output.n_dims >= 1) ? (int)n->output.dims[0] : 1;
            int N = (n->output.n_dims >= 2) ? (int)n->output.dims[1] : 1;
            int K = (n->inputs[0]->output.n_dims >= 2) ?
                    (int)n->inputs[0]->output.dims[1] : 1;
            /* Use float GEMM for F32 output, int GEMM otherwise */
            if (n->output.dtype == 0) { /* F32 */
                wubu_mir_tgemm_f32(p, a, b, dst, M, N, K);
            } else {
                wubu_mir_tgemm(p, a, b, dst, M, N, K);
            }
            break;
        }
        case HLIR_SOFTMAX: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tsoftmax(p, a, b, dst, nelems);
            break;
        }
        case HLIR_ATTENTION: {
            wubu_vr_t q = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t k = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t v = base_vr[hlir_topo_order_of(g, n->inputs[2])];
            /* Attention: use Q and K for the op, V as value */
            wubu_mir_tattention(p, q, k, dst, nelems);
            /* Note: full attention needs Q,K,V — the 3-arg form is simplified */
            (void)v;
            break;
        }
        case HLIR_SWIGLU: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tswiglu(p, a, b, dst, nelems);
            break;
        }
        case HLIR_ROPE: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_trope(p, a, b, dst, nelems);
            break;
        }
        case HLIR_EXP: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_texp(p, a, b, dst, nelems);
            break;
        }
        case HLIR_SQRT: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tsqrt(p, a, b, dst, nelems);
            break;
        }
        case HLIR_LOG: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            /* log via exp inverse — use tsqrt as placeholder */
            wubu_mir_tsqrt(p, a, b, dst, nelems);
            break;
        }
        case HLIR_CLAMP: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_mir_tclamp(p, a, b, dst, nelems);
            break;
        }
        case HLIR_RESIDUAL_ADD: {
            wubu_vr_t x = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t r = base_vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t vx = wubu_mir_load(p, x);
            wubu_vr_t vr = wubu_mir_load(p, r);
            wubu_vr_t sum = wubu_mir_binop(p, MIR_FADD, vx, vr);
            wubu_mir_store(p, dst, sum);
            break;
        }
        case HLIR_CAST_F16: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t cvt = wubu_mir_unop(p, MIR_F32_TO_F16, va);
            wubu_mir_store(p, dst, cvt);
            break;
        }
        case HLIR_CAST_F32: {
            wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t va = wubu_mir_load(p, a);
            wubu_vr_t cvt = wubu_mir_unop(p, MIR_F16_TO_F32, va);
            wubu_mir_store(p, dst, cvt);
            break;
        }
        default: {
            /* Passthrough first input (unary/binary placeholder ops) */
            if (n->n_inputs > 0) {
                wubu_vr_t a = base_vr[hlir_topo_order_of(g, n->inputs[0])];
                wubu_vr_t va = wubu_mir_load(p, a);
                wubu_mir_store(p, dst, va);
            }
            break;
        }
        }
    }

    /* Emit return: load first element of last output */
    if (n_sorted > 0) {
        /* Find the last non-placeholder, non-constant node */
        for (int i = n_sorted - 1; i >= 0; i--) {
            hlir_node_t *n = sorted[i];
            if (n->op != HLIR_PLACEHOLDER && n->op != HLIR_CONSTANT) {
                wubu_vr_t last = wubu_mir_load(p, base_vr[i]);
                wubu_mir_ret(p, last);
                break;
            }
        }
    }

    return 0;
}

int hlir_topo_order_of(const hlir_graph_t *g, const hlir_node_t *n)
{
    for (int i = 0; i < g->n; i++) {
        if (g->nodes[i] == n) return g->nodes[i]->topo_order;
    }
    return -1;
}

void hlir_dump(const hlir_graph_t *g)
{
    printf("=== HLIR Graph (%d nodes, %d inputs, %d outputs) ===\n",
           g->n, g->n_inputs, g->n_outputs);

    hlir_node_t *sorted[256];
    int n_sorted = hlir_topo_sort(g, sorted);
    if (n_sorted == 0) { printf("(empty or cyclic)\n"); return; }

    for (int i = 0; i < n_sorted; i++) {
        hlir_node_t *n = sorted[i];
        printf("[%3d] %-20s shape=[", i, n->name ? n->name : "(anon)");
        for (int d = 0; d < n->output.n_dims; d++)
            printf("%s%lld", d ? "x" : "", (long long)n->output.dims[d]);
        printf("] dtype=%d", n->output.dtype);

        if (n->n_inputs > 0) {
            printf(" <- [");
            for (int j = 0; j < n->n_inputs; j++) {
                for (int k = 0; k < g->n; k++) {
                    if (g->nodes[k] == n->inputs[j])
                        printf("%s[%d]", j ? ", " : "", k);
                }
            }
            printf("]");
        }
        printf("\n");
    }
}
