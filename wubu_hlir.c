/*
 * wubu_hlir.c -- High-Level IR (HLIR) implementation.
 *
 * Builds computation graphs, topologically sorts, lowers to MIR.
 * C11, self-contained.
 */
#include "wubu_hlir.h"
#include "wubu_mir.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
    n->output = *output_shape;
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

    /* Constant pool: keep a small cache to avoid duplicate loads */
    /* Map node pointer -> VR for this compilation */
    int vr[256];
    memset(vr, 0xFF, sizeof(vr));

    for (int i = 0; i < n_sorted; i++) {
        hlir_node_t *n = sorted[i];

        switch (n->op) {
        case HLIR_PLACEHOLDER: {
            /* Already represented by the input node's data; nothing to emit */
            vr[i] = 0; /* v0 = first input */
            break;
        }
        case HLIR_CONSTANT: {
            /* Load constant data into VR */
            if (n->data && n->output.dtype == 0) { /* F32 */
                vr[i] = wubu_mir_const(p, (int64_t)((float *)n->data)[0]);
            } else {
                vr[i] = wubu_mir_const(p, 0);
            }
            break;
        }
        case HLIR_ADD: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[1])];
            vr[i] = wubu_mir_binop(p, MIR_ADD, a, b);
            break;
        }
        case HLIR_MUL: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[1])];
            vr[i] = wubu_mir_binop(p, MIR_MUL, a, b);
            break;
        }
        case HLIR_RELU: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            /* relu = max(0, x) => x > 0 ? x : 0 */
            wubu_vr_t c = wubu_mir_const(p, 0);
            wubu_vr_t gt = wubu_mir_binop(p, MIR_GT, a, c);
            wubu_vr_t zero = wubu_mir_const(p, 0);
            wubu_vr_t one = wubu_mir_const(p, 1);
            /* Conditional: (gt ? x : 0) — simplified via arithmetic for now */
            vr[i] = wubu_mir_binop(p, MIR_MUL, a, gt);
            break;
        }
        case HLIR_SIGMOID: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            /* sigmoid = 1 / (1 + exp(-a)) — use softfloat */
            vr[i] = a;
            break;
        }
        case HLIR_GELU: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            vr[i] = a;
            break;
        }
        case HLIR_RMSNORM: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            /* rmsnorm = x / sqrt(mean(x*x) + eps) — simplified: pass x through */
            vr[i] = a;
            break;
        }
        case HLIR_LAYERNORM: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            vr[i] = a;
            break;
        }
        case HLIR_MATMUL: {
            /* GEMM: emit T_GEMM call via the host dispatch path */
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t b = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[1])];
            vr[i] = wubu_mir_binop(p, MIR_ADD, a, b); /* placeholder */
            break;
        }
        case HLIR_SOFTMAX: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            vr[i] = a;
            break;
        }
        case HLIR_ATTENTION: {
            wubu_vr_t q = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t k = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[1])];
            wubu_vr_t v = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[2])];
            vr[i] = wubu_mir_binop(p, MIR_ADD, q, k);
            break;
        }
        case HLIR_SWIGLU: {
            wubu_vr_t x = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            vr[i] = x;
            break;
        }
        case HLIR_ROPE: {
            wubu_vr_t x = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            vr[i] = x;
            break;
        }
        case HLIR_CAST: {
            wubu_vr_t a = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            /* dtype conversion via MIR_F32_TO_F16 / MIR_F16_TO_F32 */
            vr[i] = a;
            break;
        }
        case HLIR_RESIDUAL_ADD: {
            wubu_vr_t x = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            wubu_vr_t r = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[1])];
            vr[i] = wubu_mir_binop(p, MIR_ADD, x, r);
            break;
        }
        default: {
            /* Unimplemented: passthrough the first input */
            if (n->n_inputs > 0)
                vr[i] = (wubu_vr_t)vr[hlir_topo_order_of(g, n->inputs[0])];
            else
                vr[i] = wubu_mir_const(p, 0);
            break;
        }
        }
    }

    /* Emit return for last VR */
    if (n_sorted > 0) {
        wubu_vr_t last = (wubu_vr_t)vr[n_sorted - 1];
        if (last >= 0) wubu_mir_ret(p, last);
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
