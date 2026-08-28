/*
 * wubu_hlir.h -- High-Level IR (HLIR): model graph frontend.
 *
 * Represents neural network computation as a DAG of typed operations
 * over tensors. HLIR is the entry point for importing ONNX/TorchScript
 * models and lowering them to the existing MIR backend.
 *
 * Key concepts:
 *   - HLIRNode: a typed operation (matmul, add, relu, softmax, ...)
 *   - HLIRTensor: a typed multi-dimensional array descriptor
 *   - HLIRGraph: the full computation DAG
 *   - Each node has inputs/outputs (tensor refs), op type, attributes
 *   - Topological sort → linearized → lowered to MIR
 *
 * C11, self-contained.
 */

#ifndef WUBU_HLIR_H
#define WUBU_HLIR_H

#include <stdint.h>
#include <stddef.h>

/* ---- Tensor descriptor ---- */
typedef struct {
    int      n_dims;       /* number of dimensions (1-8) */
    int64_t  dims[8];      /* dimension sizes (row-major) */
    int      dtype;        /* F32=0, F16=1, BF16=2, I32=3, I8=4, I4=5 */
    int64_t  nelems;       /* total elements (computed on build) */
} hlir_tensor_t;

/* ---- Operation types ---- */
typedef enum {
    HLIR_PLACEHOLDER = 0,  /* input tensor (model input) */
    HLIR_CONSTANT,         /* constant tensor (weights, biases) */
    HLIR_ADD,              /* element-wise add */
    HLIR_MUL,              /* element-wise multiply */
    HLIR_RELU,             /* max(0, x) */
    HLIR_SIGMOID,          /* 1 / (1 + exp(-x)) */
    HLIR_TANH,             /* tanh(x) */
    HLIR_GELU,             /* x * 0.5 * (1 + erf(x / sqrt(2))) */
    HLIR_LAYERNORM,        /* LayerNorm: normalize + scale */
    HLIR_MATMUL,           /* GEMM: C = A @ B (+ bias) */
    HLIR_SOFTMAX,          /* row-wise softmax */
    HLIR_ATTENTION,        /* scaled dot-product attention (Q,K,V) */
    HLIR_RMSNORM,          /* RMS normalization */
    HLIR_SWIGLU,           /* SwiGLU: SiLU(xW) * xV */
    HLIR_ROPE,             /* Rotary position embedding */
    HLIR_CAST,             /* dtype cast */
    HLIR_RESHAPE,          /* reshape without copy */
    HLIR_TRANSPOSE,        /* swap two dims */
    HLIR_REDUCE_SUM,       /* sum over axis */
    HLIR_REDUCE_MEAN,      /* mean over axis */
    HLIR_DIV,              /* element-wise divide */
    HLIR_SUB,              /* element-wise subtract */
    HLIR_SQUARE,           /* x * x */
    HLIR_SQRT,             /* sqrt(x) */
    HLIR_EXP,              /* exp(x) */
    HLIR_LOG,              /* log(x) */
    HLIR_POW,              /* pow(x, y) */
    HLIR_CLAMP,            /* clamp(x, min, max) */
    HLIR_CONCAT,           /* concat tensors along axis */
    HLIR_SLICE,            /* extract a slice */
    HLIR_PAD,              /* pad a tensor */
    HLIR_CAST_F16,         /* F32 -> F16 */
    HLIR_CAST_F32,         /* F16 -> F32 */
    HLIR_RESIDUAL_ADD,     /* add + residual (common in transformers) */
    HLIR_PROJECTION,       /* linear projection (GEMM without bias) */
    HLIR_NODE_MAX,
} hlir_op_t;

/* ---- Node attribute ---- */
typedef struct {
    const char *key;       /* attribute name */
    float       fval;      /* float value (used for eps, alpha, axis) */
    int         ival;      /* int value (used for axis) */
} hlir_attr_t;

/* ---- Forward decl ---- */
struct hlir_node;
struct hlir_graph;

/* ---- Computation node ---- */
typedef struct hlir_node {
    hlir_op_t        op;            /* operation type */
    hlir_tensor_t    output;         /* output tensor shape + dtype */
    const char      *name;           /* human-readable name */
    hlir_attr_t     *attrs;          /* attribute array (nullable) */
    int              n_attrs;        /* number of attributes */
    struct hlir_node **inputs;       /* input nodes (DAG edges) */
    int              n_inputs;       /* number of inputs */
    int              n_users;        /* how many downstream nodes read this */
    int              topo_order;     /* set during topo sort */
    int64_t         *data;           /* optional constant data (CONSTANT nodes) */
    int              user_data;      /* scratch slot for lowering */
} hlir_node_t;

/* ---- Computation graph ---- */
typedef struct {
    hlir_node_t **nodes;  /* dynamic array of nodes */
    int           n, cap; /* count / capacity */
    hlir_node_t **inputs; /* model input placeholders */
    int           n_inputs;
    hlir_node_t **outputs; /* model output nodes */
    int           n_outputs;
} hlir_graph_t;

/* ---- Build API ---- */

/* Init/free */
void hlir_graph_init(hlir_graph_t *g);
void hlir_graph_free(hlir_graph_t *g);

/* Create a placeholder (model input). Returns the node. */
hlir_node_t *hlir_placeholder(hlir_graph_t *g, const char *name,
                              const hlir_tensor_t *shape);

/* Create a constant tensor node. data points to nelems * sizeof(float)
 * values (for F32), or packed bits for F16/BF16. Returns the node. */
hlir_node_t *hlir_constant(hlir_graph_t *g, const char *name,
                           const hlir_tensor_t *shape, const void *data);

/* Create an op node with N inputs. Variadic input count via n_inputs.
 * attrs/attr_count optional. Returns the node. */
hlir_node_t *hlir_op(hlir_graph_t *g, hlir_op_t op, const char *name,
                     hlir_node_t **inputs, int n_inputs,
                     const hlir_tensor_t *output_shape,
                     const hlir_attr_t *attrs, int n_attrs);

/* Convenience wrappers */
hlir_node_t *hlir_add(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b);
hlir_node_t *hlir_mul(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b);
hlir_node_t *hlir_relu(hlir_graph_t *g, hlir_node_t *x);
hlir_node_t *hlir_matmul(hlir_graph_t *g, hlir_node_t *a, hlir_node_t *b,
                         const hlir_tensor_t *out_shape);
hlir_node_t *hlir_softmax(hlir_graph_t *g, hlir_node_t *x, int axis);
hlir_node_t *hlir_attention(hlir_graph_t *g, hlir_node_t *q,
                            hlir_node_t *k, hlir_node_t *v,
                            float scale);
hlir_node_t *hlir_layernorm(hlir_graph_t *g, hlir_node_t *x,
                            float eps, int axis);
hlir_node_t *hlir_rmsnorm(hlir_graph_t *g, hlir_node_t *x, float eps);
hlir_node_t *hlir_gelu(hlir_graph_t *g, hlir_node_t *x);
hlir_node_t *hlir_swiglu(hlir_graph_t *g, hlir_node_t *x);
hlir_node_t *hlir_rope(hlir_graph_t *g, hlir_node_t *x, int dim);
hlir_node_t *hlir_cast(hlir_graph_t *g, hlir_node_t *x, int dtype);
hlir_node_t *hlir_residual_add(hlir_graph_t *g, hlir_node_t *x,
                               hlir_node_t *residual);

/* Mark the final node(s) as graph outputs */
void hlir_set_output(hlir_graph_t *g, hlir_node_t *node);

/* ---- helpers ---- */
hlir_tensor_t hlir_tensor(int n_dims, const int64_t *dims, int dtype);

/* Topological sort */
int hlir_topo_sort(const hlir_graph_t *g, hlir_node_t **sorted);

/* Lower graph to canonical MIR */
int hlir_lower_mir(const hlir_graph_t *g, void *prog);

/* Memory planning: compute buffer reuse plan for the graph */
typedef struct memplan memplan_t;
memplan_t *hlir_memplan(const hlir_graph_t *g);

/* Debug: dump graph as text */
void hlir_dump(const hlir_graph_t *g);

/* Topo index helper */
int hlir_topo_order_of(const hlir_graph_t *g, const hlir_node_t *n);

#endif /* WUBU_HLIR_H */
