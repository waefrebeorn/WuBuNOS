#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#define SAFETENSORS_MAX_TENSORS 1024
#define SAFETENSORS_MAX_NAME 256
#define SAFETENSORS_MAX_DIMS 16
#define SAFETENSORS_MAX_ERROR 256

/* Single tensor entry */
typedef struct {
    char     name[SAFETENSORS_MAX_NAME];
    char     dtype_str[16];          /* "F32", "F16", "BF16", "I8", etc. */
    int      dtype;                  /* HLIR dtype enum: 0=F32, 1=F16, 2=BF16, 3=I32, 4=I8 */
    int      dtype_size;             /* bytes per element */
    int      n_dims;
    int64_t  shape[SAFETENSORS_MAX_DIMS];
    size_t   data_offset;            /* offset from start of data section */
    size_t   data_size;              /* bytes of data */
    uint8_t *data;                   /* pointer into file buffer */
} safetensors_tensor_t;

/* Parsed SafeTensors file */
typedef struct {
    char                 filename[512];
    int                  n_tensors;
    safetensors_tensor_t tensors[SAFETENSORS_MAX_TENSORS];
    uint8_t             *file_data;      /* entire file in memory */
    size_t               file_size;
    char                *header_json;    /* null-terminated JSON header */
    size_t               header_size;
    char                 error[SAFETENSORS_MAX_ERROR];
} safetensors_t;

/* Load a .safetensors file. Returns 0 on success, -1 on error (st->error set). */
int safetensors_load(const char *filepath, safetensors_t *st);

/* Load safetensors from memory buffer (no file I/O).
 * buffer/sz: the raw safetensors data in memory.
 * Returns 0 on success, -1 on error (st->error set). */
int safetensors_load_from_memory(const uint8_t *buffer, size_t sz, safetensors_t *st);

/* Free resources */
void safetensors_free(safetensors_t *st);

/* Find a tensor by name */
const safetensors_tensor_t *safetensors_find(const safetensors_t *st, const char *name);

/* Print summary */
void safetensors_print(const safetensors_t *st);

#endif /* SAFETENSORS_H */
