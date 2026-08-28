#ifndef HF_HUB_H
#define HF_HUB_H

#include "safetensors.h"
#include <stddef.h>
#include <stdint.h>

/* Download a URL to memory. Returns allocated buffer (caller frees).
 * Sets *out_size to the number of bytes downloaded. NULL on error. */
uint8_t *hf_download(const char *url, size_t *out_size);

/* Download a file from HuggingFace Hub.
 * model_id: e.g. "gpt2", "microsoft/phi-2"
 * filename: e.g. "model.safetensors", "config.json"
 * Returns allocated buffer (caller frees), sets *out_size. NULL on error. */
uint8_t *hf_hub_download(const char *model_id, const char *filename,
                          size_t *out_size);

/* List safetensors files available for a model.
 * Returns a NULL-terminated array of filenames (caller frees each + array). */
char **hf_hub_list_files(const char *model_id, int max_files);

/* Download and load a model's safetensors weights.
 * Returns 0 on success, -1 on error. */
int hf_hub_load_model(const char *model_id, safetensors_t *st);

#endif /* HF_HUB_H */
