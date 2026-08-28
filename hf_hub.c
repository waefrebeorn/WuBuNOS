/*
 * hf_hub.c — HuggingFace Hub runtime for WuBuNOS.
 *
 * Downloads model files from HuggingFace Hub and loads them into
 * SafeTensors format for the compiler pipeline.
 *
 * Implements:
 *   - HF Hub API: list files, resolve download URLs
 *   - HTTP download via libcurl (linked at build time)
 *   - JSON response parsing (minimal, hand-rolled)
 *   - Integration with safetensors parser
 *
 * C11, self-contained beyond libcurl (which is universally available).
 */
#include "hf_hub.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

/* ---- HTTP download via libcurl ─── */

typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} download_buf_t;

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t total = size * nmemb;
    download_buf_t *buf = (download_buf_t *)userdata;
    if (buf->len + total > buf->cap) {
        buf->cap = (buf->len + total) * 2;
        buf->data = (uint8_t *)realloc(buf->data, buf->cap);
    }
    memcpy(buf->data + buf->len, ptr, total);
    buf->len += total;
    return total;
}

/* Download a URL to memory. Returns allocated buffer (caller frees).
 * Sets *out_size to the number of bytes downloaded. NULL on error. */
uint8_t *hf_download(const char *url, size_t *out_size) {
    CURL *curl = curl_easy_init();
    if (!curl) return NULL;

    download_buf_t buf = {0};
    buf.cap = 1024 * 1024; /* 1MB initial */
    buf.data = (uint8_t *)malloc(buf.cap);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "WuBuNOS/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L); /* 5 min for large models */

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        free(buf.data);
        return NULL;
    }

    *out_size = buf.len;
    return buf.data;
}

/* ---- HF Hub URL construction ─── */

static void hf_resolve_url(const char *model_id, const char *filename,
                           char *buf, size_t bufsz) {
    snprintf(buf, bufsz, "https://huggingface.co/%s/resolve/main/%s",
             model_id, filename);
}

/* ---- Download a model file from HF Hub ─── */

uint8_t *hf_hub_download(const char *model_id, const char *filename,
                          size_t *out_size) {
    char url[2048];
    hf_resolve_url(model_id, filename, url, sizeof(url));
    fprintf(stderr, "[hf_hub] downloading %s\n", url);
    return hf_download(url, out_size);
}

/* ---- Parse HF Hub API JSON response to find model files ─── */

/* Minimal JSON string extraction: find "key": "value" pairs */
static const char *json_find_key(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != ':') return NULL;
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Extract a JSON string value starting at p. Caller frees result. */
static char *json_extract_string(const char *p) {
    if (!p || *p != '"') return NULL;
    p++;
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    size_t len = end - p;
    char *s = (char *)malloc(len + 1);
    memcpy(s, p, len);
    s[len] = '\0';
    return s;
}

/* List safetensors files available for a model.
 * Returns a NULL-terminated array of filenames (caller frees).
 * max_files limits the number of results. */
char **hf_hub_list_files(const char *model_id, int max_files) {
    char url[2048];
    snprintf(url, sizeof(url), "https://huggingface.co/api/models/%s", model_id);

    size_t sz;
    uint8_t *data = hf_download(url, &sz);
    if (!data) return NULL;

    /* Null-terminate the JSON */
    char *json = (char *)malloc(sz + 1);
    memcpy(json, data, sz);
    json[sz] = '\0';
    free(data);

    /* Find "siblings" array which contains file info */
    const char *siblings = json_find_key(json, "siblings");
    if (!siblings) { free(json); return NULL; }

    /* Count files with .safetensors extension */
    char **files = (char **)calloc(max_files + 1, sizeof(char *));
    int n_files = 0;

    const char *p = siblings;
    while (*p && n_files < max_files) {
        /* Look for "rfilename" key */
        const char *rfilename = strstr(p, "\"rfilename\"");
        if (!rfilename) break;
        const char *val = json_find_key(rfilename, "rfilename");
        /* Actually the key is at rfilename, value follows */
        val = rfilename + strlen("\"rfilename\"");
        while (*val && (*val == ' ' || *val == ':' || *val == '\t')) val++;
        if (*val == '"') {
            char *fname = json_extract_string(val);
            if (fname) {
                /* Check if it's a safetensors file */
                size_t flen = strlen(fname);
                if (flen > 12 && strcmp(fname + flen - 12, ".safetensors") == 0) {
                    files[n_files++] = fname;
                } else {
                    free(fname);
                }
            }
        }
        p = rfilename + 1;
    }

    free(json);
    return files;
}

/* ---- Download and load a model's safetensors weights ─── */

int hf_hub_load_model(const char *model_id, safetensors_t *st) {
    /* Download the main safetensors file */
    size_t sz;
    uint8_t *data = hf_hub_download(model_id, "model.safetensors", &sz);
    if (!data) {
        /* Try sharded format: model-00001-of-00001.safetensors */
        data = hf_hub_download(model_id, "model-00001-of-00001.safetensors", &sz);
    }
    if (!data) {
        fprintf(stderr, "[hf_hub] failed to download model weights\n");
        return -1;
    }

    /* Parse the safetensors data directly from memory */
    /* We need a memory-based parser since we have the data in memory */
    int rc = safetensors_load_from_memory(data, sz, st);
    free(data);
    return rc;
}
