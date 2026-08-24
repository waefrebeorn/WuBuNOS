#ifndef STYX_INTERNAL_H
#define STYX_INTERNAL_H
#include "styx.h"
/* Internal Styx server state — stubbed for gap_audit standalone. */
static inline void styx_init(styx_server_t *srv) { (void)srv; }
static inline int styx_serve(styx_server_t *srv, const uint8_t *in, uint32_t inlen, uint8_t *out, uint32_t *outlen) {
    (void)srv; (void)in; (void)inlen;
    if (outlen) *outlen = 0;
    return 0;
}
#endif
