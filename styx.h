#ifndef STYX_H
#define STYX_H
#include <stdint.h>
/* Minimal Styx/9P protocol constants for gap_audit standalone build. */
#define STYX_MAX_MSG 8192
#define STX_TVERSION 100
#define STX_TATTACH 102
#define STX_TCLUNK 108
#define STX_TREMOVE 109
#define STX_TOPEN 110
#define STX_TCREATE 112
#define STX_TREAD 116
#define STX_TWRITE 113
#define STX_TWALK 118
#define STX_TWSTAT 120
#define STX_TAUTH 101
#define STX_RERROR 107
typedef struct { uint8_t *buf; uint32_t len; } styx_msg_t;
typedef void *styx_server_t;
#endif
