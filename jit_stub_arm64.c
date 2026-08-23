/* arm64 stub for standalone T_GEMM build — the real driver is in the OS repo. */
#include "wubu_isa_driver.h"
const wubu_isa_driver_t wubu_isa_arm64 = {
    .name = "arm64", .family = "native-jit", .exec = WUBU_ISA_NATIVE,
    .compile = NULL, .run = NULL, .describe = NULL,
};
