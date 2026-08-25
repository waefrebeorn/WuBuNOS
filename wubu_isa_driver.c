/*
 * wubu_isa_driver.c -- the ISA driver registry (the driver space).
 *
 * The driver space (wubu_isa_driver.h): every ISA is a driver. The
 * frontend emits ONE mid-level IR (wubu_mir) and N backends consume
 * it. This file is the registry that maps names to drivers.
 *
 * C11, self-contained.
 */
#include "wubu_isa_driver.h"
#include <string.h>
#include <stdatomic.h>

/* the built-in drivers, one per ISA (all extern in the header) */
extern const wubu_isa_driver_t wubu_isa_x86_64;
extern const wubu_isa_driver_t wubu_isa_arm64;
extern const wubu_isa_driver_t wubu_isa_mips;
extern const wubu_isa_driver_t wubu_isa_m68k;
extern const wubu_isa_driver_t wubu_isa_i8086;
extern const wubu_isa_driver_t wubu_isa_riscv;
extern const wubu_isa_driver_t wubu_isa_6502;
extern const wubu_isa_driver_t wubu_isa_z80;
extern const wubu_isa_driver_t wubu_isa_8051;  /* Intel 8051 (1978, the $0.10 chip) */
extern const wubu_isa_driver_t wubu_isa_avr;   /* Atmel AVR (Arduino Uno) */
extern const wubu_isa_driver_t wubu_isa_pic;   /* Microchip PIC (PIC16F877A) */
extern const wubu_isa_driver_t wubu_isa_amdgpu; /* AMD GPU (RDNA2/gfx1030) */
extern const wubu_isa_driver_t wubu_isa_ptx;  /* NVIDIA GPU (PTX/SM89) */
extern const wubu_isa_driver_t wubu_isa_vulkan; /* ANY Vulkan card (SPIR-V) */

/* Portable cache clear for JIT'd code.
 * x86-64: hardware maintains i/d cache coherency — compiler barrier only.
 * ARM64: __builtin___clear_cache would be ideal; for hosted builds the
 *        kernel's mmap(PROT_EXEC) handles it. This is a C11 barrier. */
void wubu_clear_cache(void *addr, size_t size)
{
    (void)addr; (void)size;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" ::: "memory");
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    /* C11 fallback: atomic signal fence acts as compiler barrier */
    atomic_thread_fence(memory_order_seq_cst);
#endif
}

const wubu_isa_driver_t *wubu_isa_find(const char *name)
{
    if (!name) return NULL;
    if (!strcmp(name, "x86-64") || !strcmp(name, "x86_64") ||
        !strcmp(name, "x86")) return &wubu_isa_x86_64;
    if (!strcmp(name, "arm64") || !strcmp(name, "aarch64") ||
        !strcmp(name, "armv8") || !strcmp(name, "arm"))
        return &wubu_isa_arm64;
    if (!strcmp(name, "mips") || !strcmp(name, "mipsel") ||
        !strcmp(name, "mipseb") || !strcmp(name, "mips32"))
        return &wubu_isa_mips;
    if (!strcmp(name, "8051") || !strcmp(name, "mcs-51") ||
        !strcmp(name, "mcs51") || !strcmp(name, "intel-8051"))
        return &wubu_isa_8051;
    if (!strcmp(name, "avr") || !strcmp(name, "atmega") ||
        !strcmp(name, "atmega328p") || !strcmp(name, "arduino"))
        return &wubu_isa_avr;
    if (!strcmp(name, "pic") || !strcmp(name, "pic16") ||
        !strcmp(name, "pic16f877a") || !strcmp(name, "microchip"))
        return &wubu_isa_pic;
    if (!strcmp(name, "amdgpu") || !strcmp(name, "amd") ||
        !strcmp(name, "rdna") || !strcmp(name, "gfx1030") ||
        !strcmp(name, "radeon"))
        return &wubu_isa_amdgpu;
    if (!strcmp(name, "m68k") || !strcmp(name, "68000") ||
        !strcmp(name, "motorola-68000") || !strcmp(name, "68k"))
        return &wubu_isa_m68k;
    if (!strcmp(name, "8086") || !strcmp(name, "i8086") ||
        !strcmp(name, "x86-16") || !strcmp(name, "8088"))
        return &wubu_isa_i8086;
    if (!strcmp(name, "riscv") || !strcmp(name, "rv64i") ||
        !strcmp(name, "riscv64") || !strcmp(name, "rv"))
        return &wubu_isa_riscv;
    if (!strcmp(name, "6502") || !strcmp(name, "65c02") ||
        !strcmp(name, "w65c02") || !strcmp(name, "6502x"))
        return &wubu_isa_6502;
    if (!strcmp(name, "z80") || !strcmp(name, "zilog-z80") ||
        !strcmp(name, "z180") || !strcmp(name, "8080-compat"))
        return &wubu_isa_z80;
    if (!strcmp(name, "ptx") || !strcmp(name, "nvidia") ||
        !strcmp(name, "gpu") || !strcmp(name, "cuda"))
        return &wubu_isa_ptx;
    if (!strcmp(name, "vulkan") || !strcmp(name, "vk") ||
        !strcmp(name, "spirv") || !strcmp(name, "radv") ||
        !strcmp(name, "igpu") || !strcmp(name, "740m"))
        return &wubu_isa_vulkan;
    return NULL;
}