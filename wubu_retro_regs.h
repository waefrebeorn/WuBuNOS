/*
 * wubu_retro_regs.h -- shared imaginary register file for the 8-bit/retro
 * ISA drivers (6502, Z80, 8086, ...).
 *
 * The 1970s machines each had a handful of real registers (6502: A/X/Y/S/P;
 * 8086: AX/BX/CX/DX/SI/DI/BP/SP). To lower the SAME MIR on every target we
 * pretend each virtual register owns a "cell" in an imaginary register file
 * and materialize that cell as the cheapest addressable slot on the target:
 *
 *   6502  -> zero-page byte  (zp[vr+1], 256 cells max, 1 byte each)
 *   8086  -> stack slot      ([bp - (vr+1)*2], 16-bit cells)
 *
 * The accumulator (6502 A / 8086 AX) is the single implicit operand register
 * that flows through every ALU op — the slot file just holds *state*, the
 * accumulator holds *transient values*. This is exactly the llvm-mos model
 * the Kevin-Bacon wave flagged: "imaginary register bank" = treat the
 * addressing window as a register file.
 *
 * This header formalizes that bank as a compile-time layout so all retro
 * drivers share one allocator story (max_vr → frame size) and so the
 * interpreter's memory map matches the emitter's slot map.
 *
 * No malloc. Fixed-size, constexpr layouts.
 */
#ifndef WUBU_RETRO_REGS_H
#define WUBU_RETRO_REGS_H

#include "wubu_isa_driver.h"
#include <stdint.h>

/* ---- 6502 imaginary register file (zero-page) ----
 * vr 0..max -> zp byte (vr+1). slot 0 is scratch.
 * 6502 zero page is 256 bytes; reserve 254 for vrs (slots 1..254), slot 255 = scratch. */
#define ZP_MAX_VR   254
static inline uint8_t zp_slot(wubu_vr_t vr) {
    /* vr+1; clamp so 6502 programs never walk past zp into the stack page */
    uint32_t s = (uint32_t)vr + 1;
    if (s > 254) s = 254;   /* overflow into scratch */
    return (uint8_t)s;
}
#define ZP_SCRATCH  255

/* ---- float operand slots (6502): 4-byte little-endian cells ----
 * Integers use zp_slot(vr) (1 byte). Float vrs instead live in a
 * high-ZP region aligned +4 so the 4 bytes never overlap a neighbour.
 * Float vrs are dense in [0..max_vr]; the slot base is 0xFC - vr*4
 * scanning downward from the top, clamped into the 0x80..0xFD window. */
static inline uint8_t zp_fslot(wubu_vr_t vr) {
    uint32_t off = (uint32_t)vr * 4u;
    uint32_t base = 0xFCu;                 /* top of ZP, 4-aligned */
    if (off >= (base - 0x80u)) return 0x80u;  /* clamp: low float region */
    return (uint8_t)(base - off);
}

/* ---- Z80 imaginary register file (direct 16-bit memory slots) ----
 * Z80 has a full 16-bit address space; vrs live in 2-byte memory slots.
 * slot_addr(vr) = vr*2 (little-endian 16-bit cells). */
#define Z80_SLOT_BASE 0x0000      /* vrs start at address 0; interpreter reserves 0..frame */
static inline uint16_t z80_slot_addr(wubu_vr_t vr) { return (uint16_t)((uint32_t)vr * 2); }
static inline size_t z80_frame_size(size_t max_vr) { return ((max_vr + 1) * 2); }

/* ---- 8086 imaginary register file (stack slots below BP) ----
 * Each vr -> 16-bit cell at [bp - (vr+1)*2].
 * frame = (max_vr+1)*2 + safety. */
static inline int16_t slot_disp(wubu_vr_t vr) {
    return (int16_t)(-((int64_t)(vr + 1) * 2));
}
#define SLOT_FRAME_EXTRA 32   /* bytes of scratch/stack safety margin */

/*
 * Compute the caller's stack-frame size from max_vr for the 8086 driver.
 * Exposed so wubu_isa_8086.c and the interpreter agree.
 */
static inline size_t i8086_frame_size(size_t max_vr) {
    return (max_vr + 1) * 2 + SLOT_FRAME_EXTRA;
}

/*
 * Compute the zero-page usage ceiling for the 6502 driver (number of
 * zero-page bytes claimed beyond the scratch slot).
 */
static inline size_t zp_frame_size(size_t max_vr) {
    if (max_vr > ZP_MAX_VR) max_vr = ZP_MAX_VR;
    return max_vr + 2;  /* slots 1..max_vr+1, plus scratch 255 */
}

#endif /* WUBU_RETRO_REGS_H */
