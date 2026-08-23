/*
 * holyd_bin.c  --  Raw binary (flat binary) emitter for WuBuNOS HolyD compiler.
 *
 * Writes a flat binary image with no headers — just raw machine code.
 * Used for:
 *   - Bootloaders (load at 0x7C00)
 *   - Bare-metal kernels
 *   - Shellcode
 *   - Custom boot protocols
 *
 * The code is written as-is, starting at the load address.
 * For bootable images, pad to 512 bytes and add 0x55AA signature.
 */

#include "holyd_codegen.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Write a flat binary image.
 * If bootable != 0, pad to 512 bytes and add 0x55AA boot signature.
 */
int hd_write_bin(const char *filename,
                 const uint8_t *code, size_t code_size,
                 int bootable)
{
    if (!filename || !code || code_size == 0) return -1;

    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(code, 1, code_size, f);

    if (bootable) {
        /* Pad to 510 bytes, then add 0x55AA signature */
        if (code_size < 510) {
            uint8_t zero = 0;
            for (size_t i = code_size; i < 510; i++)
                fwrite(&zero, 1, 1, f);
        }
        uint8_t sig[2] = {0x55, 0xAA};
        fwrite(sig, 1, 2, f);
    }

    fclose(f);
    return 0;
}
