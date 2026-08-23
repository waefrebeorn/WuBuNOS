/* test_x86_peephole.c -- verify discovered peephole rules.
 * Exercises the self-MOV elimination and shrink_movabs paths that the
 * peephole_superopt/ discovery loop targets. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "x86_peephole.h"

static int fails=0, tot=0;

static void t(int cond, const char *msg){ tot++; if(!cond){fails++; printf("FAIL: %s\n",msg);} }

int main(void){
    /* 1. self-MOV elimination: 48 8B C0 (mov rax,rax) -> deleted (3 bytes) */
    uint8_t selfmov[] = {0x48,0x8B,0xC0, 0xC3};  /* + ret so we have 4 bytes */
    size_t n = x86_peephole_optimize(selfmov, sizeof selfmov);
    t(n==1 && selfmov[0]==0xC3, "self-mov rax,rax should be deleted");

    /* 2. mov r, r via REX.B variant: 41 8B D8 = mov r11,r11? (reg=3,rm=3,rex_b=1 -> r11) */
    /* REX=0x41 means REX.B=1, reg=3(rm=3),base=3 -> both r11. But is_rex(0x41)=yes(>=0x40<=0x4F) */
    uint8_t selfmov11[] = {0x45,0x8B,0xDB,  /* mov r11,r11 */ 0xC3};
    size_t n2 = x86_peephole_optimize(selfmov11, sizeof selfmov11);
    t(n2==1 && selfmov11[0]==0xC3, "self-mov r11,r11 should be deleted");

    /* 3. shrink_movabs: 48 B8 + imm32-in-range -> 5-byte mov */
    uint64_t v = 42;
    uint8_t mb[] = {0x48,0xB8, (uint8_t)(v&0xFF),(uint8_t)((v>>8)&0xFF),
                    (uint8_t)((v>>16)&0xFF),(uint8_t)((v>>24)&0xFF),
                    0,0,0,0};  /* 10 bytes, high 4 zero */
    size_t n3 = x86_peephole_optimize(mb, sizeof mb);
    t(n3==5 && mb[0]==0xB8, "movabs rax,42 should shrink to 5-byte mov");

    /* 4. non-self MOV must survive */
    uint8_t realmov[] = {0x48,0x8B,0xC1, 0xC3};  /* mov rax,rcx; ret */
    size_t n4 = x86_peephole_optimize(realmov, sizeof realmov);
    t(n4==4, "real mov rax,rcx must be kept");

    printf("=== %d/%d passed (peephole self-tests) ===\n", tot-fails, tot);
    return fails ? 1 : 0;
}
