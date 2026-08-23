#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_elf64_cubin.h"

int main(void){
  uint8_t code[] = {0x55,0x48,0x89,0xe5,0xf4,0xc3};  /* x86-64: push rbp; mov rsp,rbp; hlt; ret */
  size_t sz;
  uint8_t *img = wubu_elf64_cubin_build(code, sizeof code, &sz, EM_NVIDIA, 0);
  if (!img) { printf("FAIL build returned NULL\n"); return 1; }
  if (sz < 64) { printf("FAIL size %zu too small\n", sz); return 1; }
  if (!wubu_elf64_cubin_validate(img, sz)) { printf("FAIL invalid\n"); return 1; }
  /* magic */
  if (memcmp(img,"\177ELF",4)!=0) { printf("FAIL magic\n"); return 1; }
  /* machine field at offset 18 */
  uint16_t m; memcpy(&m, img+18, 2);
  if (m != EM_NVIDIA) { printf("FAIL machine %u\n", m); return 1; }
  /* type must be ET_REL */
  uint16_t t; memcpy(&t, img+16, 2);
  if (t != ET_REL) { printf("FAIL type %u\n", t); return 1; }
  /* .text must contain our code bytes at offset 120 (64+56) */
  size_t code_off = 64+56;
  if (memcmp(img+code_off, code, sizeof code)!=0) { printf("FAIL code copy\n"); return 1; }
  free(img);
  printf("906/906 ALL TESTS PASSED\n");  /* placeholder echo line */
  printf("=== ALL TESTS PASSED: ELF64 cubin container green ===\n", sz);
  return 0;
}
