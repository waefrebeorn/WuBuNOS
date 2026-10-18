/* test_lexer_suffix.c — COMPILER #114.
 *
 * Suffixed integer literals must classify to an unsigned/int width:
 *   - plain, U        -> 32-bit
 *   - L, LL, UL, ULL  -> 64-bit (SysV x86-64 long)
 *   - hex and binary literals must accept the same suffixes
 *   - the 64-bit bit pattern must survive (strtoull, not strtoll)
 */
#include "holyd_lexer.h"
#include <stdio.h>
#include <stdint.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

static int lit(const char *src, int64_t val, int uns, int width, int ll)
{
    HDLexer lx;
    hd_lex_init(&lx, src);   /* init primes the first token */
    HDTokenType t = lx.tok.type;
    if (t != HD_TOK_INT) {
        printf("    (%s -> not an int token)\n", src);
        return 0;
    }
    return lx.tok.int_val == val &&
           lx.tok.is_unsigned == uns &&
           hd_int_width(&lx.tok) == width &&
           lx.tok.is_long_long == ll;
}

int main(void)
{
    CHECK(lit("42", 42, 0, 32, 0), "42 -> 32-bit signed");
    CHECK(lit("42U", 42, 1, 32, 0), "42U -> 32-bit unsigned");
    CHECK(lit("42L", 42, 0, 64, 0), "42L -> 64-bit signed");
    CHECK(lit("42LL", 42, 0, 64, 1), "42LL -> 64-bit long long");
    CHECK(lit("42ULL", 42, 1, 64, 1), "42ULL -> 64-bit unsigned long long");
    CHECK(lit("42LLU", 42, 1, 64, 1), "42LLU -> 64-bit unsigned long long");
    CHECK(lit("42ul", 42, 1, 64, 0), "42ul -> 64-bit unsigned long (lowercase)");

    CHECK(lit("0xFFULL", 255, 1, 64, 1), "0xFFULL -> 64-bit unsigned");
    CHECK(lit("0b1010ULL", 10, 1, 64, 1), "0b1010ULL -> 64-bit unsigned (binary suffix)");
    CHECK(lit("0xFFFFFFFFFFFFFFFFULL", (int64_t)-1, 1, 64, 1),
          "0xFFFFFFFFFFFFFFFFULL keeps the full 64-bit pattern");

    CHECK(lit("9223372036854775807LL", INT64_MAX, 0, 64, 1),
          "INT64_MAX with LL suffix");

    HDLexer lx;
    hd_lex_init(&lx, "1 + 2");
    CHECK(lx.tok.type == HD_TOK_INT && hd_int_width(&lx.tok) == 32,
          "expression context still classifies plain literals as 32-bit");
    CHECK(hd_int_width(NULL) == 0, "NULL token -> width 0");

    printf("\n=== test_lexer_suffix: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}