/*
 * test_mir_battery.c — Self-hosting battery via MIR pipeline.
 *
 * Tests the same C11 constructs as selfhost_battery.c, but uses
 * hd_eval_mir() (MIR path) instead of hd_eval() (legacy direct codegen).
 *
 * MIR path: HolyD source → parser → MIR → interpreter (oracle)
 * This tests the modern compiler pipeline end-to-end.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include "wubu_mir.h"
#include "wubu_isa_driver.h"
#include "holyd_mir_eval.h"

typedef struct {
    const char *name;
    const char *src;
    long long  expect;
} Probe;

static const Probe PROBES[] = {
    /* ---- preprocessor ---- */
    {"#define object macro", "#define KVFS_SLOT_LIVE 1\nKVFS_SLOT_LIVE;", 1},
    {"#define with expr", "#define M 3\nM*14;", 42},
    {"#define fn-like macro", "#define R8(n) ((n+7)-(7))\nR8(42);", 42},
    /* ---- arithmetic ---- */
    {"add", "20+22;", 42}, {"sub", "50-8;", 42}, {"mul", "6*7;", 42},
    {"div", "84/2;", 42}, {"mod", "45%3;", 0}, {"neg", "-42;", -42},
    {"paren", "(2+3)*4-6;", 14}, {"prec mul over add", "2+3*4;", 14},
    {"prec paren", "(2+3)*4;", 20},
    /* ---- bitwise ---- */
    {"bitand", "42 & 63;", 42}, {"bitor", "40 | 2;", 42},
    {"bitxor", "43 ^ 1;", 42}, {"shl", "21 << 1;", 42},
    {"shr", "84 >> 1;", 42}, {"bitnot", "~0;", -1},
    /* ---- comparison ---- */
    {"eq", "42==42;", 1}, {"ne", "42!=0;", 1}, {"lt", "41<42;", 1},
    {"le", "42<=42;", 1}, {"gt", "43>42;", 1}, {"ge", "42>=42;", 1},
    {"logic and", "1&&1;", 1}, {"logic or", "0||1;", 1},
    {"logic not", "!0;", 1}, {"ternary", "(1)?42:0;", 42},
    {"ternary false", "(0)?0:42;", 42},
    /* ---- compound assignment ---- */
    {"+=", "int v=30; v+=12; v;", 42},
    {"-=", "int v=50; v-=8; v;", 42},
    {"*=", "int v=6; v*=7; v;", 42},
    {"/=", "int v=84; v/=2; v;", 42},
    {"%=", "int v=45; v%=3; v;", 0},
    {"<<=", "int v=21; v<<=1; v;", 42},
    {">>=", "int v=84; v>>=1; v;", 42},
    {"var decl + use", "int x=40; x+2;", 42},
    {"multiple vars", "int a=1; int b=2; int c=3; a+b+c+36;", 42},
    {"shadow reassign", "int x=10; x=42; x;", 42},
    {"chained assign", "int a; int b; a=b=42; a;", 42},
    /* ---- functions ---- */
    {"func call", "int sq(int n){return n*n;} sq(6)+6;", 42},
    {"func 2 params", "int add(int a,int b){return a+b;} add(20,22);", 42},
    {"func local vars", "int f(int n){ int x=n*2; return x+2; } f(20);", 42},
    /* ---- control flow ---- */
    {"if", "if(1){42;}else{0;}", 42},
    {"if-else", "if(0){0;}else{42;}", 42},
    {"while", "int i=0; while(i<3){i++;} i;", 3},
    {"for", "int s=0; for(int i=0;i<3;i++){s+=i;} s;", 3},
    {"return expr", "int f(){return 42;} f();", 42},
    /* ---- arrays ---- */
    {"int array", "int a[3]; a[0]=1;a[1]=2;a[2]=3; a[2];", 3},
    {"array init", "int a[]={1,2,3}; a[2];", 3},
    /* ---- pointers ---- */
    {"ptr deref", "int x=42; int*p=&x; *p;", 42},
    {"ptr arith", "int a[]={1,2,3}; int*p=a; p++; *p;", 2},
    /* ---- structs ---- */
    {"struct decl", "struct S{int a;}; struct S s; s.a=42; s.a;", 42},
    {"struct member", "struct S{int a;int b;}; struct S s; s.a=1; s.b=2; s.a+s.b;", 3},
    /* ---- sizeof ---- */
    {"sizeof int", "sizeof(int);", 4},
    /* ---- string ---- */
    {"string literal", "\"hello\"[0];", 104},
    {"char array", "char s[4]; s[0]='h'; s[1]='i'; s[2]=0; s[0];", 104},
    /* ---- globals ---- */
    {"global var", "int g=7; int f(){return g;} f();", 7},
    {NULL, NULL, 0}
};

#define NPROBES ((int)(sizeof(PROBES)/sizeof(PROBES[0])) - 1)

static int run_isolated(const Probe *probe)
{
    const char *src = probe->src;
    long long expect = probe->expect;
    int pipefd[2];
    if (pipe(pipefd) != 0) return 2;
    pid_t pid = fork();
    if (pid < 0) return 2;
    if (pid == 0) {
        close(pipefd[0]);
        long long r = hd_eval_mir(src, NULL); /* NULL = interpreter */
        (void)!write(pipefd[1], &r, sizeof(r));
        close(pipefd[1]);
        _exit(0);
    }
    close(pipefd[1]);
    long long result = 0;
    int got = (int)read(pipefd[0], &result, sizeof(result));
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (got != (int)sizeof(result)) return 2;
    if (WIFSIGNALED(status)) return 2;
    return (result == expect) ? 0 : 1;
}

int main(void)
{
    printf("=== MIR PIPELINE SELF-HOSTING BATTERY ===\n");
    printf("HolyD → MIR → interpreter (oracle)\n\n");

    int pass = 0, wrong = 0, crash = 0;

    for (int i = 0; i < NPROBES; i++) {
        int rc = run_isolated(&PROBES[i]);
        if (rc == 0) {
            pass++;
            printf("  PASS  %-22s\n", PROBES[i].name);
        } else if (rc == 1) {
            wrong++;
            printf("  WRONG %-22s (expect %lld)\n", PROBES[i].name, PROBES[i].expect);
        } else {
            crash++;
            printf("  CRASH %-22s\n", PROBES[i].name);
        }
    }

    printf("\n=== RESULTS ===\n");
    printf("PASS:  %d\n", pass);
    printf("WRONG: %d\n", wrong);
    printf("CRASH: %d\n", crash);
    printf("TOTAL: %d\n", NPROBES);
    return (wrong + crash) ? 1 : 0;
}
