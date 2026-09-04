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
    {"nested call", "int f(int x){return x+1;} f(f(f(0)));", 3},
    {"recursion", "int fact(int n){ if(n<=1) return 1; return n*fact(n-1); } fact(5);", 120},
    {"break", "int i=0; while(1){i++; if(i>=3) break;} i;", 3},
    {"continue", "int s=0; for(int i=0;i<5;i++){ if(i==2) continue; s+=i; } s;", 8},
    {"do-while", "int i=0; do{i++;}while(i<3); i;", 3},
    {"float add", "float a=1.5; float b=2.5; a+b;", 1082130432},
    {"nested struct", "struct A{int x;}; struct A a; a.x=42; a.x;", 42},
    {"struct return", "struct A{int x;}; struct A f(int v){struct A r; r.x=v; return r;} struct A t; t=f(7); t.x;", 7},
    {"struct return 2m", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} struct S r; r=f(); r.a+r.b;", 49},
    {"struct return member", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} f().a;", 42},
    {"struct return member b", "struct S{int a;int b;}; struct S f(){struct S s; s.a=42; s.b=7; return s;} f().b;", 7},
    {"sizeof struct", "struct S{int a;int b;}; sizeof(struct S);", 8},
    {"sizeof nested struct", "struct P{int a;}; struct Q{struct P p; int b;}; sizeof(struct Q);", 8},
    {"sizeof member", "struct S{int a;int b;}; struct S s; sizeof(s.a);", 4},
    {"sizeof var", "int x; sizeof(x);", 4},
    {"sizeof array", "int a[3]; sizeof a;", 12},
    {"struct pass by value", "struct S{int a;int b;int c;}; int f(struct S x){return x.a+x.b+x.c;} struct S s; s.a=10; s.b=20; s.c=30; f(s);", 60},
    {"struct+int args", "struct S{int a;int b;int c;}; int f(struct S x, int n){return x.a+x.b+x.c+n;} struct S s; s.a=10; s.b=20; s.c=30; f(s,-8);", 52},
    {"deref int* struct", "struct S{int a;int b;}; struct S s; s.a=10; s.b=20; int* p=&s.a; *p;", 10},
    {"multi-arg call", "int f(int a,int b,int c){return a+b+c;} f(10,20,12);", 42},
    {"deep nest", "int f(int x){return x+1;} f(f(f(f(f(0)))));", 5},
    {"scope shadow", "int x=1; {int x=2; x;} x;", 1},
    {"comma op", "int a=0,b=0; (a=1,b=2); a+b;", 3},
    /* ---- additional struct edge cases ---- */
    {"struct assign", "struct S{int a;int b;}; struct S s; s.a=42; s.b=7; struct S r; r=s; r.a+r.b;", 49},
    {"struct return chain", "struct S{int a;}; struct S f(int v){struct S r; r.a=v; return r;} f(99).a;", 99},
    {"nested struct member", "struct P{int x;}; struct Q{struct P p; int y;}; struct Q q; q.p.x=42; q.y=7; q.p.x+q.y;", 49},
    {"struct sizeof longlong", "struct S{long long a; long long b;}; sizeof(struct S);", 16},
    {"struct pass 16B", "struct S{long long a; long long b;}; long long f(struct S x){return x.a+x.b;} struct S s; s.a=9; s.b=8; f(s);", 17},
    {"int+struct args", "struct S{int a;int b;int c;}; int f(int n, struct S x){return n+x.a+x.b+x.c;} struct S s; s.a=10; s.b=20; s.c=30; f(-8,s);", 52},
    {"2 struct args", "struct S{int a;int b;int c;}; int f(struct S x, struct S y){return x.a+x.b+y.c;} struct S s; s.a=10; s.b=20; s.c=30; f(s,s);", 60},
    {"arg+ret nested", "struct S{int a;int b;int c;}; int f(struct S x){return x.a+x.b+x.c;} struct S g(){struct S s; s.a=10; s.b=20; s.c=30; return s;} f(g());", 60},
    {"deref int* struct b", "struct S{int a;int b;}; struct S s; s.a=10; s.b=20; int* p=&s.b; *p;", 20},
    {"global struct", "struct S{int a;int b;}; struct S g; g.a=42; g.b=7; g.a+g.b;", 49},
    {"struct ptr arrow", "struct S{int a;int b;}; struct S s; s.a=42; s.b=7; struct S* p=&s; p->a+p->b;", 49},
    {"fn ptr member", "struct S{int (*fn)(int,int); int n;}; int add(int x,int y){return x+y;} struct S s; s.fn=add; s.n=5; s.fn(3,4);", 7},
    {"array member", "struct S{int a[3]; int n;}; struct S s; s.a[0]=1; s.a[1]=2; s.a[2]=3; s.n=3; s.a[0]+s.a[1]+s.a[2];", 6},
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