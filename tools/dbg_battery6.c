/* Isolate the globals-in-function bug. */
#include <stdio.h>
#include <stdint.h>
int wubu_run_program(const char *a, char **b, int c){ (void)a;(void)b;(void)c; return -1; }
extern int64_t hd_eval(const char *source);
static void t(const char *src, long long want){
    int64_t v = hd_eval(src);
    printf("got=%-6lld want=%-4lld %s  <- %s\n", (long long)v, want,
           v==want?"OK":"FAIL", src);
    fflush(stdout);
}
int main(void){
    t("int g=7; g;", 7);                                  /* global read module level */
    t("int f(){return 5;} f();", 5);                      /* plain func */
    t("int g=7; int f(){return 3;} f();", 3);             /* func after global */
    t("int g=7; int f(){return g;} f();", 7);
    t("int a[3]; a[0]=5; a[1]=7; int* p=a; p++; *p;", 7);
    t("int a[3]; a[0]=5; a[1]=7; int* p=a; p++; p--; *p;", 5);
    t("int a[3]; a[0]=5; a[1]=7; int* p=a; p++; p--; *p;", 5);
    t("int a[3]; a[0]=5; a[1]=7; int* p=a; p++; p--; *p;", 5);             /* THE BUG */
    return 0;
}
