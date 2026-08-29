#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "holyd_mir_eval.h"

int main(void) {
    const char *src = "int add(int a,int b){return a+b;} add(20,22);";
    int64_t r = hd_eval_mir(src, NULL);
    printf("result=%lld\n", (long long)r);
    return 0;
}
