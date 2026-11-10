/* tools/selfbuild_checksum.hc -- the pointer-walk stage, compiled natively.
 * Same body as src/runtime/selfbuild/checksum.hd but with a `main` entry so
 * the thin ELF emitter can compile it.  Returns 323 when run on the CPU,
 * matching what the wubuos loader gets for checksum.hd -- proof the emitted
 * machine code and the loader agree on pointer deref + .data string reads. */
int main() {
    char *s = "WUBU";
    char *p = s;
    int c = 0;
    while (*p != 0) {
        c = c + (int)*p;
        p = p + 1;
    }
    return c;
}
