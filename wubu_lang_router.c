/*
 * wubu_lang_router.c — Language routing implementation.
 * C11, self-contained.
 */
#include "wubu_lang_router.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <libgen.h>

/* ---- Routing table */

static const lang_routing_t routing_table[] = {
    {
        .language = LANG_HOLYC,
        .policy_flags = POLICY_ACCEPT,
        .description = HOLYC_DESC,
        .action_message = "[holyd] compiling natively on all targets",
        .container_image = NULL,
        .template_name = NULL,
    },
    {
        .language = LANG_BRAINFUCK,
        .policy_flags = POLICY_ACCEPT | POLICY_MEME | POLICY_AGI_READY,
        .description = BRAINFUCK_DESC,
        .action_message = "[brainfuck] the meme. compiled for real (x86-64 JIT).",
        .container_image = NULL,
        .template_name = NULL,
    },
    {
        .language = LANG_C,
        .policy_flags = POLICY_ACCEPT,
        .description = "C18 (C11 + bugfix patch) — accepted natively. No C23, no C++, no GNU extensions.",
        .action_message = "[c18] C18 accepted. HolyD C-subset mode. No higher than C18.",
        .container_image = NULL,
        .template_name = NULL,
    },
    {
        .language = LANG_CPP,
        .policy_flags = POLICY_REJECT,
        .description = CPP_DESC,
        .action_message = "[foreign] C++ detected. We only accept C18 (C11+bugfix). Route to container? (use -foreign)",
        .container_image = "silkeh/clang:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_RUST,
        .policy_flags = POLICY_CONTAINER,
        .description = RUST_DESC,
        .action_message = "[foreign] Rust detected. Routing to containerized rustc.",
        .container_image = "rust:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_GO,
        .policy_flags = POLICY_CONTAINER,
        .description = "Go (foreign) — routed to containerized go compiler.",
        .action_message = "[foreign] Go detected. Routing to containerized go.",
        .container_image = "golang:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_PYTHON,
        .policy_flags = POLICY_TEMPLATE | POLICY_WASM_FALLBACK,
        .description = PYTHON_DESC,
        .action_message = "[template] Python detected. JIT compiling to WASM.",
        .container_image = NULL,
        .template_name = "python_wasm",
    },
    {
        .language = LANG_JAVASCRIPT,
        .policy_flags = POLICY_TEMPLATE | POLICY_WASM_FALLBACK,
        .description = JS_DESC,
        .action_message = "[template] JavaScript detected. JIT compiling to WASM.",
        .container_image = NULL,
        .template_name = "js_wasm",
    },
    {
        .language = LANG_JAVA,
        .policy_flags = POLICY_CONTAINER,
        .description = "Java (foreign) — routed to containerized javac.",
        .action_message = "[foreign] Java detected. Routing to containerized javac.",
        .container_image = "openjdk:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_SWIFT,
        .policy_flags = POLICY_CONTAINER,
        .description = "Swift (foreign) — routed to containerized swiftc.",
        .action_message = "[foreign] Swift detected. Routing to containerized swiftc.",
        .container_image = "swift:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_FORTRAN,
        .policy_flags = POLICY_CONTAINER,
        .description = "Fortran (foreign) — routed to containerized gfortran.",
        .action_message = "[foreign] Fortran detected. Routing to containerized gfortran.",
        .container_image = "gfutils/fortran:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_ADA,
        .policy_flags = POLICY_CONTAINER,
        .description = "Ada (foreign) — routed to containerized gnat.",
        .action_message = "[foreign] Ada detected. Routing to containerized gnat.",
        .container_image = "karotte13/ada-gnat:latest",
        .template_name = NULL,
    },
    {
        .language = LANG_COBOL,
        .policy_flags = POLICY_CONTAINER,
        .description = "COBOL (foreign) — routed to containerized cobc.",
        .action_message = "[foreign] COBOL detected. Routing to containerized cobc.",
        .container_image = "gregcoleman/docker-cobol:latest",
        .template_name = NULL,
    },
};

#define N_ROUTES (sizeof(routing_table) / sizeof(routing_table[0]))

/* ---- Extension to language mapping */

typedef struct {
    const char *ext;
    lang_t      lang;
} ext_map_t;

static const ext_map_t ext_map[] = {
    {".hc",   LANG_HOLYC},
    {".holy", LANG_HOLYC},
    {".hcc",  LANG_HOLYC},
    {".bf",   LANG_BRAINFUCK},
    {".b",    LANG_BRAINFUCK},
    {".c",    LANG_C},
    {".h",    LANG_C},
    {".cpp",  LANG_CPP},
    {".cc",   LANG_CPP},
    {".cxx",  LANG_CPP},
    {".rs",   LANG_RUST},
    {".go",   LANG_GO},
    {".py",   LANG_PYTHON},
    {".js",   LANG_JAVASCRIPT},
    {".java", LANG_JAVA},
    {".swift",LANG_SWIFT},
    {".f90",  LANG_FORTRAN},
    {".f95",  LANG_FORTRAN},
    {".f03",  LANG_FORTRAN},
    {".adb",  LANG_ADA},
    {".cob",  LANG_COBOL},
    {".cbl",  LANG_COBOL},
};

/* ---- API implementation */

lang_t lang_detect(const char *filename, const char *source, size_t len) {
    if (!filename) return LANG_UNKNOWN;

    /* Check extension */
    const char *dot = strrchr(filename, '.');
    if (dot) {
        for (size_t i = 0; i < sizeof(ext_map) / sizeof(ext_map[0]); i++) {
            if (strcasecmp(dot, ext_map[i].ext) == 0)
                return ext_map[i].lang;
        }
    }

    /* Check shebang */
    if (source && len > 2 && source[0] == '#' && source[1] == '!') {
        if (strstr(source, "python")) return LANG_PYTHON;
        if (strstr(source, "node"))   return LANG_JAVASCRIPT;
        if (strstr(source, "bash"))   return LANG_UNKNOWN; /* no bash compilation */
    }

    /* Check for brainfuck signature (only +-><.,[] characters) */
    if (source && len > 0) {
        size_t bf_chars = 0, total = 0;
        for (size_t i = 0; i < len && i < 1000; i++) {
            char c = source[i];
            if (c == '+' || c == '-' || c == '>' || c == '<' ||
                c == '.' || c == ',' || c == '[' || c == ']') {
                bf_chars++;
            }
            if (!isspace((unsigned char)c)) total++;
        }
        if (total > 0 && bf_chars * 100 / total > 80)
            return LANG_BRAINFUCK;
    }

    return LANG_UNKNOWN;
}

const lang_routing_t *lang_route(lang_t lang) {
    for (size_t i = 0; i < N_ROUTES; i++) {
        if (routing_table[i].language == lang)
            return &routing_table[i];
    }
    return NULL;
}

const char *lang_describe(lang_t lang) {
    const lang_routing_t *r = lang_route(lang);
    return r ? r->description : UNKNOWN_DESC;
}

int lang_has_policy(lang_t lang, policy_flag_t flag) {
    const lang_routing_t *r = lang_route(lang);
    return r ? ((r->policy_flags & flag) != 0) : 0;
}

int lang_compile(const char *filename, const char *source, size_t len,
                 const char *output, const char *target_isa) {
    lang_t lang = lang_detect(filename, source, len);
    const lang_routing_t *route = lang_route(lang);

    if (!route) {
        fprintf(stderr, "[wubu] unknown language: %s\n", filename);
        return -1;
    }

    printf("%s\n", route->action_message);

    if (route->policy_flags & POLICY_ACCEPT) {
        /* Compile natively via holyd */
        return holyd_compile_native(filename, source, len, output, target_isa);
    }

    if (route->policy_flags & POLICY_TEMPLATE) {
        /* JIT template */
        return holyd_compile_template(route->template_name, source, len, output);
    }

    if (route->policy_flags & POLICY_CONTAINER) {
        /* Route to container */
        return holyd_compile_container(route->container_image, source, len, output);
    }

    if (route->policy_flags & POLICY_REJECT) {
        fprintf(stderr, "[wubu] %s\n", route->action_message);
        fprintf(stderr, "[wubu] tip: use -foreign flag to force containerized compilation\n");
        return -1;
    }

    return -1;
}

/* ---- Stub implementations (to be wired) */

int holyd_compile_native(const char *filename, const char *source, size_t len,
                          const char *output, const char *target_isa);
int holyd_compile_template(const char *template_name, const char *source, size_t len,
                            const char *output);
int holyd_compile_container(const char *image, const char *source, size_t len,
                              const char *output);

int holyd_compile_native(const char *filename, const char *source, size_t len,
                          const char *output, const char *target_isa) {
    /* Existing holyd compilation path */
    (void)filename; (void)source; (void)len; (void)output; (void)target_isa;
    return 0;
}

int holyd_compile_template(const char *template_name, const char *source, size_t len,
                            const char *output) {
    (void)template_name; (void)source; (void)len; (void)output;
    return 0;
}

int holyd_compile_container(const char *image, const char *source, size_t len,
                              const char *output) {
    (void)image; (void)source; (void)len; (void)output;
    return 0;
}
