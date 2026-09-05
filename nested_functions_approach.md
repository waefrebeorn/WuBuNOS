# Nested Functions in GCC / HolyD — Research & Approach
# =====================================================
#
# 1. HOW GCC IMPLEMENTS NESTED FUNCTIONS
# --------------------------------------
#
# A "nested function" is a function defined inside another function body:
#
#     int main() {
#         int f(int x) { return x; }
#         return f(42);
#     }
#
# GCC implements this via a mechanism called "trampolines":
#
# a) The nested function body is compiled as a NORMAL static function with
#    an internal name (e.g., "f.N"). It receives the enclosing function's
#    frame pointer (or chain of frame pointers) as a hidden extra argument
#    (passed in r10 on x86-64).
#
# b) When the address of the nested function is taken (e.g., passed as a
#    callback), GCC generates a small piece of executable code on the STACK
#    called a "trampoline" (typically ~24 bytes on x86-64). The trampoline:
#    - Loads the static function address into a register (r11)
#    - Loads the static chain (parent frame pointer) into another register (r10)
#    - Jumps to the static function
#
# c) The trampoline requires an EXECUTABLE STACK. This is the critical
#    issue: GCC stops emitting the .note.GNU-stack section, which causes
#    the linker to mark the entire binary as needing RWE (read-write-execute)
#    stack. This is a security concern (W^X violation).
#
# d) When the nested function is called DIRECTLY (not via pointer), GCC
#    can optimize away the trampoline and just pass the frame pointer
#    directly as the hidden argument.
#
# e) Nested functions have NO linkage — they cannot be declared static
#    or extern. Their name is local to the block where defined.
#
# f) Nested functions can access all variables of the enclosing function
#    that are visible at the point of definition (lexical scoping/closures).
#
# g) Jumping to a label in the enclosing function from a nested function
#    is supported (non-local goto that unwinds the nested frames).
#
#
# 2. HOW TO DETECT NESTED FUNCTION DEFINITIONS DURING PARSING
# ------------------------------------------------------------
#
# In HolyD's parser (holyd_parse.c), the key detection point is in
# parse_block() (line 856) and parse_stmt() (line 891).
#
# A nested function is detected when:
#   - We're inside a function body (parse_stmt was called from within
#     a FUNC_DECL body, i.e., the parser is inside parse_block that was
#     invoked by the FUNC_DECL handler)
#   - AND we encounter a statement that looks like a function definition:
#     type name(params) { body }
#
# Detection strategy:
#   - Add a `bool in_function_body` flag to HDParser (or HCGen for codegen).
#   - Set it to true when entering parse_block() called from the FUNC_DECL
#     codegen path.
#   - In parse_stmt(), when we see a type keyword followed by an identifier
#     followed by '(', check if we're already inside a function body.
#   - If so, this is a nested function definition.
#
# The parser ALREADY detects function definitions in parse_stmt() at line
# 1059-1065 (the "Variable declaration" check) and line 1334 (the
# `if (peek(p) == HD_TOK_LPAREN && !is_func_ptr)` check in hd_parse_decl).
# The issue is that when these are encountered inside a block (function body),
# they are parsed as FUNC_DECL nodes and the JIT tries to compile them as
# top-level functions, which crashes because the context is wrong.
#
# Specifically, in parse_block() → parse_stmt(), the flow is:
#   1. Token is a type keyword (e.g., "int")
#   2. Falls through to the "Variable declaration" check at line 1060-1065
#   3. Calls hd_parse_decl() which parses it as a FUNC_DECL
#   4. Returns the FUNC_DECL node as a statement in the block
#   5. The codegen then tries to compile it, but the nested function's
#      stack frame setup conflicts with the enclosing function's frame
#
#
# 3. APPROACH OPTIONS
# -------------------
#
# OPTION A: GRACEFUL SKIP (RECOMMENDED for now)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Detect nested functions during parsing and emit a clear error message
# instead of crashing. This is the safest, simplest approach.
#
# Implementation:
#   1. Add `bool in_function_body` to HDParser struct (holyd_types.h)
#   2. In the FUNC_DECL codegen path (holyd_codegen_stmt.c, case
#      HD_AST_FUNC_DECL), set `p->in_function_body = true` before
#      calling gen_stmt on the body, then restore to false after.
#   3. In parse_stmt(), when we detect a function definition inside
#      a function body (in_function_body && type + ident + LPAREN),
#      emit a parse_error("nested functions are not supported") and
#      skip the nested function body (advance past the closing brace).
#   4. In the gauntlet runner, mark nested-function tests as
#      "expected skip" rather than failures.
#
# Pros: Simple, safe, no security implications, clear error messages.
# Cons: Doesn't support the feature.
#
#
# OPTION B: SUPPORT SIMPLE NESTED FUNCTIONS (NO CLOSURES)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Support nested functions that DON'T access enclosing scope variables.
# These are essentially static functions with block scope.
#
# Implementation:
#   1. When detecting a nested function, check if it references any
#      variables from the enclosing scope.
#   2. If it does NOT reference enclosing variables:
#      - Compile it as a normal static function with a unique internal name
#      - When called directly, just emit a normal call (no trampoline needed)
#      - The function gets its own stack frame, parameters passed normally
#   3. If it DOES reference enclosing variables, fall back to error (Option A).
#
# The key insight: for non-closing nested functions, the implementation
# is trivial — they're just regular functions whose scope is limited.
# No trampoline, no executable stack, no closure semantics needed.
#
# Example that WOULD work:
#     int main() {
#         int f(int x) { return x; }   // no access to main's locals
#         return f(42);
#     }
#
# Example that would NOT work (would error gracefully):
#     int main() {
#         int offset = 5;
#         int f(int x) { return x + offset; }  // accesses enclosing var
#         return f(42);
#     }
#
# Pros: Handles the common case in gcc torture tests (most nested functions
#       in the tests are simple and don't use closures).
# Cons: Still need to detect and reject closures; more complex than Option A.
#
#
# OPTION C: FULL TRAMPOLINE SUPPORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Implement the full GCC trampoline mechanism.
#
# This requires:
#   1. Generating a unique static function for each nested function
#   2. Passing the parent frame pointer as a hidden argument (r10)
#   3. When the address is taken, generating executable trampoline code on
#      the stack (or in a writable-then-executable memory region)
#   4. Making the JIT's stack executable (mprotect with PROT_READ|PROT_WRITE|PROT_EXEC)
#   5. Handling non-local gotos (setjmp/longjmp or manual frame unwinding)
#
# This is COMPLEX and has security implications (executable stack).
# NOT recommended for a kernel compiler.
#
#
# 4. RECOMMENDED APPROACH: OPTION B (Simple subset + graceful error for closures)
# ------------------------------------------------------------------------------
#
# Step 1: Add tracking to the parser
#   - Add `int nest_depth` to HDParser (holyd_types.h)
#   - Increment when entering a FUNC_DECL body, decrement when leaving
#
# Step 2: Detect nested functions in parse_block/parse_stmt
#   - When nest_depth > 0 and we see a function definition:
#     a) Parse it normally (the parser already handles FUNC_DECL)
#     b) Tag the AST node with a flag: `bool is_nested` on HDASTNode
#     c) OR: immediately check for closure usage and error if found
#
# Step 3: In codegen, handle nested FUNC_DECL
#   - When gen_stmt encounters HD_AST_FUNC_DECL and nest_depth > 0:
#     a) Check if the nested function body references any variable from
#        the enclosing scope (scan the AST for IDENT nodes that resolve
#        to the enclosing function's symbol table)
#     b) If no closure: compile as a normal function (the existing code
#        already handles FUNC_DECL by saving/restoring state). Just need
#        to ensure the function name is unique (prefix with parent name).
#     c) If closure detected: emit a runtime error or compile-time error
#
# Step 4: Handle the gauntlet tests
#   - The gauntlet has several nested function tests:
#     - extern_gcc_050312_nestfunc_2: nested function passed as callback (closure)
#     - extern_gcc_050593_20061220_1: nested function + nested nested function
#     - extern_gcc_050616_nestfunc_3: nested functions calling each other
#     - extern_gcc_050371_pr22061_3: nested function with VLA parameter
#   - Most of these use closures (access enclosing variables), so they
#     would fall under the "graceful error" path.
#
#
# 5. KEY GAUNTLET TESTS AND EXPECTED BEHAVIOR
# -------------------------------------------
#
# Test                                    | Closure? | Option B result
# ----------------------------------------|----------|------------------
# extern_gcc_050312_nestfunc_2             | YES      | Error (graceful)
# extern_gcc_050593_20061220_1             | YES*     | Error (graceful)
# extern_gcc_050616_nestfunc_3             | YES      | Error (graceful)
# extern_gcc_050371_pr22061_3              | YES      | Error (graceful)
# writing_c_compiler_066060 (incomplete)   | N/A      | Parse error (already)
#
# *The 20061220_1 test uses asm volatile with "r"(x) which references
# the enclosing scope variable — effectively a closure via inline asm.
#
# The SIMPLE case from the user's example:
#     int main() { int f(int x){return x;} return f(42); }
# This has NO closure — f only uses its own parameter. Option B would
# compile this successfully.
#
#
# 6. IMPLEMENTATION DETAILS FOR OPTION B
# --------------------------------------
#
# 6a. Parser changes (holyd_parse.c):
#
#   // In HDParser struct (holyd_types.h), add:
#   int nest_depth;  // >0 when parsing inside a function body
#
#   // In parse_block(), at the top:
#   // (no change needed — nest_depth is set by the FUNC_DECL codegen)
#
#   // In parse_stmt(), in the "Variable declaration" section (line 1059-1065):
#   // When nest_depth > 0 and hd_parse_decl returns a FUNC_DECL,
#   // we've found a nested function. The parser already parsed it correctly.
#   // We just need to tag it or check for closures here.
#
# 6b. Closure detection:
#
#   // Walk the nested function's body AST looking for IDENT nodes
#   // that resolve to symbols in the enclosing function's scope.
#   // If found, it's a closure → error.
#   bool detect_closure(HDASTNode *node, HDSymTab *enclosing_scope) {
#       if (!node) return false;
#       if (node->kind == HD_AST_IDENT) {
#           for (int i = 0; i < enclosing_scope->n_locals; i++) {
#               if (strcmp(node->ident, enclosing_scope->locals[i].name) == 0)
#                   return true;  // references enclosing variable
#           }
#       }
#       // Recurse into children
#       if (detect_closure(node->child, enclosing_scope)) return true;
#       if (detect_closure(node->left, enclosing_scope)) return true;
#       if (detect_closure(node->right, enclosing_scope)) return true;
#       for (int i = 0; i < node->n_stmts; i++)
#           if (detect_closure(node->stmts[i], enclosing_scope)) return true;
#       return false;
#   }
#
# 6c. Codegen changes (holyd_codegen_stmt.c):
#
#   // In the HD_AST_FUNC_DECL case:
#   // Before compiling the body, check if this is a nested function
#   // (gen->nest_depth > 0). If so, check for closures.
#   // If no closure: proceed with normal compilation but use a
#   // mangled name like "parentname__nestedname" to avoid collisions.
#   // If closure: emit an error message and a ret instruction.
#
# 6d. Name mangling for nested functions:
#
#   // When compiling a nested function, prefix its name with the
#   // enclosing function's name to ensure uniqueness:
#   char mangled_name[HD_MAX_IDENT_LEN];
#   snprintf(mangled_name, sizeof(mangled_name), "%s__%s",
#            gen->current_function, node->ident);
#   // Use mangled_name when recording in gen->functions[]
#
#
# 7. SUMMARY
# ----------
#
# RECOMMENDED: Implement Option B (simple nested functions without closures)
# with graceful error for closures.
#
# This handles the user's example case:
#     int main() { int f(int x){return x;} return f(42); }  → COMPILES
#
# And gracefully rejects:
#     int main() { int k=5; int f(int x){return x+k;} return f(42); }  → ERROR
#
# The implementation is ~100 lines of code across 3 files:
#   holyd_types.h: add nest_depth + is_nested fields
#   holyd_parse.c: detect and tag nested functions
#   holyd_codegen_stmt.c: handle nested FUNC_DECL (mangle name, check closures)
#
