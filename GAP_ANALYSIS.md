# HolyD Compiler Gap Analysis

**Current gauntlet score: 2,503/4,431 (56.6%)**

## Passing Tests (2,503)
- Integer arithmetic (add, sub, mul, div, mod, bitwise, shifts)
- Comparisons (int, unsigned, float)
- Control flow (if/else, for, while, do-while, switch, break, continue)
- Functions (up to 8 args, non-recursive)
- Global variables
- Static local variables (non-recursive)
- Float arithmetic (basic operations)
- Float comparisons
- Type casts (int↔int, int→float, float→int)
- Forward declarations
- Extern declarations (basic)
- Simple structs (member access, assignment)

## Failing Tests (383)

### Chapter 13: Float (10 failures)
- NaN handling (NaN comparisons, isnan function)
- Infinity handling
- 9+ argument calling convention (stack-based args)
- Recursive functions with float args
- Rounding edge cases (ties-to-even)
- Subnormal numbers
- Negative zero

### Chapter 16: Char Types/Strings (15 failures)
- Char type (1-byte storage)
- Char pointers
- String literals
- String operations (strcpy, strcmp, etc.)
- Char expressions and conversions

### Chapter 15: Pointers (10 failures)
- Pointer arithmetic
- Nested pointer indexing (ptr_ptr[0][0])
- 2D array access
- Pointer alignment

### Chapter 10: Static/Extern (7 failures)
- Recursive functions with static locals
- Static local vs extern with same name
- Shadowed static locals

### Chapter 14: Pointer Casts (4 failures)
- Pointer↔integer casts
- Pointer dereferencing
- Pointer assignment through dereference

### Chapter 18: Unions (35 failures)
- Union memory layout
- Nested unions in structs
- Union member access

### Other Chapters (4 failures)
- Chapter 14: misc pointer operations

## Error Tests (1,547)
Tests that fail to compile. Root causes:
- Truncated test extractions from GCC/LLVM/LACC suites
- Unsupported features (vector_size, __attribute__, etc.)
- Pointer types in function signatures
- Char types
- Complex structs/unions
- GCC-specific extensions

## Architectural Gaps

### 1. Byte-Addressable Memory (needed for chars, strings)
**Impact: 15+ tests**
**Risk: HIGH** - affects all memory operations
**Approach**: Change cell size from 8 bytes to 1 byte. Add load/store byte instructions.
**Files**: wubu_isa_x86_64.c, wubu_mir.h, wubu_mir.c, holyd_mir_eval.c

### 2. Stack Frame (needed for recursion)
**Impact: 3+ tests**
**Risk: HIGH** - affects all function calls
**Approach**: Use [rbp - offset] for local variables. Allocate frame on function entry.
**Files**: wubu_isa_x86_64.c, holyd_mir_eval.c

### 3. 9+ Argument Calling Convention
**Impact: 2+ tests**
**Risk: MEDIUM** - affects function calls with many args
**Approach**: Push extra args on stack. Callee reads from [rbp + offset].
**Files**: wubu_isa_x86_64.c, holyd_mir_eval.c

### 4. NaN/Infinity Handling
**Impact: 3+ tests**
**Risk: LOW** - add special comparison logic
**Approach**: Add MIR_NAN, MIR_INF ops. Handle in comparisons.
**Files**: wubu_mir.h, wubu_isa_x86_64.c, holyd_mir_eval.c

### 5. Union Memory Layout
**Impact: 35 tests**
**Risk: HIGH** - affects struct member addressing
**Approach**: Implement union as max-size struct with shared offset 0.
**Files**: holyd_parse.c, holyd_mir_eval.c, wubu_isa_x86_64.c

### 6. Full Pointer Support
**Impact: 14+ tests**
**Risk: HIGH** - affects memory model
**Approach**: Implement pointer as address (cell index). Add load/store through pointer.
**Files**: holyd_parse.c, holyd_mir_eval.c, wubu_isa_x86_64.c

## Recommended Priority
1. **NaN/Infinity** (low risk, quick win)
2. **9+ argument calling convention** (medium risk, few tests)
3. **Stack frame** (high risk, enables recursion)
4. **Byte-addressable memory** (high risk, enables chars)
5. **Union layout** (high risk, many tests)
6. **Full pointer support** (high risk, many tests)

## Key Design Decisions
- Current model: 8-byte cells, flat memory array, no stack frame
- Target model: byte-addressable memory, stack frames, proper pointers
- Migration strategy: incremental, test after each change
