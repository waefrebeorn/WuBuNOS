# Struct/union layout — units mismatch (2026-10-19)

## What the code actually does

The **parser** is byte-accurate. In `holyd_parse.c` the struct/union case sets:

```c
t->members[n].offset = t->size;   /* regular member: BYTE offset */
t->size = ...                     /* bytes, with padding/alignment */
```

Union members all get offset 0 and the type's size is the maximum member size.

The **MIR struct table** copies those values verbatim:

```c
s->member_offsets[i] = (int)n->type->members[i].offset;   /* bytes */
s->total_size      = (int)n->type->size;                  /* bytes */
```

…but the codegen comment and its arithmetic treat both as **cells**:

```c
/* member_offsets is consumed as CELLS by the codegen
 * (mir_struct_member_offset does *8) */
```

So every consumer multiplies a byte value by 8. `struct{int;int}` has
`size=8, offsets={0,4}`, which the codegen reads as 8 cells = 64 bytes of
reservation and member offsets 0 and 32.

## Consequence

Exact-layout tests cannot pass while the units disagree. The gauntlet's
chapter-18 suite requires:

```
struct two_bytes   2      struct three_bytes  3      struct seven_bytes  7
struct wonky      19      struct twenty_bytes 20     struct  24 / 28 / 16
union  no_padding 11      union  with_padding 12     union  contains_array 36
```

Those numbers are all achievable — the parser already computes them — once the
consumers stop multiplying by 8.

## Scope of the change

- `total_size`: 26 references in `holyd_mir_eval.c`
- `member_offsets`: 9 references
- allocation paths: `struct_size`, `arr_size`, the struct-return arena copy loop
  (`ret_struct_size` cells), and the JIT's struct arguments/returns

Recommended direction: declare **bytes** the single unit for
`HDType.members[].offset`, `HDType.size`, `mir_struct_t.member_offsets` and
`mir_struct_t.total_size`; remove the `* 8` factors; allocate `ceil(bytes/8)`
cells for storage. Pointers and cells keep their 8-byte granularity, so a
struct member access adds a byte offset to a byte base — which is what
`mir_address_of` already does for `ADD`.

## Why it was not attempted in this pass

It touches the same paths that handle nested-struct member chains and
struct-by-value returns — both fixed and verified earlier in this session
(seven battery probes). Changing the unit without re-running the battery
(138 probes) plus the gauntlet chapter-18 suite risks silently un-fixing the
return ABI. The change is mechanical but must be done in one pass with both
measurements available.

## Measured: the offsets really are cells

Removing the two `* 8` conversions ("cell → byte") in
`mir_struct_member_offset` and `mir_struct_member_offset_by_index` was tried and
fails loudly:

```
WRONG struct member        (expect 3, got 515)
WRONG struct return 2m     (expect 49, got 1841)
WRONG struct pass by value (expect 60, got 1978940)
```

Those are byte/word confusion artefacts: member access then adds a cell index to
a byte base and reads a neighbouring cell. So the parser's `members[].offset`
values are **cells by construction** — `t->size` is advanced by `member_cells`,
computed per member (`ceil(element bytes / 8) * array_size`, or the inner
struct's cell size).

The rewrite therefore starts in `holyd_parse.c` (§ regular member: `offset =
t->size`, and the `member_cells` computation), not in the two consumers. The
consumers are correct for the model they were written against; the parser has to
produce byte offsets first, then every `* 8` becomes a no-op to delete in the
same commit.
