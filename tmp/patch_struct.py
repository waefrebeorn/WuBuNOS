import re

with open('holyd_parse.c', 'r') as f:
    content = f.read()

# Find the struct member layout code and replace it
# We need to change the regular member case (lines 395-401) to use byte offsets with alignment

old_code = '''                            } else {
                                /* Regular member: byte offset */
                                t->members[t->n_members].offset = t->size;  /* byte offset */
                                t->members[t->n_members].bit_offset = 0;
                                t->size += (int)((msz + 7) / 8);  /* member size in int64 cells */
                            }
                            t->align = 1;'''

new_code = '''                            } else {
                                /* Regular member: byte offset with C alignment */
                                /* Align size to member's natural alignment */
                                int mem_align = (int)msz;
                                if (mem_align > 8) mem_align = 8;
                                if (mem_align < 1) mem_align = 1;
                                if (t->size % mem_align != 0)
                                    t->size += mem_align - (t->size % mem_align);
                                t->members[t->n_members].offset = t->size;  /* byte offset */
                                t->members[t->n_members].bit_offset = 0;
                                t->size += (int)msz;  /* member size in bytes */
                                /* Track max alignment */
                                if (mem_align > t->align) t->align = mem_align;
                            }'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("Replaced regular member layout code")
else:
    print("ERROR: Could not find old code")
    # Try to find it with different whitespace
    pattern = r'\}\s*else\s*\{[^}]*/\* Regular member: byte offset \*/[^}]*t->size \+= \(int\)\(\(msz \+ 7\) / 8\);[^}]*\}'
    match = re.search(pattern, content)
    if match:
        print(f"Found match at {match.start()}-{match.end()}")
        print(repr(match.group()[:200]))
    else:
        print("No match found")

with open('holyd_parse.c', 'w') as f:
    f.write(content)
