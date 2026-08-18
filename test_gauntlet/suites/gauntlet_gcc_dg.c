/* Auto-generated gauntlet suite: gcc_dg */
/* Source: gcc-dg */

#include "wubu_test_gauntlet.h"

const test_entry_t gauntlet_gcc_dg_tests[] = {
    {"gcc_dg_001901_pr47426_2", "typedef void tfoo (int *);\ntfoo *getfoo (void);\n\nvoid\nbar (int *i)\n{\n  (*i)--;\n}\n\nint\nmain ()\n{\n  int i = 1;\n  getfoo ()(&i);\n  if (i)\n    return 1;\n  return 0;\n}", 0, TEST_CAT_INTEGER, 0, 100},
    {"gcc_dg_002209_pr61602", "int a;\nint *b = &a, **c = &b;\nint\nmain ()\n{\n  int **d = &b;\n  *d = 0;\n}", 0, TEST_CAT_INTEGER, 0, 100},
    {"gcc_dg_002619_pr47278_2", "{\n  if (foo() != 1)\n    return 1;\n  return 0", 0, TEST_CAT_COMPARISON, 0, 100},
    {"gcc_dg_002673_20110719_1", "{\n  int b = i != 0;\n  int c = ~b;\n  if (c != -1)\n    return 1;\n  return 0", 0, TEST_CAT_COMPARISON, 0, 100},
    {"gcc_dg_002903_pr53272_2", "{\n struct rtc_class_ops ops = {(void *) 0};\n  struct rtc_device dev1 = {0, &ops, 42};\n\n  if (rtc_update_irq_enable (&dev1, 1) != -22)\n    return 1;\n\n  __builtin_exit (0)", 0, TEST_CAT_COMPARISON, 0, 100},
    {"gcc_dg_002971_pr71987", "{\n  fn2 ();\n  return 0", 0, TEST_CAT_COMPARISON, 0, 100},
    {"gcc_dg_003258_pr86066", "{\n  struct A t = { 0, 0, 2 };\n L:\n  t.d = ~(~(~0 % t.d) % 2);\n  if (!t.d)\n    goto L;\n  return 0", 0, TEST_CAT_CONTROL, 0, 100},
    {"gcc_dg_003726_pr59374_3", "{\n  a.b = &a;\n  foo ();\n  if (b.b != &a)\n    return 1;\n  return 0", 0, TEST_CAT_COMPARISON, 0, 100},
    {"gcc_dg_003786_pr56965_2", "{\n  int a[3];\n  if (foo ((union U *)&a[0], (union U *)&a[0]) != 0)\n    return 1;\n  if (bar ((struct R *)&a[1], (struct R *)&a[0]) != 0)\n    return 1;\n  return 0", 0, TEST_CAT_COMPARISON, 0, 100},
};

const uint32_t gauntlet_gcc_dg_test_count = 9;
