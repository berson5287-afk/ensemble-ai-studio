"""Does the file still work after the edit?

Half of these tests are about what the checker must *not* say. A warning that
fires on healthy code gets clicked past along with the one that matters, so
every ordinary Python shape that could look like a read-before-assign is here
as a silence test.
"""

from __future__ import annotations

import sys

import pytest

from aichatlab import validate


def flags(text, name="mod.py"):
    return [p.symbol for p in validate.problems(name, text)]


# -- the bug this exists for ------------------------------------------------
def test_a_counter_incremented_but_never_initialised_is_caught():
    """The real one: an edit added `nocat += 1` and no `nocat = 0`."""
    src = """\
def confirm_sel(ids):
    n = 0
    undet = 0
    for sid in ids:
        ok, why = check(sid)
        if ok:
            n += 1
        elif why == "undetermined":
            undet += 1
        elif why == "no_category":
            nocat += 1
    msg = "Confirmed %d" % n
    if nocat:
        msg += " skipped %d" % nocat
    return msg
"""
    found = validate.problems("replypilot.pyw", src)

    assert [p.symbol for p in found] == ["confirm_sel:nocat"]
    assert found[0].kind == "unbound"
    assert "UnboundLocalError" in found[0].message
    assert found[0].line == 11


def test_a_plain_read_before_assignment_is_caught():
    assert flags("def f():\n    print(total)\n    total = 1\n") == ["f:total"]


def test_the_sibling_counter_that_was_initialised_is_not_flagged():
    src = "def f(rows):\n    n = 0\n    for r in rows:\n        n += 1\n    return n\n"

    assert flags(src) == []


# -- silences: ordinary code that must never be flagged ---------------------
def test_parameters_are_not_unbound():
    """They hold a value before the first line runs, whatever the order."""
    src = "def f(parent, title, app):\n    return parent, title, app\n"

    assert flags(src) == []


def test_star_args_are_not_unbound():
    assert flags("def f(*args, **kw):\n    return args, kw\n") == []


def test_a_conditional_binding_read_afterwards_is_left_alone():
    """Real risk, but deciding it needs to know which branch ran."""
    src = """\
def f(flag):
    if flag:
        value = 1
    return value
"""
    assert flags(src) == []


def test_a_closure_reading_an_enclosing_name_is_not_flagged():
    src = """\
def outer():
    total = 0
    def inner():
        return total
    return inner
"""
    assert flags(src) == []


def test_a_nested_function_assigned_later_is_not_flagged():
    """Callbacks routinely refer to names defined further down the body."""
    src = """\
def build():
    def on_click():
        return label
    label = "hi"
    return on_click
"""
    assert flags(src) == []


def test_comprehension_targets_are_not_flagged():
    """`[x for x in items]` puts the read of x to the left of the bind."""
    assert flags("def f(items):\n    return [x for x in items]\n") == []
    assert flags("def f(i):\n    return {k: v for k, v in i}\n") == []
    assert flags("def f(i):\n    return {x for x in i}\n") == []
    assert flags("def f(i):\n    return (y for y in i)\n") == []


def test_a_for_loop_target_is_not_flagged():
    src = "def f(rows):\n    for row in rows:\n        print(row)\n"

    assert flags(src) == []


def test_tuple_unpacking_binds_both_names():
    src = "def f(pair):\n    a, b = pair\n    return a + b\n"

    assert flags(src) == []


def test_global_and_nonlocal_are_not_locals():
    assert flags("def f():\n    global c\n    c += 1\n") == []
    src = """\
def outer():
    c = 0
    def inner():
        nonlocal c
        c += 1
    return inner
"""
    assert flags(src) == []


def test_an_imported_name_is_bound():
    src = "def f():\n    import json\n    return json.dumps({})\n"

    assert flags(src) == []
    src = "def f():\n    from os import path\n    return path.sep\n"
    assert flags(src) == []


def test_an_except_alias_is_bound():
    src = """\
def f():
    try:
        go()
    except ValueError as exc:
        return str(exc)
"""
    assert flags(src) == []


def test_a_with_alias_is_bound():
    src = "def f(p):\n    with open(p) as handle:\n        return handle.read()\n"

    assert flags(src) == []


def test_a_walrus_binds():
    src = "def f(i):\n    if (n := len(i)):\n        return n\n"

    assert flags(src) == []


def test_a_star_import_switches_the_check_off():
    """It could have brought the name in; we cannot know."""
    src = "def f():\n    from os.path import *\n    return sep\n"

    assert flags(src) == []


@pytest.mark.skipif(sys.version_info < (3, 10),
                    reason="match statements do not parse before 3.10 — and "
                           "on 3.9 reporting them as a syntax error is the "
                           "truthful answer, since the interpreter running "
                           "the app cannot import such a file either")
def test_a_match_statement_switches_the_check_off():
    src = """\
def f(x):
    match x:
        case {"a": found}:
            return found
    return None
"""
    assert flags(src) == []


def test_locals_and_eval_switch_the_check_off():
    assert flags("def f():\n    print(v)\n    v = locals()\n") == []
    assert flags("def f(s):\n    print(v)\n    v = eval(s)\n") == []


# -- syntax and JSON --------------------------------------------------------
def test_a_file_that_stops_parsing_is_reported():
    found = validate.problems("m.py", "def f(:\n    pass\n")

    assert len(found) == 1
    assert found[0].kind == "syntax"
    assert "no longer parses" in found[0].message


def test_broken_json_is_reported():
    found = validate.problems("data.json", '{"a": 1,}')

    assert found and found[0].kind == "json"


def test_good_json_is_quiet():
    assert validate.problems("data.json", '{"a": [1, 2]}') == []


def test_a_file_we_cannot_check_is_left_alone():
    assert validate.problems("notes.md", "# hello\n(((") == []
    assert validate.language("notes.md") == ""
    assert validate.language("a.pyw") == "python"


# -- only what the edit introduced ------------------------------------------
def test_a_pre_existing_problem_is_not_blamed_on_the_edit():
    before = "def f():\n    print(x)\n    x = 1\n"
    after = "def f():\n    print(x)\n    x = 1\n    return x\n"

    assert validate.introduced("m.py", before, after) == []


def test_a_newly_introduced_problem_is_reported():
    before = "def f():\n    n = 0\n    return n\n"
    after = "def f():\n    n = 0\n    m += 1\n    return n\n"

    found = validate.introduced("m.py", before, after)

    assert [p.symbol for p in found] == ["f:m"]


def test_a_name_never_assigned_in_the_function_is_left_alone():
    """It is a global lookup, not a local — possibly defined elsewhere."""
    src = "def f():\n    n = 0\n    n += SOME_CONSTANT\n    return n\n"

    assert flags(src) == []


def test_an_already_broken_file_suppresses_everything():
    """With no parse of the original there is no honest baseline."""
    before = "def f(:\n"
    after = "def f():\n    print(y)\n    y = 1\n"

    assert validate.introduced("m.py", before, after) == []


def test_a_brand_new_file_is_still_checked():
    after = "def f():\n    total += 1\n    return total\n"

    found = validate.introduced("new.py", "", after)

    assert [p.symbol for p in found] == ["f:total"]


def test_summarise_says_one_thing_then_counts():
    found = validate.problems(
        "m.py", "def f():\n    print(a)\n    a = 1\n\ndef g():\n    print(b)\n    b = 2\n")

    assert len(found) == 2
    text = validate.summarise(found)
    assert "and 1 more" in text
    assert validate.summarise([]) == ""


def test_line_numbers_point_at_the_first_use():
    found = validate.problems("m.py", "def f():\n    x = 1\n    print(y)\n    y = 2\n")

    assert found[0].line == 3


# -- code stranded after a return ------------------------------------------
# What a mis-anchored insertion looks like from the inside: the model wrote a
# whole new function body and it was anchored above the old one, which is now
# unreachable. It compiles, so nothing else catches it.
def test_a_new_body_inserted_above_the_old_one_is_caught():
    src = """\
def next_drip_in(self, now=None):
    if not self.sent_log:
        return self.drip_sec()
    return max(0, self.drip_sec())
    if not self.sent_log:
        return 0
    return 1
"""
    found = [p for p in validate.problems("m.py", src)
             if p.kind == "unreachable"]

    assert found
    assert found[0].line == 5
    assert "can never run" in found[0].message


def test_unreachable_after_raise_break_and_continue():
    for tail in ("raise ValueError('x')", "break", "continue"):
        src = ("def f(rows):\n"
               "    for row in rows:\n"
               f"        {tail}\n"
               "        print(row)\n")
        found = [p for p in validate.problems("m.py", src)
                 if p.kind == "unreachable"]
        assert found, tail


def test_a_return_at_the_end_of_a_block_is_fine():
    src = ("def f(x):\n"
           "    if x:\n"
           "        return 1\n"
           "    return 2\n")

    assert [p for p in validate.problems("m.py", src)
            if p.kind == "unreachable"] == []


def test_a_return_in_one_branch_does_not_strand_the_other():
    src = ("def f(x):\n"
           "    if x:\n"
           "        return 1\n"
           "    else:\n"
           "        return 2\n"
           "\n"
           "def g():\n"
           "    return 3\n")

    assert [p for p in validate.problems("m.py", src)
            if p.kind == "unreachable"] == []


def test_only_the_edit_is_blamed_for_dead_code():
    before = "def f():\n    return 1\n    print('dead')\n"
    after = before + "\ndef g():\n    return 2\n"

    assert validate.introduced("m.py", before, after) == []


# -- a class body is not in scope for its own methods ----------------------
# The favourite failure of a model asked to "add logging": the import looks
# like it belongs to the class, the diff reads perfectly, and every
# instantiation raises NameError. Nothing else catches it, because the name is
# never assigned in the method and so looks like an ordinary global.
LEAKY = """\
class Engine:
    import logging

    def __init__(self, store):
        self.store = store
        logging.basicConfig(filename="a.log")

    def go(self, mid):
        logging.info("sent %s" % mid)
"""


def test_an_import_in_a_class_body_used_bare_in_a_method_is_caught():
    found = [p for p in validate.problems("m.py", LEAKY)
             if p.kind == "class-scope"]

    assert len(found) == 2, "once per method that would raise"
    assert "NameError" in found[0].message
    assert found[0].symbol == "Engine.__init__:logging"


def test_a_module_level_import_is_visible_and_silent():
    src = ("import logging\n"
           "class E:\n"
           "    def f(self):\n"
           "        logging.info(1)\n")

    assert flags(src) == []


def test_a_class_attribute_reached_properly_is_silent():
    for body in ("        return self.LIMIT\n", "        return E.LIMIT\n"):
        src = "class E:\n    LIMIT = 5\n    def f(self):\n" + body
        assert flags(src) == [], body


def test_a_method_local_of_the_same_name_is_silent():
    src = ("class E:\n"
           "    tally = 0\n"
           "    def f(self):\n"
           "        tally = 1\n"
           "        return tally\n")

    assert flags(src) == []


def test_a_shadowed_builtin_is_not_reported():
    src = ("class E:\n"
           "    list = []\n"
           "    def f(self):\n"
           "        return list((1, 2))\n")

    assert flags(src) == []


def test_a_name_bound_at_module_level_too_is_silent():
    """The method sees the module one, so nothing raises."""
    src = ("import os\n"
           "class E:\n"
           "    os = 1\n"
           "    def f(self):\n"
           "        return os.sep\n")

    assert flags(src) == []


def test_only_a_newly_leaked_name_is_blamed_on_the_edit():
    before = "class E:\n    def f(self):\n        return 1\n"

    found = validate.introduced("m.py", before, LEAKY)

    assert [p.kind for p in found] == ["class-scope", "class-scope"]
