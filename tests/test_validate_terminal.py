"""Compound statements whose every path terminates are terminal.

The live edit that proved it: a model pasted a whole new function body —
ending in a try/except where every branch returns — above the original, and
the duplicate below sailed through, because a `try` is not a `return` even
when nothing inside it ever falls through.
"""

from __future__ import annotations

from aichatlab import validate


def unreachable(src):
    return [p for p in validate.problems("m.py", src)
            if p.kind == "unreachable"]


LIVE_SHAPE = """\
def classify_llm(subject):
    if not subject:
        return None
    try:
        obj = parse(subject)
        return obj, "ok"
    except Exception:
        return None
    if not subject:
        return None
    return old_path(subject)
"""


def test_the_live_shape_is_now_caught():
    found = unreachable(LIVE_SHAPE)

    assert found, "the duplicated body below the try/except must be flagged"
    assert found[0].line == 9


def test_a_try_whose_body_can_fall_through_is_not_terminal():
    src = """\
def f(x):
    try:
        x = load(x)
    except Exception:
        return None
    return x
"""
    assert unreachable(src) == []


def test_a_try_with_a_non_returning_handler_is_not_terminal():
    src = """\
def f(x):
    try:
        return load(x)
    except Exception:
        log(x)
    return None
"""
    assert unreachable(src) == []


def test_if_else_where_both_branches_return_is_terminal():
    src = """\
def f(x):
    if x:
        return 1
    else:
        return 2
    print('stranded')
"""
    found = unreachable(src)

    assert found and found[0].line == 6


def test_an_if_without_else_is_never_terminal():
    src = """\
def f(x):
    if x:
        return 1
    return 2
"""
    assert unreachable(src) == []


def test_a_returning_finally_is_terminal():
    src = """\
def f(x):
    try:
        x = load(x)
    finally:
        return x
    print('stranded')
"""
    found = unreachable(src)

    assert found and found[0].line == 6


def test_a_with_block_ending_in_return_is_terminal():
    src = """\
def f(p):
    with open(p) as h:
        return h.read()
    print('stranded')
"""
    found = unreachable(src)

    assert found and found[0].line == 4


def test_try_else_decides_the_fallthrough_path():
    src = """\
def f(x):
    try:
        y = load(x)
    except Exception:
        return None
    else:
        return y
    print('stranded')
"""
    found = unreachable(src)

    assert found and found[0].line == 8
