"""Where everything lives — and the wall it breaks.

A file bigger than the context window was uneditable by every path.  The
tree knows where each function spans, so the second pass can carry the
function instead of the file.  Verbatim extraction is the property every
test here guards: one reflowed space and every FIND anchor misses.
"""

from __future__ import annotations

from aichatlab import codetree

NESTED = '''\
import os

def top(a):
    return a

class Engine:
    LIMIT = 5

    def confirm(self, sid):
        if sid is None:
            return None
        return sid

    def undo(self):
        return True

def tail():
    return 0
'''


def test_the_tree_nests_methods_under_their_class():
    nodes = codetree.tree(NESTED)

    assert [n.name for n in nodes] == ["top", "Engine", "tail"]
    engine = nodes[1]
    assert engine.kind == "class"
    assert [c.name for c in engine.children] == ["confirm", "undo"]


def test_line_ranges_are_exact():
    nodes = codetree.tree(NESTED)

    confirm = codetree.find(nodes, "confirm")
    assert (confirm.start, confirm.end) == (9, 12)
    assert codetree.find(nodes, "top").start == 3


def test_find_accepts_a_qualified_name():
    nodes = codetree.tree(NESTED)

    assert codetree.find(nodes, "Engine.confirm") is confirm_node(nodes)
    assert codetree.find(nodes, "Other.confirm") is None


def confirm_node(nodes):
    return nodes[1].children[0]


def test_find_misses_honestly():
    assert codetree.find(codetree.tree(NESTED), "imaginary") is None


def test_render_shows_structure_with_ranges():
    text = codetree.render(codetree.tree(NESTED), "engine.py", 18)

    assert "engine.py  (18 lines)" in text
    assert "class Engine" in text
    assert "confirm  [9–12]" in text


def test_a_region_is_a_verbatim_slice_of_the_file():
    found = codetree.region(NESTED, "confirm", context=2)

    assert found is not None
    excerpt, first, last = found
    lines = NESTED.splitlines()
    assert excerpt == "\n".join(lines[first - 1:last]), "verbatim or useless"
    assert "def confirm" in excerpt
    assert first <= 9 and last >= 12


def test_a_region_clamps_at_the_file_edges():
    excerpt, first, last = codetree.region(NESTED, "top", context=100)

    assert first == 1
    assert last == len(NESTED.splitlines())


def test_region_for_a_missing_name_is_none():
    assert codetree.region(NESTED, "imaginary") is None


def test_region_by_terms_centres_on_the_first_hit():
    found = codetree.region_by_terms(NESTED, ["limit"])

    assert found is not None
    excerpt, first, last = found
    assert "LIMIT" in excerpt


def test_region_by_terms_misses_honestly():
    assert codetree.region_by_terms(NESTED, ["zzz"]) is None
    assert codetree.region_by_terms(NESTED, []) is None


def test_a_file_ast_cannot_parse_still_gets_a_tree():
    broken = "def alpha():\n    return (\n\ndef beta():\n    return 2\n"

    nodes = codetree.tree(broken)

    assert [n.name for n in nodes] == ["alpha", "beta"]
    assert nodes[0].start == 1 and nodes[1].start == 4


def test_the_excerpt_prompt_says_where_it_is_and_what_it_demands():
    from aichatlab.intent import Change, build_change_prompt

    prompt = build_change_prompt(
        Change("guard confirm() against None", "big.pyw"),
        "def confirm(sid):\n    return sid\n", "INSTRUCTIONS",
        excerpt_of=(1700, 1760, 3800),
        file_tree="big.pyw  (3800 lines)\n  class App  [10-3700]")

    assert "lines 1700–1760" in prompt
    assert "3800 lines" in prompt
    assert "too large to send whole" in prompt
    assert "copied exactly from this excerpt" in prompt
    assert "class App" in prompt


def test_a_whole_file_prompt_is_unchanged():
    from aichatlab.intent import Change, build_change_prompt

    prompt = build_change_prompt(
        Change("guard confirm()", "small.py"), "def confirm():\n    pass\n",
        "INSTRUCTIONS")

    assert "complete contents of small.py" in prompt
    assert "too large" not in prompt


def test_closures_inside_methods_are_on_the_tree():
    """`confirm_sel` in the real 147KB file lives three levels down —
    a closure in a method in a class — and was invisible to a tree that
    only descended into classes."""
    src = ("class App:\n"
           "    def open_dialog(self):\n"
           "        def confirm_sel():\n"
           "            return 1\n"
           "        return confirm_sel\n")
    nodes = codetree.tree(src)

    assert codetree.find(nodes, "confirm_sel") is not None
    assert codetree.region(src, "confirm_sel", context=1) is not None
