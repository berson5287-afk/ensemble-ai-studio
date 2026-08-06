"""Where everything lives: files → classes → functions, with line ranges.

The symbol index says what a file *defines*; the map says what it is *for*.
Neither says *where*, and where is the missing piece for the one wall the
edit pipeline could never climb: a file bigger than the context window.
`replypilot.pyw` is 147,000 characters — no request that needed it could
carry it, so its functions were simply uneditable by every path.

A tree with line ranges fixes that at the root.  When the second pass knows
the function it was asked to change spans lines 1705–1731, it can send that
region — verbatim, with context — instead of the whole file.  The anchors
still match, because the excerpt *is* the file's own text; the window stops
mattering, because a function fits where a file cannot.

Deterministic throughout: ast for Python, a definition-line scan for
everything else.  A tree is only worth trusting if it cannot be wrong about
where things are.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# Lines of surrounding file shown either side of an extracted region, so the
# model can see what its change sits between.
REGION_CONTEXT = 30

# Above this a file goes to the second pass as a region, not whole.
# Chosen against a 32k window: the file, the instructions, the
# evidence and the reply all share it.
FITS_WHOLE_TOKENS = 8000

# The fallback definition-line scan, for files ast cannot parse.
DEF_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?:async\s+)?(?:def|class|function|fn|sub)\s+"
    r"(?P<name>\w+)", re.MULTILINE)


@dataclass
class Node:
    """One thing a file defines, and exactly where."""

    name: str
    kind: str                   # "class" | "function"
    start: int                  # 1-based, inclusive
    end: int
    children: list = field(default_factory=list)

    @property
    def qualified(self) -> str:
        return self.name


def tree(text: str) -> list[Node]:
    """The file's structure, in source order.

    ast gives exact spans for Python.  The fallback scans definition lines
    and closes each span at the next definition of equal or lesser
    indentation — approximate, but approximate in the safe direction, since
    a region is always re-read from the real text it points into.
    """
    try:
        parsed = ast.parse(text or "")
    except (SyntaxError, ValueError, RecursionError):
        return _fallback_tree(text or "")
    return _walk(parsed.body)


def _walk(body) -> list[Node]:
    found: list[Node] = []
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = Node(
                name=statement.name, kind="function",
                start=statement.lineno,
                end=getattr(statement, "end_lineno", statement.lineno))
            # Functions nest functions — every Tkinter dialog is a method
            # full of closures, and `confirm_sel` in a 147,000-character
            # file lives three levels down.  A tree that cannot see it
            # cannot excerpt it, and the wall stays up.
            node.children = _walk(statement.body)
            found.append(node)
        elif isinstance(statement, ast.ClassDef):
            node = Node(name=statement.name, kind="class",
                        start=statement.lineno,
                        end=getattr(statement, "end_lineno",
                                    statement.lineno))
            node.children = _walk(statement.body)
            found.append(node)
    return found


def _fallback_tree(text: str) -> list[Node]:
    lines = text.splitlines()
    hits = [(m.start(), len(m.group("indent")), m.group("name"))
            for m in DEF_LINE.finditer(text)]
    nodes: list[Node] = []
    positions = [text[:offset].count("\n") + 1 for offset, _i, _n in hits]
    for index, (offset, indent, name) in enumerate(hits):
        start = positions[index]
        end = len(lines)
        for later in range(index + 1, len(hits)):
            if hits[later][1] <= indent:
                end = positions[later] - 1
                break
        nodes.append(Node(name=name, kind="function", start=start, end=end))
    return nodes


def render(nodes, name: str = "", total_lines: int = 0) -> str:
    """The tree as text a model (or a person) can navigate."""
    lines = []
    if name:
        header = name + (f"  ({total_lines} lines)" if total_lines else "")
        lines.append(header)
    def _emit(items, depth):
        for node in items:
            lines.append("  " * depth
                         + f"{'class ' if node.kind == 'class' else ''}"
                           f"{node.name}  [{node.start}–{node.end}]")
            _emit(node.children, depth + 1)
    _emit(nodes, 1 if name else 0)
    return "\n".join(lines)


def find(nodes, name: str) -> Node | None:
    """A node by bare or Class.method name, depth first, source order."""
    wanted = name.split(".")
    def _search(items, path):
        for node in items:
            here = path + [node.name]
            if here[-len(wanted):] == wanted:
                return node
            hit = _search(node.children, here)
            if hit is not None:
                return hit
        return None
    return _search(nodes, [])


def region(text: str, name: str,
           context: int = REGION_CONTEXT) -> tuple[str, int, int] | None:
    """(verbatim excerpt, first line, last line) around one function.

    Verbatim matters more than anything here: the second pass copies its
    FIND anchors from what it is shown, and an excerpt that differed from
    the file by so much as a reflowed space would make every anchor miss.
    The excerpt is a pure slice of the file's own lines, no numbering, no
    trimming — the numbers travel in the return value, not the text.
    """
    node = find(tree(text), name)
    if node is None:
        return None
    lines = (text or "").splitlines()
    first = max(1, node.start - context)
    last = min(len(lines), node.end + context)
    return "\n".join(lines[first - 1:last]), first, last


def region_by_terms(text: str, terms) -> tuple[str, int, int] | None:
    """A region centred on the first line matching any claim term.

    The fallback for a change that named no function the tree can find —
    better a window onto the right neighbourhood than a file that cannot
    be sent at all.
    """
    lines = (text or "").splitlines()
    for number, line in enumerate(lines, 1):
        lowered = line.lower()
        if any(term in lowered for term in terms or []):
            first = max(1, number - REGION_CONTEXT * 2)
            last = min(len(lines), number + REGION_CONTEXT * 2)
            return "\n".join(lines[first - 1:last]), first, last
    return None
