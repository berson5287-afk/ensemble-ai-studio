"""Does the file still work after the edit?

The review window answers "is this the change I wanted". It cannot answer "does
this run", and those are different questions with the same-looking diff. A
five-line patch that adds `nocat += 1` to a function that never initialises
`nocat` reads perfectly: the lines are plausible, the indentation is right, the
surrounding code is real. It also breaks the function on every call, and the
first anyone knows is a traceback in an app that worked an hour ago.

So the resulting file is compiled and scanned before the diff is shown, and
anything the edit *introduced* is put above it in red. Two rules shape the
whole module:

*Only new problems count.* A file with a pre-existing complaint is not the
edit's fault, and blaming the edit for it teaches people to ignore the
warnings. Everything is measured as the difference between before and after.

*Silence must mean something.* A checker that cries wolf is worse than no
checker, because the one time it is right it gets clicked past with all the
others. So the analysis is deliberately conservative: it reports the cases it
is certain about and stays quiet everywhere else. A missed bug costs one bad
edit; a false alarm costs the feature.
"""

from __future__ import annotations

import ast
import builtins
import json
from dataclasses import dataclass
from pathlib import Path

PYTHON_SUFFIXES = {".py", ".pyw"}
JSON_SUFFIXES = {".json"}


@dataclass(frozen=True)
class Problem:
    """Something wrong with the file the edit would produce."""

    kind: str               # "syntax" | "unbound" | "json"
    message: str
    line: int = 0
    symbol: str = ""        # what it is about, for before/after comparison

    @property
    def key(self) -> tuple:
        """Identity that survives lines moving around."""
        return (self.kind, self.symbol)

    def describe(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        return where + self.message


def language(name: str) -> str:
    suffix = Path(str(name)).suffix.lower()
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in JSON_SUFFIXES:
        return "json"
    return ""


def problems(name: str, text: str) -> list[Problem]:
    """Everything wrong with this file, as far as we can tell cheaply."""
    kind = language(name)
    if kind == "python":
        return _python(text)
    if kind == "json":
        return _json(text)
    return []


def introduced(name: str, before: str, after: str) -> list[Problem]:
    """Only the problems this edit added.

    A file that was already broken cannot be judged, so a pre-existing syntax
    error suppresses everything: with no parse of the original there is no
    honest baseline, and reporting the lot would blame the edit for the state
    it found.
    """
    if not language(name):
        return []
    was = problems(name, before) if before else []
    if any(problem.kind in ("syntax", "json") for problem in was):
        return []                   # nothing to measure against
    known = {problem.key for problem in was}
    return [problem for problem in problems(name, after)
            if problem.key not in known]


def summarise(found) -> str:
    """One line for a status bar or a log."""
    found = list(found)
    if not found:
        return ""
    if len(found) == 1:
        return found[0].describe()
    return f"{found[0].describe()} (and {len(found) - 1} more)"


# -- JSON -------------------------------------------------------------------
def _json(text: str) -> list[Problem]:
    try:
        json.loads(text)
    except ValueError as exc:
        line = getattr(exc, "lineno", 0) or 0
        return [Problem("json", f"this is no longer valid JSON — {exc}",
                        line=line, symbol="json")]
    return []


# -- Python -----------------------------------------------------------------
def _python(text: str) -> list[Problem]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [Problem(
            "syntax",
            f"this no longer parses as Python — {exc.msg}",
            line=exc.lineno or 0, symbol="syntax")]
    except (ValueError, RecursionError, MemoryError):
        # Null bytes and pathological nesting.  Not our business to explain.
        return []
    found: list[Problem] = []
    for function in _functions(tree):
        found.extend(_unbound_in(function))
    found.extend(_unreachable_in(tree))
    found.extend(_class_scope_leaks(tree))
    return found


NESTED_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.Lambda)


def _bound_here(block) -> set:
    """Names this block binds *in its own scope*.

    Stopping at nested functions and classes is the whole point. Walking
    through them puts a class body's own imports into the module's set, the
    two then cancel out, and the check this exists for finds nothing.
    """
    names: set = set()
    pending = list(block)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
            continue
        if isinstance(node, NESTED_SCOPE):
            # Its name belongs to this scope; its body does not.
            names.add(getattr(node, "name", ""))
            names.discard("")
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
            continue
        pending.extend(ast.iter_child_nodes(node))
    return names


def _class_scope_leaks(tree) -> list[Problem]:
    """Names bound in a class body and then used bare inside its methods.

    Python does not put the class body in the lookup chain for its own
    methods, so this —

        class Engine:
            import logging
            def __init__(self):
                logging.basicConfig(...)

    — raises NameError on every instantiation. It is a favourite of models
    asked to "add logging": the import looks like it belongs to the class, the
    diff reads perfectly, and the class stops being constructible. Nothing
    else here catches it, because the name is never assigned *in the method*
    and so looks like an ordinary global.
    """
    module_level = _bound_here(getattr(tree, "body", []))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        in_class = _bound_here(node.body) - module_level
        if not in_class:
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                continue
            scan = _ScopeScan()
            scan.enter(statement)
            for (line, _col, _seq), name, kind in scan.events:
                if kind != "read" or name not in in_class:
                    continue
                if name in scan.bound or name in dir(builtins):
                    continue
                found.append(Problem(
                    "class-scope",
                    f"“{name}” is defined in the body of class "
                    f"“{node.name}”, which its methods cannot see — a bare "
                    f"reference to it raises NameError. Move it to the top of "
                    f"the file, or reach it as self.{name}",
                    line=line, symbol=f"{node.name}.{statement.name}:{name}"))
                break
    return sorted(found, key=lambda p: p.line)


# Statements after which nothing in the same block can run.
TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _unreachable_in(tree) -> list[Problem]:
    """Code left stranded after a return, raise, break or continue.

    This is what a mis-anchored insertion looks like from the inside. A model
    told to change a function writes the whole new body and anchors it at the
    top of the old one; the app inserts it, and the original body is still
    there underneath — after the new `return`, where it can never run again.
    The file compiles, the diff looks like an ordinary rewrite, and the
    function quietly has different behaviour and twice the length.

    Nothing subtle is attempted: only a statement that directly follows a
    terminal one in the same block, which is unreachable by definition rather
    than by analysis.
    """
    found = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for first, second in zip(block, block[1:]):
                if not isinstance(first, TERMINAL):
                    continue
                where = getattr(first, "__class__").__name__.lower()
                found.append(Problem(
                    "unreachable",
                    f"this line can never run — it comes straight after a "
                    f"`{where}` in the same block, so the code above it "
                    f"returns first. An edit that inserted a new version of a "
                    f"function above the old one looks exactly like this",
                    line=getattr(second, "lineno", 0),
                    symbol=f"unreachable:{getattr(second, 'lineno', 0)}"))
                break
    return sorted(found, key=lambda p: p.line)


def _functions(tree) -> list:
    return [node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


# Constructs whose binding rules are subtle enough that getting them slightly
# wrong would produce false alarms.  A function containing one is skipped
# entirely rather than guessed at.
def _too_subtle(node) -> bool:
    for child in ast.walk(node):
        if isinstance(child, getattr(ast, "Match", ())):
            return True
        if isinstance(child, ast.Name) and child.id in ("locals", "globals",
                                                        "vars", "eval", "exec"):
            return True
    return False


def _unbound_in(function) -> list[Problem]:
    """Locals whose first appearance in the function is a read.

    This is the one case that can be called with certainty from source order
    alone: if the *first* thing a function does with a local name is read it,
    no path exists on which it was set first. Anything conditional — bound in
    one branch, read after — is left alone, because deciding that needs to
    know which branch runs.
    """
    if _too_subtle(function):
        return []
    scan = _ScopeScan()
    scan.enter(function)
    if scan.uses_star_import:
        return []

    first: dict = {}
    for _position, name, kind in sorted(scan.events, key=lambda e: e[0]):
        if name not in first:
            first[name] = kind

    found = []
    for name, kind in first.items():
        if kind != "read":
            continue
        if name in scan.declared or name not in scan.bound:
            continue                    # global/nonlocal, or not a local
        if name in scan.pre_bound:
            continue                    # a parameter: bound before line one
        line = min(pos[0] for pos, other, _k in scan.events if other == name)
        found.append(Problem(
            "unbound",
            f"“{name}” is read before it is given a value, so this raises "
            f"UnboundLocalError — it is assigned inside "
            f"“{function.name}” but never initialised",
            line=line, symbol=f"{function.name}:{name}"))
    return sorted(found, key=lambda p: p.line)


class _ScopeScan(ast.NodeVisitor):
    """Reads and bindings in one function's own scope, in source order.

    Nested functions, lambdas and class bodies are separate scopes: their
    names are bound here, but what happens inside them is not this scope's
    business and is deliberately not descended into. A closure reading an
    enclosing variable is not an error, and treating it as one would light up
    every callback in the file.
    """

    def __init__(self) -> None:
        self.events: list = []          # ((line, col, seq), name, kind)
        self.bound: set = set()
        # Names that already hold a value when the first statement runs, so a
        # read of one can never be too early however the source is ordered.
        self.pre_bound: set = set()
        self.declared: set = set()
        self.uses_star_import = False
        self._seq = 0

    # -- recording ---------------------------------------------------------
    def _at(self, node) -> tuple:
        self._seq += 1
        return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0),
                self._seq)

    def _read(self, name: str, node) -> None:
        self.events.append((self._at(node), name, "read"))

    def _bind(self, name: str, node) -> None:
        self.bound.add(name)
        self.events.append((self._at(node), name, "bind"))

    def enter(self, function) -> None:
        """Parameters bind before anything in the body can run."""
        args = function.args
        every = (list(args.posonlyargs) + list(args.args) +
                 list(args.kwonlyargs))
        for arg in every:
            self.pre_bound.add(arg.arg)
            self.bound.add(arg.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                self.pre_bound.add(extra.arg)
                self.bound.add(extra.arg)
        for statement in function.body:
            self.visit(statement)

    # -- names -------------------------------------------------------------
    def visit_Name(self, node) -> None:
        if isinstance(node.ctx, ast.Store):
            self._bind(node.id, node)
        elif isinstance(node.ctx, ast.Del):
            self.bound.add(node.id)
        else:
            self._read(node.id, node)

    def visit_AugAssign(self, node) -> None:
        # `x += 1` reads x and then rebinds it, in that order.  This is the
        # shape the whole module exists for.
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._read(node.target.id, node.target)
            self._bind(node.target.id, node.target)
        else:
            self.visit(node.target)

    def visit_Global(self, node) -> None:
        self.declared.update(node.names)

    def visit_Nonlocal(self, node) -> None:
        self.declared.update(node.names)

    # -- imports -----------------------------------------------------------
    def _imported(self, node) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.uses_star_import = True
                continue
            self._bind(alias.asname or alias.name.split(".")[0], node)

    visit_Import = _imported
    visit_ImportFrom = _imported

    # -- separate scopes ---------------------------------------------------
    def _defines(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        args = getattr(node, "args", None)
        if args is not None:
            for default in list(args.defaults) + list(args.kw_defaults):
                if default is not None:
                    self.visit(default)
        for base in getattr(node, "bases", []):
            self.visit(base)
        for keyword in getattr(node, "keywords", []):
            self.visit(keyword.value)
        self._bind(node.name, node)         # the body is not ours to read

    visit_FunctionDef = _defines
    visit_AsyncFunctionDef = _defines
    visit_ClassDef = _defines

    def visit_Lambda(self, node) -> None:
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    # -- comprehensions ----------------------------------------------------
    def _comprehension(self, node) -> None:
        """Targets bind before the element expression, whatever the source says.

        `[x for x in items]` puts the read of `x` to the left of the bind, so
        plain source order would report every comprehension in the file.
        """
        for generator in node.generators:
            self.visit(generator.iter)
            self._bind_target(generator.target, node)
        for generator in node.generators:
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_DictComp = _comprehension

    def _bind_target(self, target, at) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                self._bind(node.id, at)

    # -- statements that bind a bare string --------------------------------
    def visit_ExceptHandler(self, node) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._bind(node.name, node)
        for statement in node.body:
            self.visit(statement)
