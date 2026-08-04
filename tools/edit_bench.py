"""Does this model actually produce edits this app can apply?

The edit log says what happened after the fact. This asks the question on
purpose: it sends a real request about a real folder, through the same system
prompt and the same parser the app uses, and reports the verdict the app would
have reached. Nothing is ever written — the point is to find out whether a
diff would have been offered, not to change anything.

    python tools/edit_bench.py --folder ../Replyit --model qwen2.5-coder:32b \
        --ask "In replypilot_classify_engine.py, ..."

Two things make it worth having. Comparing models on "can it be applied" is a
different question from comparing them on how good the prose is, and it is the
one that decides whether the editing feature works at all. And a prompt change
can be measured here in one run instead of being guessed at over an afternoon
of real use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import edits as edit_tools  # noqa: E402
from aichatlab import validate  # noqa: E402
from aichatlab.client import OllamaClient  # noqa: E402
from aichatlab.config import CONVERSATIONAL_PROMPT  # noqa: E402
from aichatlab.folderscan import Limits, build_block, select, survey  # noqa: E402


def folder_block(root: Path, budget_chars: int) -> tuple[str, list]:
    """The same attachment the app builds when you attach a folder."""
    found = survey(root)
    selection = select(found, Limits.everything())
    block = build_block(root, selection.chosen, len(found.entries),
                        char_budget=budget_chars)
    return block, [entry.relative for entry in selection.chosen]


def run(folder: Path, model: str, server_url: str, ask: str,
        budget_chars: int, num_predict: int, detailed: bool,
        timeout: int, second_pass: str = "", only: str = "") -> dict:
    """One request, through the app's own prompt and parser.

    `second_pass` names a file and switches to the shape the app uses after a
    described change is approved: the complete file, one change, prose
    refused. It is a different question from the first pass and worth
    measuring separately — a folder block truncates a big file to fit, and a
    model cannot copy anchor lines it was never shown.
    """
    if second_pass:
        from aichatlab.intent import Change, build_change_prompt

        current = edit_tools.read_current(folder, second_pass)
        if not current:
            raise ValueError(f"cannot read {second_pass} from {folder}")
        sent = [second_pass]
        system = CONVERSATIONAL_PROMPT
        user = build_change_prompt(
            Change(description=ask, file=second_pass), current,
            edit_tools.instructions(detailed=True))
    elif only:
        # What retrieval does once a folder is too big to send whole: the few
        # files this question needs, complete, instead of all of them with the
        # middle cut out. A model cannot copy an anchor it was never shown.
        sent = [name.strip() for name in only.split(",") if name.strip()]
        parts = []
        for name in sent:
            text = edit_tools.read_current(folder, name)
            if not text:
                raise ValueError(f"cannot read {name} from {folder}")
            parts.append(f"--- {name} ---\n{text}\n--- end of {name} ---")
        block = (f"[Folder: {folder.name} — {len(sent)} file(s) chosen for "
                 f"this question]\n\n" + "\n\n".join(parts))
        system = "\n\n".join([
            CONVERSATIONAL_PROMPT,
            edit_tools.instructions(detailed=detailed),
        ])
        user = f"{block}\n\n{ask}"
    else:
        block, sent = folder_block(folder, budget_chars)
        system = "\n\n".join([
            CONVERSATIONAL_PROMPT,
            edit_tools.instructions(detailed=detailed),
        ])
        user = f"{block}\n\n{ask}"

    client = OllamaClient(server_url, server="bench", timeout=timeout)
    started = time.time()
    result = client.chat(
        model,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        options={"temperature": 0.1,
                 "num_predict": edit_tools.allowance_for_edits(num_predict),
                 "num_ctx": 32768},
    )
    elapsed = time.time() - started
    reply = result.text or ""

    # -- the app's pipeline, in the app's order --------------------------
    proposed, rejected = edit_tools.parse(reply, truncated=result.truncated)
    allowed, refused = edit_tools.check(folder, proposed)
    rejected = list(rejected) + list(refused)
    patches, bad = edit_tools.parse_patches(reply, truncated=result.truncated)
    patched, unmatched = edit_tools.resolve_patches(
        folder, patches, known=sorted(sent))
    rejected += list(bad) + list(unmatched)
    known = {e.name for e in allowed}
    allowed += [e for e in patched if e.name not in known]
    allowed = [e for e in allowed if not edit_tools.unchanged(folder, e)]

    flaws = {}
    for edit in allowed:
        found = validate.introduced(
            edit.name, edit_tools.read_current(folder, edit.name),
            edit.new_text)
        if found:
            flaws[edit.name] = found

    if allowed and not flaws:
        verdict = "APPLICABLE"
    elif allowed:
        verdict = "APPLICABLE-BUT-BROKEN"
    elif rejected:
        verdict = "REJECTED"
    else:
        verdict = "NO-BLOCKS"

    return {
        "verdict": verdict, "reply": reply, "elapsed": elapsed,
        "model": model, "truncated": result.truncated,
        "eval_tokens": getattr(result, "eval_tokens", None),
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "allowed": allowed, "rejected": rejected, "flaws": flaws,
        "sent": sent, "prompt_chars": len(system) + len(user),
    }


def report(out: dict, show_reply: int = 0) -> None:
    from aichatlab.editdebug import shape, shape_note

    print(f"  model     : {out['model']}")
    print(f"  verdict   : {out['verdict']}")
    print(f"  took      : {out['elapsed']:.0f}s"
          f"   prompt {out['prompt_tokens'] or '?'} tok"
          f" / reply {out['eval_tokens'] or '?'} tok"
          f"{'  (CUT OFF)' if out['truncated'] else ''}")
    info = shape(out["reply"])
    print(f"  shape     : {shape_note(info) or 'well-formed edit blocks'}")
    print(f"  markers   : edit_blocks={info['edit_blocks']} "
          f"file_blocks={info['file_openers']} fences={info['code_fences']}")
    for edit in out["allowed"]:
        mark = "✖ breaks" if out["flaws"].get(edit.name) else "✔ applies"
        print(f"    {mark}  {edit.name}")
        for problem in out["flaws"].get(edit.name, []):
            print(f"              {problem.describe()}")
    for item in out["rejected"]:
        print(f"    ✖ refused {item.name}: {item.reason}")
    if show_reply:
        print("  --- reply " + "-" * 50)
        for line in out["reply"][:show_reply].splitlines():
            print("  | " + line)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="repeatable, to compare models")
    parser.add_argument("--server", default="http://100.106.60.4:11434")
    parser.add_argument("--ask", required=True)
    parser.add_argument("--budget-chars", type=int, default=48_000)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--detailed", action="store_true",
                        help="use the second-pass instructions with the example")
    parser.add_argument("--show-reply", type=int, default=0)
    parser.add_argument("--only", default="",
                        help="FILE[,FILE] — send just these, complete, as "
                             "retrieval does for a big folder")
    parser.add_argument("--second-pass", default="",
                        help="FILE — send the whole file, one change, as the "
                             "app does after a described change is approved")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"No such folder: {folder}")
        return 2

    print(f"folder: {folder}")
    print(f"ask   : {args.ask}\n")
    tally = {}
    for model in args.model:
        print("=" * 68)
        try:
            out = run(folder, model, args.server, args.ask,
                      args.budget_chars, args.num_predict, args.detailed,
                      args.timeout, args.second_pass, args.only)
        except Exception as exc:                      # noqa: BLE001
            print(f"  model     : {model}\n  FAILED    : {str(exc)[:200]}")
            tally[model] = "ERROR"
            continue
        report(out, args.show_reply)
        tally[model] = out["verdict"]
    print("=" * 68)
    for model, verdict in tally.items():
        print(f"  {verdict:<22} {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
