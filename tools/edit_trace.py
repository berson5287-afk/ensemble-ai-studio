"""Read the edit-check trace log from outside the app.

    python tools/edit_trace.py              # the last 20 checks
    python tools/edit_trace.py --all        # everything in the file
    python tools/edit_trace.py --failures   # only the runs that wrote nothing
    python tools/edit_trace.py --reply 3    # the reply trace 3 was reading

The app writes one JSON object per line to ~/.ai_chat_lab_edit_trace.jsonl
every time it checks a reply for edits. This prints it in a form a person can
read, which is the whole reason the file exists: "it listed the changes and
did nothing" is one sentence covering half a dozen different failures, and the
only way to tell them apart is to look at what the check actually decided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab.editdebug import (  # noqa: E402
    TRACE_PATH,
    VERDICTS,
    read_traces,
)


def render(index: int, trace: dict, show_reply: bool = False) -> str:
    verdict = trace.get("verdict", "?")
    lines = [f"#{index}  [{trace.get('at', '')}]  {verdict} — "
             f"{trace.get('verdict_text') or VERDICTS.get(verdict, '')}"]
    if trace.get("project"):
        lines.append(f"    folder: {trace['project']}")
    flags = []
    if trace.get("truncated"):
        flags.append("reply was cut off at the token cap")
    if trace.get("second_pass"):
        flags.append("second pass")
    if not trace.get("had_target"):
        flags.append("no model recorded to retry against")
    if flags:
        lines.append("    " + "; ".join(flags))
    if trace.get("shape_note"):
        lines.append(f"    shape: {trace['shape_note']}")
    shape = trace.get("reply_shape") or {}
    if shape:
        lines.append("    counts: " + ", ".join(
            f"{key}={value}" for key, value in shape.items()))
    for step in trace.get("steps") or []:
        bit = f"    {step.get('name')}: {step.get('count')}"
        names = step.get("names") or []
        if names:
            bit += f" ({', '.join(names[:6])}{', …' if len(names) > 6 else ''})"
        if step.get("detail"):
            bit += f" — {step['detail']}"
        lines.append(bit)
    for item in trace.get("rejections") or []:
        lines.append(f"    refused {item.get('name')}: {item.get('reason')}")
    if trace.get("claimed_to_write"):
        lines.append(f"    ⚠ the model claimed to have written: "
                     f"“{trace['claimed_to_write']}”")
    if show_reply and trace.get("reply_head"):
        lines.append("    --- reply ---")
        lines += [f"    | {line}"
                  for line in trace["reply_head"].splitlines()]
    return "\n".join(lines)


def main() -> int:
    # The Windows console defaults to cp1252, which cannot encode the arrows
    # and warning signs in these messages — printing one raises rather than
    # degrading. A debugging tool that crashes on its own output is no use.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None,
                        help=f"trace file (default {TRACE_PATH})")
    parser.add_argument("--all", action="store_true",
                        help="every trace in the file, not just the tail")
    parser.add_argument("--failures", action="store_true",
                        help="only checks that produced no diff")
    parser.add_argument("--reply", type=int, default=None, metavar="N",
                        help="also print the reply text of trace N")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    traces = read_traces(args.path, limit=0 if args.all else args.limit)
    if not traces:
        target = args.path or TRACE_PATH
        print(f"No traces in {target}.")
        print("Attach a folder, ask a model to change a file, and this fills "
              "in. If it stays empty, the edit check is never running — "
              "check that a folder is attached and editing is on.")
        return 1

    shown = [(i, t) for i, t in enumerate(traces)]
    if args.failures:
        shown = [(i, t) for i, t in shown if not t.get("offered")]
        if not shown:
            print(f"All {len(traces)} check(s) reached a diff. "
                  f"Nothing failed.")
            return 0

    tally: dict = {}
    for _, trace in shown:
        key = trace.get("verdict", "?")
        tally[key] = tally.get(key, 0) + 1
    print(f"{len(shown)} check(s) — " + ", ".join(
        f"{count} {verdict}" for verdict, count
        in sorted(tally.items(), key=lambda kv: -kv[1])))
    print()
    for index, trace in shown:
        print(render(index, trace, show_reply=(index == args.reply)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
