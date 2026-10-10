"""Command-line interface.

    python -m qgen import data/bank.json                  # load questions (+ tags) into the collection
    python -m qgen add "What is total internal reflection?" --tag chapter=light --tag marks=2
    python -m qgen list --tag topic=reflection
    python -m qgen check "Define the reflection of light."  # independent duplicate check
    python -m qgen generate "Reflection of light" -n 5 --tag chapter=light   # generate → check → store
    python -m qgen eval                                    # accuracy on corpus + calibration pairs

The collection lives in $QGEN_DB (default runtime/questions.db).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .store import QuestionStore


def _tags(pairs: list[str] | None) -> dict:
    out: dict[str, list[str]] = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qgen")
    ap.add_argument("--db", default=os.environ.get("QGEN_DB", "runtime/questions.db"))
    ap.add_argument("--offline", action="store_true", help="no LLM calls (judge → human review)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import"); p.add_argument("path")
    p = sub.add_parser("add"); p.add_argument("text"); p.add_argument("--tag", action="append"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("list"); p.add_argument("--tag", action="append")
    p = sub.add_parser("check"); p.add_argument("text", nargs="+")
    p = sub.add_parser("generate"); p.add_argument("topic"); p.add_argument("-n", type=int, default=5)
    p.add_argument("--tag", action="append", help="tags for the new questions")
    p.add_argument("--scope", action="append", help="tag filter for the context pool, e.g. chapter=light")
    sub.add_parser("eval")
    args = ap.parse_args(argv)
    if args.offline:
        os.environ["QGEN_OFFLINE"] = "1"

    from .detector import DuplicateDetector

    store = QuestionStore(args.db)
    if args.cmd == "import":
        data = json.loads(open(args.path, encoding="utf-8").read())
        items = data["questions"] if isinstance(data, dict) else data
        for q in items:
            store.add(q["text"], q.get("tags"), source=q.get("source", "human"))
        print(f"imported {len(items)}; collection has {store.count()} questions")
    elif args.cmd == "add":
        v = DuplicateDetector(store).build().check(args.text)
        if v.label != "unique" and not args.force:
            print(f"not added: {v.label} of {v.matched_id} ({v.matched_text}) — {v.reason}\nuse --force to add anyway")
            return 2
        q = store.add(args.text, _tags(args.tag))
        print(f"added {q.id} {q.tags}")
    elif args.cmd == "list":
        for q in store.all(_tags(args.tag)):
            print(f"{q.id}  [{q.status}] {q.text}  {json.dumps(q.tags)}")
    elif args.cmd == "check":
        det = DuplicateDetector(store).build()
        for text in args.text:
            v = det.check(text)
            print(f"{v.label:<19} {text}")
            if v.matched_id:
                print(f"{'':<19} ↳ {v.matched_id}: {v.matched_text}")
            print(f"{'':<19}   {v.reason}")
    elif args.cmd == "generate":
        from .pipeline import generate_and_store

        rep = generate_and_store(store, args.topic, args.n, tags=_tags(args.scope), new_tags=_tags(args.tag))
        ctx = rep.generation.context
        print(f"context: {ctx.relevant} relevant questions → {len(ctx.concepts)} concepts, {ctx.shown} shown "
              f"(≈{ctx.est_tokens} tokens)")
        for c in rep.generation.rejected:
            print(f"  rejected in generation: {c.question}  ({c.reason})")
        for c, v in zip(rep.generation.accepted, rep.verdicts):
            print(f"  {v.label:<19} {c.question}" + (f"  ↳ {v.matched_id}" if v.matched_id else ""))
        print(f"stored {len(rep.stored)}, review {len(rep.review)}, dropped {len(rep.dropped)}")
    elif args.cmd == "eval":
        from . import evaluate as E
        from .llm import llm_available

        for flag in [False] + ([True] if llm_available() else []):
            rows = E.eval_corpus(flag)
            m = E.eval_pairs(flag)
            print(f"{'LLM judge' if flag else 'offline'}: corpus {sum(r['ok'] for r in rows)}/{len(rows)}, "
                  f"calibration accuracy {m['accuracy']:.0%}, dup precision {m['dup_precision']:.0%}, "
                  f"dup recall {m['dup_recall']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
