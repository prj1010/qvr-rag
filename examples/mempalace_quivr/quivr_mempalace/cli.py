"""Command-line entry point for trying the memory layer without an LLM."""

from __future__ import annotations

import argparse
import os
import sys

from .store import MemoryRecord, MempalaceMemoryStore, MemoryStoreError


def _store(args: argparse.Namespace) -> MempalaceMemoryStore:
    palace_path = args.palace or os.environ.get(
        "MEMPALACE_PALACE_PATH", "~/.mempalace/palace"
    )
    return MempalaceMemoryStore(palace_path=palace_path, wing=args.wing)


def _remember(args: argparse.Namespace) -> int:
    memory = MemoryRecord(
        content=args.content,
        wing=args.wing,
        room=args.room,
        source_file=args.source_file,
        added_by=args.added_by,
    )
    drawer_id = _store(args).add(memory)
    print(drawer_id)
    return 0


def _search(args: argparse.Namespace) -> int:
    matches = _store(args).search(
        args.query,
        wing=args.wing,
        room=args.room,
        n_results=args.limit,
    )
    if not matches:
        print("No memories found.")
        return 0
    for index, match in enumerate(matches, start=1):
        print(f"[{index}] {match.wing}/{match.room} similarity={match.similarity:.3f}")
        print(match.content)
        print()
    return 0


def _wake_up(args: argparse.Namespace) -> int:
    context = _store(args).wake_up(wing=args.wing)
    print(context or "No wake-up context found.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use MemPalace from a Quivr project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    remember = subparsers.add_parser("remember", help="store a verbatim memory")
    remember.add_argument("content")
    remember.add_argument("--wing", required=True)
    remember.add_argument("--room", default="conversation")
    remember.add_argument("--source-file", default="")
    remember.add_argument("--added-by", default="quivr-mempalace")
    remember.add_argument("--palace")
    remember.set_defaults(handler=_remember)

    search = subparsers.add_parser("search", help="semantically search memories")
    search.add_argument("query")
    search.add_argument("--wing")
    search.add_argument("--room")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--palace")
    search.set_defaults(handler=_search)

    wake_up = subparsers.add_parser("wake-up", help="load compact session context")
    wake_up.add_argument("--wing")
    wake_up.add_argument("--palace")
    wake_up.set_defaults(handler=_wake_up)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (MemoryStoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
