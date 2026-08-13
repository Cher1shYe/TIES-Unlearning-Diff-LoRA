"""Command-line entry point for the frozen Stage 2 subprocess monitor."""

import argparse
from pathlib import Path
import sys

from canonical.monitoring import monitor_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor a Stage 2 command without changing its experiment settings."
    )
    parser.add_argument("--events", required=True, type=Path, help="JSONL output path")
    parser.add_argument(
        "--watch",
        required=True,
        action="append",
        type=Path,
        help="file or directory whose changes count as job progress (repeatable)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command argv after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        raise SystemExit("a command is required after --")
    return monitor_command(
        command,
        cwd=Path.cwd(),
        events_path=arguments.events,
        watched_paths=arguments.watch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
