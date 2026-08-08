"""Dependency-light CLI for validated Stage-2 evidence export."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and package lightweight Stage 2 A100 evidence.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-expectations", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from canonical.freeze import build_evidence_archive
    result = build_evidence_archive(args.repo_root, args.output, expectations_path=args.source_expectations)
    print(result["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
