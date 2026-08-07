"""Dependency-light CLI for packaging the exact clean Stage-2 source commit."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package an exact clean Git commit for Stage 2 Colab.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from canonical.source_package import build_source_package

    metadata = build_source_package(args.repo_root, args.protocol, args.output)
    print(metadata["bundle_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
