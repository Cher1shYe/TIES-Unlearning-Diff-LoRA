"""Dependency-light CLI for Stage-2 no-weight transport verification."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Stage 2 evidence ZIP before optionally extracting ties_results."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from canonical.evidence_transport import verify_evidence_archive

    report = verify_evidence_archive(args.archive, extract_dir=args.extract_dir)
    print(report["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
