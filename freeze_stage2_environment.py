"""Dependency-light CLI for creating or independently verifying a Stage-2 freeze."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze validated Stage 2 Colab A100 evidence.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--smoke-root", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from canonical.freeze import build_freeze_bundle, verify_freeze_bundle

    if args.verify_only:
        if args.fresh or any(value is not None for value in (args.protocol, args.smoke_root, args.source_archive, args.commands)):
            raise SystemExit("--verify-only accepts only --output-dir")
        report = verify_freeze_bundle(args.output_dir)
    else:
        missing = [name for name, value in (("--protocol", args.protocol), ("--smoke-root", args.smoke_root), ("--source-archive", args.source_archive), ("--commands", args.commands)) if value is None]
        if missing:
            raise SystemExit(f"missing required creation arguments: {', '.join(missing)}")
        if not args.fresh:
            raise SystemExit("freeze creation requires --fresh")
        report = build_freeze_bundle(args.protocol, args.smoke_root, args.output_dir, args.repo_root,
                                     source_archive_path=args.source_archive, commands_path=args.commands)
    print(report["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
