"""Frozen canonical_v1 experiment entry point."""

import argparse
import json
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the frozen canonical_v1 core experiment matrix."
    )
    parser.add_argument("--stage", required=True, choices=("core",))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="initialize only a new or empty output directory; never delete existing results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.protocol.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {args.protocol}")

    # These modules are dependency-light. PyTorch, Transformers, Datasets, and
    # NumPy are imported only when the selected backend operation actually runs.
    from canonical.backend import RealCanonicalBackend
    from canonical.runner import run_core

    result = run_core(
        args.protocol,
        args.output_dir,
        RealCanonicalBackend(),
        fresh=args.fresh,
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        repo_root=Path(__file__).resolve().parent,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
