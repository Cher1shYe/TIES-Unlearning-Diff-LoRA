"""CLI for dependency-light validation of a completed Stage-2 smoke root."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate strict Stage 2 smoke artifacts.")
    parser.add_argument("--root", type=Path, required=True, help="completed smoke artifact root")
    parser.add_argument("--conditions", nargs="+", required=True, help="expected method tags")
    parser.add_argument(
        "--canonical-dir", type=Path, default=Path("ties_results/canonical_v1"),
        help="formal result directory, which must be absent or empty",
    )
    return parser


def _markdown(report: dict) -> str:
    lines = ["# Stage 2 Smoke Validation", "", f"State: `{report['state']}`", "", "## Checks", ""]
    lines.extend(f"- {name}: `{entry['state']}`" for name, entry in report["checks"].items())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Delayed import guarantees `--help` works without torch/transformers/datasets.
    from canonical.artifacts import write_json
    from canonical.stage2_validation import validate_smoke_root

    report = validate_smoke_root(args.root, expected_conditions=args.conditions, canonical_dir=args.canonical_dir)
    write_json(args.root / "stage2_validation.json", report)
    (args.root / "stage2_validation.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(report["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
