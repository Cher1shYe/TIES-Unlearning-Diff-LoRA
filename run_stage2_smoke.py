"""GPU-locked entry point for the isolated Stage 2 smoke matrix."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from canonical.artifacts import write_json
from canonical.runner import run_condition_matrix
from canonical.smoke import (
    PRIMARY_CONDITIONS,
    REPEAT_CONDITIONS,
    SMOKE_PROFILE_NAME,
    assert_stage2_output_path,
    build_smoke_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the GPU-locked Stage 2 smoke condition matrix."
    )
    parser.add_argument("--mode", required=True, choices=("primary", "repeat_full_sr"))
    parser.add_argument(
        "--environment", required=True, choices=("local_rtx5080", "colab_a100")
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="initialize only a new or empty smoke output directory; never delete results",
    )
    return parser


def require_expected_gpu(environment: str) -> str:
    """Require the environment-specific CUDA device before running smoke training."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 smoke requires CUDA")
    name = torch.cuda.get_device_name(0)
    expected = "RTX 5080" if environment == "local_rtx5080" else "A100"
    if expected not in name:
        raise RuntimeError(f"Expected {expected}, found {name}")
    return name


def _condition_tags(mode: str) -> tuple[str, ...]:
    return PRIMARY_CONDITIONS if mode == "primary" else REPEAT_CONDITIONS


def _command_record(
    *,
    mode: str,
    environment: str,
    argv: list[str],
    condition_tags: tuple[str, ...],
    gpu_name: str,
) -> dict[str, object]:
    return {
        "schema_version": "stage2_smoke_commands_v1",
        "mode": mode,
        "environment": environment,
        "argv": argv,
        "expected_condition_tags": list(condition_tags),
        "profile_name": SMOKE_PROFILE_NAME,
        "gpu_name": gpu_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    if not args.protocol.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {args.protocol}")

    output_dir = assert_stage2_output_path(args.output_dir, repo_root)
    gpu_name = require_expected_gpu(args.environment)
    condition_tags = _condition_tags(args.mode)
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(argv if argv is not None else sys.argv[1:]),
    ]

    if args.fresh and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--fresh requires a new or empty output directory")
    if args.fresh or not (output_dir / "commands.json").exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "commands.json",
            _command_record(
                mode=args.mode,
                environment=args.environment,
                argv=invocation,
                condition_tags=condition_tags,
                gpu_name=gpu_name,
            ),
        )

    # Import the real backend only after parsing and GPU enforcement so --help
    # remains usable in dependency-light environments.
    from canonical.backend import RealCanonicalBackend

    result = run_condition_matrix(
        args.protocol,
        output_dir,
        RealCanonicalBackend(build_smoke_config(output_dir)),
        seeds=(42,),
        condition_tags=condition_tags,
        matrix_schema_version="stage2_smoke_matrix_v1",
        fresh=args.fresh,
        command=invocation,
        repo_root=repo_root,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
