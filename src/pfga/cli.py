"""Command-line entry point for PFGA experiments."""

import argparse
import json
import logging
from typing import Optional, Sequence

from pfga.config import load_config
from pfga.trainer import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Personalized Fair Graph Assigner"
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    summary = run_experiment(config)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "test_periods"}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
