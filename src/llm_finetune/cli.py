from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dill import PicklingWarning
from .finetune_tool import FineTuner
import warnings

warnings.filterwarnings("ignore", category=PicklingWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fine-tuning with a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_train_epochs from config.",
    )
    parser.add_argument(
        "--save-gguf",
        action="store_true",
        help="Enable GGUF export regardless of config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.save_gguf:
        overrides["save_gguf"] = True

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Loading config from %s", config_path)
    tuner = FineTuner(str(config_path), **overrides)
    tuner.train()


if __name__ == "__main__":
    main()
