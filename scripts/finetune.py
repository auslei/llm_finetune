#!/usr/bin/env python3
"""
CLI for supervised fine-tuning using SFTTrainer (DocumentFineTune).
"""
import argparse
import logging
import sys

from llm_finetune.finetune_tool import DocumentFineTune

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a base LLM with LoRA via SFTTrainer."
    )
    parser.add_argument(
        "--training_data_path", "-i", required=True,
        help="Path to .jsonl file or directory with train/test splits"
    )
    parser.add_argument(
        "--model_name", "-n", required=True,
        help="Name for saving checkpoints and weights"
    )
    parser.add_argument(
        "--base_model", "-b", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        help="Base model identifier or path"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--mode", choices=["pretrain", "instruct"], default="pretrain",
        help="Training mode (pretrain or instruct)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--save_gguf", action="store_true",
        help="Also export the model to GGUF format after training"
    )
    args = parser.parse_args()

    try:
        tuner = DocumentFineTune(
            training_data_path=args.training_data_path,
            model_name=args.model_name,
            base_model_path=args.base_model,
            max_seq_length=args.max_seq_length,
            training_mode=args.mode,
        )
        tuner.train(num_train_epochs=args.epochs)
        if args.save_gguf:
            tuner.save_gguf()
    except Exception:
        logger.exception("An error occurred during fine-tuning:")
        sys.exit(1)


if __name__ == "__main__":
    main()