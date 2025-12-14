from __future__ import annotations

import logging
import os
import builtins
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from datasets import load_dataset
from packaging import version as pkg_version
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from .tuner_config import FineTuneConfig, load_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _ensure_torch_version_patch() -> None:
    """Some Unsloth builds expect builtins.is_torch_version to exist."""
    if hasattr(builtins, "is_torch_version"):
        return

    def is_torch_version(op: str, ver: str) -> bool:  # type: ignore
        return eval(
            f"pkg_version.parse(torch.__version__) {op} pkg_version.parse('{ver}')"
        )

    builtins.is_torch_version = is_torch_version  # type: ignore[attr-defined]


class FineTuner:
    """
    Unified fine-tuner supporting both SFT (instruct) and CPT (pretrain) modes.
    
    Memory Optimization Features:
    - Supports streaming mode for incremental dataset loading (set streaming=True in config)
    - Batched tokenization with configurable batch sizes to reduce memory spikes
    - Efficient data processing pipeline to minimize memory footprint
    """

    def __init__(self, config: FineTuneConfig | Mapping[str, Any] | str | None = None, **overrides: Any) -> None:
        _ensure_torch_version_patch()
        self.config = load_config(config, overrides)
        self.model: Any = None
        self.tokenizer: Any = None
        self.train_dataset = None
        self.val_dataset = None
        self._dataset_uses_text_field = False
        self._loaded = False

    def load(self) -> None:
        """Load model and training data. Call once before training."""
        if self._loaded:
            return
        self._load_model()
        self._load_training_data()
        self._loaded = True

    def _load_model(self) -> None:
        logger.info("✅ Loading model %s", self.config.base_model)

        model_path: str | Path = self.config.base_model
        if self.config.resume_from_lora:
            model_path = Path(self.config.resume_from_lora)
            logger.info("Resuming from provided LoRA weights at %s", model_path)
        elif self.config.model_output_path.exists():
            logger.info("LoRA weights found at %s, continuing from them", self.config.model_output_path)
            model_path = self.config.model_output_path

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.seed,
        )

        logger.info("✅ Model loaded and LoRA applied.")

    def _load_training_data(self) -> None:
        """
        Load training data from JSONL files.
        
        Supports both traditional (in-memory) and streaming modes:
        - Traditional mode: Loads entire dataset into memory, supports train/test splitting
        - Streaming mode: Loads data incrementally, reduces memory usage, requires pre-split files
        """
        logger.info("✅ Loading training data for mode: %s", self.config.mode)

        path = Path(self.config.training_data_path)
        if path.is_dir():
            train_file = path / "train.jsonl"
            test_file = path / "test.jsonl"
            if not train_file.exists():
                raise FileNotFoundError(f"train.jsonl not found in directory: {path}")
            files = {"train": str(train_file)}
            if test_file.exists():
                files["test"] = str(test_file)
        elif path.is_file():
            suffixes = path.suffixes
            is_jsonl = path.suffix == ".jsonl"
            is_jsonl_gz = len(suffixes) >= 2 and suffixes[-2:] == [".jsonl", ".gz"]

            if is_jsonl or is_jsonl_gz:
                files = {"train": str(path)}
            else:
                raise ValueError(
                    f"Invalid training_data_path: {path}. Must be directory or .jsonl/.jsonl.gz file."
                )
        else:
            raise ValueError(f"Invalid training_data_path: {path}. Must be directory or .jsonl/.jsonl.gz file.")

        # Enable streaming mode to reduce memory consumption
        dataset = load_dataset("json", data_files=files, streaming=self.config.streaming)

        if self.config.mode == "instruct":
            ds_train = dataset["train"]
            # Handle train/test split based on streaming mode
            if "test" not in dataset and self.config.val_split > 0:
                if self.config.streaming:
                    # For streaming datasets, skip auto-splitting and use provided test set only
                    logger.warning(
                        "Streaming mode enabled: auto-splitting not supported. "
                        "Validation split disabled unless test file is provided."
                    )
                    self.train_dataset = ds_train
                    self.val_dataset = None
                else:
                    split = ds_train.train_test_split(test_size=self.config.val_split, seed=self.config.seed)
                    self.train_dataset = split["train"]
                    self.val_dataset = split["test"]
            else:
                self.train_dataset = dataset["train"]
                self.val_dataset = dataset.get("test")

            if "conversations" in self.train_dataset.column_names:
                self._validate_conversation_data(self.train_dataset)
                # Use batched processing with explicit batch_size to reduce memory spikes
                self.train_dataset = self.train_dataset.map(
                    self._format_chat_conversation, 
                    batched=True,
                    batch_size=1000,
                    remove_columns=["conversations"]
                )
                if self.val_dataset:
                    self.val_dataset = self.val_dataset.map(
                        self._format_chat_conversation, 
                        batched=True,
                        batch_size=1000,
                        remove_columns=["conversations"]
                    )
                self._dataset_uses_text_field = True
            elif "text" in self.train_dataset.column_names:
                self._validate_text_data(self.train_dataset)
                self._dataset_uses_text_field = True
            else:
                self._validate_instruct_columns(self.train_dataset)
                self._dataset_uses_text_field = False
        else:
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset.get("test")
            self._dataset_uses_text_field = True
            self._validate_text_data(self.train_dataset)

        # Handle logging for streaming vs non-streaming datasets
        if self.config.streaming:
            logger.info("✅ Data loaded in streaming mode (size: unknown until consumed)")
        else:
            logger.info(
                "✅ Data loaded. Train size: %s, Val size: %s",
                len(self.train_dataset),
                len(self.val_dataset) if self.val_dataset else 0,
            )

    def train(self, num_train_epochs: Optional[int] = None) -> None:
        """Train the model using SFTTrainer with mode-aware config."""
        self.load()
        logger.info("🚀 Starting training (%s mode)", self.config.mode)

        dataset_text_field = "text" if self._dataset_uses_text_field else self.config.dataset_text_field
        formatting_func = None if self._dataset_uses_text_field else self._format_instruct_prompts
        packing = self.config.packing if self.config.packing is not None else self.config.mode == "pretrain"

        training_args = SFTConfig(
            output_dir=str(self.config.checkpoint_path),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=num_train_epochs or self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            seed=self.config.seed,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field=dataset_text_field,
            formatting_func=formatting_func,
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=max(1, os.cpu_count() // 2),
            packing=packing,
            args=training_args,
        )

        trainer.train()
        self.config.model_output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.config.model_output_path)
        self.tokenizer.save_pretrained(self.config.model_output_path)
        logger.info("✅ LoRA weights saved to %s", self.config.model_output_path)

        if self.config.save_gguf:
            self.save_gguf()

    def save_gguf(self) -> None:
        """Export the LoRA-tuned model to GGUF format."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before saving GGUF.")
        logger.info("Saving GGUF to %s...", self.config.model_output_path)
        self.config.model_output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_gguf(
            self.config.model_output_path,
            self.tokenizer,
            quantization_method=self.config.quantization_method,
        )
        self.tokenizer.save_pretrained(self.config.model_output_path)
        logger.info("✅ Model successfully saved in GGUF format.")

    def _format_instruct_prompts(self, examples: Mapping[str, list[str]]) -> list[str]:
        """Standard Alpaca/Chat formatter for instruct mode."""
        texts: list[str] = []
        for instr, inp, out in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", []),
        ):
            text = f"Instruction:\n{instr}\n\nInput:\n{inp}\n\nResponse:\n{out}<|endoftext|>"
            texts.append(text)
        return texts

    def _format_chat_conversation(self, examples: Mapping[str, Any]) -> Mapping[str, list[str]]:
        """Map chat-style conversations into a single text field using the tokenizer's template."""
        texts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    def _validate_instruct_columns(self, dataset: Any) -> None:
        missing = [c for c in ("instruction", "input", "output") if c not in dataset.column_names]
        if missing:
            raise ValueError(f"Instruct mode expects columns instruction/input/output, missing: {missing}")
        
        # For streaming datasets, skip length check and use take() instead of select()
        if self.config.streaming:
            sample = list(dataset.take(5))
            if not sample:
                raise ValueError("Training dataset is empty.")
        else:
            if len(dataset) == 0:
                raise ValueError("Training dataset is empty.")
            sample = dataset.select(range(min(5, len(dataset))))
        
        for row in sample:
            if not str(row.get("instruction", "")).strip() or not str(row.get("output", "")).strip():
                raise ValueError("Instruct rows must have non-empty 'instruction' and 'output' values.")

    def _validate_text_data(self, dataset: Any) -> None:
        if "text" not in dataset.column_names:
            raise ValueError("Pretrain mode expects a 'text' column.")
        
        # For streaming datasets, skip length check and use take() instead of select()
        if self.config.streaming:
            sample = list(dataset.take(5))
            if not sample:
                raise ValueError("Training dataset is empty.")
        else:
            if len(dataset) == 0:
                raise ValueError("Training dataset is empty.")
            sample = dataset.select(range(min(5, len(dataset))))
        
        for row in sample:
            if not str(row.get("text", "")).strip():
                raise ValueError("Text rows must have non-empty 'text' values.")

    def _validate_conversation_data(self, dataset: Any) -> None:
        if "conversations" not in dataset.column_names:
            raise ValueError("Chat-style datasets must include a 'conversations' column.")
        
        # For streaming datasets, skip length check and use take() instead of select()
        if self.config.streaming:
            sample = list(dataset.take(3))
            if not sample:
                raise ValueError("Training dataset is empty.")
        else:
            if len(dataset) == 0:
                raise ValueError("Training dataset is empty.")
            sample = dataset.select(range(min(3, len(dataset))))
        
        for row in sample:
            conv = row.get("conversations", [])
            if not isinstance(conv, list) or not conv:
                raise ValueError("Each row must contain a non-empty list of conversation turns.")


# Backwards-compatible aliases
DocumentFineTune = FineTuner
UniversalFineTuner = FineTuner
