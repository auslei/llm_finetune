import os
import logging
import builtins
from pathlib import Path
from packaging import version as pkg_version

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

# Module-level logger (configured by caller)
logger = logging.getLogger(__name__)

def is_torch_version(op, ver):
    """Compatibility patch for torch version checks used in Unsloth."""
    return eval(f"pkg_version.parse(torch.__version__) {op} pkg_version.parse('{ver}')")

builtins.is_torch_version = is_torch_version

class DocumentFineTune:
    def __init__(
        self,
        training_data_path: str,
        model_name: str,
        base_model_path: str = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length: int = 2048,
        training_mode: str = "pretrain",
        seed: int = 40,
    ):
        """
        Initializes fine-tuning pipeline: loads model and training data.

        Args:
            training_data_path: Directory or JSONL file of training data.
            model_name: Name for saving checkpoints and weights.
            base_model_path: Base model identifier or path.
            max_seq_length: Maximum sequence length.
            training_mode: 'pretrain' or 'instruct'.
            seed: RNG seed.
        """
        self.base_model_path = base_model_path
        self.training_data_path = Path(training_data_path)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.training_mode = training_mode

        self.model_checkpoints = Path(f"checkpoints/{model_name}")
        self.model_lora_weights_location = Path(f"models/{model_name}")

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

        self.load_model()
        self.load_training_data()

    def load_model(self):
        """
        Loads the base model and applies LoRA using Unsloth utilities.
        """
        logger.info(f"✅ Loading model: {self.base_model_path}...")

        model_path = self.base_model_path
        if self.model_lora_weights_location.exists():
            logger.info(
                f"Loading from existing LoRA weights at {self.model_lora_weights_location}"
            )
            model_path = self.model_lora_weights_location

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=self.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("✅ Model loaded and LoRA applied.")

    def load_training_data(self):
        """
        Loads JSONL data, splits into train/test, and applies chat template if needed.
        """
        logger.info(f"✅ Loading training data for mode: {self.training_mode}")

        path = self.training_data_path
        files = {}
        if path.is_dir():
            train_file = path / "train.jsonl"
            test_file = path / "test.jsonl"
            if not train_file.exists():
                raise FileNotFoundError(f"train.jsonl not found in directory: {path}")
            files["train"] = str(train_file)
            if test_file.exists():
                files["test"] = str(test_file)
        elif path.is_file() and path.suffix == ".jsonl":
            files = {"train": str(path)}
        else:
            raise ValueError(
                f"Invalid training_data_path: {path}. Must be directory or .jsonl file."
            )

        dataset = load_dataset("json", data_files=files)

        if self.training_mode == "instruct":
            ds_train = dataset["train"]
            if "test" not in dataset:
                split = ds_train.train_test_split(test_size=0.1, seed=self.seed)
                self.train_dataset = split["train"]
                self.val_dataset = split["test"]
            else:
                self.train_dataset = dataset["train"]
                self.val_dataset = dataset["test"]

            if "conversations" in self.train_dataset.column_names:
                def format_fn(examples):
                    texts = []
                    for convo in examples["conversations"]:
                        texts.append(
                            self.tokenizer.apply_chat_template(
                                convo,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                        )
                    return {"text": texts}

                self.train_dataset = self.train_dataset.map(format_fn, batched=True)
                if self.val_dataset:
                    self.val_dataset = self.val_dataset.map(format_fn, batched=True)
        else:
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset.get("test")

        logger.info(
            f"✅ Data loaded. Train size: {len(self.train_dataset)},"
            f" Val size: {len(self.val_dataset) if self.val_dataset else 0}"
        )

    def train(self, num_train_epochs: int = 2):
        """
        Run supervised fine-tuning with SFTTrainer.
        """
        logger.info("Starting training...")
        training_args = SFTConfig(
            output_dir=str(self.model_checkpoints),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=num_train_epochs,
            warmup_steps=10,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.seed,
            report_to="none",
            eval_strategy="epoch" if self.val_dataset else "no",
            eval_steps=2 if self.val_dataset else None,
            save_strategy="epoch",
            load_best_model_at_end=bool(self.val_dataset),
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=os.cpu_count() or 1,
            packing=False,
            args=training_args,
        )
        trainer.train()
        self.model_lora_weights_location.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_lora_weights_location)
        logger.info(f"✅ LoRA weights saved to {self.model_lora_weights_location}")

    def save_gguf(self):
        """
        Export the LoRA-tuned model to GGUF format.
        """
        logger.info(f"Saving GGUF to {self.model_lora_weights_location}...")
        self.model_lora_weights_location.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained_gguf(
            self.model_lora_weights_location,
            self.tokenizer,
            quantization_method="q4_k_m",
        )
        self.tokenizer.save_pretrained(self.model_lora_weights_location)
        logger.info("✅ Model successfully saved in GGUF format.")