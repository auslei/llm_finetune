from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import (
    TrainingArguments, Trainer,
    EarlyStoppingCallback, DataCollatorForLanguageModeling
)
import torch
import os
from packaging import version as pkg_version
import builtins
from pathlib import Path
from trl import SFTConfig, SFTTrainer

# Compatibility patch for torch version checks used in Unsloth
def is_torch_version(op, ver):
    return eval(f"pkg_version.parse(torch.__version__) {op} pkg_version.parse('{ver}')")
builtins.is_torch_version = is_torch_version


class DocumentFineTune:
    def __init__(self,
                 training_data_path: str,
                 model_name: str,
                 base_model_path: str = "unsloth/mistral-7b-bnb-4bit",
                 max_seq_length: int = 2048,
                 seed: int = 40):
        """
        Setup model and training configuration.
        """
        self.base_model_path = base_model_path
        self.training_data_path = Path(training_data_path)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.seed = seed

        self.model_checkpoints = Path(f"checkpoints/{model_name}")
        self.model_lora_weights_location = Path(f"models/{model_name}")

        self.training_args = TrainingArguments(
            output_dir=str(self.model_checkpoints),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            seed=self.seed,
        )

        self.load_model()
        self.load_training_data()

    def load_model(self):
        """
        Load model and apply LoRA with Unsloth.
        """
        print("✅ Loading model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None
        )

    def load_training_data(self):
        """
        Load JSONL training data, tokenize, and prepare splits.
        Supports either single file (auto split) or train/test files.
        """
        print("✅ Loading training data...")

        # Determine available files
        path = self.training_data_path
        if path.is_dir():
            train_file = path / "train.jsonl"
            test_file = path / "test.jsonl"
            if not train_file.exists():
                raise FileNotFoundError("train.jsonl not found in directory.")
            files = {"train": str(train_file)}
            if test_file.exists():
                files["test"] = str(test_file)
        elif path.is_file():
            files = {"train": str(path)}
        else:
            raise ValueError("Invalid training_data_path provided.")

        dataset = load_dataset("json", data_files=files)

        def tokenize(example):
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens


        # if "test" in dataset:
        #     self.train_dataset = dataset["train"].filter(lambda x: len(x["text"]) > 60)
        #     self.val_dataset = dataset["test"].filter(lambda x: len(x["text"]) > 60)
        # else:
        #     # Auto-split if only one file
        # dataset = dataset["train"].filter(lambda x: len(x["text"]) > 60)

        split = dataset["train"].train_test_split(test_size=0.1, seed=self.seed)
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]

        # this handles Q & A data
        if "conversations" in dataset["train"].column_names:
            def formatting_prompts_func(examples):
                all_convos = examples["conversations"]  # List of conversations in the batch
                texts = [
                    self.tokenizer.apply_chat_template(
                        convo, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    for convo in all_convos
                ]
                return {"text": texts}
            self.train_dataset = self.train_dataset.map(formatting_prompts_func, batched=True)
            self.val_dataset = self.val_dataset.map(formatting_prompts_func, batched=True)

        self.train_dataset = self.train_dataset.map(tokenize, batched=True, remove_columns=["text"])
        self.val_dataset = self.val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    


    def train_1(self):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.train_dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = SFTConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 60,
                learning_rate = 2e-4,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = self.model_checkpoints,
                report_to = "none", # Use this for WandB etc
            ),
        )

        self.model.save_pretrained(
            self.model_lora_weights_location,
            save_method="lora",
            safe_serialization=False
        )

    def train(self):
        """
        Fine-tune the model using HuggingFace Trainer.
        """
        print("✅ Training started...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            data_collator=data_collator,
        )

        trainer.train()

        self.model.save_pretrained(
            self.model_lora_weights_location,
            save_method="lora",
            safe_serialization=False
        )

    def save_gguf(self):
        """
        Export LoRA fine-tuned model to GGUF for Ollama / llama.cpp.
        """
        self.model.save_pretrained_gguf(
            self.model_lora_weights_location,
            self.tokenizer,
            quantization_method="q4_k_m"
        )
        self.tokenizer.save_pretrained(self.model_lora_weights_location)
