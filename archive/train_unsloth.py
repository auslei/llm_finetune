from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import (
    TrainingArguments, Trainer,
    EarlyStoppingCallback, DataCollatorForLanguageModeling
)
import torch
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
                 training_mode = 'pretrain',
                 seed: int = 40):
        """
        Setup model and training configuration.
        """
        self.base_model_path = base_model_path
        self.training_data_path = Path(training_data_path)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.training_mode = training_mode

        self.model_checkpoints = Path(f"checkpoints/{model_name}")
        self.model_lora_weights_location = Path(f"models/{model_name}")

        self.load_training_data()
        self.load_model()        

    def load_model(self):
        """
        Load model and apply LoRA with Unsloth.
        """
        print(f"✅ Loading model: {self.base_model_path}...")
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
        print(f"✅ Loading training data {self.training_mode}...")

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
            raise ValueError("Invalid training_data_path provided: {path}")

        dataset = load_dataset("json", data_files=files)
        print(dataset['train'][0])

        def tokenize(example):
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        if self.training_mode == "instruct":
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
        elif self.training_mode == "pretrain":
            self.train_dataset = dataset["train"]
            self.val_dataset = None
        else:
            raise Exception(f"Training Mode: {self.training_mode} does not exist!")


    def train(self):
        
        print(self.train_dataset[0])

        # ⚙️ TrainingArguments (no evaluation)
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=2,
            warmup_steps=10,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(self.model_checkpoints),
            report_to="none",
        )


        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args =args
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
