from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict # Import DatasetDict for type hinting
import torch
import os
import logging # Import logging module
from packaging import version as pkg_version
import builtins
from pathlib import Path
from trl import SFTConfig, SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Compatibility patch for torch version checks used in Unsloth
# This helps Unsloth correctly identify PyTorch versions in some environments.
def is_torch_version(op, ver):
    return eval(f"pkg_version.parse(torch.__version__) {op} pkg_version.parse('{ver}')")
builtins.is_torch_version = is_torch_version


class DocumentFineTune:
    def __init__(self,
                 training_data_path: str,
                 model_name: str,
                 base_model_path: str = "unsloth/mistral-7b-bnb-4bit",
                 max_seq_length: int = 2048,
                 training_mode: str = 'pretrain', # Added type hint for clarity
                 seed: int = 40):
        """
        Initializes the DocumentFineTune class, setting up paths, model parameters,
        and loading the model and training data.

        Args:
            training_data_path (str): Path to the training data (directory or single JSONL file).
            model_name (str): Name for the fine-tuned model and its checkpoints.
            base_model_path (str): Path or name of the base model to load (default: Unsloth Mistral-7B).
            max_seq_length (int): Maximum sequence length for tokenization (default: 2048).
            training_mode (str): Mode of training ('pretrain' or 'instruct') (default: 'pretrain').
            seed (int): Random seed for reproducibility (default: 40).
        """
        self.base_model_path = base_model_path
        self.training_data_path = Path(training_data_path)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.training_mode = training_mode

        # Define paths for saving model checkpoints and final LoRA weights
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
        Loads the base language model and tokenizer, then applies LoRA adaptations
        using Unsloth's optimized methods.
        """
        logger.info(f"✅ Loading model: {self.base_model_path}...")
        # Load the base model and tokenizer from HuggingFace
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=torch.float16, # Use float16 for faster training and less memory
            load_in_4bit=True, # Load in 4-bit quantization for memory efficiency
        )

        # Apply LoRA (Low-Rank Adaptation) to the model for efficient fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16, # LoRA attention dimension
            target_modules=[ # Modules to apply LoRA to
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=32, # LoRA scaling factor
            lora_dropout=0, # Dropout percentage for LoRA layers
            bias="none", # Type of bias to add to LoRA layers
            use_gradient_checkpointing=True, # Enable gradient checkpointing for memory saving
            random_state=self.seed, # Random seed for LoRA initialization
            use_rslora=False, # Whether to use Reversible LoRA
            loftq_config=None # LoFTQ configuration for quantization-aware fine-tuning
        )
        logger.info("✅ Model loaded and LoRA applied.")

    def load_training_data(self):
        """
        Loads JSONL training data from the specified path, handles train/test splits,
        and applies chat templates if the training mode is 'instruct'.
        """
        logger.info(f"✅ Loading training data for mode: {self.training_mode} from {self.training_data_path}...")

        # Determine if the path is a directory containing train/test.jsonl or a single file
        path = self.training_data_path
        files = {}
        if path.is_dir():
            train_file = path / "train.jsonl"
            test_file = path / "test.jsonl"
            if not train_file.exists():
                logger.error(f"Error: train.jsonl not found in directory: {path}")
                raise FileNotFoundError(f"train.jsonl not found in directory: {path}")
            files["train"] = str(train_file)
            if test_file.exists():
                files["test"] = str(test_file)
                logger.info(f"Found train.jsonl and test.jsonl in {path}")
            else:
                logger.warning(f"test.jsonl not found in {path}. Only training on train.jsonl.")
        elif path.is_file():
            if not path.suffix == ".jsonl":
                logger.error(f"Error: Provided file is not a .jsonl file: {path}")
                raise ValueError(f"Provided file is not a .jsonl file: {path}")
            files = {"train": str(path)}
            logger.info(f"Loading single training file: {path}")
        else:
            logger.error(f"Error: Invalid training_data_path provided: {path}. Must be a directory or a .jsonl file.")
            raise ValueError(f"Invalid training_data_path provided: {path}")

        # Load dataset from JSONL file(s)
        try:
            dataset = load_dataset("json", data_files=files)
            logger.info(f"Initial dataset loaded. Example: {dataset['train'][0]}")
        except Exception as e:
            logger.critical(f"Failed to load dataset from {files}: {e}")
            raise

        # Conditional processing based on training mode
        if self.training_mode == "instruct":
            # For instruct tuning, split the dataset into train and validation
            if "train" not in dataset:
                logger.error("Error: 'train' split not found in dataset for 'instruct' mode.")
                raise ValueError("Dataset must contain a 'train' split for 'instruct' mode.")

            # If only a train file was provided, create a test split from it
            if "test" not in dataset:
                split_ratio = 0.1
                logger.info(f"Creating train/validation split (test_size={split_ratio}) from training data.")
                split = dataset["train"].train_test_split(test_size=split_ratio, seed=self.seed)
                self.train_dataset = split["train"]
                self.val_dataset = split["test"]
            else:
                self.train_dataset = dataset["train"]
                self.val_dataset = dataset["test"]
                logger.info("Using provided train and test splits for 'instruct' mode.")


            # Handle Q&A data by applying chat template to format conversations
            if "conversations" in self.train_dataset.column_names:
                logger.info("Applying chat template for 'conversations' field...")
                def formatting_prompts_func(examples):
                    all_convos = examples["conversations"]
                    texts = []
                    for i, convo in enumerate(all_convos):
                        try:
                            # logger.info(f"Processing conversation {i} in batch: {convo}") # Too verbose for large batches
                            text = self.tokenizer.apply_chat_template(
                                convo,
                                tokenize=False,
                                add_generation_prompt=False
                            )
                            texts.append(text)
                        except Exception as e:
                            logger.error(f"Error applying chat template to conversation {i} in batch: {convo}. Error: {e}")
                            # Depending on severity, you might want to raise, skip, or return an empty string
                            texts.append("") # Or raise the exception if you want to stop on error
                    return {"text": texts}

                # Map the formatting function to both train and validation datasets
                self.train_dataset = self.train_dataset.map(formatting_prompts_func, batched=True)                
                if self.val_dataset:
                    self.val_dataset = self.val_dataset.map(formatting_prompts_func, batched=True)
                logger.info(self.train_dataset[0])
                logger.info("Chat template applied successfully.")
            else:
                logger.warning("No 'conversations' column found for 'instruct' mode. Ensure 'text' column is pre-formatted for instruction tuning.")

        elif self.training_mode == "pretrain":
            # For pre-training, typically the dataset contains long-form text.
            # No chat template or separate validation set (unless provided explicitly) is needed.
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset.get("test") # Use .get() to avoid KeyError if 'test' split doesn't exist
            if self.val_dataset:
                logger.info("Using provided 'train' and 'test' splits for 'pretrain' mode.")
            else:
                logger.info("Using only 'train' split for 'pretrain' mode (no validation).")

        else:
            logger.critical(f"Error: Training Mode: '{self.training_mode}' is not supported. Choose 'pretrain' or 'instruct'.")
            raise ValueError(f"Training Mode: '{self.training_mode}' does not exist!")

        logger.info(f"✅ Training data loaded. Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")


    def train(self):
        """
        Initiates the fine-tuning process using the SFTTrainer from TRL.
        """
        logger.info("Starting model training with SFTTrainer...")
        logger.info(f"First training example: {self.train_dataset[0]}")

        # ⚙️ SFTConfig: Configuration for Supervised Fine-Tuning
        # This replaces the need for transformers.TrainingArguments when using SFTTrainer.
        training_args = SFTConfig(
            output_dir=str(self.model_checkpoints), # Directory to save checkpoints and final model
            per_device_train_batch_size=2, # Batch size per GPU
            gradient_accumulation_steps=4, # Accumulate gradients over N steps
            learning_rate=2e-5, # Initial learning rate
            num_train_epochs=30, # Number of training epochs
            warmup_steps=10, # Number of warmup steps for learning rate scheduler
            logging_steps=10, # Log training progress every N steps
            optim="adamw_8bit", # Optimizer to use (8-bit AdamW for memory efficiency)
            weight_decay=0.01, # L2 regularization
            lr_scheduler_type="linear", # Learning rate scheduler type
            seed=self.seed, # Random seed for reproducibility
            report_to="none", # Disable reporting to services like Weights & Biases
            # You can add evaluation arguments here if self.val_dataset is available
            # For example:
            eval_strategy="epoch" if self.val_dataset else "no",
            eval_steps=2 if self.val_dataset else None,
            save_strategy="epoch",
            load_best_model_at_end=True if self.val_dataset else False,
        )
        logger.info(f"SFTConfig initialized: {training_args}")

        # Initialize SFTTrainer with the model, tokenizer, datasets, and configurations
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset, # Pass validation dataset if available
            dataset_text_field="text", # Column in the dataset containing the text to train on
            max_seq_length=self.max_seq_length, # Max sequence length for SFTTrainer's internal tokenization
            dataset_num_proc=os.cpu_count() or 1, # Number of processes for dataset processing
            packing=False, # Whether to pack multiple short sequences into one (false for long documents)
            args=training_args # Pass the SFTConfig object
        )
        logger.info("SFTTrainer initialized. Starting training...")

        # Start the training process
        trainer.train()
        logger.info("✅ Training complete.")

        # Save the fine-tuned LoRA weights
        self.model_lora_weights_location.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self.model.save_pretrained(
            self.model_lora_weights_location,
            #save_method="lora", # Save only the LoRA weights
            #safe_serialization=False # Disable safe serialization for potentially faster saving
        )
        logger.info(f"✅ LoRA weights saved to {self.model_lora_weights_location}")


    def save_gguf(self):
        """
        Exports the LoRA fine-tuned model into GGUF format, suitable for
        inference with Ollama or llama.cpp.
        """
        logger.info(f"Converting and saving model to GGUF format in {self.model_lora_weights_location}...")
        self.model_lora_weights_location.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self.model.save_pretrained_gguf(
            self.model_lora_weights_location,
            self.tokenizer,
            quantization_method="q4_k_m" # Common quantization method for GGUF
        )
        # Also save the tokenizer for GGUF model
        self.tokenizer.save_pretrained(self.model_lora_weights_location)
        logger.info("✅ Model successfully saved in GGUF format.")

# Example Usage (consider wrapping this in a main block if this is a script)
if __name__ == "__main__":
    # Define your training parameters
    TRAINING_DATA_PATH = "data/pirate/pirate_pretrain_strong.jsonl" # Path to your JSONL file or directory
    TRAINING_DATA_PATH = "data/pirate/pirate_instruct.jsonl" # Path to your JSONL file or directory
    MODEL_NAME = "pirate_instruct"
    BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit" # Or "unsloth/mistral-7b-bnb-4bit"
    BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 512 # Adjust based on your data and GPU memory
    TRAINING_MODE = "instruct" # "pretrain" for long documents, "instruct" for Q&A/chat

    try:
        fine_tuner = DocumentFineTune(
            training_data_path=TRAINING_DATA_PATH,
            model_name=MODEL_NAME,
            base_model_path=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            training_mode=TRAINING_MODE
        )

        fine_tuner.train()

        # If you want to convert to GGUF after training
        # fine_tuner.save_gguf()

    except Exception as e:
        logger.exception("An error occurred during fine-tuning:") # Logs the full traceback