from datasets import load_dataset
import json
from pathlib import Path
from llm_finetune.finetune_tool import DocumentFineTune

# Step 1: Download dataset from Hugging Face
dataset = load_dataset("AISPIN/shiji-70liezhuan")

# Step 2: Convert to 'conversations' format for Unsloth-style instruction tuning
output_dir = Path("data/shiji")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "shiji_instruct.jsonl"

with output_file.open("w", encoding="utf-8") as f:
    for example in dataset["train"]:
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        output_text = example.get("output", "").strip()

        # Format as conversations
        conversations = [
            {"role": "user", "content": f"{instruction}\n{input_text}".strip()},
            {"role": "assistant", "content": output_text}
        ]
        json.dump({"conversations": conversations}, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Dataset saved to {output_file.resolve()}")

# Step 3: Run fine-tuning using your library
TRAINING_DATA_PATH = str(output_file)
MODEL_NAME = "llama3.2-chinese-shiji"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
TRAINING_MODE = "instruct"
NUM_OF_EPOCHS = 3

# Instantiate and run training
trainer = DocumentFineTune(
    training_data_path=TRAINING_DATA_PATH,
    model_name=MODEL_NAME,
    base_model_path=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    training_mode=TRAINING_MODE,
)

trainer.train(num_train_epochs=NUM_OF_EPOCHS)