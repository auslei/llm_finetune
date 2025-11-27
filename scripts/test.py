import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.llm_finetune.finetune_tool import FineTuner

# Load the ready-made reasoning dataset
#dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")


# It already has 'question', 'complex_reasoning', and 'response' columns.
# Just map it to your format:
def format_to_detective(row):
    return {
        "instruction": "Answer the medical query using clinical reasoning.",
        "input": row['Question'],
        "output": f"Thinking: {row['Complex_CoT']}\nResponse: {row['Response']}"
    }

#dataset = dataset.map(format_to_detective)
#dataset.to_json("data/medical_o1_reasoning.jsonl")




if __name__ == "__main__":
    config_path = os.environ.get("FINETUNE_CONFIG", "config/finetune.example.yaml")
    tuner = FineTuner(config_path)
    tuner._load_training_data("data/medical_o1_reasoning.jsonl.gz")
    tuner.train()
