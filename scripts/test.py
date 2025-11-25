import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.llm_finetune.finetune_tool import UniversalFineTuner

# Load the ready-made reasoning dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")


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


uft = UniversalFineTuner(
    training_data_path="data/medical_o1_reasoning.jsonl",
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    mode="instruct")

uft.train()
    
    
