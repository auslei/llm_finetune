from pathlib import Path
from unsloth import FastLanguageModel
import torch
import os
import gc

os.environ['PYTORCH_CUDA_EXPANDABLE_SEGMENTS'] = "1"


def load_combined_model(model_name: str, max_seq_length: int = 2048):
    lora_path = Path(f"models/{model_name}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at: {lora_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True
    )

    return model, tokenizer

model_name = "anthony-pretrain"
lora_path = Path(f"models/{model_name}")
model, tokenizer = load_combined_model(model_name)

# Explicitly clear cache and collect garbage
torch.cuda.empty_cache()
gc.collect()
torch.cuda.memory_summary(device=None, abbreviated=False)

# Merge LoRA weights
model = model.merge_and_unload()  # This merges LoRA and removes the LoRA weights
torch.cuda.empty_cache()
gc.collect()
torch.cuda.memory_summary(device=None, abbreviated=False)

try:
    gguf_out_path = Path(f"gguf/{model_name}")
    gguf_out_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        str(gguf_out_path / "model.gguf"),
        tokenizer,
        quantization_method="q4_k_m",
        maximum_memory_usage=0.8  # Reduce slightly to be safer
    )
    print("GGUF conversion successful!")

except RuntimeError as e:
    print(f"Error during GGUF conversion: {e}")
    print("Consider reducing maximum_memory_usage further or using a machine with more VRAM.")

finally:
    # Clean up after the conversion attempt
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.memory_summary(device=None, abbreviated=False)