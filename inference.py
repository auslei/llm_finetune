from unsloth import FastLanguageModel
import torch
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the paths and model details - these should match your training setup
BASE_MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"  # Ensure this matches your trained base model
MODEL_NAME = "bible_kjv"  # Ensure this matches your trained model's name
LORA_WEIGHTS_LOCATION = Path(f"models/{MODEL_NAME}")  # Where your LoRA weights are saved
MAX_SEQ_LENGTH = 512


def run_chatbot_inference(model_path: str, lora_weights_path: Path):
    logger.info(f"Loading base model: {model_path}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        logger.info("Base model and tokenizer loaded.")
    except Exception as e:
        logger.critical(f"Failed to load base model: {e}")
        sys.exit(1)

    logger.info(f"Loading LoRA adapter from: {lora_weights_path}...")
    try:
        model.load_adapter(str(lora_weights_path))
        logger.info("LoRA adapter loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load LoRA weights. Ensure the path is correct and weights exist: {e}")
        sys.exit(1)

    logger.info("Starting chatbot. Type 'exit' to quit.")
    messages = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            logger.info("Exiting chatbot.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            logger.info("Generating response...")
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            logger.info(f"Generated response tokens: {outputs.shape[1]}")
            messages.append({"role": "assistant", "content": response})
            print(f"Bot: {response}")

        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            continue


if __name__ == "__main__":
    if not LORA_WEIGHTS_LOCATION.exists():
        logger.error(f"Error: LoRA weights directory not found at {LORA_WEIGHTS_LOCATION}. "
                     "Please ensure training was successful and weights were saved.")
        sys.exit(1)

    run_chatbot_inference(BASE_MODEL_PATH, LORA_WEIGHTS_LOCATION)
