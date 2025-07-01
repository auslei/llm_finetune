from unsloth import FastLanguageModel
import torch
from pathlib import Path
import logging
import sys


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_chatbot_inference(lora_weights_path: Path, max_seq_length: int = 512):
    logger.info(f"Loading LoRA model from: {lora_weights_path}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_weights_path),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        logger.info(f"Model and tokenizer loaded successfully from: {lora_weights_path}")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        sys.exit(1)

    FastLanguageModel.for_inference(model)

    logger.info("Chatbot ready. Type 'exit' to quit.")
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
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=pad_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            logger.info(f"Generated {outputs.shape[1]} tokens.")
            messages.append({"role": "assistant", "content": response})
            print(f"Bot: {response}")

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            continue

if __name__ == "__main__":
    MODEL_NAME = "zarnian"
    MAX_SEQ_LENGTH = 512
    LORA_WEIGHTS_LOCATION = Path(f"models/{MODEL_NAME}")

    if not LORA_WEIGHTS_LOCATION.exists():
        logger.error(f"LoRA weights not found at {LORA_WEIGHTS_LOCATION}.")
        sys.exit(1)

    run_chatbot_inference(LORA_WEIGHTS_LOCATION, MAX_SEQ_LENGTH)
