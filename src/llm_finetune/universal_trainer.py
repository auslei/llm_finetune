from datasets import Dataset
import json
import time

def generate_universal_dataset(
    chunks: list[str],
    output_file: str,
    mode: str = "instruct", # Options: "instruct" or "pretrain"
    teacher_model: str = "qwen2.5:14b", # Only used if mode="instruct"
    client = None, # Your Ollama client
) -> dict:
    """
    Universal function to generate training data for ANY Fine-Tuning mode.
    
    Args:
        mode="pretrain": Returns raw chunks for Continual Pre-Training (CPT).
        mode="instruct": Uses Teacher LLM to generate Logic/Q&A pairs (SFT).
    """
    data_records = []
    
    print(f"🔄 Starting Data Generation in [{mode.upper()}] mode...")

    # --- MODE A: CONTINUAL PRE-TRAINING (CPT) ---
    if mode == "pretrain":
        # For CPT, the "input" is just the raw text itself.
        # We wrap it in a dict with a 'text' key as required by SFTTrainer.
        for p in chunks:
            if len(p.strip()) > 50: # Filter empty noise
                data_records.append({"text": p.strip()})
        
        print(f"✅ CPT: Prepared {len(data_records)} raw text chunks.")

    # --- MODE B: INSTRUCTION TUNING (SFT) ---
    elif mode == "instruct":
        # Your existing logic: Use Teacher to create synthetic Q&A
        prompt_template = (
            "Read the text. Create a training example where an AI converses with a User to extract info.\n"
            "Output JSON list with keys: 'instruction', 'dialogue', 'thought_process', 'extraction'.\n"
            "Text: {snippet}...\nJSON ONLY."
        )
        
        for idx, p in enumerate(chunks):
            snippet = p.strip().replace("\n", " ")[:2000]
            try:
                # Call Teacher (Ollama)
                res = client.generate(
                    model=teacher_model, 
                    prompt=prompt_template.format(snippet=snippet), 
                    options={"temperature": 0.4}
                )
                
                # Parse JSON (Simplified for brevity)
                text = res.get("response", "")
                start, end = text.find('['), text.rfind(']')
                if start != -1 and end != -1:
                    pairs = json.loads(text[start:end+1])
                    for item in pairs:
                        # Format for Unsloth/Alpaca
                        data_records.append({
                            "instruction": item['instruction'],
                            "input": item['dialogue'],
                            "output": f"Thinking: {item['thought_process']}\nResponse: {json.dumps(item['extraction'])}"
                        })
            except Exception as e:
                print(f"⚠️ Error chunk {idx}: {e}")

        print(f"✅ Instruct: Generated {len(data_records)} synthetic samples.")

    # --- SAVE ---
    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data_records:
            f.write(json.dumps(entry) + "\n")
            
    return {"path": output_file, "count": len(data_records)}



class UniversalFineTuner:
    def __init__(
        self, 
        training_data_path: str, 
        model_name: str, 
        mode: str = "instruct"  # "instruct" or "pretrain"
    ):
        self.mode = mode
        self.training_data_path = training_data_path
        self.base_model_name = "unsloth/Qwen2.5-1.5B-Instruct"
        
        # 1. Dynamic Target Modules
        # CPT needs to train embeddings to learn new domain terms. SFT does not.
        if self.mode == "pretrain":
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj", 
                                   "embed_tokens", "lm_head"] # <--- Added for CPT
        else:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]

        # 2. Load Model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.base_model_name,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )

        # 3. Add LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 64 if mode == "instruct" else 128, # Higher rank for CPT (more knowledge to absorb)
            target_modules = self.target_modules,
            lora_alpha = 32,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

    def train(self):
        # Load Data
        dataset = load_dataset("json", data_files={"train": self.training_data_path}, split="train")

        # 4. Configure Trainer based on Mode
        if self.mode == "instruct":
            # INSTRUCT MODE: Uses your formatting function & NO packing
            packing = False
            dataset_text_field = None
            formatting_func = self._format_instruct_prompts # Defined below
            learning_rate = 2e-4
        else:
            # PRETRAIN MODE: Raw text packing & simple text field
            packing = True
            dataset_text_field = "text" # The column we created in generate_universal_dataset
            formatting_func = None
            learning_rate = 5e-5 # Lower LR for CPT is usually safer

        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            dataset_text_field = dataset_text_field,
            formatting_func = formatting_func,
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = packing, # <--- The Magic Switch
            args = SFTConfig(
                output_dir = "outputs",
                per_device_train_batch_size = 8 if self.mode=="instruct" else 4,
                gradient_accumulation_steps = 2,
                num_train_epochs = 3,
                learning_rate = learning_rate,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 10,
                optim = "adamw_8bit",
                seed = 3407,
            ),
        )
        
        trainer.train()
        self.model.save_pretrained(f"models/{self.mode}_model")
        self.tokenizer.save_pretrained(f"models/{self.mode}_model")

    def _format_instruct_prompts(self, examples):
        # Your standard Alpaca/Chat formatter
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = f"Instruction:\n{instr}\n\nInput:\n{inp}\n\nResponse:\n{out}<|endoftext|>"
            texts.append(text)
        return texts