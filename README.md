# 🧠 LLM Finetune Framework (Unsloth)

This repository provides a complete pipeline for fine-tuning and continually pretraining local LLMs (e.g., Mistral or LLaMA 3.2) using [Unsloth](https://github.com/unslothai/unsloth). It supports both instruction-style fine-tuning and continual pretraining using synthetic or domain-specific data.

---

## 🚀 Features

- ⚡️ Fast finetuning and inference using Unsloth's patched models
- 🧾 Continual pretraining from scratch using `.txt` or `.jsonl`
- 🤖 Instruction fine-tuning with multi-turn chat format
- 🗂 Modular pipeline: `data_prep.py`, `train.py`, `inference.py`
- 🧪 Test your custom model via terminal chatbot interface
- ✅ Sample finetunes: Pirate Instruct, Zarnian Lore, CV domain expertise

---

## 🗂 Project Structure
llm_finetune/
├── data/ # Preprocessed training datasets (.jsonl)
├── docs/ # Raw source texts (.txt or PDFs)
├── models/ # Saved LoRA weights
├── scripts/
│ ├── data_prep.py # Chunk, clean, and prepare training data
│ ├── train.py # Run Unsloth-based pretraining or finetuning
│ └── inference.py # Simple chatbot for inference
├── requirements.txt
└── README.md

## 📄 1. Preparing Your Data

**Supported formats:**

- `.txt` for continual pretraining
- `.jsonl` with chat format for instruction finetuning

### Continual Pretraining

```bash
python data_prep.py \
  --input_file docs/zarnian_lore.txt \
  --output_dir data/Zarnian/ \
  --chunk_size 1500 \
  --overlap 200 \
  --mode pretrain
```

### Instruction Finetuning

```bash
python data_prep.py \
  --input_file docs/pirate_chat.json \
  --output_dir data/Pirate/ \
  --mode instruct
```

## 🏋️‍♂️ 2. Training

```
python train.py \
  --model_name unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --train_dataset data/Pirate/pretrain.jsonl \
  --output_dir models/pirate_instruct \
  --epochs 10 \
  --max_seq_length 512
```

## 💬 3. Inference

```bash
python inference.py --model_dir models/pirate_instruct

You: What is the meaning of life?
Bot: Ah matey, 'tis to chase dreams, swig rum, and leave no treasure unclaimed!
```


## 📚 Examples

- Zarnian Lore: A fictional world injected via continual pretraining
- Pirate Instruct: Instruction-style chat finetune with pirate roleplay
- Anthony CV: Finetune on domain expertise using extracted structured experience from a PDF

## 🛠 Requirements
- Python 3.11+
- CUDA-enabled GPU recommended (8GB+)
- transformers, unsloth, datasets, torch, pdfplumber
- [TODO] Mac MLX-ML training

## 📎 Notes

- Uses Unsloth for native 4-bit loading and patched fast training
- Avoid large chunk_size if dataset is small
- Instruction-style fine-tune works best with chat-format .jsonl

## 📄 License
This project is under the MIT License.

## 🙏 Credits
- Unsloth
- HuggingFace Transformers
- Anthony Sun
- ChatGPT for this helpful README.md