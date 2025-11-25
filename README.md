# 🧠 LLM Finetune Framework (Unsloth)

This repository provides a complete pipeline for fine-tuning and continually pretraining local LLMs (e.g., Mistral or LLaMA 3.2) using [Unsloth](https://github.com/unslothai/unsloth). It supports both instruction-style fine-tuning and continual pretraining using synthetic or domain-specific data.

---

## 🚀 Features

- ⚡️ Fast finetuning and inference using Unsloth's patched models
- 🧾 Continual pretraining from scratch using `.txt` or `.jsonl`
- 🤖 Instruction fine-tuning with multi-turn chat format
- 🗂 Modular pipeline (CLI): `scripts/prepare_data.py`, `scripts/finetune.py`, `scripts/chat_interface.py`, `scripts/finetune_shiji.py`
- 🖥 GUI interface via Streamlit: `app.py`
- 🧪 Test your custom model via terminal chatbot interface
- ✅ Sample finetunes: Pirate Instruct, Zarnian Lore, CV domain expertise


## 📂 Project Structure

```
. (root)
├── app.py                     # Streamlit GUI for data-prep, training & chat
├── scripts/                  # CLI entrypoints
│   ├── prepare_data.py       # Chunk & prepare pretrain/instruct data
│   ├── finetune.py           # Supervised fine-tuning via SFTTrainer
│   ├── chat_interface.py     # Terminal-based chat interface
│   └── finetune_shiji.py     # Shiji dataset example pipeline
├── src/                      # Importable Python package
│   └── llm_finetune/
│       ├── __init__.py
│       ├── data_prep_tools.py
│       └── finetune_tool.py
├── notebooks/                # Jupyter notebooks
├── data/                     # Prepared JSONL datasets
├── docs/                     # Raw docs (.txt/.pdf) and guides
├── models/                   # Saved LoRA weights
├── outputs/                  # Checkpoints, logs (git-ignored)
├── llama.cpp/                # llama.cpp artifacts (git-ignored)
├── mac/                      # macOS build artifacts (git-ignored)
└── unsloth_compiled_cache/   # Cache directory (git-ignored)
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
conda env create -f environment_core.yml
conda activate unsloth3.11
# or with pip: pip install -e .
```

### 2. Run the Streamlit GUI
```bash
export PYTHONPATH=src:$PYTHONPATH
streamlit run app.py
```

### 3. Use the CLI
```bash
# 3.1 Prepare data
python scripts/prepare_data.py \
  --input_file docs/zarnian_lore.txt \
  --output_dir data/Zarnian \
  --mode pretrain

# 3.2 Fine-tune model
python scripts/finetune.py \
  --training_data_path data/Zarnian/pretrain.jsonl \
  --model_name Zarnian \
  --base_model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --mode pretrain \
  --epochs 3

# 3.3 Chat in terminal
python scripts/chat_interface.py \
  --model_dir models/Zarnian

# 3.4 Shiji example
python scripts/finetune_shiji.py
```

## 📄 License
This project is under the MIT License.

## 🙏 Credits
- Unsloth
- HuggingFace Transformers
- Anthony Sun