# 🧠 LLM Finetune Framework (Unsloth)

This repository provides a complete pipeline for fine-tuning and continually pretraining local LLMs (e.g., Mistral or LLaMA 3.2) using [Unsloth](https://github.com/unslothai/unsloth). It supports both instruction-style fine-tuning and continual pretraining using synthetic or domain-specific data.

---

## 🚀 Features

- ⚡️ Fast finetuning and inference using Unsloth's patched models
- 🧾 Continual pretraining from scratch using `.txt` or `.jsonl`
- 🤖 Instruction fine-tuning with multi-turn chat format
- 💾 Memory-optimized dataset loading with streaming mode for large datasets
- 📦 Batched tokenization to prevent memory spikes
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

## 💾 Memory Optimization

For large datasets that don't fit in memory, enable streaming mode to load data incrementally:

### Using Streaming Mode (Python API)
```python
from llm_finetune.finetune_tool import FineTuner

tuner = FineTuner(
    training_data_path="data/large_dataset.jsonl",
    model_name="MyModel",
    base_model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    mode="pretrain",
    streaming=True  # Enable streaming mode
)
tuner.train()
```

### Using Configuration File
Create a YAML config file with streaming enabled:
```yaml
training_data_path: "data/large_dataset.jsonl"
mode: "pretrain"
base_model: "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
model_name: "MyModel"
streaming: true
dataset_batch_size: 1000  # Optional: adjust based on available memory (default: 1000)
num_train_epochs: 3
```

Then run:
```bash
python -m llm_finetune.cli --config config.yaml
```

### Memory Optimization Features
- **Streaming Mode**: Loads data incrementally instead of loading entire dataset into memory
- **Batched Tokenization**: Processes data in configurable batches (default: 1000 samples) to prevent memory spikes
- **Column Removal**: Automatically removes unused columns during preprocessing to reduce memory footprint
- **Efficient Validation**: Uses minimal sampling for dataset validation
- **Configurable Batch Size**: Adjust `dataset_batch_size` parameter to optimize for your available memory

### Important Notes
- When using streaming mode, automatic train/test splitting is disabled. Provide separate train and test files if validation is needed.
- Streaming datasets don't report total size until consumed.
- For datasets under 1GB, traditional loading may be faster.

## 📄 License
This project is under the MIT License.

## 🙏 Credits
- Unsloth
- HuggingFace Transformers
- Anthony Sun