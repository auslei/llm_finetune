import sys
import yaml
from pathlib import Path
from tools.unsloth_finetuning import DocumentFineTune

import os
os.environ["UNSLOTH_USE_CUT_CROSSENTROPY"] = "0"

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str):
    config = load_config(config_path)
    
    train_cfg = config["train"]    
    train_mode = train_cfg.get("mode", "pretrain")
    data_path = train_cfg["data_dir"]
    
    if train_mode == "pretrain":
        data_path = Path(data_path) / "pretrain.jsonl"
    elif train_mode == "instruct":
        data_path = Path(data_path) / "instruct.jsonl"
    else:
        raise Exception(f"{train_mode} training not supported")

    training_args = train_cfg['training_args']

    trainer = DocumentFineTune(
        training_data_path = str(data_path),
        model_name = train_cfg["model_name"],
        base_model_path = train_cfg.get("base_model_path"),
        max_seq_length = train_cfg.get("max_seq_length", 2048),
        seed=train_cfg.get("seed", 42),
    )
    trainer.train()
    

    # --- Step 3: Export (Optional) ---
    if config.get("export", {}).get("convert_to_gguf", False):
        trainer.save_gguf()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_pipeline.py config/your_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
