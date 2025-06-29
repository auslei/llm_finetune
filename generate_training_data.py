import sys
import yaml
from pathlib import Path
from tools.data_prep import make_pretrain_data, make_instruct_data


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str):
    config = load_config(config_path)
    mode = config.get("mode", "pretrain")

    data_cfg = config["data"]
    source = data_cfg["input_pdf"]
    output = data_cfg["data_dir"]
    output_pretrain = Path(output) / "pretrain.jsonl"
    output_instruct = Path(output) / "instruct.jsonl"

    # generate coniunal pretraining data
    if mode in ["pretrain", "both"]:
        print("Generating pretrain data...")
        make_pretrain_data(
            source=source,
            output_file=str(output_pretrain),
            entity=data_cfg["entity"],
            doc_type=data_cfg["doc_type"],
            min_len=data_cfg.get("min_length", 40),
            mode=data_cfg.get("text_mode", "simple"),
            inject=True,
        )

    # generate instruct data
    if mode in ["instruct", "both"]:
        print("Generating instruct data...")
        instruct_cfg = config.get("instruct", {})
        make_instruct_data(
            source=source,
            output_file=output_instruct,
            entity=data_cfg["entity"],
            doc_type=data_cfg["doc_type"],
            min_len=data_cfg.get("min_length", 40),
            mode=data_cfg.get("text_mode", "columns"),
            model=instruct_cfg.get("ollama_model", "mistral"),
            max_q=instruct_cfg.get("max_questions", 3),
            delay=instruct_cfg.get("delay", 0.5))

    if mode not in ["pretrain", "instruct", "both"]:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_pipeline.py config/your_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
