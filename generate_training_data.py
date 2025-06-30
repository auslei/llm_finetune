import sys
import yaml
from pathlib import Path
from tools.data_prep import make_pretrain_data, make_instruct_data


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str):
    config = load_config(config_path)
    
    data_cfg = config["data"]
    mode   = data_cfg.get("mode", "pretrain")
    source = data_cfg["input_pdf"]
    output = data_cfg["data_dir"]
    output_pretrain = Path(output) / "pretrain.jsonl"
    output_instruct = Path(output) / "instruct.jsonl"


    chunk_size = data_cfg.get("chunk_size", 1500)
    chunk_overlap = data_cfg.get("chunk_overlap", 200)
    inject = bool(data_cfg["inject"])

    # generate coniunal pretraining data
    if mode in ["pretrain", "both"]:
        print("Generating pretrain data...")
        make_pretrain_data(
            source=source,
            output_file=str(output_pretrain),
            entity=data_cfg["entity"],
            doc_type=data_cfg["doc_type"],            
            mode=data_cfg.get("text_mode", "simple"),
            inject=True,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
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
            mode=data_cfg.get("text_mode", "columns"),
            model=instruct_cfg.get("ollama_model", "mistral"),
            max_q=instruct_cfg.get("max_questions", 3),
            delay=instruct_cfg.get("delay", 0.5),
            inject = inject)

    if mode not in ["pretrain", "instruct", "both"]:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_pipeline.py config/your_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
