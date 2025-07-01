import sys
import yaml
from pathlib import Path
from tools.data_prep import make_pretrain_data,

 # generate coniunal pretraining data
def generate(source, output_path, mode, entity, doc_type, text_mode, inject, chunk_size, chunk_overlap):

    cfig = {
        "source": source,
        "output_path": output_path,        
        "entity": entity,
        "doc_type": doc_type,
        "text_mode": text_mode,
        "inject": inject,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    if mode in ["pretrain", "both"]:
        print("Generating pretrain data...")
        make_pretrain_data(
            **cfig
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
    source = "docs/zarnian_lore.txt"
    output_path = "data/zarnian_lore"
    mode = "pretrain"
    entity = "zarnian lore"
    doc_type = "zarnian_lore.txt"
    text_mode = "simple"
    inject = True
    chunk_size = 1500
    chunk_overlap = 200
    
    generate(source, output_path, mode, entity, doc_type, text_mode, inject, chunk_size, chunk_overlap)