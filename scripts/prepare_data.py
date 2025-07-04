#!/usr/bin/env python3
"""
CLI entrypoint for preparing LLM training data (pretraining or instruction tuning).
This script imports reusable routines from data_prep_tools.py.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from llm_finetune.data_prep_tools import (
    read_and_chunk_document,
    save_json_dataset,
    make_pretrain_data,
    make_instruct_data,
)

# Configure logging for CLI
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training data for LLM pretraining and instruction tuning."
    )
    parser.add_argument(
        "--input_file", "-i", required=True, type=Path,
        help="Path to input file (.txt, .pdf, .json, .jsonl)"
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, type=Path,
        help="Directory to save output jsonl files"
    )
    parser.add_argument(
        "--mode", "-m", required=True, choices=["pretrain", "instruct"],
        help="Operation mode: 'pretrain' or 'instruct'"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1024,
        help="Chunk size (characters or tokens depending on chunk_method)"
    )
    parser.add_argument(
        "--overlap", type=int, default=200,
        help="Chunk overlap (characters or tokens depending on chunk_method)"
    )
    parser.add_argument(
        "--pdf_mode", choices=["simple", "columns"], default="simple",
        help="PDF parsing mode when reading PDF files"
    )
    parser.add_argument(
        "--entity", type=str, default="Unknown",
        help="Entity name to inject into data"
    )
    parser.add_argument(
        "--doc_type", type=str, default="document",
        help="Document type to inject into data"
    )
    parser.add_argument(
        "--inject_identity", dest="inject_identity", action="store_true", default=True,
        help="Inject identity information into pretraining chunks"
    )
    parser.add_argument(
        "--no_inject", dest="inject_identity", action="store_false",
        help="Do not inject identity information into pretraining chunks"
    )
    parser.add_argument(
        "--chunk_method", choices=["characters", "tokens"], default="characters",
        help="Chunking method: 'characters' or 'tokens'"
    )
    parser.add_argument(
        "--dedup", dest="dedup", action="store_true", default=True,
        help="Deduplicate overlapping chunks"
    )
    parser.add_argument(
        "--no_dedup", dest="dedup", action="store_false",
        help="Do not deduplicate overlapping chunks"
    )
    parser.add_argument(
        "--encoding", type=str, default="gpt2",
        help="Tiktoken encoding name for token-based chunking"
    )
    parser.add_argument(
        "--model", type=str, default="mistral",
        help="Ollama model name for instruct generation"
    )
    parser.add_argument(
        "--max_q", type=int, default=3,
        help="Maximum number of Q&A pairs to generate per chunk"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay (seconds) between Ollama API calls"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_file = args.input_file

    # Pretraining mode: split raw text/pdfs or reformat existing json/jsonl
    if args.mode == "pretrain":
        if data_file.suffix.lower() in [".txt", ".pdf"]:
            chunks = read_and_chunk_document(
                data_file,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
                mode=args.pdf_mode,
                chunk_method=args.chunk_method,
                dedup=args.dedup,
                encoding_name=args.encoding,
            )
            logger.info(f"Generated {len(chunks)} text chunks for pretraining")
            out_file = args.output_dir / "pretrain.jsonl"
            make_pretrain_data(
                chunks,
                output_file=out_file,
                entity=args.entity,
                doc_type=args.doc_type,
                inject=args.inject_identity,
            )
        elif data_file.suffix.lower() in [".json", ".jsonl"]:
            records = []
            if data_file.suffix.lower() == ".json":
                with data_file.open("r", encoding="utf-8") as f:
                    records = json.load(f)
            else:
                with data_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        records.append(json.loads(line))
            if not records:
                logger.error(f"No records found in {data_file}")
                sys.exit(1)
            out_file = args.output_dir / "pretrain.jsonl"
            save_json_dataset(records, out_file)

    # Instruction mode: generate or reformat instruction data
    elif args.mode == "instruct":
        if data_file.suffix.lower() in [".txt", ".pdf"]:
            chunks = read_and_chunk_document(
                data_file,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
                mode=args.pdf_mode,
                chunk_method=args.chunk_method,
                dedup=args.dedup,
                encoding_name=args.encoding,
            )
            logger.info(f"Generated {len(chunks)} text chunks for instruct tuning")
            out_file = args.output_dir / "instruct.jsonl"
            make_instruct_data(
                chunks,
                output_file=out_file,
                model=args.model,
                max_q=args.max_q,
                delay=args.delay,
                entity=args.entity,
                doc_type=args.doc_type,
            )
        elif data_file.suffix.lower() in [".json", ".jsonl"]:
            records = []
            if data_file.suffix.lower() == ".json":
                with data_file.open("r", encoding="utf-8") as f:
                    records = json.load(f)
            else:
                with data_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        records.append(json.loads(line))
            if not records:
                logger.error(f"No records found in {data_file}")
                sys.exit(1)
            out_file = args.output_dir / "instruct.jsonl"
            save_json_dataset(records, out_file)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()