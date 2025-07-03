import json
import time
import logging
from pathlib import Path
from typing import Union, Optional, List
from datasets import Dataset
from ollama import Client
import pdfplumber
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Client()  # Ollama API client

def load_text_file(path: Union[str, Path], mode: str = "simple") -> str:
    """
    Load text from a .txt or .pdf file.
    If PDF, optionally parse in 'columns' or 'simple' mode.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(str(path)) as pdf:
            if mode == "columns":
                pages = []
                for page in pdf.pages:
                    words = page.extract_words(use_text_flow=True)
                    left = [w["text"] for w in words if w["x0"] < page.width / 2]
                    right = [w["text"] for w in words if w["x0"] >= page.width / 2]
                    pages.append("\n".join([" ".join(left), " ".join(right)]))
                return "\n".join(pages)
            return "\n".join([page.extract_text(x_tolerance=3, layout=True) or "" for page in pdf.pages])
    else:
        return path.read_text(encoding="utf-8")



def chunk_text_by_tokens(
    large_text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "gpt2",
) -> List[str]:
    """Chunk text by tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(large_text)

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - chunk_overlap
    return chunks


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Deduplicate chunks while preserving order."""
    seen = set()
    unique = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def chunk_text_by_characters(
    large_text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: Optional[List[str]] = None,
    dedup: bool = True
) -> List[str]:

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    """Split large text into character-based chunks using LangChain splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators
    )
    chunks = splitter.split_text(large_text)
    return deduplicate_chunks(chunks) if dedup else chunks


def add_identity(text: str, entity: str, doc_type: str) -> str:
    """Prepend identity information to a paragraph."""
    return f"This paragraph is from the {doc_type} of {entity}.\n\n{text.strip()}"


def save_json_dataset(records: Union[List[dict], Dataset], path: Union[str, Path]) -> dict:
    """
    Save a list of dicts or HuggingFace Dataset to a JSONL file.
    Returns a dict with the saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(records, Dataset):
        records = records.to_list()

    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {path}")
    return {"path": str(path)}

def read_and_chunk_document(
    source: Union[str, Path],
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "simple",
    chunk_method: str = "characters",
    dedup: bool = True,
    encoding_name: str = "gpt2",
) -> List[str]:
    """
    Read and chunk a document from file path.
    chunk_method: 'characters' or 'tokens'
    """
    text = load_text_file(source, mode)
    logger.info(f"{source} loaded with {len(text)} characters")
    if chunk_method == "tokens":
        chunks = chunk_text_by_tokens(text, chunk_size, chunk_overlap, encoding_name)
        if dedup:
            chunks = deduplicate_chunks(chunks)
    else:
        chunks = chunk_text_by_characters(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dedup=dedup,
        )
    logger.info(f"Split text into {len(chunks)} chunks using {chunk_method}")
    return chunks


def make_pretrain_data(
    chunks: List[str],
    output_file: Union[str, Path],
    entity: str = "Unknown",
    doc_type: str = "document",
    inject: bool = True,
) -> dict:
    """Prepare text chunks for causal-style pretraining."""
    items = [{"text": add_identity(p, entity, doc_type) if inject else p} for p in chunks]
    if not items:
        raise ValueError("No valid paragraphs found.")
    
    ds = Dataset.from_list(items)
    return save_json_dataset(ds, output_file)


def make_instruct_data(
    chunks: List[str],
    output_file: Union[str, Path],
    model: str = "mistral",
    max_q: int = 3,
    delay: float = 0.5,
    entity: str = "Unknown",
    doc_type: str = "document"
) -> dict:
    """Generate instruction-style Q&A pairs using Ollama."""
    items = []

    for p in chunks:
        prompt = (
            f"Given the following paragraph from {entity}'s {doc_type}, generate up to {max_q} realistic questions and answers "
            f"in JSON format as a list of conversation pairs.\n\nParagraph:\n{p}\n\nFormat:\n"
            "[\n  {\"role\": \"user\", \"content\": \"...\"},\n  {\"role\": \"assistant\", \"content\": \"...\"},\n  ...\n]"
        )
        try:
            logger.info("Sending prompt to model...")
            res = client.generate(model=model, prompt=prompt)
            pairs = json.loads(res["response"].strip())
            if isinstance(pairs, list):
                items.append({"conversations": pairs})
                logger.debug(f"Generated pairs: {pairs}")
        except Exception as e:
            logger.warning(f"Failed on paragraph: {e}")
        time.sleep(delay)

    if not items:
        raise ValueError("No Q&A data was generated.")

    ds = Dataset.from_list(items)
    logger.info(f"Generated {len(items)} instruction records")
    return save_json_dataset(ds, output_file)


if __name__ == "__main__":
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
    parser.set_defaults(inject_identity=True)
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
    parser.set_defaults(dedup=True)
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
    data_file = Path(args.input_file)

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
