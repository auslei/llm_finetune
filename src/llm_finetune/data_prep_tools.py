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

# Library logger (configured in CLI)
logger = logging.getLogger(__name__)

# Ollama API client
client = Client()

def load_text_file(path: Union[str, Path], mode: str = "simple") -> str:
    """
    Load text from a .txt or .pdf file.
    If PDF, optionally parse in 'columns' or 'simple' mode.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(str(path)) as pdf:
            if mode == "columns":
                pages: List[str] = []
                for page in pdf.pages:
                    words = page.extract_words(use_text_flow=True)
                    left = [w["text"] for w in words if w["x0"] < page.width / 2]
                    right = [w["text"] for w in words if w["x0"] >= page.width / 2]
                    pages.append("\n".join([" ".join(left), " ".join(right)]))
                return "\n".join(pages)
            return "\n".join(
                page.extract_text(x_tolerance=3, layout=True) or "" for page in pdf.pages
            )
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
    unique: List[str] = []
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
    dedup: bool = True,
) -> List[str]:
    """Split large text into character-based chunks using LangChain splitter."""
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    chunks = splitter.split_text(large_text)
    return deduplicate_chunks(chunks) if dedup else chunks


def add_identity(text: str, entity: str, doc_type: str) -> str:
    """Prepend identity information to a paragraph."""
    return f"[ENTITY: {entity}] [TYPE: {doc_type}]\n\n{text.strip()}"


def save_json_dataset(
    records: Union[List[dict], Dataset], path: Union[str, Path]
) -> dict:
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
    doc_type: str = "document",
) -> dict:
    """Generate instruction-style Q&A pairs using Ollama, with JSON extraction and validation."""
    items: List[dict] = []

    for idx, p in enumerate(chunks):
        snippet = p.strip().replace("\n", " ")[:2000]
        prompt = (
            f"You are a helpful Q&A assistant. Produce up to {max_q} question-answer pairs in strict JSON format "
            f"as an array of objects with 'question' and 'answer' fields.\n\n"
            f"Paragraph:\n{snippet}...\n\nRespond with JSON only, no extra text."
        )
        try:
            logger.info(f"Generating Q&A for chunk {idx}...")
            res = client.generate(model=model, prompt=prompt)
            text = res.get("response", "")
            start = text.find('[')
            end = text.rfind(']')
            raw = text[start : end + 1] if 0 <= start < end else text.strip()
            pairs = json.loads(raw)
            valid: List[dict] = []
            for qa in pairs if isinstance(pairs, list) else []:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    valid.append({
                        'question': str(qa['question']).strip(),
                        'answer': str(qa['answer']).strip(),
                    })
            if valid:
                items.append({
                    'entity': entity,
                    'doc_type': doc_type,
                    'chunk_index': idx,
                    'question_answer_pairs': valid,
                })
                logger.debug(f"Chunk {idx} valid Q&A count: {len(valid)}")
            else:
                logger.warning(f"No valid Q&A extracted for chunk {idx}")
        except Exception as e:
            logger.warning(f"Error generating Q&A for chunk {idx}: {e}")
        time.sleep(delay)

    if not items:
        raise ValueError("No Q&A data was generated.")

    ds = Dataset.from_list(items)
    logger.info(f"Generated {len(items)} instruction records")
    return save_json_dataset(ds, output_file)