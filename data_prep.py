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



def chunk_text_by_tokens(large_text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunk text by tokens using tiktoken."""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(large_text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(enc.decode(chunk))
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
    #return f"This paragraph is from the {doc_type} of {entity}.\n\n{text.strip()}"
    return f"[ENTITY: {entity}] [TYPE: {doc_type}]\n\n{text.strip()}"


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
    chunk_size,
    chunk_overlap,
    mode: str = "simple",
) -> List[str]:
    """Read and chunk a document from file path."""
    text = load_text_file(source, mode)
    logger.info(f"{source} loaded with {len(text)} characters")
    return chunk_text_by_characters(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


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
    
    return save_json_dataset(Dataset.from_list(items), output_file)


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
    return save_json_dataset(Dataset.from_list(items), output_file)


if __name__ == "__main__":

    INPUT_DOC = "docs/zarnian_lore.txt"
    MODEL = "Zarnian"
    ENTITY = "Oracle Veyra"
    DOC_TYPE = "Zarnian Lore"
    OLLAMA_MODEL = "mistral"

    DATA_PATH = Path(f"data/{MODEL}")
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    TEXT_CHUNK_SIZE = 640
    TEXT_CHUNK_OVERLAP = 100

    logger.info(f"Reading and chunking document: {INPUT_DOC}")
    chunks = read_and_chunk_document(
        INPUT_DOC,
        mode="simple",
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP
    )

    logger.info(f"Generated {len(chunks)} text chunks")

    if True:
        logger.info("Generating pretrain dataset...")
        make_pretrain_data(
            chunks,
            output_file=DATA_PATH / "pretrain.jsonl",
            entity=ENTITY,
            doc_type=DOC_TYPE
        )

    if False:
        logger.info("Generating instruct dataset...")
        make_instruct_data(
            chunks,
            output_file=DATA_PATH / "instruct.jsonl",
            entity=ENTITY,
            doc_type=DOC_TYPE,
            model=OLLAMA_MODEL
        )
