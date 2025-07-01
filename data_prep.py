import re
import json
import time
from pathlib import Path
from typing import Union, Optional, List
from datasets import Dataset
from ollama import Client
import pdfplumber
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


client = Client()  # Ollama API client

def read_pdf(path: Union[str, Path], mode: str = "simple") -> str:
    """
    Extract text from PDF, with optional column-based parsing.
    """
    with pdfplumber.open(str(path)) as pdf:
        if mode == "columns":
            pages = []
            for page in pdf.pages:
                words = page.extract_words(use_text_flow=True)
                left = [w["text"] for w in words if w["x0"] < page.width / 2]
                right = [w["text"] for w in words if w["x0"] >= page.width / 2]
                pages.append("\n".join([" ".join(left), " ".join(right)]))
            return "\n".join(pages)
        else:
            return "\n".join([page.extract_text(x_tolerance=3, layout=True) or "" for page in pdf.pages])

def read_text(path: Union[str, Path], mode: str = "simple") -> str:
    """
    Load raw text from file (PDF or text).
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return read_pdf(path, mode=mode)
    return path.read_text(encoding="utf-8")


def chunk_text_by_tokens(large_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunk text by tokens with specified chunk size and overlap using tiktoken.

    :param text: The input text to chunk.
    :param chunk_size: Maximum number of tokens per chunk.
    :param chunk_overlap: Number of overlapping tokens between chunks.
    :return: List of text chunks.
    """
    # Load GPT-3 tokenizer (uses GPT-2 tokenizer internally, compatible with GPT-3/4 and other models)
    enc = tiktoken.get_encoding("gpt2")

    # Tokenize the input text into token IDs
    tokens = enc.encode(large_text)

    # Create chunks with overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        decoded_chunk = enc.decode(chunk)

        # Add the decoded chunk to the list
        chunks.append(decoded_chunk)

        # Move the start pointer forward by (chunk_size - chunk_overlap) to create overlap
        start += chunk_size - chunk_overlap
    return chunks


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Deduplicate chunks but preserve orders
    """
    seen = set()
    unique = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def chunk_text_by_charcters(large_text: str,  chunk_size: int = 512, chunk_overlap: int =64, 
                            separators: list[str] = ["\n\n", "\n", " ", ""], dedup = True) -> list[str]:
        """
        Split large text into smaller chunks. Uses configuration values by default, 
        but the user can override chunk_size, chunk_overlap, and separators.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )

        chunks = text_splitter.split_text(large_text)

        if dedup:
            chunks = deduplicate_chunks(chunks)
        return chunks

def add_identity(text: str, entity: str, doc_type: str) -> str:
    """
    Prepend identity info to a paragraph.
    """
    return f"This paragraph is from the {doc_type} of {entity}.\n\n{text.strip()}"

def save_jsonl(records: List[dict], path: Path) -> None:
    """
    Save list of dicts to JSONL file.
    """
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_dataset(ds: Dataset, output_file: Union[str, Path]) -> dict:
    """
    Save dataset as train/val JSONL files or single file.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    save_jsonl(ds, output_file)
    saved["pretrain"] = str(output_file)
    print(f"Training file saved at: {output_file}")

    return saved

def read_and_chunk_document(source: Union[str, Path], mode: str = "simple", chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """
    Read and chunk a document.
    """
    text = read_text(source, mode)
    return chunk_text_by_charcters(large_text = text, chunk_size = chunk_size, chunk_overlap =  chunk_overlap)

def make_pretrain_data(
    chunks: list[str],
    output_file: Union[str, Path],
    entity: str = "Unknown",
    doc_type: str = "document",
    inject: bool = True,
    ) -> dict:
    """
    Prepare chunked text for causal-style fine-tuning using a uniform overlapping split.
    """    
    items = []
    for p in chunks:       
        if inject:
            p = add_identity(p, entity, doc_type)
        items.append({"text": p})

    if not items:
        raise ValueError("No valid paragraphs found.")

    ds = Dataset.from_list(items)
    return save_dataset(ds, output_file)

def make_instruct_data(
    chunks: list[str],
    output_file: Union[str, Path],
    model: str = "mistral",
    max_q: int = 3,
    delay: float = 0.5
) -> dict:
    """
    Generate instruction-style Q&A data from document using Ollama.
    """    

    results = []
    for p in chunks:
        prompt = (
            f"Given the following paragraph from {entity}'s {doc_type}, generate up to {max_q} realistic questions and answers "
            f"in JSON format as a list of conversation pairs.\n\nParagraph:\n{p}\n\nFormat:\n"
            "[\n  {\"role\": \"user\", \"content\": \"...\"},\n  {\"role\": \"assistant\", \"content\": \"...\"},\n  ...\n]"
        )
        try:
            print(f"Generating question with prompt:\n{prompt}")
            res = client.generate(model=model, prompt=prompt)
            pairs = json.loads(res["response"].strip())
            if isinstance(pairs, list):
                results.append({"conversations": pairs})
                print(pairs)
        except Exception as e:
            print(f"[WARN] Failed on paragraph: {e}")
        time.sleep(delay)

    if not results:
        raise ValueError("No Q&A data was generated.")

    ds = Dataset.from_list(results)
    return save_dataset(ds, output_file)


