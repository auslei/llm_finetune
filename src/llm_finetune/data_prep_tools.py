import json
import time
import logging
import os
from pathlib import Path
from typing import Union, Optional, List, Dict

from datasets import Dataset
from ollama import Client
import pdfplumber
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

import base64
from io import BytesIO
from pdf2image import convert_from_path
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars set another way

from rich.logging import RichHandler
from rich.console import Console

# Library logger (configured in CLI)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Configure rich console with timestamp
console = Console()
rich_handler = RichHandler(
    console=console,
    show_time=True,
    show_path=False,
    markup=True,
    rich_tracebacks=True,
    level=logging.INFO
)
rich_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(rich_handler)
logger.propagate = False  # Prevent duplicate logs from parent loggers

# Constants
MAX_SNIPPET_LENGTH = 2000
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def get_ollama_client(host: Optional[str] = None) -> Client:
    """Get Ollama client with configurable host from environment or parameter."""
    if host is None:
        host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    return Client(host=host)


# Ollama API client - use environment variable or default
client = get_ollama_client()

def load_text_file(path: Union[str, Path], mode: str = "simple") -> str:
    """
    Load text from a .txt or .pdf file.
    If PDF, optionally parse in 'columns' or 'simple' mode.
    
    Args:
        path: Path to the file (.txt or .pdf)
        mode: Parsing mode for PDFs ('simple' or 'columns')
    
    Returns:
        Extracted text content
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix.lower() == ".pdf":
        try:
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
        except Exception as e:
            raise ValueError(f"Error reading PDF file {path}: {e}")
    elif path.suffix.lower() == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Error reading text file {path}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .txt and .pdf are supported.")


def chunk_text_by_tokens(
    large_text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "gpt2",
) -> List[str]:
    """Chunk text by tokens using tiktoken.
    
    Args:
        large_text: Text to chunk
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        encoding_name: Tiktoken encoding name
    
    Returns:
        List of text chunks
    
    Raises:
        ValueError: If chunk_size or chunk_overlap are invalid
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    
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
    """Split large text into character-based chunks using LangChain splitter.
    
    Args:
        large_text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of overlapping characters
        separators: List of separators for splitting
        dedup: Whether to deduplicate chunks
    
    Returns:
        List of text chunks
    
    Raises:
        ValueError: If parameters are invalid
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    
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
) -> Dict[str, Union[str, int]]:
    """
    Save a list of dicts or HuggingFace Dataset to a JSONL file.
    
    Args:
        records: List of dictionaries or HuggingFace Dataset
        path: Output file path
    
    Returns:
        Dictionary with 'path' and 'count' keys
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(records, Dataset):
        records = records.to_list()

    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {path}")
    return {"path": str(path), "count": len(records)}


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
    
    Args:
        source: Path to source file
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        mode: PDF parsing mode ('simple' or 'columns')
        chunk_method: 'characters' or 'tokens'
        dedup: Whether to deduplicate chunks
        encoding_name: Tiktoken encoding (only for tokens method)
    
    Returns:
        List of text chunks
    
    Raises:
        ValueError: If chunk_method is invalid
    """
    if chunk_method not in ["characters", "tokens"]:
        raise ValueError(f"chunk_method must be 'characters' or 'tokens', got '{chunk_method}'")
    
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


def extract_json_array(text: str) -> List[dict]:
    """Extract JSON array from text response with robust parsing.
    
    Args:
        text: Text containing JSON array
    
    Returns:
        Parsed JSON array
    
    Raises:
        ValueError: If no valid JSON array found
    """
    try:
        # Try direct parse first
        result = json.loads(text)
        if isinstance(result, list):
            return result
        raise ValueError("Response is not a JSON array")
    except json.JSONDecodeError:
        # Fallback to bracket extraction
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in response")
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
            raise ValueError("Extracted content is not a JSON array")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")


def make_pretrain_data(
    chunks: List[str],
    output_file: Union[str, Path],
    entity: str = "Unknown",
    doc_type: str = "document",
    inject: bool = True,
) -> Dict[str, Union[str, int]]:
    """Prepare text chunks for causal-style pretraining.
    
    Args:
        chunks: List of text chunks
        output_file: Output JSONL file path
        entity: Entity name for identity injection
        doc_type: Document type for identity injection
        inject: Whether to inject identity information
    
    Returns:
        Dictionary with output path and record count
    
    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("chunks list is empty")
    
    items = [{"text": add_identity(p, entity, doc_type) if inject else p} for p in chunks]
    if not items:
        raise ValueError("No valid paragraphs found.")

    ds = Dataset.from_list(items)
    return save_json_dataset(ds, output_file)


def make_instruct_data_QA(
    chunks: List[str],
    output_file: Union[str, Path],
    model: str = "qwen2.5:14b",
    max_q: int = 3,
    delay: float = 0.5,
    entity: str = "Unknown",
    doc_type: str = "document",
) -> Dict[str, Union[str, int]]:
    """Generate instruction-style Q&A pairs using Ollama, with JSON extraction and validation.
    
    Args:
        chunks: List of text chunks to process
        output_file: Output JSONL file path
        model: Ollama model name
        max_q: Maximum questions per chunk
        delay: Delay between API calls (seconds)
        entity: Entity name for metadata
        doc_type: Document type for metadata
    
    Returns:
        Dictionary with output path and record count
    
    Raises:
        ValueError: If no data was generated or chunks is empty
    """
    if not chunks:
        raise ValueError("chunks list is empty")
    
    items: List[dict] = []

    for idx, p in tqdm(enumerate(chunks), total=len(chunks)):
        snippet = p.strip().replace("\n", " ")[:MAX_SNIPPET_LENGTH]
        prompt = (
            f"You are a helpful Q&A assistant. Produce up to {max_q} question-answer pairs in strict JSON format "
            f"as an array of objects with 'question' and 'answer' fields.\n\n"
            f"Paragraph:\n{snippet}...\n\nRespond with JSON only, no extra text."
        )
        try:
            logger.info(f"Generating Q&A for chunk {idx}...")
            
            res = client.generate(
                model=model, 
                prompt=prompt.format(snippet=snippet), 
                options={
                    "temperature": 0.4,
                    "num_ctx": 2048,  # <--- LIMIT THIS. Default is often 4096 or higher.
                    "num_gpu": 99     # <--- Force all layers to GPU
                }
            )
            
            text = res.get("response", "")
            
            logger.info(f"Ollama response for chunk {idx}: {text[:100]}...")
            
            # Use the extracted helper function
            pairs = extract_json_array(text)
            
            
            valid: List[dict] = []
            for qa in pairs:
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
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error for chunk {idx}: {e}")
        except ValueError as e:
            logger.warning(f"Validation error for chunk {idx}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error generating Q&A for chunk {idx}: {e}")
        
        time.sleep(delay)

    if not items:
        raise ValueError("No Q&A data was generated.")

    ds = Dataset.from_list(items)
    logger.info(f"Generated {len(items)} instruction records")
    return save_json_dataset(ds, output_file)


def make_instruct_data(
    chunks: List[str],
    output_file: Union[str, Path],
    model: str = "qwen2.5:14b",
    max_q: int = 2,
    delay: float = 0.5,
    entity: str = "Unknown",
    doc_type: str = "document",
) -> Dict[str, Union[str, int]]:
    """
    Generate 'Reasoning Extraction' training data using Ollama.
    Creates conversational slot-filling data with multi-turn dialogues.
    
    Args:
        chunks: List of text chunks to process
        output_file: Output JSONL file path
        model: Ollama model name (recommend 14B for dialogues)
        max_q: Maximum dialogues per chunk (lower than Q&A due to length)
        delay: Delay between API calls (seconds)
        entity: Entity name for metadata
        doc_type: Document type for metadata
    
    Returns:
        Dictionary with output path and record count
    
    Raises:
        ValueError: If no data was generated or chunks is empty
    """
    if not chunks:
        raise ValueError("chunks list is empty")
    
    items: List[dict] = []

    for idx, p in tqdm(enumerate(chunks), total=len(chunks)):
        snippet = p.strip().replace("\n", " ")[:MAX_SNIPPET_LENGTH]
        
        prompt = (
            f"Read the text below. Create {max_q} training examples where an AI Assistant "
            f"must have a conversation with a User to extract specific information found in the text.\n\n"
            f"TEXT: {snippet}...\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Create a realistic multi-turn 'dialogue' where the User gives info piece-by-piece.\n"
            f"2. The 'extraction' must be the final JSON object the Assistant would generate.\n"
            f"3. Output a strict JSON list of objects with these keys:\n"
            f"   - 'instruction': What the AI is trying to do (e.g. 'Extract config settings').\n"
            f"   - 'dialogue': The conversation history (User: ... System: ...).\n"
            f"   - 'thought_process': The AI's internal logic before the final answer.\n"
            f"   - 'extraction': The final JSON result.\n\n"
            f"Respond with JSON ONLY."
        )

        try:
            logger.info(f"Generating Dialogue Data for chunk {idx}...")
            res = client.generate(
                model=model, 
                prompt=prompt.format(snippet=snippet), 
                options={
                    "temperature": 0.4,
                    "num_ctx": 2048,  # <--- LIMIT THIS. Default is often 4096 or higher.
                    "num_gpu": 99     # <--- Force all layers to GPU
                }
            )
            logger.info(f"Ollama response for chunk {idx}: {res.get('response', '')[:100]}...")
            text = res.get("response", "")
            
            # Use the extracted helper function
            pairs = extract_json_array(text)
            
            valid: List[dict] = []
            required_keys = ('instruction', 'dialogue', 'thought_process', 'extraction')
            
            for item in pairs:
                if isinstance(item, dict) and all(k in item for k in required_keys):
                    # Format into the standard Alpaca/Unsloth format
                    valid.append({
                        'instruction': str(item['instruction']).strip(),
                        'input': str(item['dialogue']).strip(), 
                        'output': f"Thinking: {str(item['thought_process']).strip()}\nResponse: {json.dumps(item['extraction'])}"
                    })
            
            if valid:
                for v in valid:
                    v['source_entity'] = entity
                items.extend(valid)
                logger.debug(f"Chunk {idx}: Generated {len(valid)} dialogues")
            else:
                logger.warning(f"No valid dialogue objects found for chunk {idx}")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error for chunk {idx}: {e}")
        except ValueError as e:
            logger.warning(f"Validation error for chunk {idx}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error processing chunk {idx}: {e}")
        
        time.sleep(delay)

    if not items:
        raise ValueError("No data was generated.")

    ds = Dataset.from_list(items)
    logger.info(f"Generated {len(items)} conversational records")
    return save_json_dataset(ds, output_file)

def get_global_context(text_start: str, client) -> str:
    """
    Reads the first page and extracts the Name/Entity.
    """
    prompt = (
        f"Read this document header and extract the main Subject Name (e.g. candidate name, company name).\n"
        f"Return ONLY the name as a string. If unknown, return 'Unknown Entity'.\n\n"
        f"Text:\n{text_start[:1000]}"
    )
    
    # Use a cheap/fast model for this (7b is fine)
    res = client.generate(model="qwen2.5:7b", prompt=prompt)
    return res.get("response", "").strip()


def analyze_page_visually(
    pdf_path: str, 
    page_number: int, 
    model: str = "qwen3-vl:8b"
) -> str:
    """
    Converts a specific PDF page to an image and asks vision model to extract logic.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to analyze (1-indexed)
        model: Vision-capable Ollama model name
    
    Returns:
        Model response text containing extracted information
    
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If page_number is invalid
        
    Example:
        >>> result = analyze_page_visually("manual.pdf", 5)
        >>> data = json.loads(result)
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")
    
    try:
        # Convert PDF Page to Image
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if not images:
            raise ValueError(f"Could not extract page {page_number} from PDF")
        img = images[0]
        
        # Convert Image to Base64 for Ollama
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prompt with Vision Support
        prompt = (
            "You are an expert technical analyst. Look at this page from a manual.\n"
            "Extract 3 Logical Training Examples based on the visual diagrams and text.\n"
            "Ignore headers/footers. If you see a diagram/chart, explicitly describe its logic.\n"
            "Output format: JSON list with keys: 'instruction', 'logic', 'response'."
        )
        
        logger.info(f"Analyzing page {page_number} of {pdf_path} visually...")
        response = client.generate(
            model=model,
            prompt=prompt,
            images=[img_str],
            options={"temperature": 0.2}
        )
        
        return response['response']
    except Exception as e:
        raise ValueError(f"Error analyzing page {page_number}: {e}")