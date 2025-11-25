"""
Test suite for data_prep_tools module.
Tests core functionality including chunking, file loading, JSON parsing, and data generation.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from src.llm_finetune.data_prep_tools import (
    load_text_file,
    chunk_text_by_tokens,
    chunk_text_by_characters,
    deduplicate_chunks,
    add_identity,
    save_json_dataset,
    read_and_chunk_document,
    extract_json_array,
    make_pretrain_data,
    get_ollama_client,
    MAX_SNIPPET_LENGTH,
    DEFAULT_OLLAMA_HOST,
)


class TestGetOllamaClient:
    """Test Ollama client configuration."""
    
    def test_default_host(self):
        """Test client uses default host when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            client = get_ollama_client()
            assert client._client.base_url == DEFAULT_OLLAMA_HOST
    
    def test_env_var_host(self):
        """Test client uses OLLAMA_HOST from environment."""
        test_host = "http://test.host:11434"
        with patch.dict(os.environ, {"OLLAMA_HOST": test_host}):
            client = get_ollama_client()
            assert client._client.base_url == test_host
    
    def test_explicit_host_override(self):
        """Test explicit host parameter overrides environment."""
        test_host = "http://explicit.host:11434"
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://env.host:11434"}):
            client = get_ollama_client(host=test_host)
            assert client._client.base_url == test_host


class TestLoadTextFile:
    """Test file loading functionality."""
    
    def test_load_text_file_success(self, tmp_path):
        """Test loading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file.\nWith multiple lines."
        test_file.write_text(test_content, encoding="utf-8")
        
        result = load_text_file(test_file)
        assert result == test_content
    
    def test_load_text_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_text_file("nonexistent_file.txt")
    
    def test_load_unsupported_file_type(self, tmp_path):
        """Test error with unsupported file type."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_text_file(test_file)
    
    @patch('pdfplumber.open')
    def test_load_pdf_simple_mode(self, mock_pdf_open, tmp_path):
        """Test loading PDF in simple mode."""
        # Mock PDF structure
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page 1 content"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        
        mock_pdf_open.return_value = mock_pdf
        
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")
        
        result = load_text_file(pdf_file, mode="simple")
        assert result == "Page 1 content"
        mock_page.extract_text.assert_called_once()


class TestChunkingFunctions:
    """Test text chunking functionality."""
    
    def test_chunk_text_by_tokens_valid(self):
        """Test token-based chunking with valid parameters."""
        text = "This is a test sentence. " * 50  # ~250 tokens
        chunks = chunk_text_by_tokens(text, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 0
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
    
    def test_chunk_text_by_tokens_invalid_size(self):
        """Test token chunking with invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text_by_tokens("test", chunk_size=0, chunk_overlap=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text_by_tokens("test", chunk_size=-5, chunk_overlap=0)
    
    def test_chunk_text_by_tokens_invalid_overlap(self):
        """Test token chunking with invalid overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_text_by_tokens("test", chunk_size=10, chunk_overlap=-1)
    
    def test_chunk_text_by_tokens_overlap_too_large(self):
        """Test token chunking when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            chunk_text_by_tokens("test", chunk_size=10, chunk_overlap=10)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            chunk_text_by_tokens("test", chunk_size=10, chunk_overlap=15)
    
    def test_chunk_text_by_characters_valid(self):
        """Test character-based chunking."""
        text = "This is a test.\n\n" * 50
        chunks = chunk_text_by_characters(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 0
        assert isinstance(chunks, list)
    
    def test_chunk_text_by_characters_invalid_params(self):
        """Test character chunking with invalid parameters."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text_by_characters("test", chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            chunk_text_by_characters("test", chunk_size=10, chunk_overlap=10)
    
    def test_chunk_text_by_characters_no_dedup(self):
        """Test character chunking without deduplication."""
        text = "test " * 20
        chunks = chunk_text_by_characters(text, chunk_size=10, chunk_overlap=5, dedup=False)
        assert len(chunks) > 0
    
    def test_deduplicate_chunks(self):
        """Test chunk deduplication while preserving order."""
        chunks = ["chunk1", "chunk2", "chunk1", "chunk3", "chunk2"]
        result = deduplicate_chunks(chunks)
        
        assert result == ["chunk1", "chunk2", "chunk3"]
        assert len(result) == 3


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_add_identity(self):
        """Test identity information prepending."""
        text = "Sample text content"
        result = add_identity(text, entity="TestEntity", doc_type="test_doc")
        
        assert "[ENTITY: TestEntity]" in result
        assert "[TYPE: test_doc]" in result
        assert "Sample text content" in result
    
    def test_add_identity_strips_whitespace(self):
        """Test that add_identity strips extra whitespace."""
        text = "  Sample text  \n\n"
        result = add_identity(text, entity="Test", doc_type="doc")
        
        assert result.endswith("Sample text")


class TestSaveJsonDataset:
    """Test JSON dataset saving."""
    
    def test_save_json_dataset_from_list(self, tmp_path):
        """Test saving list of dicts to JSONL."""
        records = [
            {"field1": "value1", "field2": 123},
            {"field1": "value2", "field2": 456},
        ]
        output_file = tmp_path / "output" / "test.jsonl"
        
        result = save_json_dataset(records, output_file)
        
        assert output_file.exists()
        assert result["path"] == str(output_file)
        assert result["count"] == 2
        
        # Verify content
        with open(output_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0]) == records[0]
    
    def test_save_json_dataset_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        output_file = tmp_path / "level1" / "level2" / "test.jsonl"
        records = [{"test": "data"}]
        
        save_json_dataset(records, output_file)
        
        assert output_file.exists()
        assert output_file.parent.exists()


class TestExtractJsonArray:
    """Test JSON extraction from text."""
    
    def test_extract_json_array_direct_parse(self):
        """Test extraction when JSON is clean."""
        text = '[{"key": "value"}, {"key": "value2"}]'
        result = extract_json_array(text)
        
        assert len(result) == 2
        assert result[0]["key"] == "value"
    
    def test_extract_json_array_with_prefix(self):
        """Test extraction with text before JSON."""
        text = 'Here is the data:\n[{"key": "value"}]\nEnd of data'
        result = extract_json_array(text)
        
        assert len(result) == 1
        assert result[0]["key"] == "value"
    
    def test_extract_json_array_no_array_found(self):
        """Test error when no JSON array found."""
        with pytest.raises(ValueError, match="No JSON array found"):
            extract_json_array("This is just text with no JSON")
    
    def test_extract_json_array_invalid_json(self):
        """Test error with malformed JSON."""
        with pytest.raises(ValueError, match="(Failed to parse JSON|No JSON array found)"):
            extract_json_array('[{"key": "value"')  # Missing closing bracket
    
    def test_extract_json_array_not_array(self):
        """Test error when JSON is not an array."""
        with pytest.raises(ValueError, match="not a JSON array"):
            extract_json_array('{"key": "value"}')


class TestReadAndChunkDocument:
    """Test document reading and chunking."""
    
    def test_read_and_chunk_document_characters(self, tmp_path):
        """Test reading and chunking with character method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content. " * 100)
        
        chunks = read_and_chunk_document(
            source=test_file,
            chunk_size=100,
            chunk_overlap=20,
            chunk_method="characters"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
    
    def test_read_and_chunk_document_tokens(self, tmp_path):
        """Test reading and chunking with token method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content. " * 100)
        
        chunks = read_and_chunk_document(
            source=test_file,
            chunk_size=50,
            chunk_overlap=10,
            chunk_method="tokens"
        )
        
        assert len(chunks) > 0
    
    def test_read_and_chunk_document_invalid_method(self, tmp_path):
        """Test error with invalid chunk method."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        with pytest.raises(ValueError, match="chunk_method must be 'characters' or 'tokens'"):
            read_and_chunk_document(
                source=test_file,
                chunk_size=100,
                chunk_overlap=20,
                chunk_method="invalid"
            )


class TestMakePretrainData:
    """Test pretrain data generation."""
    
    def test_make_pretrain_data_with_inject(self, tmp_path):
        """Test pretrain data generation with identity injection."""
        chunks = ["Chunk 1 content", "Chunk 2 content"]
        output_file = tmp_path / "pretrain.jsonl"
        
        result = make_pretrain_data(
            chunks=chunks,
            output_file=output_file,
            entity="TestEntity",
            doc_type="test",
            inject=True
        )
        
        assert output_file.exists()
        assert result["count"] == 2
        
        # Verify content has identity
        with open(output_file, "r") as f:
            first_line = json.loads(f.readline())
            assert "[ENTITY: TestEntity]" in first_line["text"]
    
    def test_make_pretrain_data_without_inject(self, tmp_path):
        """Test pretrain data generation without identity injection."""
        chunks = ["Chunk 1 content"]
        output_file = tmp_path / "pretrain.jsonl"
        
        make_pretrain_data(chunks, output_file, inject=False)
        
        with open(output_file, "r") as f:
            first_line = json.loads(f.readline())
            assert first_line["text"] == "Chunk 1 content"
            assert "[ENTITY:" not in first_line["text"]
    
    def test_make_pretrain_data_empty_chunks(self, tmp_path):
        """Test error with empty chunks list."""
        with pytest.raises(ValueError, match="chunks list is empty"):
            make_pretrain_data([], tmp_path / "output.jsonl")


class TestConstants:
    """Test module constants."""
    
    def test_max_snippet_length(self):
        """Test MAX_SNIPPET_LENGTH is defined."""
        assert MAX_SNIPPET_LENGTH == 2000
    
    def test_default_ollama_host(self):
        """Test DEFAULT_OLLAMA_HOST is defined."""
        assert DEFAULT_OLLAMA_HOST == "http://localhost:11434"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
