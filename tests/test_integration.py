"""
Integration tests for data_prep_tools module.
Tests end-to-end workflows and Ollama API integration.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm_finetune.data_prep_tools import (
    make_instruct_data,
    make_instruct_data_ex,
    analyze_page_visually,
)


@pytest.mark.integration
class TestMakeInstructData:
    """Integration tests for instruction data generation."""
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_make_instruct_data_success(self, mock_client, tmp_path):
        """Test successful Q&A generation."""
        # Mock Ollama response
        mock_response = {
            'response': '[{"question": "What is this?", "answer": "Test answer"}]'
        }
        mock_client.generate.return_value = mock_response
        
        chunks = ["This is a test chunk about testing."]
        output_file = tmp_path / "instruct.jsonl"
        
        result = make_instruct_data(
            chunks=chunks,
            output_file=output_file,
            model="test-model",
            max_q=1,
            delay=0.0,
        )
        
        assert output_file.exists()
        assert result["count"] == 1
        
        # Verify generated content
        with open(output_file, "r") as f:
            record = json.loads(f.readline())
            assert "question_answer_pairs" in record
            assert len(record["question_answer_pairs"]) == 1
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_make_instruct_data_malformed_json(self, mock_client, tmp_path):
        """Test handling of malformed JSON responses."""
        mock_response = {'response': 'Not valid JSON at all'}
        mock_client.generate.return_value = mock_response
        
        chunks = ["Test chunk"]
        output_file = tmp_path / "instruct.jsonl"
        
        # Should raise error when no valid data generated
        with pytest.raises(ValueError, match="No Q&A data was generated"):
            make_instruct_data(chunks, output_file, delay=0.0)
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_make_instruct_data_partial_success(self, mock_client, tmp_path):
        """Test when some chunks succeed and others fail."""
        # First call succeeds, second fails
        mock_client.generate.side_effect = [
            {'response': '[{"question": "Q1?", "answer": "A1"}]'},
            {'response': 'Invalid JSON'},
        ]
        
        chunks = ["Chunk 1", "Chunk 2"]
        output_file = tmp_path / "instruct.jsonl"
        
        result = make_instruct_data(chunks, output_file, delay=0.0)
        
        # Should save the successful chunk
        assert result["count"] == 1
    
    def test_make_instruct_data_empty_chunks(self, tmp_path):
        """Test error with empty chunks."""
        with pytest.raises(ValueError, match="chunks list is empty"):
            make_instruct_data([], tmp_path / "output.jsonl")


@pytest.mark.integration
class TestMakeInstructDataEx:
    """Integration tests for extended instruction data generation."""
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_make_instruct_data_ex_success(self, mock_client, tmp_path):
        """Test successful dialogue generation."""
        mock_response = {
            'response': json.dumps([{
                'instruction': 'Extract user info',
                'dialogue': 'User: Hi\nAssistant: Hello',
                'thought_process': 'Analyzing input',
                'extraction': {'name': 'John'}
            }])
        }
        mock_client.generate.return_value = mock_response
        
        chunks = ["User profile: John Smith, age 30"]
        output_file = tmp_path / "instruct_ex.jsonl"
        
        result = make_instruct_data_ex(
            chunks=chunks,
            output_file=output_file,
            model="test-model",
            max_q=1,
            delay=0.0,
        )
        
        assert output_file.exists()
        assert result["count"] == 1
        
        # Verify Alpaca format
        with open(output_file, "r") as f:
            record = json.loads(f.readline())
            assert "instruction" in record
            assert "input" in record
            assert "output" in record
            assert "Thinking:" in record["output"]
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_make_instruct_data_ex_missing_fields(self, mock_client, tmp_path):
        """Test handling when required fields are missing."""
        # Missing 'extraction' field
        mock_response = {
            'response': json.dumps([{
                'instruction': 'Test',
                'dialogue': 'Test dialogue',
                'thought_process': 'Thinking'
            }])
        }
        mock_client.generate.return_value = mock_response
        
        chunks = ["Test"]
        output_file = tmp_path / "instruct_ex.jsonl"
        
        with pytest.raises(ValueError, match="No data was generated"):
            make_instruct_data_ex(chunks, output_file, delay=0.0)


@pytest.mark.integration
@pytest.mark.slow
class TestAnalyzePageVisually:
    """Integration tests for visual PDF analysis."""
    
    def test_analyze_page_visually_file_not_found(self):
        """Test error when PDF doesn't exist."""
        with pytest.raises(FileNotFoundError):
            analyze_page_visually("nonexistent.pdf", 1)
    
    def test_analyze_page_visually_invalid_page_number(self, tmp_path):
        """Test error with invalid page number."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf")
        
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            analyze_page_visually(str(pdf_file), 0)
    
    @patch('src.llm_finetune.data_prep_tools.convert_from_path')
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_analyze_page_visually_success(self, mock_client, mock_convert, tmp_path):
        """Test successful page analysis."""
        # Mock image conversion
        from PIL import Image
        mock_image = Image.new('RGB', (100, 100))
        mock_convert.return_value = [mock_image]
        
        # Mock Ollama response
        mock_client.generate.return_value = {
            'response': '[{"instruction": "test", "logic": "test logic", "response": "test response"}]'
        }
        
        # Create dummy PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")
        
        result = analyze_page_visually(str(pdf_file), 1)
        
        assert isinstance(result, str)
        assert len(result) > 0
        mock_convert.assert_called_once()
        mock_client.generate.assert_called_once()


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_full_pretrain_pipeline(self, tmp_path):
        """Test complete pretrain data generation pipeline."""
        from src.llm_finetune.data_prep_tools import (
            read_and_chunk_document,
            make_pretrain_data,
        )
        
        # Create test document
        test_doc = tmp_path / "source.txt"
        test_doc.write_text("This is a test document. " * 50)
        
        # Chunk the document
        chunks = read_and_chunk_document(
            source=test_doc,
            chunk_size=100,
            chunk_overlap=20,
            chunk_method="characters"
        )
        
        # Generate pretrain data
        output_file = tmp_path / "pretrain.jsonl"
        result = make_pretrain_data(
            chunks=chunks,
            output_file=output_file,
            entity="TestDoc",
            doc_type="test",
            inject=True
        )
        
        assert output_file.exists()
        assert result["count"] == len(chunks)
        
        # Verify format
        with open(output_file, "r") as f:
            first_record = json.loads(f.readline())
            assert "text" in first_record
            assert "[ENTITY: TestDoc]" in first_record["text"]
    
    @patch('src.llm_finetune.data_prep_tools.client')
    def test_full_instruct_pipeline(self, mock_client, tmp_path):
        """Test complete instruction data generation pipeline."""
        from src.llm_finetune.data_prep_tools import (
            read_and_chunk_document,
            make_instruct_data,
        )
        
        # Mock responses for each chunk
        mock_client.generate.return_value = {
            'response': '[{"question": "Test Q?", "answer": "Test A"}]'
        }
        
        # Create test document
        test_doc = tmp_path / "source.txt"
        test_doc.write_text("Test content for questions. " * 30)
        
        # Chunk the document
        chunks = read_and_chunk_document(
            source=test_doc,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Generate instruction data
        output_file = tmp_path / "instruct.jsonl"
        result = make_instruct_data(
            chunks=chunks,
            output_file=output_file,
            model="test-model",
            delay=0.0
        )
        
        assert output_file.exists()
        assert result["count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
