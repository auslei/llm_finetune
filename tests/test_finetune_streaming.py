"""
Tests for streaming dataset loading and memory optimization features.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.llm_finetune.finetune_tool import FineTuner
from src.llm_finetune.tuner_config import FineTuneConfig


@pytest.fixture
def sample_instruct_data():
    """Sample instruction data for testing."""
    return [
        {"instruction": "What is AI?", "input": "", "output": "AI stands for Artificial Intelligence."},
        {"instruction": "Define ML", "input": "", "output": "ML stands for Machine Learning."},
        {"instruction": "Explain NLP", "input": "", "output": "NLP stands for Natural Language Processing."},
    ]


@pytest.fixture
def sample_pretrain_data():
    """Sample pretrain data for testing."""
    return [
        {"text": "This is the first chunk of text for pretraining."},
        {"text": "This is the second chunk of text for pretraining."},
        {"text": "This is the third chunk of text for pretraining."},
    ]


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return [
        {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"}
            ]
        },
    ]


@pytest.mark.unit
class TestStreamingConfiguration:
    """Test streaming configuration options."""
    
    def test_default_streaming_disabled(self):
        """Test that streaming is disabled by default."""
        config = FineTuneConfig(
            training_data_path="test.jsonl",
            mode="instruct"
        )
        assert config.streaming is False
    
    def test_enable_streaming(self):
        """Test enabling streaming mode."""
        config = FineTuneConfig(
            training_data_path="test.jsonl",
            mode="instruct",
            streaming=True
        )
        assert config.streaming is True
    
    def test_streaming_from_dict(self):
        """Test streaming config from dictionary."""
        config = FineTuneConfig.from_dict({
            "training_data_path": "test.jsonl",
            "mode": "pretrain",
            "streaming": True
        })
        assert config.streaming is True


@pytest.mark.integration
class TestNonStreamingDataLoading:
    """Test traditional non-streaming data loading (baseline)."""
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_load_instruct_data_non_streaming(self, mock_load_dataset, mock_model, tmp_path, sample_instruct_data):
        """Test loading instruction data without streaming."""
        # Create test data file
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_instruct_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = MagicMock()
        mock_dataset.column_names = ["instruction", "input", "output"]
        mock_dataset.__len__.return_value = len(sample_instruct_data)
        mock_dataset.select.return_value = sample_instruct_data
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model components
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config and load data
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="instruct",
            streaming=False,
            val_split=0.0
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify load_dataset was called with streaming=False
        mock_load_dataset.assert_called_once()
        call_kwargs = mock_load_dataset.call_args.kwargs
        assert call_kwargs.get("streaming") is False
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_load_pretrain_data_non_streaming(self, mock_load_dataset, mock_model, tmp_path, sample_pretrain_data):
        """Test loading pretrain data without streaming."""
        # Create test data file
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_pretrain_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = len(sample_pretrain_data)
        mock_dataset.select.return_value = sample_pretrain_data
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model components
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config and load data
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="pretrain",
            streaming=False
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify load_dataset was called with streaming=False
        mock_load_dataset.assert_called_once()
        call_kwargs = mock_load_dataset.call_args.kwargs
        assert call_kwargs.get("streaming") is False


@pytest.mark.integration
class TestStreamingDataLoading:
    """Test streaming data loading for memory optimization."""
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_load_data_with_streaming(self, mock_load_dataset, mock_model, tmp_path, sample_pretrain_data):
        """Test that streaming mode is enabled when configured."""
        # Create test data file
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_pretrain_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock streaming dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        # Streaming datasets don't support len()
        mock_dataset.__len__.side_effect = TypeError("Streaming dataset doesn't support len()")
        # Mock take() for validation
        mock_dataset.take.return_value = sample_pretrain_data[:3]
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model components
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config with streaming enabled
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="pretrain",
            streaming=True
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify load_dataset was called with streaming=True
        mock_load_dataset.assert_called_once()
        call_kwargs = mock_load_dataset.call_args.kwargs
        assert call_kwargs.get("streaming") is True
        
        # Verify dataset was loaded
        assert tuner.train_dataset is not None
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_streaming_skips_auto_split(self, mock_load_dataset, mock_model, tmp_path, sample_instruct_data):
        """Test that streaming mode skips automatic train/test splitting."""
        # Create test data file (single file, no separate test set)
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_instruct_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock streaming dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "input", "output"]
        mock_dataset.take.return_value = sample_instruct_data[:3]
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model components
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config with streaming and val_split
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="instruct",
            streaming=True,
            val_split=0.2  # This should be ignored in streaming mode
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify that val_dataset is None (no auto-split in streaming mode)
        assert tuner.val_dataset is None
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_streaming_with_separate_test_file(self, mock_load_dataset, mock_model, tmp_path, sample_instruct_data):
        """Test streaming mode with separate train and test files."""
        # Create train and test files
        train_dir = tmp_path / "data"
        train_dir.mkdir()
        train_file = train_dir / "train.jsonl"
        test_file = train_dir / "test.jsonl"
        
        with open(train_file, "w") as f:
            for item in sample_instruct_data[:2]:
                f.write(json.dumps(item) + "\n")
        
        with open(test_file, "w") as f:
            f.write(json.dumps(sample_instruct_data[2]) + "\n")
        
        # Mock streaming datasets
        mock_train_dataset = MagicMock()
        mock_train_dataset.column_names = ["instruction", "input", "output"]
        mock_train_dataset.take.return_value = sample_instruct_data[:2]
        
        mock_test_dataset = MagicMock()
        mock_test_dataset.column_names = ["instruction", "input", "output"]
        mock_test_dataset.take.return_value = [sample_instruct_data[2]]
        
        mock_ds_dict = {"train": mock_train_dataset, "test": mock_test_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model components
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config with streaming
        config = FineTuneConfig(
            training_data_path=str(train_dir),
            mode="instruct",
            streaming=True
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify both datasets are loaded
        assert tuner.train_dataset is not None
        assert tuner.val_dataset is not None


@pytest.mark.integration
class TestBatchedTokenization:
    """Test batched tokenization features."""
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_conversation_formatting_batched(self, mock_load_dataset, mock_model, tmp_path, sample_conversation_data):
        """Test that conversation formatting uses batched processing."""
        # Create test data file
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_conversation_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["conversations"]
        mock_dataset.__len__.return_value = len(sample_conversation_data)
        mock_dataset.take.return_value = sample_conversation_data
        
        # Mock the map function to capture batched parameter
        map_calls = []
        def mock_map(*args, **kwargs):
            map_calls.append(kwargs)
            result = MagicMock()
            result.column_names = ["text"]
            return result
        mock_dataset.map = mock_map
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted text"
        mock_model.from_pretrained.return_value = (MagicMock(), mock_tokenizer)
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="instruct",
            streaming=True
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        tuner._load_training_data()
        
        # Verify map was called with batched=True and batch_size
        assert len(map_calls) > 0
        assert map_calls[0].get("batched") is True
        assert map_calls[0].get("batch_size") == 1000


@pytest.mark.unit
class TestValidationWithStreaming:
    """Test data validation works with both streaming and non-streaming modes."""
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_validate_text_data_streaming(self, mock_load_dataset, mock_model, tmp_path, sample_pretrain_data):
        """Test text data validation with streaming dataset."""
        # Create test data
        data_file = tmp_path / "train.jsonl"
        with open(data_file, "w") as f:
            for item in sample_pretrain_data:
                f.write(json.dumps(item) + "\n")
        
        # Mock streaming dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.take.return_value = sample_pretrain_data[:3]
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config with streaming
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="pretrain",
            streaming=True
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        
        # Should not raise an error
        tuner._load_training_data()
        
        # Verify take() was called for validation
        mock_dataset.take.assert_called()
    
    @patch('src.llm_finetune.finetune_tool.FastLanguageModel')
    @patch('src.llm_finetune.finetune_tool.load_dataset')
    def test_validate_empty_dataset_streaming(self, mock_load_dataset, mock_model, tmp_path):
        """Test that empty dataset raises error in streaming mode."""
        # Create empty data file
        data_file = tmp_path / "train.jsonl"
        data_file.write_text("")
        
        # Mock empty streaming dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.take.return_value = []  # Empty
        
        mock_ds_dict = {"train": mock_dataset}
        mock_load_dataset.return_value = mock_ds_dict
        
        # Mock model
        mock_model.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_model.get_peft_model.return_value = MagicMock()
        
        # Create config with streaming
        config = FineTuneConfig(
            training_data_path=str(data_file),
            mode="pretrain",
            streaming=True
        )
        
        tuner = FineTuner(config)
        tuner._load_model()
        
        # Should raise error for empty dataset
        with pytest.raises(ValueError, match="Training dataset is empty"):
            tuner._load_training_data()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
