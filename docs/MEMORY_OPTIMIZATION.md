# Memory Optimization Changes

## Overview
This document describes the memory optimization changes made to address high memory consumption during dataset loading and fine-tuning.

## Changes Made

### 1. Configuration (`src/llm_finetune/tuner_config.py`)
- Added `streaming: bool = False` parameter to `FineTuneConfig` class
- Default is `False` to maintain backward compatibility
- Users can enable streaming mode by setting `streaming=True`

### 2. Dataset Loading (`src/llm_finetune/finetune_tool.py`)

#### Streaming Mode Support
- Modified `load_dataset()` call to accept `streaming` parameter
- When streaming is enabled, datasets are loaded incrementally rather than loading entire dataset into memory

#### Batched Tokenization
- Updated conversation formatting to use explicit `batch_size=1000` parameter
- Added `remove_columns` parameter to drop unused columns after processing
- Reduces memory spikes during tokenization phase

#### Validation Methods
- Updated all three validation methods to work with both streaming and non-streaming datasets:
  - `_validate_instruct_columns()`
  - `_validate_text_data()`
  - `_validate_conversation_data()`
- Streaming datasets use `.take()` instead of `.select()` for sampling
- Skip `len()` checks on streaming datasets (not supported)

#### Smart Train/Test Splitting
- In streaming mode, automatic train/test splitting is disabled (requires loading full dataset)
- Users are warned and should provide separate train and test files
- Non-streaming mode continues to support automatic splitting

### 3. Documentation (`README.md`)
- Added "Memory Optimization" section with usage examples
- Documented streaming mode usage for both Python API and YAML config
- Explained limitations and best practices

### 4. Tests
Created comprehensive test suite:

#### `tests/test_finetune_streaming.py`
- Tests for streaming configuration
- Tests for streaming vs non-streaming data loading
- Tests for batched tokenization
- Tests for validation with streaming datasets
- Total: 10 test cases

#### `tests/test_memory_profiling.py`
- Memory usage comparison tests
- Batched processing verification
- Streaming mechanism verification
- Column removal verification
- Total: 6 test cases

## Memory Optimization Benefits

### 1. Streaming Mode
- **Before**: Entire dataset loaded into RAM at once
- **After**: Data loaded incrementally as needed
- **Impact**: Reduces peak memory usage proportionally to dataset size

### 2. Batched Tokenization
- **Before**: Potentially processing large batches without explicit limits
- **After**: Fixed batch size of 1000 samples
- **Impact**: Prevents memory spikes during tokenization

### 3. Column Removal
- **Before**: All columns kept in memory throughout processing
- **After**: Unused columns removed immediately after use
- **Impact**: Reduces memory footprint of transformed datasets

## Usage Examples

### Enable Streaming via Python API
```python
from llm_finetune.finetune_tool import FineTuner

tuner = FineTuner(
    training_data_path="large_dataset.jsonl",
    model_name="MyModel",
    base_model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    mode="pretrain",
    streaming=True  # Enable streaming mode
)
tuner.train()
```

### Enable Streaming via Config File
```yaml
# config.yaml
training_data_path: "large_dataset.jsonl"
mode: "pretrain"
streaming: true
```

```bash
python -m llm_finetune.cli --config config.yaml
```

## Important Considerations

### When to Use Streaming Mode
- Large datasets (>1GB) that don't fit comfortably in RAM
- Systems with limited memory
- Production environments with strict memory limits

### When NOT to Use Streaming Mode
- Small datasets (<1GB)
- When you need automatic train/test splitting
- When you need to shuffle the entire dataset randomly

### Limitations
1. No automatic train/test splitting in streaming mode
2. Cannot get total dataset size until fully consumed
3. Some operations that require full dataset access are not available

## Testing

All tests pass syntax validation. To run tests with dependencies installed:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run only streaming tests
pytest tests/test_finetune_streaming.py

# Run memory profiling tests
pytest tests/test_memory_profiling.py -v -s
```

## Backward Compatibility

All changes are backward compatible:
- Default `streaming=False` maintains existing behavior
- Existing code works without modification
- New features are opt-in via configuration

## Future Enhancements

Potential future improvements:
1. Automatic memory monitoring to suggest streaming mode
2. Dynamic batch size based on available memory
3. Hybrid mode that streams validation data while loading training data in memory
4. Progress bars showing memory usage during training
