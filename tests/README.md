# Tests

This directory contains the test suite for the `llm_finetune` project.

## Structure

- `test_data_prep_tools.py` - Unit tests for core data preparation functions
- `test_integration.py` - Integration tests for end-to-end workflows
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

### Install test dependencies

```bash
uv pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run specific test categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run specific test files

```bash
# Run only data prep tests
pytest tests/test_data_prep_tools.py

# Run only integration tests
pytest tests/test_integration.py
```

### Run with coverage

```bash
pytest --cov=src/llm_finetune --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in a browser.

### Run verbose mode

```bash
pytest -v
```

## Test Categories

Tests are marked with the following categories:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests that may involve mocked external services
- `@pytest.mark.slow` - Tests that take longer to run

## Writing New Tests

### Basic test structure

```python
import pytest
from src.llm_finetune.data_prep_tools import function_to_test

def test_function_behavior():
    """Test description."""
    result = function_to_test(input_data)
    assert result == expected_output
```

### Using fixtures

```python
def test_with_fixture(sample_text, tmp_path):
    """Test using shared fixtures."""
    # sample_text and tmp_path are provided by conftest.py
    assert len(sample_text) > 0
```

### Mocking external services

```python
from unittest.mock import patch

@patch('src.llm_finetune.data_prep_tools.client')
def test_with_mock(mock_client):
    """Test with mocked Ollama client."""
    mock_client.generate.return_value = {'response': 'test'}
    # Test code here
```

## Key Test Areas

1. **Input Validation** - Ensure functions validate parameters correctly
2. **Error Handling** - Test error conditions and exceptions
3. **Edge Cases** - Test boundary conditions and unusual inputs
4. **Integration** - Test end-to-end workflows
5. **Mocking** - Test external API interactions without actual calls

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure all tests pass before submitting pull requests.

## Coverage Goals

- Aim for >80% code coverage
- All critical paths should be tested
- Error handling should be tested
