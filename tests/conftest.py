"""
Pytest configuration and shared fixtures for test suite.
"""

import os
import pytest
from pathlib import Path


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return "This is a sample text for testing. " * 20


@pytest.fixture
def sample_chunks():
    """Provide sample text chunks."""
    return [
        "First chunk of text with some content.",
        "Second chunk of text with different content.",
        "Third chunk discussing various topics.",
    ]


@pytest.fixture
def mock_ollama_response():
    """Provide a mock Ollama API response."""
    return {
        'response': '[{"question": "What is this?", "answer": "This is a test."}]'
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_pdf_content():
    """Provide minimal PDF binary content for testing."""
    # This is a minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Page) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000317 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
409
%%EOF"""


@pytest.fixture
def create_sample_pdf(tmp_path, sample_pdf_content):
    """Create a sample PDF file for testing."""
    def _create_pdf(filename="test.pdf"):
        pdf_path = tmp_path / filename
        pdf_path.write_bytes(sample_pdf_content)
        return pdf_path
    return _create_pdf


# Markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
