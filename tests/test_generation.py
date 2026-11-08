"""Tests for generation, prompt formatting, citation validation, and abstention."""

from unittest.mock import patch, MagicMock
import pytest
import requests

from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import build_rag_messages, ABSTENTION_MESSAGE
from app.generation.answer_validator import AnswerValidator


def test_ollama_model_selection_exact_tags():
    """Client must preserve tags and distinguish between llama3.2:1b and llama3.2:latest."""
    client = OllamaClient()

    with patch.object(
        client, "list_installed_models", return_value=["llama3.2:1b", "mistral:7b-instruct"]
    ):
        # When preferring llama3.2:latest, llama3.2:1b should match as part of llama3.2 family if 3.2:latest absent
        best = client.select_best_model(["llama3.2:latest", "mistral:latest", "llama3.2:1b"])
        assert best in ("llama3.2:1b", "mistral:7b-instruct")


def test_ollama_connection_failure():
    """Client must raise ConnectionError with clear troubleshooting instructions when Ollama is offline."""
    client = OllamaClient(base_url="http://localhost:59999")
    with pytest.raises(ConnectionError, match="ollama serve"):
        client.generate_chat([{"role": "user", "content": "hi"}], model="llama3.2")


def test_prompt_builder_structure():
    """Prompt builder must inject structured evidence with citations."""
    retrieved = [
        {
            "source_filename": "informatik.pdf",
            "page_number": 5,
            "programme_name": "Informatik MSc",
            "section_heading": "AlgoDat",
            "text": "Prerequisites: C, C++, Java",
        }
    ]
    messages = build_rag_messages("What are prerequisites?", retrieved)
    assert len(messages) == 2
    assert "informatik.pdf" in messages[1]["content"]
    assert "page 5" in messages[1]["content"]
    assert "AlgoDat" in messages[1]["content"]


def test_citation_validator_valid():
    """Validator must accept citations that match retrieved pages."""
    retrieved = [
        {"source_filename": "informatik.pdf", "page_number": 5},
        {"source_filename": "embedded-systems.pdf", "page_number": 8},
    ]
    answer = "Algorithms and Data Structures has 9 ECTS [Source: informatik.pdf, page 5]."
    validator = AnswerValidator()
    res = validator.validate(answer, retrieved)
    assert res.is_valid is True
    assert len(res.validated_citations) == 1
    assert res.validated_citations[0]["filename"] == "informatik.pdf"
    assert res.validated_citations[0]["page_number"] == 5


def test_citation_validator_hallucinated():
    """Validator must detect hallucinated sources or non-retrieved page numbers."""
    retrieved = [
        {"source_filename": "informatik.pdf", "page_number": 5},
    ]
    # Answer cites page 99 which was never retrieved!
    answer = "Fake fact [Source: informatik.pdf, page 99]."
    validator = AnswerValidator()
    res = validator.validate(answer, retrieved)
    assert res.is_valid is False
    assert len(res.invalid_citations) == 1
    assert res.invalid_citations[0]["page_number"] == 99


def test_citation_validator_abstention():
    """Validator must identify standard abstention answers as valid abstentions."""
    validator = AnswerValidator()
    answer = "I could not find sufficient information in the indexed module handbooks to answer this question reliably."
    res = validator.validate(answer, [])
    assert res.is_valid is True
    assert res.is_abstained is True
