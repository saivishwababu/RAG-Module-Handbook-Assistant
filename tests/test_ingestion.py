"""Tests for PDF loading, error handling, page preservation, and text extraction."""

import tempfile
from pathlib import Path
import pytest

from app.ingestion.pdf_loader import PDFLoader, compute_file_hash
from app.ingestion.text_extractor import clean_text_spacing, detect_language
from app.storage.document_registry import DocumentRegistry


def test_pdf_loader_invalid_file():
    """Loader must raise FileNotFoundError for nonexistent file."""
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf("nonexistent_file.pdf")


def test_pdf_loader_empty_file(tmp_path):
    """Loader must reject zero-byte files."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    loader = PDFLoader()
    with pytest.raises(ValueError, match="completely empty"):
        loader.load_pdf(empty_pdf)


def test_pdf_loader_unsupported_extension(tmp_path):
    """Loader must reject non-pdf extensions."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Some text")
    loader = PDFLoader()
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_pdf(txt_file)


def test_pdf_page_number_preservation():
    """Verify that actual PDF pages are preserved with correct 1-based indexing."""
    pdf_path = Path("data/visual_computing.pdf")
    if not pdf_path.exists():
        pytest.skip("data/visual_computing.pdf not found in local workspace")

    loader = PDFLoader()
    doc = loader.load_pdf(pdf_path)

    assert doc.total_pages == 36
    assert len(doc.pages) == 36
    for idx, page in enumerate(doc.pages):
        assert page.page_number == idx + 1, "Page numbers must strictly be 1-indexed"


def test_clean_text_spacing():
    """Verify recovery of lost spaces without mangling standard text."""
    samples = [
        ("ComputerGraphics", "Computer Graphics"),
        ("AlgorithmsandDataStructures", "Algorithms and Data Structures"),
        ("Semester1", "Semester 1"),
        ("9ECTS", "9 ECTS"),
        ("Introductionto", "Introduction to"),
        ("Signal&System", "Signal & System"),
    ]
    for inp, expected in samples:
        res = clean_text_spacing(inp)
        assert expected in res or res == expected, f"Failed for '{inp}': got '{res}'"


def test_language_detection():
    """Verify German vs English detection."""
    de_text = "Modulhandbuch Fakultät für Mathematik und Informatik Leistungspunkte Prüfungsleistung"
    en_text = "Faculty of Mathematics and Computer Science Module Descriptions Entrance Requirements"

    assert detect_language(de_text) == "de"
    assert detect_language(en_text) == "en"


def test_document_registry_idempotency(tmp_path):
    """Registry must skip unchanged files and re-index modified files."""
    reg_file = tmp_path / "registry.json"
    registry = DocumentRegistry(reg_file)

    fname = "sample.pdf"
    h1 = "hash_v1"
    model = "all-MiniLM-L6-v2"

    # Initially should reindex
    assert registry.should_reindex(fname, h1, model, 800, 150) is True

    # Record document
    registry.record_document(
        filename=fname,
        file_hash=h1,
        total_pages=10,
        num_chunks=20,
        num_modules=5,
        programme_name="Informatik MSc",
        language="en",
        embedding_model=model,
        chunk_size=800,
        chunk_overlap=150,
    )

    # Identical request -> should NOT reindex
    assert registry.should_reindex(fname, h1, model, 800, 150) is False

    # Force reindex -> should reindex
    assert registry.should_reindex(fname, h1, model, 800, 150, force_reindex=True) is True

    # Modified file hash -> should reindex
    assert registry.should_reindex(fname, "hash_v2_changed", model, 800, 150) is True

    # Changed embedding model -> should reindex
    assert registry.should_reindex(fname, h1, "different-model", 800, 150) is True
