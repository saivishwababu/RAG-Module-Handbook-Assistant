"""PDF text extraction with page number preservation."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class PDFPage:
    page_number: int
    text: str


@dataclass
class LoadedDocument:
    path: Path
    pages: List[PDFPage]
    total_pages: int
    file_hash: str


def compute_file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PDFLoader:
    def load_pdf(self, path: str | Path) -> LoadedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file format: {path.suffix}")
        if path.stat().st_size == 0:
            raise ValueError("PDF file is completely empty")
        try:
            with pdfplumber.open(path) as pdf:
                pages = [PDFPage(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, 1)]
        except Exception as exc:
            raise ValueError(f"Could not read PDF {path}: {exc}") from exc
        return LoadedDocument(path, pages, len(pages), compute_file_hash(path))
