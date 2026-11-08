"""Robust document chunking with strict length enforcement, section boundary preservation, and complete metadata."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from app.ingestion.text_extractor import clean_text_spacing, extract_page_heading


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_filename: str
    programme_name: str
    page_number: int
    text: str
    section_heading: str
    document_language: str
    content_hash: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of chunk content for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_oversized_paragraph(para: str, max_size: int, overlap: int) -> List[str]:
    """
    Split a paragraph that exceeds max_size along sentence or word boundaries,
    ensuring no fragment exceeds max_size.
    """
    para = para.strip()
    if len(para) <= max_size:
        return [para] if para else []

    # If single unbroken continuous token without spaces/newlines
    if " " not in para and "\n" not in para:
        frags = []
        step = max(1, max_size - overlap)
        for i in range(0, len(para), step):
            f = para[i : i + max_size]
            if f:
                frags.append(f)
            if i + max_size >= len(para):
                break
        return frags

    # Split by sentence end punctuation (.!?)
    sentences = re.split(r"(?<=[.!?])\s+", para)
    fragments: List[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If a single sentence is larger than max_size, split by words
        if len(sent) > max_size:
            if current:
                fragments.append(current)
                current = ""
            words = sent.split(" ")
            word_buf = ""
            for w in words:
                if len(word_buf) + len(w) + 1 <= max_size:
                    word_buf = f"{word_buf} {w}".strip()
                else:
                    if word_buf:
                        fragments.append(word_buf)
                        # Overlap from end of word_buf
                        overlap_words = word_buf[-overlap:].strip() if len(word_buf) > overlap else ""
                        word_buf = f"{overlap_words} {w}".strip()
                    else:
                        fragments.append(w[:max_size])
                        word_buf = w[max_size:]
            if word_buf:
                fragments.append(word_buf)
            continue

        if len(current) + len(sent) + 1 <= max_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                fragments.append(current)
                overlap_text = current[-overlap:].strip() if len(current) > overlap else ""
                current = f"{overlap_text} {sent}".strip()
            else:
                current = sent

    if current and current not in fragments:
        fragments.append(current)

    return [f for f in fragments if f.strip()]


class DocumentChunker:
    """Token-aware, size-enforced chunker with rich page-level metadata."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_chars: int = 40,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk_page(
        self,
        page_text: str,
        source_filename: str,
        programme_name: str,
        page_number: int,
        document_id: str,
        language: str = "en",
        section_heading: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Split a single page's text into strict chunks with full metadata."""
        cleaned = clean_text_spacing(page_text)
        if not cleaned or len(cleaned) < self.min_chunk_chars:
            return []

        heading = section_heading or extract_page_heading(cleaned)

        # Split on double newlines (paragraphs)
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

        # Ensure no individual paragraph exceeds chunk_size
        paras: List[str] = []
        for p in raw_paras:
            if len(p) > self.chunk_size:
                paras.extend(split_oversized_paragraph(p, self.chunk_size, self.chunk_overlap))
            else:
                paras.append(p)

        chunks: List[DocumentChunk] = []
        current = ""
        chunk_index = 0

        for para in paras:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current and len(current) >= self.min_chunk_chars:
                    c_id = f"{source_filename}__p{page_number}__c{chunk_index}"
                    c_hash = compute_text_hash(current)
                    tokens = max(1, len(current.split()))
                    chunk = DocumentChunk(
                        chunk_id=c_id,
                        document_id=document_id,
                        source_filename=source_filename,
                        programme_name=programme_name,
                        page_number=page_number,
                        text=current,
                        section_heading=heading,
                        document_language=language,
                        content_hash=c_hash,
                        token_count=tokens,
                        metadata={
                            "chunk_id": c_id,
                            "document_id": document_id,
                            "source_filename": source_filename,
                            "source": source_filename,  # backward compatibility
                            "programme_name": programme_name,
                            "page_number": page_number,
                            "section_heading": heading,
                            "language": language,
                            "content_hash": c_hash,
                            "token_count": tokens,
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Carry over overlap
                overlap_text = (
                    current[-self.chunk_overlap:].strip()
                    if len(current) > self.chunk_overlap
                    else ""
                )
                current = f"{overlap_text}\n\n{para}".strip() if overlap_text else para

        # Last chunk on page
        if current and len(current) >= self.min_chunk_chars:
            c_id = f"{source_filename}__p{page_number}__c{chunk_index}"
            c_hash = compute_text_hash(current)
            tokens = max(1, len(current.split()))
            chunk = DocumentChunk(
                chunk_id=c_id,
                document_id=document_id,
                source_filename=source_filename,
                programme_name=programme_name,
                page_number=page_number,
                text=current,
                section_heading=heading,
                document_language=language,
                content_hash=c_hash,
                token_count=tokens,
                metadata={
                    "chunk_id": c_id,
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "source": source_filename,
                    "programme_name": programme_name,
                    "page_number": page_number,
                    "section_heading": heading,
                    "language": language,
                    "content_hash": c_hash,
                    "token_count": tokens,
                },
            )
            chunks.append(chunk)

        return chunks
