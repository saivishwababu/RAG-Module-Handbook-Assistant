"""Text extraction, normalization, spacing recovery, and language detection."""

import re
from typing import Tuple


def clean_text_spacing(text: str) -> str:
    """
    Recover lost spaces from PDF extraction without corrupting valid words.
    Fixes:
    - camelCase mergers: e.g. 'ComputerGraphics' -> 'Computer Graphics'
    - Punctuation spacing: e.g. 'Science,and' -> 'Science, and'
    - Slashes and ampersands: 'Transmission&Signal' -> 'Transmission & Signal'
    """
    if not text:
        return ""

    # Replace strange unicode spaces / zero-width spaces
    text = text.replace("\u200b", "").replace("\xa0", " ")

    # Normalize carriage returns and multiple newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 1: CamelCase (lowercase immediately followed by uppercase)
    # e.g., 'ComputerGraphics' -> 'Computer Graphics'
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Step 2: Digits adjacent to letters where appropriate
    # e.g., 'Semester1' -> 'Semester 1', '9ECTS' -> '9 ECTS'
    text = re.sub(r"(\d+)([A-Za-z]{2,})", r"\1 \2", text)
    text = re.sub(r"([A-Za-z]{2,})(\d+)", r"\1 \2", text)

    # Step 3: Fused connectives at word boundary
    # e.g., 'Algorithmsand Data' -> 'Algorithms and Data'
    # 'Introductionto' -> 'Introduction to'
    text = re.sub(r"([a-z]{3,})(and|und|with|mit|for|für)\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(to|für|von|mit|with)([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z]{3,})(to|of|in)\b", r"\1 \2", text, flags=re.IGNORECASE)

    # Step 4: Spacing around punctuation
    text = re.sub(r"\s*[/]\s*", " / ", text)
    text = re.sub(r"([a-zA-Z0-9])([,&])([a-zA-Z0-9])", r"\1 \2 \3", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    # Normalize whitespace while preserving single newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Collapse 3+ newlines to 2
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_language(text: str) -> str:
    """
    Detect document or page language (English 'en' or German 'de') based on
    Saarland University academic keywords.
    """
    if not text:
        return "en"

    text_lower = text.lower()
    german_indicators = [
        "modulhandbuch",
        "fakultät für mathematik",
        "leistungspunkte",
        "regelstudiensemester",
        "studiensem",
        "prüfungsleistung",
        "lehrveranstaltungen",
        "arbeitsaufwand",
        "lernziele",
        "modulverantwortliche",
        "dozent",
        "zulassungsvoraussetzungen",
        "dauer",
        "turnus",
        "benotet",
        "vorlesung",
        "übung",
        "praktikum",
        "seminar",
        "pflichtmodul",
        "wahlpflicht",
    ]

    english_indicators = [
        "module descriptions",
        "faculty of mathematics",
        "computer science",
        "workload",
        "entrance requirements",
        "assessments",
        "exams",
        "course types",
        "weekly hours",
        "responsible",
        "lecturers",
        "aims",
        "competences",
        "learning outcomes",
        "content",
        "literature",
        "prerequisites",
        "lecture",
        "tutorial",
    ]

    de_count = sum(1 for kw in german_indicators if kw in text_lower)
    en_count = sum(1 for kw in english_indicators if kw in text_lower)

    return "de" if de_count > en_count else "en"


def extract_page_heading(text: str) -> str:
    """Extract first non-trivial heading line from page text."""
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:4]:
        if len(line) >= 4 and not line.startswith("---") and not line.isdigit():
            # If line is header banner like 'FACULTY OF MATHEMATICS', look for next
            if "FACULTY OF" in line.upper() or "FAKULTÄT" in line.upper():
                continue
            if "MODULEDESCRIPTIONS" in line.upper() or "MODULHANDBUCH" in line.upper():
                continue
            return line
    return lines[0] if lines else ""
