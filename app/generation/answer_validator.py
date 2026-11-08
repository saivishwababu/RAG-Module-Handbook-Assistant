"""Validate citations in generated answers against retrieved evidence."""
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List
from app.generation.prompt_builder import ABSTENTION_MESSAGE

CITATION = re.compile(r"\[Source:\s*([^,\]]+),\s*page\s+(\d+)\]", re.I)

@dataclass
class ValidationResult:
    is_valid: bool
    is_abstained: bool = False
    validated_citations: List[Dict[str, Any]] = field(default_factory=list)
    invalid_citations: List[Dict[str, Any]] = field(default_factory=list)

class AnswerValidator:
    def validate(self, answer: str, retrieved: List[Dict[str, Any]]) -> ValidationResult:
        if answer.strip() == ABSTENTION_MESSAGE:
            return ValidationResult(True, True)
        valid_keys = {(str(x.get("source_filename", x.get("source", x.get("filename", "")))).lower(), int(x.get("page_number", 0))) for x in retrieved}
        ok, bad = [], []
        for filename, page in CITATION.findall(answer):
            item = {"filename": filename.strip(), "page_number": int(page)}
            (ok if (item["filename"].lower(), item["page_number"]) in valid_keys else bad).append(item)
        return ValidationResult(bool(ok) and not bad, False, ok, bad)
