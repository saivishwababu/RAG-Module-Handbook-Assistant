"""Document-level and page-level metadata extraction."""

from pathlib import Path
from typing import Dict, Any

PROGRAMME_MAPPING = {
    "embedded-systems.pdf": ("Embedded Systems MSc", "en"),
    "entrepreneurial-cybersecurity.pdf": ("Entrepreneurial Cybersecurity MSc", "de"),
    "informatik.pdf": ("Informatik MSc", "en"),
    "mathematics-and-computer-science.pdf": ("Mathematics and Computer Science MSc", "en"),
    "medieninformatik.pdf": ("Medieninformatik MSc", "en"),
    "visual_computing.pdf": ("Visual Computing MSc", "en"),
}


def extract_programme_info(filename: str, first_page_text: str = "") -> tuple[str, str]:
    """Return (programme_name, primary_language) for a given document."""
    fname_lower = Path(filename).name.lower()
    if fname_lower in PROGRAMME_MAPPING:
        return PROGRAMME_MAPPING[fname_lower]

    # Inferred from cover page text
    text_lower = first_page_text.lower()
    if "entrepreneurial cybersecurity" in text_lower:
        return "Entrepreneurial Cybersecurity MSc", "de"
    elif "embedded systems" in text_lower:
        return "Embedded Systems MSc", "en"
    elif "medieninformatik" in text_lower:
        return "Medieninformatik MSc", "en"
    elif "mathematics and computer science" in text_lower:
        return "Mathematics and Computer Science MSc", "en"
    elif "visual computing" in text_lower:
        return "Visual Computing MSc", "en"
    elif "informatik" in text_lower:
        return "Informatik MSc", "en"

    # Default to cleaned filename without extension
    clean_name = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    return f"{clean_name} MSc", "en"
