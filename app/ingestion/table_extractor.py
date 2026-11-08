"""Extraction and structured parsing of module descriptions from table layouts."""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.ingestion.text_extractor import clean_text_spacing

logger = logging.getLogger(__name__)


@dataclass
class ModuleRecord:
    name: str
    code: Optional[str]
    ects: Optional[int]
    sws: Optional[int]
    semester: Optional[str]
    cycle: Optional[str]
    duration: Optional[str]
    responsible: Optional[str]
    lecturers: Optional[str]
    prerequisites: Optional[str]
    assessments: Optional[str]
    course_types: Optional[str]
    workload: Optional[str]
    grade: Optional[str]
    language: Optional[str]
    aims: Optional[str]
    content: Optional[str]
    literature: Optional[str]
    source_filename: str
    programme_name: str
    page_number: int
    raw_evidence: str
    content_hash: str = ""


FIELD_REGEX_PATTERNS = [
    (
        "responsible",
        r"(?:responsible|Modulverantwortliche/r)\s+(.*?)(?=(?:lecturers|Dozent/inn/en|entrancerequirements|Zulassungsvoraussetzungen)|$)",
    ),
    (
        "lecturers",
        r"(?:lecturers|Dozent/inn/en)\s+(.*?)(?=(?:entrancerequirements|Zulassungsvoraussetzungen|assessments/exams|Leistungskontrollen/Prüfungen)|$)",
    ),
    (
        "prerequisites",
        r"(?:entrancerequirements|Zulassungsvoraussetzungen)\s+(.*?)(?=(?:assessments/exams|Leistungskontrollen/Prüfungen|coursetypes/weeklyhours|Lehrveranstaltungen/SWS)|$)",
    ),
    (
        "assessments",
        r"(?:assessments/exams|Leistungskontrollen/Prüfungen)\s+(.*?)(?=(?:coursetypes/weeklyhours|Lehrveranstaltungen/SWS|totalworkload|Arbeitsaufwand)|$)",
    ),
    (
        "course_types",
        r"(?:coursetypes/weeklyhours|Lehrveranstaltungen/SWS)\s+(.*?)(?=(?:totalworkload|Arbeitsaufwand|grade|Modulnote)|$)",
    ),
    (
        "workload",
        r"(?:totalworkload|Arbeitsaufwand)\s+(.*?)(?=(?:grade|Modulnote|language|Sprache)|$)",
    ),
    (
        "grade",
        r"(?:grade|Modulnote)\s+(.*?)(?=(?:language|Sprache|aims/competencestobedeveloped|Lernziele/Kompetenzen)|$)",
    ),
    (
        "language",
        r"(?:language|Sprache)\s+(.*?)(?=(?:aims/competencestobedeveloped|Lernziele/Kompetenzen|content|Inhalt)|$)",
    ),
    (
        "aims",
        r"(?:aims/competencestobedeveloped|Lernziele/Kompetenzen)\s+(.*?)(?=(?:content|Inhalt|literature&reading|Literatur)|$)",
    ),
    (
        "content",
        r"(?:content|Inhalt)\s+(.*?)(?=(?:literature&reading|Literatur)|$)",
    ),
    (
        "literature",
        r"(?:literature&reading|Literatur)\s+(.*?)$",
    ),
]


def split_module_name_and_code(header_str: str) -> tuple[str, Optional[str]]:
    """Split header string into cleaned module name and optional module code."""
    header_str = header_str.strip().replace("\n", " ")
    # Look for module code at the end (e.g. 'AlgorithmsandDataStructures AlgoDat', 'Computer Graphics CG')
    m = re.search(r"^(.*?)\s+([A-Z][A-Za-z0-9_\-\+]{1,10})$", header_str)
    if m:
        raw_name = m.group(1).strip()
        code = m.group(2).strip()
        # Avoid false positives like degree levels
        if code.lower() in ("msc", "bsc", "ii", "iii", "iv", "v", "phd"):
            return clean_text_spacing(header_str), None
        return clean_text_spacing(raw_name), code
    return clean_text_spacing(header_str), None


def parse_module_stats(stats_text: str, workload_text: str = "") -> Dict[str, Any]:
    """
    Extract SWS, ECTS, semester, cycle, and duration from table stats text
    and cross-verify with workload text.
    """
    res: Dict[str, Any] = {
        "sws": None,
        "ects": None,
        "semester": None,
        "cycle": None,
        "duration": None,
    }

    # Cross-check ECTS from workload text if available (e.g., '= 270 h (= 9 ECTS)', '9 ECTS', '15 ECTS')
    workload_ects = None
    if workload_text:
        m_wk = re.search(r"(\d{1,2})\s*(?:ECTS|CP|Leistungspunkte)", workload_text, re.IGNORECASE)
        if m_wk:
            workload_ects = int(m_wk.group(1))

    # Search for stats numbers
    lines = [l.strip() for l in stats_text.splitlines() if l.strip()]
    for line in lines:
        # Check for English or German stats row
        # English: '1-3 4 6 9' (semester, std.semester, sws, ects)
        # German: '2 / 3 / 4 4 je 10 je 15'
        if re.search(r"je\s+(\d+)\s+je\s+(\d+)", line):
            m = re.search(r"je\s+(\d+)\s+je\s+(\d+)", line)
            res["sws"] = int(m.group(1))
            res["ects"] = int(m.group(2))
        elif re.search(r"(\d+(?:-\d+)?)\s+(\d+)\s+(\d+)\s+(\d+)", line):
            m = re.search(r"(\d+(?:-\d+)?)\s+(\d+)\s+(\d+)\s+(\d+)", line)
            res["semester"] = m.group(1)
            res["sws"] = int(m.group(3))
            res["ects"] = int(m.group(4))

        # Check for cycle / duration keywords
        line_clean = clean_text_spacing(line)
        if any(w in line_clean.lower() for w in ("every", "semester", "jährlich", "turnus")):
            if "semester" in line_clean.lower() and not res["duration"]:
                res["duration"] = line_clean
            if ("year" in line_clean.lower() or "jährlich" in line_clean.lower() or "annual" in line_clean.lower()) and not res["cycle"]:
                res["cycle"] = line_clean

    # Fallback / cross-validation: if stats row missed ECTS, take workload ECTS
    if res["ects"] is None and workload_ects is not None:
        res["ects"] = workload_ects

    # Cross-check SWS from stats text if still None
    if res["sws"] is None:
        m_sws = re.search(r"(\d{1,2})\s*SWS", stats_text, re.IGNORECASE)
        if m_sws:
            res["sws"] = int(m_sws.group(1))

    return res


def extract_modules_from_tables(
    tables: List[List[List[Optional[str]]]],
    source_filename: str,
    programme_name: str,
    page_number: int,
) -> List[ModuleRecord]:
    """Parse module records from table cells on a PDF page."""
    records: List[ModuleRecord] = []

    for table in tables:
        if not table or len(table) < 2:
            continue

        # Look for the module header row (usually table[0])
        # and stats row (usually table[1])
        row0_cells = [c.strip() for c in table[0] if c and c.strip()]
        if not row0_cells:
            continue

        header_text = row0_cells[0]
        # Skip non-module tables (e.g. general overview, table of contents)
        stats_cell_text = ""
        for row in table[1:3]:
            for cell in row:
                if cell and ("ECTS" in cell or "SWS" in cell or "Leistungspunkte" in cell or "CP" in cell):
                    stats_cell_text += " " + cell

        if not stats_cell_text:
            continue

        # Extract name and code
        name, code = split_module_name_and_code(header_text)
        if not name or len(name) < 3:
            continue

        # Collect detailed field content from remaining table rows
        body_cell_text = ""
        for row in table[1:]:
            for cell in row:
                if cell:
                    body_cell_text += "\n" + cell

        fields: Dict[str, str] = {}
        for key, pattern in FIELD_REGEX_PATTERNS:
            m = re.search(pattern, body_cell_text, re.DOTALL | re.IGNORECASE)
            if m:
                fields[key] = clean_text_spacing(m.group(1))

        # Parse stats
        stats = parse_module_stats(stats_cell_text, fields.get("workload", ""))

        record = ModuleRecord(
            name=name,
            code=code,
            ects=stats.get("ects"),
            sws=stats.get("sws"),
            semester=stats.get("semester"),
            cycle=stats.get("cycle"),
            duration=stats.get("duration"),
            responsible=fields.get("responsible"),
            lecturers=fields.get("lecturers"),
            prerequisites=fields.get("prerequisites"),
            assessments=fields.get("assessments"),
            course_types=fields.get("course_types"),
            workload=fields.get("workload"),
            grade=fields.get("grade"),
            language=fields.get("language"),
            aims=fields.get("aims"),
            content=fields.get("content"),
            literature=fields.get("literature"),
            source_filename=source_filename,
            programme_name=programme_name,
            page_number=page_number,
            raw_evidence=clean_text_spacing(body_cell_text),
        )
        records.append(record)

    return records
