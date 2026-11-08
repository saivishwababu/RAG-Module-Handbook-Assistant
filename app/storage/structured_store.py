"""SQLite structured store for validated module specifications and structured SQL queries."""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.ingestion.table_extractor import ModuleRecord

logger = logging.getLogger(__name__)


class StructuredModuleStore:
    """Manages verified academic module records in SQLite."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._get_connection() as con:
            # Check if existing table has legacy schema
            cur = con.execute("PRAGMA table_info(modules)")
            columns = {row[1] for row in cur.fetchall()}
            if columns and "semester" not in columns:
                logger.info("Upgrading legacy SQLite schema to verified schema...")
                con.execute("DROP TABLE modules")

            con.execute("""
                CREATE TABLE IF NOT EXISTS modules (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT NOT NULL,
                    code            TEXT,
                    ects            INTEGER,
                    sws             INTEGER,
                    semester        TEXT,
                    cycle           TEXT,
                    duration        TEXT,
                    responsible     TEXT,
                    lecturers       TEXT,
                    prerequisites   TEXT,
                    assessments     TEXT,
                    course_types    TEXT,
                    workload        TEXT,
                    grade           TEXT,
                    language        TEXT,
                    aims            TEXT,
                    content         TEXT,
                    literature      TEXT,
                    source          TEXT NOT NULL,
                    programme       TEXT NOT NULL,
                    page_number     INTEGER NOT NULL,
                    raw_evidence    TEXT,
                    content_hash    TEXT
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_modules_source ON modules(source)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_modules_ects ON modules(ects)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_modules_sws ON modules(sws)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_modules_code ON modules(code)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_modules_name ON modules(name)")
            con.commit()

    def delete_by_source(self, source_filename: str) -> int:
        """Safely delete all existing module records for a document."""
        with self._get_connection() as con:
            cur = con.execute("DELETE FROM modules WHERE source = ?", (source_filename,))
            deleted = cur.rowcount
            con.commit()
            return deleted

    def insert_modules(self, records: List[ModuleRecord]) -> int:
        """Insert verified module records inside a transaction."""
        if not records:
            return 0
        with self._get_connection() as con:
            rows = [
                (
                    r.name,
                    r.code,
                    r.ects,
                    r.sws,
                    r.semester,
                    r.cycle,
                    r.duration,
                    r.responsible,
                    r.lecturers,
                    r.prerequisites,
                    r.assessments,
                    r.course_types,
                    r.workload,
                    r.grade,
                    r.language,
                    r.aims,
                    r.content,
                    r.literature,
                    r.source_filename,
                    r.programme_name,
                    r.page_number,
                    r.raw_evidence,
                    r.content_hash,
                )
                for r in records
            ]
            cur = con.executemany(
                """
                INSERT INTO modules (
                    name, code, ects, sws, semester, cycle, duration,
                    responsible, lecturers, prerequisites, assessments,
                    course_types, workload, grade, language, aims,
                    content, literature, source, programme, page_number,
                    raw_evidence, content_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            con.commit()
            return cur.rowcount

    def count_modules(
        self,
        ects: Optional[int] = None,
        sws: Optional[int] = None,
        programme: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        """Accurately count modules matching criteria."""
        query = "SELECT COUNT(*) FROM modules WHERE 1=1"
        params: List[Any] = []
        if ects is not None:
            query += " AND ects = ?"
            params.append(ects)
        if sws is not None:
            query += " AND sws = ?"
            params.append(sws)
        if programme is not None:
            query += " AND (programme LIKE ? OR source LIKE ?)"
            params.extend([f"%{programme}%", f"%{programme}%"])
        if source is not None:
            query += " AND source = ?"
            params.append(source)

        with self._get_connection() as con:
            return con.execute(query, params).fetchone()[0]

    def list_modules(
        self,
        ects: Optional[int] = None,
        sws: Optional[int] = None,
        programme: Optional[str] = None,
        code: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List modules matching filters with complete validated attributes."""
        query = "SELECT * FROM modules WHERE 1=1"
        params: List[Any] = []
        if ects is not None:
            query += " AND ects = ?"
            params.append(ects)
        if sws is not None:
            query += " AND sws = ?"
            params.append(sws)
        if code is not None:
            query += " AND UPPER(code) = UPPER(?)"
            params.append(code.strip())
        if programme is not None:
            query += " AND (programme LIKE ? OR source LIKE ?)"
            params.extend([f"%{programme}%", f"%{programme}%"])

        query += " ORDER BY programme, name LIMIT ?"
        params.append(limit)

        with self._get_connection() as con:
            rows = con.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def find_module_by_code_or_name(self, query_str: str) -> List[Dict[str, Any]]:
        """Exact or substring match for module name or code."""
        query_str = query_str.strip()
        with self._get_connection() as con:
            # First try exact code
            rows = con.execute(
                "SELECT * FROM modules WHERE UPPER(code) = UPPER(?) ORDER BY programme",
                (query_str,),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]

            # Try exact name match
            rows = con.execute(
                "SELECT * FROM modules WHERE LOWER(name) = LOWER(?) ORDER BY programme",
                (query_str,),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]

            # Try prefix / substring
            rows = con.execute(
                "SELECT * FROM modules WHERE name LIKE ? OR code LIKE ? ORDER BY programme LIMIT 10",
                (f"%{query_str}%", f"%{query_str}%"),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_sources_summary(self) -> List[Dict[str, Any]]:
        """Get summary breakdown of indexed modules by source file."""
        with self._get_connection() as con:
            rows = con.execute(
                """
                SELECT source, programme, COUNT(*) as count,
                       COUNT(DISTINCT name) as unique_names,
                       GROUP_CONCAT(DISTINCT ects) as ects_values
                FROM modules
                GROUP BY source
                ORDER BY source
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def clear(self) -> None:
        """Purge all records."""
        with self._get_connection() as con:
            con.execute("DELETE FROM modules")
            con.commit()
