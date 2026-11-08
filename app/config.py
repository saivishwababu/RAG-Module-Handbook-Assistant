"""Environment-backed application configuration."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    db_dir: Path = Path(os.getenv("CHROMA_DB_PATH", str(ROOT / "chroma_db")))
    registry_path: Path = Path(os.getenv("DOCUMENT_REGISTRY_PATH", str(ROOT / "chroma_db" / "registry.json")))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "6"))

settings = Settings()
