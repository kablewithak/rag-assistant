from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    retrieval_mmr: bool = _bool(os.getenv("RETRIEVAL_MMR"), True)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))


SETTINGS = Settings()
