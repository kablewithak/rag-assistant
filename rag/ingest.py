from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


def _read_markdown_files(kb_dir: str) -> List[tuple[str, str]]:
    paths = sorted(Path(kb_dir).glob("*.md"))
    docs: List[tuple[str, str]] = []
    for p in paths:
        docs.append((p.name, p.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return docs


def build_index(
    kb_dir: str,
    persist_dir: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> Chroma:
    docs = _read_markdown_files(kb_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    texts, metadatas = [], []
    for title, text in docs:
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metadatas.append({"source": title})

    embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
    vectordb = Chroma.from_texts(
        texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb
