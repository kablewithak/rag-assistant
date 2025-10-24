from __future__ import annotations

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


def get_retriever(persist_dir: str, embed_model_name: str, k: int = 4, mmr: bool = True):
    embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k})
    retriever.search_type = "mmr" if mmr else "similarity"
    return retriever
