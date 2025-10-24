from __future__ import annotations

from typing import List, Tuple

from langchain.prompts import PromptTemplate

from .llm import LocalLLM
from .retriever import get_retriever

BASE_TEMPLATE = """You are a helpful assistant. Use the sources to answer the user's question.
Cite sources as a bullet list at the end like:
Sources:
- <filename>

Question: {question}

Relevant snippets:
{context}

Answer concisely and factually.
"""


def _format_context(docs) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{src}] {d.page_content[:600]}")
    return "\n".join(lines)


def ask(
    question: str,
    persist_dir: str,
    embed_model_name: str,
    model_name: str,
    k: int = 4,
    mmr: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> Tuple[str, List[str]]:
    retriever = get_retriever(persist_dir, embed_model_name=embed_model_name, k=k, mmr=mmr)
    docs = retriever.get_relevant_documents(question)
    context = _format_context(docs)
    sources = [d.metadata.get("source", "unknown") for d in docs]

    prompt = PromptTemplate.from_template(BASE_TEMPLATE).format(question=question, context=context)
    llm = LocalLLM(model=model_name, temperature=temperature, top_p=top_p, num_predict=max_tokens)
    answer = llm.generate(prompt)
    return answer, sources
