from __future__ import annotations

import typer

from .config import SETTINGS
from .ingest import build_index
from .pipeline import ask

app = typer.Typer(add_completion=False)


@app.command()
def ingest(
    kb_dir: str = typer.Option("kb", help="Directory with .md files"),
    persist_dir: str = typer.Option(SETTINGS.chroma_dir, help="Chroma persistence directory"),
    chunk_size: int = typer.Option(SETTINGS.chunk_size, help="Chunk size"),
    chunk_overlap: int = typer.Option(SETTINGS.chunk_overlap, help="Chunk overlap"),
    embed_model: str = typer.Option(SETTINGS.embed_model, help="Embedding model"),
):
    build_index(kb_dir, persist_dir, embed_model, chunk_size, chunk_overlap)
    typer.echo(f"✅ Ingest complete. Index at: {persist_dir}")


@app.command()
def askq(
    question: str = typer.Argument(..., help="Your question"),
    persist_dir: str = typer.Option(SETTINGS.chroma_dir, help="Chroma persistence directory"),
    k: int = typer.Option(SETTINGS.retrieval_k, help="Top-k results"),
    mmr: bool = typer.Option(SETTINGS.retrieval_mmr, help="Use maximal marginal relevance"),
    model: str = typer.Option(SETTINGS.ollama_model, help="Ollama model name"),
    show_sources: bool = typer.Option(True, help="Print sources at the end"),
):
    answer, sources = ask(
        question,
        persist_dir=persist_dir,
        embed_model_name=SETTINGS.embed_model,
        model_name=model,
        k=k,
        mmr=mmr,
        temperature=SETTINGS.temperature,
        top_p=SETTINGS.top_p,
        max_tokens=SETTINGS.max_tokens,
    )
    typer.echo(answer)
    if show_sources:
        typer.echo("\nSources:")
        for s in dict.fromkeys(sources):
            typer.echo(f"- {s}")


if __name__ == "__main__":
    app()
