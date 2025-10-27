from __future__ import annotations

import typer

from .config import SETTINGS
from .eval_harness import run_eval
from .ingest import build_index
from .pipeline import ask
from .sweep import sweep as run_sweep

app = typer.Typer(add_completion=False)


@app.command()
def ingest(
    kb_dir: str = typer.Option("kb", help="Directory with .md files"),
    persist_dir: str = typer.Option(SETTINGS.chroma_dir, help="Chroma persistence directory"),
    chunk_size: int = typer.Option(SETTINGS.chunk_size, help="Chunk size"),
    chunk_overlap: int = typer.Option(SETTINGS.chunk_overlap, help="Chunk overlap"),
    embed_model: str = typer.Option(SETTINGS.embed_model, help="Embedding model"),
) -> None:
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
) -> None:
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


@app.command(name="eval")  # expose as `eval` on the CLI
def eval_cmd(
    queries: str = typer.Option("docs/eval_queries.yaml", help="YAML with id/query/nuggets"),
    persist_dir: str = typer.Option(SETTINGS.chroma_dir, help="Chroma persistence directory"),
    k: int = typer.Option(SETTINGS.retrieval_k, help="Top-k"),
    mmr: bool = typer.Option(SETTINGS.retrieval_mmr, help="Use MMR"),
    report_path: str = typer.Option("docs/eval/report.md", help="Output markdown report"),
) -> None:
    out = run_eval(
        persist_dir=persist_dir,
        embed_model_name=SETTINGS.embed_model,
        k=k,
        mmr=mmr,
        queries_path=queries,
        report_path=report_path,
    )
    typer.echo(f"✅ Eval complete: hit_rate={out['hit_rate']:.2f} → {out['report_path']}")


@app.command(name="sweep")
def sweep_cmd(
    kb_dir: str = typer.Option("kb", help="KB directory"),
    queries: str = typer.Option("docs/eval_queries.yaml", help="YAML queries"),
    chunk_sizes: str = typer.Option("400,700,1000", help="Comma list of ints"),
    overlaps: str = typer.Option("50,100,200", help="Comma list of ints"),
    ks: str = typer.Option("3,4,6", help="Comma list of ints"),
    mmrs: str = typer.Option("true,false", help="Comma list of booleans"),
) -> None:
    def parse_ints(s: str):
        return [int(x) for x in s.split(",") if x.strip()]

    def parse_bools(s: str):
        return [
            x.strip().lower() in ("1", "true", "t", "yes", "y") for x in s.split(",") if x.strip()
        ]

    out = run_sweep(
        kb_dir=kb_dir,
        queries_path=queries,
        chunk_sizes=parse_ints(chunk_sizes),
        overlaps=parse_ints(overlaps),
        ks=parse_ints(ks),
        mmrs=parse_bools(mmrs),
    )
    best = out["best"]
    if best:
        typer.echo(
            f"✅ Sweep done. Best → chunk={best['chunk_size']} overlap={best['overlap']} "
            f"k={best['k']} mmr={best['mmr']} (hit_rate={best['hit_rate']:.2f})"
        )
    else:
        typer.echo("Sweep completed but returned no results (check queries file).")
    typer.echo("Reports: docs/eval/sweep.md, docs/eval/sweep.csv")


if __name__ == "__main__":
    app()


@app.command()
def eval(
    queries: str = typer.Option("docs/eval_queries.yaml", help="YAML with id/query/nuggets"),
    persist_dir: str = typer.Option(SETTINGS.chroma_dir, help="Chroma persistence directory"),
    k: int = typer.Option(SETTINGS.retrieval_k, help="Top-k"),
    mmr: bool = typer.Option(SETTINGS.retrieval_mmr, help="Use MMR"),
    report_path: str = typer.Option("docs/eval/report.md", help="Output markdown report"),
):
    out = run_eval(
        persist_dir=persist_dir,
        embed_model_name=SETTINGS.embed_model,
        k=k,
        mmr=mmr,
        queries_path=queries,
        report_path=report_path,
    )
    typer.echo(f"✅ Eval complete: hit_rate={out['hit_rate']:.2f} → {out['report_path']}")
