# Quickstart

## CLI help
$ python -m rag.cli --help
<first ~10 lines of help output>

## Ingest
$ python -m rag.cli ingest --kb-dir kb --persist-dir .chroma --chunk-size 700 --chunk-overlap 100
✅ Ingest complete. Index at: .chroma

## Ask
$ python -m rag.cli askq "Explain Retrieval-Augmented Generation in one sentence." --persist-dir .chroma --k 4 --mmr --show-sources
<answer text>
Sources:
- project_faq.md
- glossary_ai_rag.md
...
