# Project FAQ
**What does this project do?**  
It answers questions using a local knowledge base. Documents are chunked, embedded, and indexed in Chroma; a retriever finds the most relevant chunks; an LLM composes an answer with citations.

**Can I run this offline?**  
Yes. Embeddings and vector search are local. The LLM runs locally via Ollama once models are downloaded.
