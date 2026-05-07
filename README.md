# Neural RAG

Implementation of **retrieval-augmented generation** using **neural (dense) retrieval**: documents are embedded with a transformer encoder and retrieved by vector similarity, then fed to an LLM for grounded answers.

## What this repo contains

- **Embedding index**: [Chroma](https://www.trychroma.com/) + [sentence-transformers](https://www.sbert.net/) (default model: `all-MiniLM-L6-v2`).
- **RAG loop**: retrieve top-k chunks → build prompt → call OpenAI-compatible chat completion API.

## Setup

```bash
cd neural-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or another provider that exposes an OpenAI-compatible API).

## Usage

Index text files under a directory (recursive):

```bash
neural-rag ingest ./docs --collection my_kb
```

Ask a question:

```bash
neural-rag ask "What is the refund policy?" --collection my_kb
```

## Roadmap

- Hybrid retrieval (BM25 + dense) and re-ranking
- Evaluation harness (recall@k, answer faithfulness)
- Async batch ingestion and streaming answers

## License

MIT
