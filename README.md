# GNN-RAG

Python scaffold for **[GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning](https://arxiv.org/abs/2405.20139)** (Mavromatis & Karypis, 2024).

The paper’s pipeline is:

1. **Subgraph** around question-linked entities (k-hop neighborhood; configurable).
2. **GNN retrieval**: a question-conditioned GNN scores KG entities as answer candidates (ω(q,r)–style relation gating; LM embeddings from sentence-transformers).
3. **Reasoning paths**: shortest paths from topic entities to top candidates, **verbalized** as `head → relation → tail → …`.
4. **LLM**: answers using those paths only (RAG-style prompt aligned with §4.2 of the paper).

The reference implementation and trained checkpoints for **ReaRev** + WebQSP/CWQ live in the authors’ repo: [github.com/cmavro/GNN-RAG](https://github.com/cmavro/GNN-RAG). This package implements the **same retrieval → verbalize → generate** structure for custom triple files; training WebQSP-scale models is out of scope here.

## Setup

Use **Python 3.11 or 3.12** (PyTorch + sentence-transformers).

```bash
cd neural-rag
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or another OpenAI-compatible API).

## Knowledge graph format

- **JSONL**: one object per line: `{"head": "...", "relation": "...", "tail": "..."}`
- **CSV**: columns `head`, `relation`, `tail`

## Usage

Ingest triplets (stored under `data/kgs/<collection>.json`):

```bash
neural-rag ingest examples/sample.kg.jsonl --collection demo
```

Ask a question (entity linking is simple overlap with KG entity strings; use clear entity names in the KG):

```bash
neural-rag ask "How long is the refund window?" --collection demo
```

Optional **trained GNN weights** (PyTorch `state_dict` for `QuestionConditionedGNN`):

```bash
export NEURAL_RAG_GNN_WEIGHTS=/path/to/gnn.pt
# or
neural-rag ask "..." --collection demo --weights /path/to/gnn.pt
```

Without weights, the GNN uses random initialization (structure demo only). For production KGQA, train or port checkpoints per the paper / upstream code.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `NEURAL_RAG_LM_MODEL` | sentence-transformers model (default `all-MiniLM-L6-v2`) |
| `NEURAL_RAG_SUBGRAPH_HOPS` | k-hop neighborhood (default `3`) |
| `NEURAL_RAG_TOP_K` | top candidate entities from GNN for path extraction |
| `NEURAL_RAG_GNN_WEIGHTS` | path to optional GNN checkpoint |
| `NEURAL_RAG_DATA_DIR` | where KG JSON files are stored (default `data/kgs`) |

## License

MIT
