"""GNN-RAG orchestration: subgraph → GNN candidates → shortest paths → LLM (arXiv:2405.20139)."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from neural_rag.config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    GNN_WEIGHTS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SUBGRAPH_HOPS,
)
from neural_rag.gnn_retriever import retrieve_answer_candidates
from neural_rag.kg_data import KnowledgeGraph, kg_path_for_collection
from neural_rag.linking import link_question_entities
from neural_rag.reasoning_paths import shortest_reasoning_paths, verbalize_paths
from neural_rag.subgraph import extract_k_hop_subgraph, subgraph_edges


def load_kg(collection: str) -> KnowledgeGraph:
    path = kg_path_for_collection(Path(DATA_DIR), collection)
    if not path.is_file():
        raise FileNotFoundError(f"No KG found at {path}. Run `neural-rag ingest` first.")
    return KnowledgeGraph.load(path)


def build_retrieval_context(
    kg: KnowledgeGraph,
    question: str,
    top_k: int,
    weights_path: Path | None,
) -> tuple[str, list[int], list[int]]:
    """Returns verbalized reasoning paths, topic entity ids, candidate answer ids."""
    topics = link_question_entities(question, kg)
    node_subset = extract_k_hop_subgraph(kg, topics, SUBGRAPH_HOPS)
    if not node_subset:
        node_subset = set(topics)
    if not node_subset:
        return "(no linked entities in KG)", [], []

    src, dst, rel = subgraph_edges(kg, node_subset)
    nodes_sorted = sorted(node_subset)
    candidates = retrieve_answer_candidates(
        kg,
        question,
        nodes_sorted,
        src,
        dst,
        rel,
        top_k=top_k,
        weights_path=weights_path,
    )
    paths = shortest_reasoning_paths(kg, topics, candidates, max_paths=max(8, top_k * 4))
    verbalized = verbalize_paths(paths) if paths else "(no paths between topics and candidates)"
    return verbalized, topics, candidates


def answer(
    question: str,
    collection: str,
    top_k: int = DEFAULT_TOP_K,
    weights_path: Path | None = None,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    wp = weights_path or (Path(GNN_WEIGHTS) if GNN_WEIGHTS else None)
    kg = load_kg(collection)
    reasoning_paths, _topics, _cands = build_retrieval_context(kg, question, top_k, wp)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    system = (
        "You answer questions using only the provided KG reasoning paths. "
        "If they are insufficient, say so briefly."
    )
    user = (
        "Based on the reasoning paths, answer the question.\n\n"
        f"Reasoning Paths:\n{reasoning_paths}\n\n"
        f"Question:\n{question}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()
