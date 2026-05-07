from __future__ import annotations

import networkx as nx

from neural_rag.kg_data import KnowledgeGraph


def extract_k_hop_subgraph(kg: KnowledgeGraph, seeds: list[int], max_hops: int) -> set[int]:
    """BFS over the directed multigraph up to max_hops from any seed."""
    if not seeds:
        return set()
    nodes: set[int] = set(seeds)
    frontier = list(seeds)
    for _ in range(max_hops):
        nxt: list[int] = []
        for u in frontier:
            for _, v, _ in kg.graph.out_edges(u, keys=True):
                if v not in nodes:
                    nodes.add(v)
                    nxt.append(v)
            for pred, succ, _ in kg.graph.in_edges(u, keys=True):
                if succ != u:
                    continue
                if pred not in nodes:
                    nodes.add(pred)
                    nxt.append(pred)
        frontier = nxt
        if not frontier:
            break
    return nodes


def subgraph_edges(kg: KnowledgeGraph, node_subset: set[int]) -> tuple[list[int], list[int], list[int]]:
    """Return edge lists (src, dst, rel_id) restricted to nodes in subset."""
    src, dst, rel = [], [], []
    for h, t, r in kg.triplets:
        if h in node_subset and t in node_subset:
            src.append(h)
            dst.append(t)
            rel.append(r)
    return src, dst, rel


def pagerank_seeds(kg: KnowledgeGraph, seeds: list[int], alpha: float = 0.85, max_nodes: int = 512) -> set[int]:
    """Personalized PageRank-style expansion (paper mentions PageRank for dense subgraphs)."""
    if not seeds:
        return set()
    G = kg.graph
    if G.number_of_nodes() == 0:
        return set(seeds)
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        pr = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=100)
    except Exception:
        return extract_k_hop_subgraph(kg, seeds, max_hops=3)
    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    keep = {n for n, _ in ranked[:max_nodes]}
    return keep | set(seeds)
