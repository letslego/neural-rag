"""Shortest KG paths and verbalization as in GNN-RAG §4.2 (arXiv:2405.20139)."""

from __future__ import annotations

import networkx as nx

from neural_rag.kg_data import KnowledgeGraph


def _hop_triplet(kg: KnowledgeGraph, x: int, y: int) -> tuple[str, str, str]:
    """One hop along the undirected walk, oriented as in the stored KG triplet."""
    G = kg.graph
    if G.has_edge(x, y):
        k = next(iter(G[x][y]))
        r = G[x][y][k].get("relation", "")
        return kg.id_entity[x], r, kg.id_entity[y]
    if G.has_edge(y, x):
        k = next(iter(G[y][x]))
        r = G[y][x][k].get("relation", "")
        return kg.id_entity[y], r, kg.id_entity[x]
    return kg.id_entity[x], "", kg.id_entity[y]


def shortest_reasoning_paths(
    kg: KnowledgeGraph,
    topic_entities: list[int],
    candidate_answers: list[int],
    max_paths: int = 32,
) -> list[list[tuple[str, str, str]]]:
    """
    Shortest paths between question entities and GNN-scored answer candidates (paper §4).
    """
    G = kg.graph
    if G.number_of_nodes() == 0:
        return []

    und = G.to_undirected()
    paths_out: list[list[tuple[str, str, str]]] = []

    for s in topic_entities:
        for a in candidate_answers:
            if s == a:
                continue
            try:
                node_path = nx.shortest_path(und, s, a)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            hops: list[tuple[str, str, str]] = []
            for i in range(len(node_path) - 1):
                hops.append(_hop_triplet(kg, node_path[i], node_path[i + 1]))
            if hops:
                paths_out.append(hops)
            if len(paths_out) >= max_paths:
                return paths_out
    return paths_out


def verbalize_paths(paths: list[list[tuple[str, str, str]]]) -> str:
    """Paper-style verbalization: head → relation → tail → …"""
    lines: list[str] = []
    for hops in paths:
        parts: list[str] = []
        for i, (h, r, t) in enumerate(hops):
            if i == 0:
                parts.append(f"{h} → {r} → {t}")
            else:
                parts.append(f"→ {r} → {t}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
