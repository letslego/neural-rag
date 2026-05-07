from __future__ import annotations

import re

from neural_rag.kg_data import KnowledgeGraph


def link_question_entities(question: str, kg: KnowledgeGraph, max_entities: int = 8) -> list[int]:
    """
    Weak supervision–style entity linking: rank KG entities by token overlap with the question.
    The paper uses stronger linked entities (e.g., off-the-shelf EL); this keeps the demo self-contained.
    """
    q = question.lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    scores: list[tuple[float, int]] = []
    for eid, name in enumerate(kg.id_entity):
        n = name.lower()
        if len(n) < 2:
            continue
        if n in q:
            scores.append((10.0 + len(n), eid))
            continue
        nt = set(re.findall(r"[a-z0-9]+", n))
        if not nt:
            continue
        overlap = len(q_tokens & nt)
        if overlap:
            scores.append((float(overlap) + 0.01 * len(nt), eid))
    scores.sort(reverse=True)
    seen: set[int] = set()
    out: list[int] = []
    for _, eid in scores:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
        if len(out) >= max_entities:
            break
    if not out and kg.id_entity:
        out.append(0)
    return out
