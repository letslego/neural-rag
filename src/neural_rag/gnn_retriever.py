"""
Question-conditioned GNN retrieval as in GNN-RAG (Eq. 1–3, arXiv:2405.20139).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_rag.config import GNN_HIDDEN, GNN_LAYERS
from neural_rag.kg_data import KnowledgeGraph
from neural_rag.lm import encode_question, encode_texts


class QuestionConditionedGNN(nn.Module):
    """Sum aggregation with ω(q,r) implemented as an MLP on [rel_h || q_h]."""

    def __init__(self, lm_dim: int, hidden: int = GNN_HIDDEN, num_layers: int = GNN_LAYERS):
        super().__init__()
        self.num_layers = num_layers
        self.rel_proj = nn.Linear(lm_dim, hidden)
        self.q_proj = nn.Linear(lm_dim, hidden)
        self.omega_mlps = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                for _ in range(num_layers)
            ]
        )
        self.msg_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.self_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.score_head = nn.Linear(hidden, 2)
        self._init_h = nn.Linear(lm_dim, hidden)

    def forward(
        self,
        h: torch.Tensor,
        q_lm: torch.Tensor,
        rel_lm: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: [N, H] initial node states.
        q_lm: [D], rel_lm: [R, D].
        Edge tensors index into local nodes / relation ids.
        Returns logits [N, 2].
        """
        q_h = self.q_proj(q_lm)
        rel_h = self.rel_proj(rel_lm)
        num_nodes = h.size(0)

        for layer_idx in range(self.num_layers):
            omega_in = torch.cat([rel_h[edge_rel], q_h.unsqueeze(0).expand(rel_h[edge_rel].size(0), -1)], dim=-1)
            omega_e = torch.sigmoid(self.omega_mlps[layer_idx](omega_in)).squeeze(-1)
            msg_src = self.msg_linears[layer_idx](h[edge_src])
            weighted = omega_e.unsqueeze(-1) * msg_src
            agg = torch.zeros(num_nodes, weighted.size(1), device=h.device, dtype=h.dtype)
            agg.index_add_(0, edge_dst, weighted)
            h = self.layer_norms[layer_idx](h + agg + self.self_linears[layer_idx](h))
            h = F.relu(h)

        return self.score_head(h)


def init_node_features(ent_lm: torch.Tensor, module: QuestionConditionedGNN) -> torch.Tensor:
    return module._init_h(ent_lm)


def retrieve_answer_candidates(
    kg: KnowledgeGraph,
    question: str,
    nodes_sorted: list[int],
    src: list[int],
    dst: list[int],
    rel: list[int],
    top_k: int,
    weights_path: Path | None,
    device: torch.device | None = None,
) -> list[int]:
    """
    Run GNN on the subgraph induced by nodes_sorted; return global entity ids with highest
    P(answer | GNN) among nodes in the subgraph.
    """
    dev = device or torch.device("cpu")
    if not nodes_sorted:
        return []

    old_to_new = {o: i for i, o in enumerate(nodes_sorted)}
    n_local = len(nodes_sorted)

    if not src:
        return nodes_sorted[: min(top_k, len(nodes_sorted))]

    es = torch.tensor([old_to_new[s] for s in src], dtype=torch.long, device=dev)
    ed = torch.tensor([old_to_new[d] for d in dst], dtype=torch.long, device=dev)
    er = torch.tensor(rel, dtype=torch.long, device=dev)

    ent_texts = [kg.id_entity[i] for i in nodes_sorted]
    ent_lm = encode_texts(ent_texts).to(dev)
    rel_lm = encode_texts(kg.id_relation).to(dev)
    q_lm = encode_question(question).to(dev)
    lm_dim = int(ent_lm.shape[1])

    model = QuestionConditionedGNN(lm_dim=lm_dim).to(dev)
    if weights_path and weights_path.is_file():
        try:
            state = torch.load(weights_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=dev)
        model.load_state_dict(state, strict=False)

    model.eval()
    with torch.no_grad():
        h = init_node_features(ent_lm, model)
        logits = model(h, q_lm, rel_lm, es, ed, er)
        probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu()

    order_local = torch.argsort(probs, descending=True).tolist()
    global_ids = [nodes_sorted[i] for i in order_local[:top_k]]
    return global_ids
