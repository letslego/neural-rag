from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class KnowledgeGraph:
    """Triplet KG with string entities/relations and a NetworkX MultiDiGraph over integer node ids."""

    entity_id: dict[str, int] = field(default_factory=dict)
    id_entity: list[str] = field(default_factory=list)
    relation_id: dict[str, int] = field(default_factory=dict)
    id_relation: list[str] = field(default_factory=list)
    triplets: list[tuple[int, int, int]] = field(default_factory=list)
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    def entity_to_id(self, name: str) -> int:
        if name not in self.entity_id:
            idx = len(self.id_entity)
            self.entity_id[name] = idx
            self.id_entity.append(name)
            self.graph.add_node(idx, label=name)
        return self.entity_id[name]

    def relation_to_id(self, name: str) -> int:
        if name not in self.relation_id:
            idx = len(self.id_relation)
            self.relation_id[name] = idx
            self.id_relation.append(name)
        return self.relation_id[name]

    def add_triplet(self, head: str, relation: str, tail: str) -> None:
        h = self.entity_to_id(head.strip())
        t = self.entity_to_id(tail.strip())
        r = self.relation_to_id(relation.strip())
        self.triplets.append((h, t, r))
        self.graph.add_edge(h, t, key=r, relation=self.id_relation[r])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_id": self.entity_id,
            "id_entity": self.id_entity,
            "relation_id": self.relation_id,
            "id_relation": self.id_relation,
            "triplets": self.triplets,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        raw = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        kg.entity_id = {k: int(v) for k, v in raw["entity_id"].items()}
        kg.id_entity = list(raw["id_entity"])
        kg.relation_id = {k: int(v) for k, v in raw["relation_id"].items()}
        kg.id_relation = list(raw["id_relation"])
        kg.triplets = [tuple(t) for t in raw["triplets"]]
        kg.graph = nx.MultiDiGraph()
        for n, name in enumerate(kg.id_entity):
            kg.graph.add_node(n, label=name)
        for h, t, r in kg.triplets:
            kg.graph.add_edge(h, t, key=r, relation=kg.id_relation[r])
        return kg


def load_triplets_file(path: Path) -> list[tuple[str, str, str]]:
    """Load (head, relation, tail) from .jsonl or .csv."""
    triplets: list[tuple[str, str, str]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            triplets.append((str(row["head"]), str(row["relation"]), str(row["tail"])))
    elif suffix == ".csv":
        import csv

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                triplets.append((row["head"].strip(), row["relation"].strip(), row["tail"].strip()))
    else:
        raise ValueError(f"Unsupported KG file type: {path}")
    return triplets


def kg_path_for_collection(data_dir: Path, collection: str) -> Path:
    return data_dir / f"{collection}.json"
