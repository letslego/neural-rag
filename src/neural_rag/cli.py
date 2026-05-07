from pathlib import Path

import typer

from neural_rag.config import DATA_DIR, DEFAULT_TOP_K, GNN_WEIGHTS
from neural_rag.gnn_rag import answer
from neural_rag.kg_data import KnowledgeGraph, kg_path_for_collection, load_triplets_file

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True, help="Triplet file or directory"),
    collection: str = typer.Option("default", help="Name for this knowledge graph"),
):
    """Load triplets (head, relation, tail) and persist a KG for GNN-RAG."""
    if path.is_dir():
        files = sorted(path.glob("**/*.jsonl")) + sorted(path.glob("**/*.csv"))
        triplets: list[tuple[str, str, str]] = []
        for f in files:
            triplets.extend(load_triplets_file(f))
    else:
        triplets = load_triplets_file(path)
    if not triplets:
        typer.echo(f"No triplets found in {path}", err=True)
        raise typer.Exit(code=1)

    kg = KnowledgeGraph()
    for h, r, t in triplets:
        kg.add_triplet(h, r, t)

    out = kg_path_for_collection(Path(DATA_DIR), collection)
    kg.save(out)
    typer.echo(
        f"Saved KG {collection!r}: {len(kg.id_entity)} entities, "
        f"{len(kg.id_relation)} relations, {len(kg.triplets)} triplets → {out}"
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
    collection: str = typer.Option("default", help="KG name from ingest"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Top GNN answer candidates for path extraction"),
    weights: Path | None = typer.Option(
        None,
        exists=False,
        help="Optional PyTorch state_dict for QuestionConditionedGNN",
    ),
):
    """GNN retrieval over a subgraph, shortest-path verbalization, then LLM answer."""
    wp = weights if weights is not None else (Path(GNN_WEIGHTS) if GNN_WEIGHTS else None)
    result = answer(question, collection, top_k=top_k, weights_path=wp)
    typer.echo(result)


if __name__ == "__main__":
    app()
