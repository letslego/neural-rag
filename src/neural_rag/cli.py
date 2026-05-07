from pathlib import Path

import typer

from neural_rag.config import DEFAULT_TOP_K
from neural_rag.rag import answer
from neural_rag.store import ingest_files

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    collection: str = typer.Option("default", help="Chroma collection name"),
    pattern: str = typer.Option("**/*.txt", help="Glob relative to path"),
    chunk_size: int = typer.Option(800, help="Max characters per chunk"),
):
    """Index .txt files under PATH into the vector store."""
    files = sorted(path.glob(pattern))
    if not files:
        typer.echo(f"No files matched {pattern!r} under {path}", err=True)
        raise typer.Exit(code=1)
    n = ingest_files(files, collection, chunk_size=chunk_size)
    typer.echo(f"Ingested {n} chunks into collection {collection!r}.")


@app.command()
def ask(
    question: str = typer.Argument(...),
    collection: str = typer.Option("default", help="Chroma collection name"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Number of chunks to retrieve"),
):
    """Retrieve context and answer with the configured LLM."""
    result = answer(question, collection, top_k=top_k)
    typer.echo(result)


if __name__ == "__main__":
    app()
