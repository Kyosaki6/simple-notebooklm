import json
from pathlib import Path

import typer

from ..export import export
from ..filters import filters_to_dict
from ..indexing import ingest as ingest_data_dir
from ..learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from ..rag import answer, retrieve

app = typer.Typer()


def _parse_filters(filters: str | None):
    if not filters:
        return None
    try:
        return json.loads(filters)
    except json.JSONDecodeError:
        # shorthand "file.pdf:3" -> filename+page
        if ":" in filters:
            fn, pg = filters.rsplit(":", 1)
            try:
                return {"filename": fn.strip(), "page": int(pg)}
            except ValueError:
                pass
        return {"filename": filters}


def _emit(result, output, fmt):
    if output:
        path = export(result, fmt=fmt, output=Path(output))
        typer.echo(f"Saved to {path}")
    else:
        typer.echo(export(result, fmt=fmt))


def _print_answer(text):
    typer.echo(text)


def _print_sources(chunks):
    for i, c in enumerate(chunks, start=1):
        typer.echo(f"[S{i}] {c.metadata.filename} p.{c.metadata.page}: {c.text[:200]}...")


@app.command()
def ingest(recreate: bool = False):
    count = ingest_data_dir(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")


@app.command()
def ask(question: str, k: int | None = None, filters: str | None = None):
    result = answer(question, k=k, filters=_parse_filters(filters))
    _print_answer(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(question: str, k: int | None = None, filters: str | None = None,
                    as_json: bool = False):
    chunks = retrieve(question, k=k, filters=_parse_filters(filters))
    typer.echo(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))


@app.command("summarize")
def summarize(document: str | None = None, query: str | None = None,
              filters: str | None = None, k: int | None = None,
              output: str | None = None, fmt: str = "text"):
    result = summarize_learning(document=document, query=query,
                                filters=_parse_filters(filters), k=k)
    _emit(result, output, fmt)


@app.command("quiz")
def quiz(document: str | None = None, query: str | None = None,
         filters: str | None = None, count: int | None = None,
         k: int | None = None, output: str | None = None, fmt: str = "text"):
    result = generate_quiz(document=document, query=query, filters=_parse_filters(filters),
                           count=count, k=k)
    _emit(result, output, fmt)


@app.command("flashcards")
def flashcards(document: str | None = None, query: str | None = None,
               filters: str | None = None, count: int | None = None,
               k: int | None = None, output: str | None = None, fmt: str = "text"):
    result = generate_flashcards(document=document, query=query,
                                 filters=_parse_filters(filters), count=count, k=k)
    _emit(result, output, fmt)


if __name__ == "__main__":
    app()
