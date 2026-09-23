import csv
import json
from pathlib import Path

from ..config import settings
from ..indexing import ingest
from ..rag import answer
from ..schemas import RagAnswer
from .chunking_strategies import all_strategies
from .ragas_evaluator import run_evaluation, summary_metrics


def load_benchmark(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_strategy(strategy, output_dir: Path, test_cases: list[dict]) -> dict:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)
    result_out: dict = {"strategy_id": strategy.strategy_id,
                        "chunk_count": chunk_count, "summary_metrics": {}}

    try:
        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider="vllm")
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)
    except Exception as exc:
        result_out["error"] = str(exc)
    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    return result_out


def main(benchmark: str = "src/evaluation/benchmark_rag.csv", out: str = "outputs/chunking"):
    from ..store import get_embeddings

    cases = load_benchmark(Path(benchmark))
    strategies = all_strategies(embeddings=get_embeddings())
    for s in strategies:
        print(f"Evaluating {s.strategy_id}...")
        print(_evaluate_strategy(s, Path(out), cases))


if __name__ == "__main__":
    import typer

    typer.run(main)
