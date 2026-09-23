from sentence_transformers import CrossEncoder

from ..llm import invoke_llm
from ..rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from ..schemas import RagAnswer

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def get_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def answer_with_reranker(question: str, collection_name: str, reranker: CrossEncoder,
                         initial_k: int = 15, rerank_k: int = 5,
                         filters=None) -> RagAnswer:
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.",
        )
    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked)
    text = invoke_llm(prompt)
    return RagAnswer(question=question, answer=text.strip(),
                     citations=format_citations(reranked), chunks=reranked)


def main(benchmark: str = "src/evaluation/benchmark_rag.csv",
         collection: str | None = None, out: str = "outputs/reranking.json"):
    import json
    from pathlib import Path

    from .run_chunking import load_benchmark
    from .ragas_evaluator import run_evaluation, summary_metrics

    cases = load_benchmark(Path(benchmark))
    reranker = get_reranker()
    coll = collection
    result = run_evaluation(cases,
                            answer_fn=lambda q: answer_with_reranker(q, coll, reranker),
                            llm_provider="vllm")
    df = result.to_pandas()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary_metrics(df), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import typer

    typer.run(main)
