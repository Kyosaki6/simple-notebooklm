import json

from pydantic import ValidationError

from .config import settings
from .rag import fetch_all_chunks, format_citations, render_prompt, retrieve
from .schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, Summary

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


def _resolve_target(document, query, filters, k, retrieval_k):
    effective_filters = dict(filters or {})
    if document:
        effective_filters["filename"] = document
    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        return chunks, "query", query
    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target
    return fetch_all_chunks(filters=None), "corpus", None


def _parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # strip ```json ... ``` fences (possibly with language tag)
        lines = cleaned.split("\n")
        # drop opening fence
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        cleaned = cleaned.removesuffix("```").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # LLM often adds prose around JSON — extract first {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"LLM did not return valid JSON: {e}. Raw preview: {cleaned[:500]!r}"
                ) from e
        else:
            raise RuntimeError(
                f"LLM did not return valid JSON. Raw preview: {cleaned[:500]!r}"
            )
    if not isinstance(obj, (dict, list)):
        raise RuntimeError("Expected JSON object or array.")
    return obj


def _validate_summary_payload(payload: dict, raw: str = "") -> tuple[str, list[str]]:
    if isinstance(payload, list):
        # some LLMs return [{"summary": ...}] — accept single-element list
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            raise RuntimeError(
                f"Expected summary object, got list. Raw preview: {raw[:500]!r}"
            )
    # accept alternative keys real LLMs often use
    summary = payload.get("summary", "")
    for alt in ("summary_text", "overview", "tom_tat", "tóm_tắt", "text"):
        if not str(summary).strip() and payload.get(alt):
            summary = payload.get(alt)
    summary = str(summary or "").strip()
    key_points = payload.get("key_points")
    for alt in ("keypoints", "keyPoints", "points", "bullets", "y_chinh", "ý_chính"):
        if key_points is None and payload.get(alt) is not None:
            key_points = payload.get(alt)
    key_points = [str(x) for x in (key_points or [])]
    if not summary:
        raise RuntimeError(
            f"Empty summary produced. Payload keys: {list(payload.keys())}. "
            f"Raw preview: {raw[:500]!r}"
        )
    return summary, key_points


def _invoke_json(prompt: str, label: str, retries: int = 1):
    """Invoke LLM and parse JSON, retrying once with a stricter instruction.

    Returns (payload, raw). Raises RuntimeError with raw preview on failure
    so uvicorn logs show what the LLM actually returned.
    """
    from .llm import invoke_llm

    last_err: Exception | None = None
    current = prompt
    for attempt in range(retries + 1):
        raw = invoke_llm(current)
        try:
            payload = _parse_json(raw)
            return payload, raw
        except RuntimeError as e:
            last_err = e
            print(f"[learning:{label}] attempt {attempt + 1} parse failed: {e}")
            print(f"[learning:{label}] raw preview: {raw[:1000]!r}")
        current = (
            prompt
            + "\n\nCHỈ trả về JSON hợp lệ, không thêm lời dẫn, "
            "không dùng markdown, không để chuỗi rỗng."
        )
    raise RuntimeError(f"LLM did not return valid JSON for {label}: {last_err}")


def _validate_items(payload, key, model_class, dedup_field, label, valid_markers, raw: str = ""):
    raw_items = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        raise RuntimeError(
            f"No valid {label} produced. Payload keys: "
            f"{list(payload.keys()) if isinstance(payload, dict) else type(payload)}. "
            f"Raw preview: {raw[:500]!r}"
        )
    items, seen = [], set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))
    if not items:
        raise RuntimeError(f"No valid {label} produced.")
    return items


def summarize(document=None, query=None, filters=None, k=None) -> Summary:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.summarize_retrieval_k
    )
    if not chunks:
        return Summary(scope=scope, target=target, summary="Không có nội dung phù hợp.", key_points=[])
    if len(chunks) <= settings.summarize_batch_size:
        prompt = render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)
        payload, raw = _invoke_json(prompt, "summarize-single")
        summary_text, key_points = _validate_summary_payload(payload, raw)
    else:
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            batch = chunks[start: start + settings.summarize_batch_size]
            payload, raw = _invoke_json(
                render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch),
                f"summarize-map-{start // settings.summarize_batch_size}",
            )
            summary_text, key_points = _validate_summary_payload(payload, raw)
            partials.append({"summary": summary_text, "key_points": key_points})
        payload, raw = _invoke_json(
            render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials),
            "summarize-reduce",
        )
        summary_text, key_points = _validate_summary_payload(payload, raw)
    return Summary(scope=scope, target=target, summary=summary_text, key_points=key_points,
                   citations=format_citations(chunks), chunks=chunks)


def generate_quiz(document=None, query=None, filters=None, count=None, k=None) -> QuizSet:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)
    payload, raw = _invoke_json(prompt, "quiz")
    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers, raw)
    return QuizSet(scope=scope, target=target, items=items, chunks=chunks,
                   citations=format_citations(chunks))


def generate_flashcards(document=None, query=None, filters=None, count=None, k=None) -> FlashcardSet:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    n = count or settings.flashcards_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload, raw = _invoke_json(prompt, "flashcards")
    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers, raw)
    return FlashcardSet(scope=scope, target=target, cards=cards, chunks=chunks,
                        citations=format_citations(chunks))
