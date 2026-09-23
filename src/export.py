from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .schemas import FlashcardSet, QuizSet, RagAnswer, Summary

ExportFormat = Literal["text", "md", "json"]


def _to_markdown(model: BaseModel) -> str:
    if isinstance(model, RagAnswer):
        lines = [f"## Hỏi đáp: {model.question}", "", model.answer, "", "### Nguồn"]
        for c in model.citations:
            lines.append(f"- [{c.source_marker}] {c.filename} (trang {c.page})")
        return "\n".join(lines) + "\n"
    if isinstance(model, Summary):
        lines = [f"# Tóm tắt ({model.scope}: {model.target or 'toàn bộ'})", "",
                 model.summary, "", "## Ý chính"]
        lines += [f"- {kp}" for kp in model.key_points]
        return "\n".join(lines) + "\n"
    if isinstance(model, QuizSet):
        lines = [f"# Quiz ({len(model.items)} câu)", ""]
        for i, it in enumerate(model.items, 1):
            lines.append(f"## Câu {i}: {it.question}")
            for j, opt in enumerate(it.options):
                mark = "✅" if j == it.correct_index else "  "
                lines.append(f"{mark} {chr(65+j)}. {opt}")
            lines.append(f"*Giải thích: {it.explanation}*")
            lines.append("")
        return "\n".join(lines)
    if isinstance(model, FlashcardSet):
        lines = [f"# Flashcards ({len(model.cards)} thẻ)", ""]
        for i, c in enumerate(model.cards, 1):
            lines.append(f"## Thẻ {i}: {c.front}")
            lines.append(f"**Đáp án:** {c.back}")
            if c.hint:
                lines.append(f"*Gợi ý: {c.hint}*")
            lines.append("")
        return "\n".join(lines)
    return model.model_dump_json(indent=2)


def export(model: BaseModel, *, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")
    if output is None:
        return text
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
