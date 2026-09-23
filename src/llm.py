from functools import lru_cache

from langchain_core.messages import HumanMessage

from .config import settings


def _build_hf_local():
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, dtype=torch.bfloat16)
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
    )
    text_gen.generation_config.max_new_tokens = settings.hf_max_new_tokens
    text_gen.generation_config.do_sample = settings.llm_temperature > 0
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


class _EchoLLM:
    """Offline fallback so UI/CLI/eval work without GPU or API keys."""

    def invoke(self, messages):
        import json

        prompt = messages[0].content if messages else ""
        snippet = prompt[:1200].replace("\n", " ")
        # Only inspect the instruction line — chunk content below may
        # contain words like "quiz"/"flashcard"/"items" and cause
        # misclassification (e.g. summarize returning quiz JSON).
        header = prompt.split("\n", 1)[0].lower()

        if "trắc nghiệm" in header or "quiz" in header:
            content = json.dumps({
                "items": [{
                    "question": "Demo question?",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "Echo fallback.",
                    "source_markers": ["S1"],
                }]
            }, ensure_ascii=False)
        elif "flashcard" in header:
            content = json.dumps({
                "cards": [{
                    "front": "Demo front",
                    "back": "Demo back",
                    "hint": None,
                    "topic": "demo",
                    "source_markers": ["S1"],
                }]
            }, ensure_ascii=False)
        else:
            # json.dumps escapes quotes/newlines in snippet — naive
            # f-string interpolation here previously produced invalid JSON
            # whenever chunk text contained `"` (JSONDecodeError at col ~187).
            content = json.dumps({
                "summary": ("Demo (echo LLM): no real LLM configured. "
                            "Set RAG_LLM_PROVIDER=gemini|vllm|hf_local."),
                "key_points": [f"Prompt preview: {snippet[:200]}"],
            }, ensure_ascii=False)

        class R:
            pass
        r = R()
        r.content = content
        return r


@lru_cache(maxsize=4)
def get_llm(provider=None):
    provider = provider or settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    if provider == "vllm":
        return _build_vllm()
    if provider == "echo":
        return _EchoLLM()
    raise ValueError(f"Unknown llm_provider '{provider}'")


def invoke_llm(prompt: str, provider=None) -> str:
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
