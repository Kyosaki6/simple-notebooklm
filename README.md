# Simple NotebookLM (RAG Learning System)

Hỏi đáp + tóm tắt + quiz + flashcards trên PDF cá nhân, theo kiến trúc RAG.

## Chạy nhanh

```bash
pip install -r requirements.txt
cp .env.example .env  # sửa RAG_LLM_PROVIDER=echo để chạy offline

# 1. Bỏ PDF vào data/
# 2. Index
python -m src.interfaces.cli ingest
# 3. Hỏi đáp
python -m src.interfaces.cli ask "LoRA là gì?"
# 4. API
uvicorn src.interfaces.api:app --reload --port 8000
# 5. UI (cần API đang chạy)
streamlit run src/interfaces/ui.py
```

Đặt `RAG_LLM_PROVIDER=gemini` (+ `GOOGLE_API_KEY`) hoặc `vllm` để dùng LLM thật.
Mặc định `echo` là LLM giả để demo offline.

## Đánh giá

```bash
python -m src.evaluation.run_chunking --benchmark src/evaluation/benchmark_rag.csv --out outputs/chunking
python -m src.evaluation.run_reranking --benchmark src/evaluation/benchmark_rag.csv --out outputs/reranking.json
```
