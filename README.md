# Simple NotebookLM (RAG Learning System)

Hỏi đáp + tóm tắt + quiz + flashcards trên PDF cá nhân, theo kiến trúc RAG
(Qdrant + sentence-transformers, LLM cắm qua provider).

## Yêu cầu

- Python 3.10+ (khuyên 3.11/3.12)

## Cấu trúc

```text
.
├── requirements.txt
├── pyproject.toml
├── .env.example        # Mẫu cấu hình (copy thành .env)
├── data/               # Bỏ PDF vào đây rồi index
├── storage/qdrant/     # Index Qdrant (tái tạo được, không commit)
├── outputs/            # Kết quả đánh giá (tái tạo được, không commit)
└── src/
    ├── interfaces/     # CLI (cli.py), API FastAPI (api.py), UI Streamlit (ui.py)
    ├── evaluation/     # Script đánh giá chunking / reranking
    └── ...             # config, indexing, rag, store, llm, ...
```

## Chạy nhanh

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # sửa RAG_LLM_PROVIDER=echo để chạy offline

# 1. Bỏ PDF vào data/
# 2. Index
python -m src.interfaces.cli ingest
# 3. Hỏi đáp
python -m src.interfaces.cli ask "LoRA là gì?"
# 4. API
uvicorn src.interfaces.api:app --reload --port 8000
# 5. UI (cần API đang chạy, terminal khác)
streamlit run src/interfaces/ui.py
```

CLI còn hỗ trợ `summarize`, `quiz`, `flashcards`, `debug-retrieval`. Chạy
`python -m src.interfaces.cli --help` để xem chi tiết.

API endpoints: `/documents`, `/upload`, `/ask`, `/summarize`, `/quiz`,
`/flashcards`, `/health`. Mở `/docs` sau khi chạy API để thử trực tiếp.

Đặt `RAG_LLM_PROVIDER=gemini` (+ `GOOGLE_API_KEY`) hoặc `vllm` / `hf_local`
để dùng LLM thật. Mặc định `echo` là LLM giả để demo offline.

## Đánh giá

```bash
python -m src.evaluation.run_chunking --benchmark src/evaluation/benchmark_rag.csv --out outputs/chunking
python -m src.evaluation.run_reranking --benchmark src/evaluation/benchmark_rag.csv --out outputs/reranking.json
```

## Ghi chú

- `.env` chứa API key — không bao giờ commit, chỉ commit `.env.example`.
- Không commit `venv/`, `__pycache__/`, `storage/qdrant/`, `outputs/`, `*.log`
  (xem `.gitignore`): index và outputs chạy lại là tái tạo được.
