# Web Nâng Cao (Advanced Web) — HK1

**Sinh viên:** Huỳnh Quang Thịnh — **MSSV:** 24100847

Bài thực hành môn Web Nâng Cao: ứng dụng NLP với Streamlit, hệ RAG
kiểu NotebookLM, và bài tập PHP buổi 01.

## Nội dung repo

```text
.
├── app.py                  # Project 1.1: Dịch văn bản + Sửa lỗi chính tả (Streamlit)
├── requirements.txt        # Deps cho app.py
├── notebooklm/             # Project: Simple NotebookLM (RAG: hỏi đáp, tóm tắt, quiz, flashcards trên PDF)
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── .env.example        # Mẫu cấu hình (copy thành .env)
│   ├── data/               # Bỏ PDF vào đây rồi index
│   └── src/
│       ├── interfaces/     # CLI (cli.py), API FastAPI (api.py), UI Streamlit (ui.py)
│       ├── evaluation/     # Script đánh giá chunking / reranking
│       └── ...             # config, indexing, rag, store, llm, ...
└── xampp/htdocs/buoi01/    # Bài PHP buổi 01 (index, text, bang, bmi, xeploai)
```

Tài liệu đề bài (PDF) nằm ở thư mục gốc, không bắt buộc khi chạy code.

## Yêu cầu

- Python 3.10+ (khuyên 3.11/3.12)
- PHP 8+ + XAMPP/Apache (chỉ cho `xampp/htdocs/buoi01`)
- Git

## 1. NLP Applications — `app.py`

Hai tab: **Dịch văn bản** (auto-detect + GoogleTranslator, fallback MyMemory,
cache + retry khi bị 429) và **Sửa lỗi chính tả** (pyspellchecker, hỗ trợ
`en, es, fr, pt, de, ru, ar, eu, lv, nl`).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

> Nhập tối thiểu 3 ký tự. Nếu Google báo `Too Many Requests`, đợi 10–20s
> rồi bấm lại — kết quả đã được cache theo giờ.

## 2. Simple NotebookLM — `notebooklm/`

Hỏi đáp + tóm tắt + quiz + flashcards trên PDF cá nhân, kiến trúc RAG
(Qdrant + sentence-transformers, LLM cắm qua provider).

```bash
cd notebooklm
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

Đặt `RAG_LLM_PROVIDER=gemini` (+ `GOOGLE_API_KEY`) hoặc `vllm` để dùng LLM
thật. Mặc định `echo` là LLM giả để demo offline. Xem thêm
[`notebooklm/README.md`](notebooklm/README.md).

Đánh giá chunking / reranking:

```bash
python -m src.evaluation.run_chunking --benchmark src/evaluation/benchmark_rag.csv --out outputs/chunking
python -m src.evaluation.run_reranking --benchmark src/evaluation/benchmark_rag.csv --out outputs/reranking.json
```

## 3. PHP buổi 01 — `xampp/htdocs/buoi01/`

Copy `xampp/htdocs/buoi01/` vào `htdocs` của XAMPP rồi mở:

```text
http://localhost/buoi01/index.php
```

Các file: `index.php`, `text.php`, `bang.php`, `bmi.php`, `xeploai.php`.

## Vì sao `git status` gọn nhẹ

`.gitignore` ở thư mục gốc đã loại các thứ không nên upload GitHub:

- `venv/`, `.venv/`, `__pycache__/`, `*.pyc`
- `.env`, `secrets.toml`, key/pem
- `notebooklm/storage/`, `notebooklm/outputs/`, `*.sqlite`, `*.log`
  (index Qdrant, outputs đánh giá — chạy lại là tái tạo được)
- `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`, file lock LibreOffice

> `venv` nặng hàng trăm MB — **không commit**. Người clone chỉ cần
> `pip install -r requirements.txt` là đủ.

## Đưa lên GitHub

```bash
cd "Advance Web"
git init
git add .gitignore README.md requirements.txt app.py notebooklm xampp
git status              # kiểm tra: không có venv/, .env, storage/, outputs/
git commit -m "Web nang cao: NLP Streamlit + NotebookLM RAG + PHP buoi01"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Nếu lỡ commit nhầm file nhạy cảm/nặng (`venv/`, `.env`, `storage/`):

```bash
git rm -r --cached venv notebooklm/storage notebooklm/outputs .env
git commit -m "Remove ignored files"
```

## Ghi chú

- `notebooklm/.env` chứa API key — không bao giờ commit, chỉ commit `.env.example`.
- PDF trong `notebooklm/data/` (~MB) vẫn được giữ để demo; nếu repo nặng,
  có thể xóa PDF mẫu trước khi push.
