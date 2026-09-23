import sys
from pathlib import Path

import httpx
import streamlit as st

# `streamlit run src/interfaces/ui.py` executes this file as a top-level
# script (__package__ is None), so relative imports like `from ..config`
# fail with "attempted relative import with no known parent package".
# Insert the project root on sys.path and use absolute imports instead.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import settings  # noqa: E402
from src.interfaces.styles import GLOBAL_CSS  # noqa: E402

_API = settings.api_url


def _api(method, path, **kwargs):
    try:
        return httpx.request(method, f"{_API}{path}", timeout=120, **kwargs)
    except httpx.ConnectError:
        st.error(
            f"Không kết nối được API `{_API}{path}` (Connection refused). "
            "Hãy mở 1 terminal khác và chạy:\n\n"
            "`uvicorn src.interfaces.api:app --reload --port 8000`"
        )
        return None
    except httpx.TimeoutException:
        st.error(f"API `{path}` timeout quá 120s. Thử lại sau.")
        return None


def _check_api() -> bool:
    """Return True if backend reachable, else show banner and return False."""
    try:
        r = httpx.get(f"{_API}/health", timeout=5)
        if r.status_code == 200:
            return True
    except httpx.ConnectError:
        pass
    except Exception:
        pass
    st.error(
        f"Backend API không khả dụng tại `{_API}`. "
        "Chạy backend trước rồi reload trang này:\n\n"
        "`uvicorn src.interfaces.api:app --reload --port 8000`"
    )
    return False


def _sidebar():
    st.sidebar.title("📚 NotebookLM đơn giản")
    up = st.sidebar.file_uploader("Tải PDF", type=["pdf"])
    if up and st.sidebar.button("Index PDF"):
        r = _api("POST", "/upload", files={"file": (up.name, up.getvalue())})
        if r is None:
            pass
        elif r.status_code == 200:
            st.sidebar.success(r.json())
        else:
            st.sidebar.error(r.text)
    filenames, page = [], None
    try:
        r = _api("GET", "/documents")
        if r is None:
            docs = []
        else:
            docs = r.json() if r.status_code == 200 else []
    except Exception:
        docs = []
    if docs:
        st.sidebar.subheader("Tài liệu đã index")
        names = [d["filename"] for d in docs]
        filenames = st.sidebar.multiselect("Phạm vi", names)
        page = st.sidebar.number_input("Trang (0 = tất cả)", min_value=0, value=0) or None
    return filenames, page


def _filters(filenames, page):
    f = {}
    if len(filenames) == 1:
        f["filename"] = filenames[0]
    elif len(filenames) > 1:
        f["filenames"] = filenames
    if page and not f.get("filenames"):
        f["page"] = int(page)
    return f or None


def _tab_chat(filenames, page):
    q = st.text_input("Câu hỏi trên tài liệu")
    if st.button("Hỏi", type="primary") and q:
        body = {"question": q, "filters": _filters(filenames, page)}
        r = _api("POST", "/ask", json=body)
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.text)
            return
        r = r.json()
        st.markdown(r.get("answer", ""))
        with st.expander("Nguồn"):
            for c in r.get("citations", []):
                st.caption(f"[{c['source_marker']}] {c['filename']} trang {c['page']}")


def _tab_summary(filenames, page):
    doc = filenames[0] if len(filenames) == 1 else None
    if st.button("Tóm tắt"):
        body = {"document": doc, "filters": _filters(filenames, page)}
        r = _api("POST", "/summarize", json=body)
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.text)
            return
        r = r.json()
        st.markdown(r.get("summary", ""))
        st.markdown("**Ý chính:**")
        for kp in r.get("key_points", []):
            st.markdown(f"- {kp}")


def _tab_quiz(filenames, page):
    n = st.slider("Số câu", 1, 20, 8)
    if st.button("Tạo Quiz"):
        body = {"count": n, "filters": _filters(filenames, page)}
        if len(filenames) == 1:
            body["document"] = filenames[0]
        r = _api("POST", "/quiz", json=body)
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.text)
            return
        r = r.json()
        for i, it in enumerate(r.get("items", []), 1):
            st.markdown(f"**Câu {i}: {it['question']}**")
            st.radio("Chọn", it["options"], key=f"q{i}")
            with st.expander("Giải thích"):
                st.write(it["explanation"])


def _tab_flashcards(filenames, page):
    n = st.slider("Số thẻ", 1, 30, 15)
    if st.button("Tạo Flashcards"):
        body = {"count": n, "filters": _filters(filenames, page)}
        if len(filenames) == 1:
            body["document"] = filenames[0]
        r = _api("POST", "/flashcards", json=body)
        if r is None:
            return
        if r.status_code != 200:
            st.error(r.text)
            return
        r = r.json()
        for i, c in enumerate(r.get("cards", []), 1):
            with st.expander(f"Thẻ {i}: {c['front']}"):
                st.write(c["back"])


def run():
    st.set_page_config(page_title="RAG Learning System", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    if not _check_api():
        st.stop()
    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)


if __name__ == "__main__":
    run()
