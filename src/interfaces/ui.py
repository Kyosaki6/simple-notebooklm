import httpx
import streamlit as st

from ..config import settings
from .styles import GLOBAL_CSS

_API = settings.api_url


def _api(method, path, **kwargs):
    return httpx.request(method, f"{_API}{path}", timeout=120, **kwargs)


def _sidebar():
    st.sidebar.title("📚 NotebookLM đơn giản")
    up = st.sidebar.file_uploader("Tải PDF", type=["pdf"])
    if up and st.sidebar.button("Index PDF"):
        r = _api("POST", "/upload", files={"file": (up.name, up.getvalue())})
        st.sidebar.success(r.json() if r.status_code == 200 else r.text)
    filenames, page = [], None
    try:
        r = _api("GET", "/documents")
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
        r = _api("POST", "/ask", json=body).json()
        st.markdown(r.get("answer", ""))
        with st.expander("Nguồn"):
            for c in r.get("citations", []):
                st.caption(f"[{c['source_marker']}] {c['filename']} trang {c['page']}")


def _tab_summary(filenames, page):
    doc = filenames[0] if len(filenames) == 1 else None
    if st.button("Tóm tắt"):
        body = {"document": doc, "filters": _filters(filenames, page)}
        r = _api("POST", "/summarize", json=body).json()
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
        r = _api("POST", "/quiz", json=body).json()
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
        r = _api("POST", "/flashcards", json=body).json()
        for i, c in enumerate(r.get("cards", []), 1):
            with st.expander(f"Thẻ {i}: {c['front']}"):
                st.write(c["back"])


def run():
    st.set_page_config(page_title="RAG Learning System", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)


run()
