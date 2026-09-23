import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .schemas import ChunkMetadata
from .store import ensure_collection, get_vector_store


def _document_id(path: Path) -> str:
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)
    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def _splitter(chunk_size=None, chunk_overlap=None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def build_chunks(pdf_paths, chunk_size=None, chunk_overlap=None, chunker=None):
    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(Path(path)))
    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    per_doc_counter = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def discover_pdfs() -> list[Path]:
    if not settings.data_dir.exists():
        return []
    return sorted(settings.data_dir.glob("*.pdf"))


def index_chunks(chunks, collection_name=None) -> int:
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def ingest(recreate=False, collection_name=None, chunker=None, chunk_size=None,
           chunk_overlap=None) -> int:
    pdfs = discover_pdfs()
    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(pdfs, chunker=chunker, chunk_size=chunk_size,
                          chunk_overlap=chunk_overlap)
    return index_chunks(chunks, collection_name=collection_name)


def save_and_ingest_pdf(file_bytes: bytes, filename: str):
    safe_name = Path(filename).name
    dest = settings.data_dir / safe_name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    return {"filename": safe_name, "chunks_indexed": index_chunks(chunks)}


def list_documents():
    from .schemas import DocumentInfo
    from .store import get_client

    client = get_client()
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return []
    # aggregate via scroll
    seen: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(collection_name=name, limit=512, offset=offset,
                                       with_payload=True, with_vectors=False)
        if not points:
            break
        for p in points:
            meta = (p.payload or {}).get("metadata") or {}
            fn = meta.get("filename", "?")
            d = seen.setdefault(fn, {"document_id": meta.get("document_id", ""),
                                     "pages": set(), "chunks": 0})
            d["pages"].add(meta.get("page", 0))
            d["chunks"] += 1
        if offset is None:
            break
    return [DocumentInfo(filename=k, document_id=v["document_id"],
                         pages=len(v["pages"]), chunks=v["chunks"]) for k, v in seen.items()]
