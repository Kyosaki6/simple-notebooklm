from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
except ImportError:  # pragma: no cover - resolved at install time
    HuggingFaceEmbeddings = None  # type: ignore
    QdrantVectorStore = None  # type: ignore


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu" if settings.hf_device < 0 else f"cuda:{settings.hf_device}"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir))


def get_vector_store(collection_name: str | None = None):
    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name or settings.qdrant_collection,
        embedding=get_embeddings(),
    )


INDEXED_PAYLOAD_FIELDS = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}


def ensure_collection(recreate: bool = False, collection_name: str | None = None):
    client = get_client()
    name = collection_name or settings.qdrant_collection
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
    payload_schema = client.get_collection(name).payload_schema or {}
    for field, schema in INDEXED_PAYLOAD_FIELDS.items():
        if payload_schema.get(field) is None:
            try:
                client.create_payload_index(name, field_name=field, field_schema=schema)
            except Exception:
                pass


def scroll_all(collection_name: str, scroll_filter=None, limit: int = 256):
    """Yield pages of points (lists) matching filter, using Qdrant scroll."""
    client = get_client()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        yield points
        if offset is None:
            break
