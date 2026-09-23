from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
]

_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]


@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict


@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list | None = None

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)


@dataclass(frozen=True)
class SemanticChunkerWrapper:
    """Wrapper for LangChain SemanticChunker."""

    embeddings: object
    breakpoint_type: str = "percentile"

    def _splitter(self):
        from langchain_experimental.text_splitter import SemanticChunker

        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)


def all_strategies(embeddings=None) -> list[ChunkingStrategy]:
    out = [ChunkingStrategy(sid, RecursiveChunker(cs, co), {"chunk_size": cs, "chunk_overlap": co})
           for sid, cs, co in _RECURSIVE_CONFIGS]
    if embeddings is not None:
        for sid, btype in _SEMANTIC_CONFIGS:
            out.append(ChunkingStrategy(sid, SemanticChunkerWrapper(embeddings, btype),
                                        {"breakpoint_type": btype}))
    return out
