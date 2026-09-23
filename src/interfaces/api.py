from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..filters import MetadataFilter, filters_to_dict
from ..indexing import list_documents, save_and_ingest_pdf
from ..learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from ..rag import answer
from ..schemas import DocumentInfo, FlashcardSet, QuizSet, RagAnswer, Summary, UploadResponse


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    pass


app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "RAG Learning API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/documents", "/upload", "/ask", "/summarize", "/quiz", "/flashcards"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return save_and_ingest_pdf(content, file.filename or "upload.pdf")


@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(req.question, k=req.k, filters=filters_to_dict(req.filters))


@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    try:
        return summarize_learning(document=req.document, query=req.query,
                                  filters=filters_to_dict(req.filters), k=req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    try:
        return generate_quiz(document=req.document, query=req.query,
                             filters=filters_to_dict(req.filters), count=req.count, k=req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest):
    try:
        return generate_flashcards(document=req.document, query=req.query,
                                   filters=filters_to_dict(req.filters), count=req.count, k=req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
