from pydantic import BaseModel, Field

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    document_id: str | None = None

class Source(BaseModel):
    document: str
    page: int
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: str
    conversation_id: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks_created: int
    status: str

class RetrievalRequest(BaseModel):
    question: str = Field(..., min_length=1)


class RetrievedChunk(BaseModel):
    document: str
    page: int
    chunk_number: int
    text: str
    relevance_score: float