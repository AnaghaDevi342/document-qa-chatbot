from fastapi import APIRouter, Depends, File, UploadFile

from .auth import verify_token
from .constants import HTTP_OK
from .schemas import LoginRequest, TokenResponse, DocumentUploadResponse, RetrievalRequest, RetrievedChunk, ChatRequest, ChatResponse
from .services import AuthService, DocumentService, RetrievalService, ChatService

router = APIRouter()

@router.post(
    "/auth/login",
    response_model=TokenResponse,
    status_code=HTTP_OK,
)
def login(request: LoginRequest):

    token = AuthService.login(
        username=request.username,
        password=request.password,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


@router.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(verify_token),
):
    return await DocumentService.process_pdf(file=file, username=username)
"""
@router.post(
    "/documents/search",
    response_model=list[RetrievedChunk],
)
def search_documents(
    request: RetrievalRequest,
    username: str = Depends(verify_token),
):
    return RetrievalService.retrieve(
        question=request.question,
        username=username,
    )
"""
@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    username: str = Depends(verify_token),
):
    return ChatService.chat(
        question=request.question,
        username=username,
        conversation_id=request.conversation_id,
    )