import uuid
from pathlib import Path
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from .config import settings
from .constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    HTTP_BAD_GATEWAY,
    HTTP_UNAUTHORIZED,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_SERVICE_UNAVAILABLE,
    INDEX_NAME,
    MAX_FILE_SIZE_MB,
    TOP_K,
    UPLOAD_DIRECTORY,
)

class AuthService:
    """Handle user authentication."""

    @staticmethod
    def login(
        username: str,
        password: str,
    ) -> str:

        if (
            username != settings.app_username
            or password != settings.app_password
        ):
            raise HTTPException(
                status_code=HTTP_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        from .auth import create_access_token

        return create_access_token(username)

class EmbeddingService:
    """Azure OpenAI embedding service using LangChain."""

    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=(
                settings.azure_openai_embedding_deployment
            ),
            api_version=settings.azure_openai_version,
        )

class VectorStoreService:
    """Manage LangChain Elasticsearch vector store."""

    def __init__(self):
        self.embeddings = embedding_service.embeddings

        self.vector_store = ElasticsearchStore(
            es_url=settings.elasticsearch_url,
            index_name=INDEX_NAME,
            embedding=self.embeddings,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": TOP_K,
            }
        )

class DocumentService:
    """Process uploaded PDFs using LangChain."""

    @staticmethod
    def _validate_file(file):
        if not file.filename:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Filename is required",
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Only PDF files are supported",
            )

    @staticmethod
    def _split_documents(
        documents: list[Document],
    ) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        return splitter.split_documents(
            documents
        )

    @staticmethod
    async def process_pdf(
        file,
        username: str,
    ) -> dict:

        DocumentService._validate_file(file)

        try:
            contents = await file.read()

            max_size = (
                MAX_FILE_SIZE_MB * 1024 * 1024
            )

            if len(contents) > max_size:
                raise HTTPException(
                    status_code=HTTP_BAD_REQUEST,
                    detail=(
                        f"File size exceeds "
                        f"{MAX_FILE_SIZE_MB} MB limit"
                    ),
                )

            document_id = str(uuid.uuid4())

            upload_directory = Path(
                UPLOAD_DIRECTORY
            )

            upload_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_path = (
                upload_directory
                / f"{document_id}.pdf"
            )

            with open(file_path, "wb") as output_file:
                output_file.write(contents)

            # -----------------------------------------
            # LangChain PDF loading
            # -----------------------------------------

            loader = PyPDFLoader(
                str(file_path)
            )

            documents = loader.load()

            # -----------------------------------------
            # Add application metadata
            # -----------------------------------------

            for document in documents:
                document.metadata.update(
                    {
                        "document_id": document_id,
                        "user_id": username,
                        "filename": file.filename,
                        "document_type": "pdf",
                    }
                )

            # -----------------------------------------
            # LangChain chunking
            # -----------------------------------------

            chunks = (
                DocumentService._split_documents(
                    documents
                )
            )

            # -----------------------------------------
            # Index into Elasticsearch
            # -----------------------------------------

            vector_store_service.vector_store.add_documents(
                chunks
            )

            return {
                "document_id": document_id,
                "filename": file.filename,
                "pages": len(documents),
                "chunks_created": len(chunks),
                "status": "indexed",
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_INTERNAL_SERVER_ERROR,
                detail="Failed to process PDF document",
            ) from exc

class RetrievalService:
    """Retrieve relevant documents using LangChain."""

    @staticmethod
    def retrieve(
        question: str, username: str,
    ) -> list[Document]:

        try:
            retriever = (
                vector_store_service
                .get_retriever()
            )

            return retriever.invoke(
                question
            )

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve documents",
            ) from exc

class ToolService:
    """Create LangChain tools."""

    @staticmethod
    def create_search_tool():
        retriever = (
            vector_store_service
            .get_retriever()
        )

        @tool
        def search_documents(
            query: str,
        ) -> str:
            """
            Search the uploaded documents for information
            relevant to the user's question.
            """

            documents = retriever.invoke(
                query
            )

            if not documents:
                return (
                    "No relevant information was "
                    "found in the uploaded documents."
                )

            context_parts = []

            for index, document in enumerate(
                documents,
                start=1,
            ):
                context_parts.append(
                    f"""
SOURCE {index}
Document: {
    document.metadata.get(
        "filename",
        Path(document.metadata.get("source", "unknown")).name,
    )
}
Page: {
    int(document.metadata.get("page",0)) +1
}

Content:
{document.page_content}
""".strip()
                )

            return "\n\n".join(
                context_parts
            )

        return search_documents

class ChatModelService:
    """Azure OpenAI chat model."""

    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_endpoint=(
                settings.azure_openai_endpoint
            ),
            api_key=(
                settings.azure_openai_api_key
            ),
            azure_deployment=(
                settings.azure_openai_chat_deployment
            ),
            api_version=(
                settings.azure_openai_version
            ),
            temperature=0,
        )

class ChatService:
    """Handle chat requests using the LangChain agent."""

    @staticmethod
    def chat(
        question: str,
        username: str,
        conversation_id: str | None = None,
    ) -> dict:

        try:
            from .agent import agent_service

            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            result = agent_service.invoke(
                question=question,
                conversation_id=conversation_id,
            )

            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": "medium",
                "conversation_id": conversation_id,
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to process chat request",
            ) from exc

embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()