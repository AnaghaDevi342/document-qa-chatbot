from pathlib import Path
import uuid

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from fastapi import HTTPException, UploadFile
from openai import OpenAI
from PyPDF2 import PdfReader

from .auth import create_access_token
from .config import settings
from .constants import (
    ALLOWED_FILE_EXTENSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCUMENT_STATUS_INDEXED,
    EMBEDDING_DIMENSIONS,
    HTTP_BAD_GATEWAY,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_UNAUTHORIZED,
    INDEX_NAME,
    MAX_FILE_SIZE_MB,
    TOP_K,
    UPLOAD_DIRECTORY,
)


class AuthService:
    """Business logic related to authentication."""

    @staticmethod
    def login(username: str, password: str) -> str:
        if (
            username != settings.app_username
            or password != settings.app_password
        ):
            raise HTTPException(
                status_code=HTTP_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        return create_access_token(username)


class EmbeddingService:
    """Generate embeddings using Azure OpenAI."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=(
                f"{settings.azure_openai_endpoint}"
                "/openai/v1/"
            ),
        )

    def generate_embedding(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                model=settings.azure_openai_embedding_deployment,
                input=text,
            )

            embedding = response.data[0].embedding

            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected embedding dimension "
                    f"{EMBEDDING_DIMENSIONS}, "
                    f"received {len(embedding)}"
                )

            return embedding

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_BAD_GATEWAY,
                detail="Failed to generate document embedding",
            ) from exc


class ElasticsearchService:
    """Handle Elasticsearch indexing and retrieval."""

    def __init__(self):
        self.client = Elasticsearch(
            settings.elasticsearch_url
        )

    def check_connection(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def create_index(self) -> None:
        try:
            if self.client.indices.exists(
                index=INDEX_NAME
            ):
                return

            mapping = {
                "properties": {
                    "document_id": {
                        "type": "keyword"
                    },
                    "user_id": {
                        "type": "keyword"
                    },
                    "filename": {
                        "type": "keyword"
                    },
                    "page": {
                        "type": "integer"
                    },
                    "chunk_number": {
                        "type": "integer"
                    },
                    "text": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }

            self.client.indices.create(
                index=INDEX_NAME,
                mappings=mapping,
            )

        except Exception as exc:
            print(f"Type: {type(exc)}")
            print(f"Error: {exc}")
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to create Elasticsearch index",
            ) from exc

    def index_chunks(
        self,
        chunks: list[dict],
    ) -> None:
        try:
            actions = []

            for chunk in chunks:
                actions.append(
                    {
                        "_index": INDEX_NAME,
                        "_id": chunk["chunk_id"],
                        "_source": {
                            "document_id": chunk["document_id"],
                            "user_id": chunk["user_id"],
                            "filename": chunk["filename"],
                            "page": chunk["page"],
                            "chunk_number": chunk["chunk_number"],
                            "text": chunk["text"],
                            "embedding": chunk["embedding"],
                        },
                    }
                )

            if actions:
                bulk(
                    self.client,
                    actions,
                    refresh="wait_for",
                )

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to index document chunks",
            ) from exc

    def search_similar_chunks(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = TOP_K,
    ) -> list[dict]:

        try:
            response = self.client.search(
                index=INDEX_NAME,
                knn={
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": 50,
                    "filter": {
                        "term": {
                            "user_id": user_id
                        }
                    },
                },
                source=[
                    "document_id",
                    "filename",
                    "page",
                    "chunk_number",
                    "text",
                ],
            )

            results = []
            seen_chunks = set()

            for hit in response["hits"]["hits"]:
                source = hit["_source"]

                chunk_key=(source["document_id"], source["page"], source["chunk_number"])

                if chunk_key in seen_chunks:
                    continue

                seen_chunks.add(chunk_key)

                results.append(
                    {
                        "document_id": source["document_id"],
                        "document": source["filename"],
                        "page": source["page"],
                        "chunk_number": source["chunk_number"],
                        "text": source["text"],
                        "relevance_score": hit["_score"],
                    }
                )

            return results

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve relevant documents",
            ) from exc

    def delete_document(
        self,
        document_id: str,
        user_id: str,
    ) -> None:
        try:
            self.client.delete_by_query(
                index=INDEX_NAME,
                query={
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "document_id": document_id
                                }
                            },
                            {
                                "term": {
                                    "user_id": user_id
                                }
                            },
                        ]
                    }
                },
            )

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_SERVICE_UNAVAILABLE,
                detail="Failed to delete document from Elasticsearch",
            ) from exc


class DocumentService:
    """Business logic for PDF processing."""

    @staticmethod
    def _validate_file(file: UploadFile) -> None:
        if not file.filename:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Filename is required",
            )

        extension = Path(
            file.filename
        ).suffix.lower()

        if extension != ALLOWED_FILE_EXTENSION:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Only PDF files are supported",
            )

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[str]:
        """Split text into overlapping chunks."""

        if not text.strip():
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - chunk_overlap

        return chunks

    @staticmethod
    async def process_pdf(
        file: UploadFile,
        username: str,
    ) -> dict:
        """Save, process, embed and index an uploaded PDF."""

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

            with open(
                file_path,
                "wb",
            ) as output_file:
                output_file.write(contents)

            reader = PdfReader(
                str(file_path)
            )

            all_chunks = []
            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                page_text = (
                    page.extract_text() or ""
                )

                page_chunks = (
                    DocumentService._chunk_text(
                        page_text
                    )
                )

                for chunk_number, chunk in enumerate(
                    page_chunks,
                    start=1,
                ):
                    chunk_id = (
                        f"{document_id}_"
                        f"{page_number}_"
                        f"{chunk_number}"
                    )

                    all_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "user_id": username,
                            "filename": file.filename,
                            "page": page_number,
                            "chunk_number": chunk_number,
                            "text": chunk,
                        }
                    )

            if not all_chunks:
                raise HTTPException(
                    status_code=HTTP_BAD_REQUEST,
                    detail=(
                        "No readable text was found "
                        "in the PDF"
                    ),
                )

            for chunk in all_chunks:
                chunk["embedding"] = (
                    embedding_service.generate_embedding(
                        chunk["text"]
                    )
                )

            elasticsearch_service.create_index()

            elasticsearch_service.index_chunks(
                all_chunks
            )

            return {
                "document_id": document_id,
                "filename": file.filename,
                "pages": len(reader.pages),
                "chunks_created": len(all_chunks),
                "status": DOCUMENT_STATUS_INDEXED,
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_INTERNAL_SERVER_ERROR,
                detail="Failed to process PDF document",
            ) from exc

class RetrievalService:
    """Business logic for semantic document retrieval."""

    @staticmethod
    def retrieve(
        question: str,
        username: str,
    ) -> list[dict]:

        try:
            query_embedding = (
                embedding_service.generate_embedding(
                    question
                )
            )

            return (
                elasticsearch_service.search_similar_chunks(
                    query_embedding=query_embedding,
                    user_id=username,
                    top_k=TOP_K,
                )
            )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve relevant documents",
            ) from exc 

class ChatService:
    """Handle RAG-based document question answering."""

    _conversations: dict[str, list[dict]] = {}

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        """Build context from retrieved document chunks."""

        context_parts = []

        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"""
SOURCE {index}
Document: {chunk["document"]}
Page: {chunk["page"]}
Chunk: {chunk["chunk_number"]}

Content:
{chunk["text"]}
""".strip()
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
        conversation_history: list[dict],
    ) -> list[dict]:
        """Build grounded chat messages."""

        system_prompt = """
You are a document question-answering assistant.

You MUST answer the user's question ONLY using the provided document
context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present in the provided context, say:
   "I don't have enough information in the uploaded documents to answer
   this question."
4. Keep the answer concise and directly relevant.
5. Use the conversation history only to understand references such as
   "it", "that", or "the previous project".
6. Do not treat conversation history as a source of facts.
7. The document context is the only source of factual information.
""".strip()

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        messages.extend(conversation_history)

        messages.append(
            {
                "role": "user",
                "content": f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
""".strip(),
            }
        )

        return messages

    @staticmethod
    def _calculate_confidence(
        chunks: list[dict],
    ) -> str:
        """Calculate a simple confidence level from retrieval scores."""

        if not chunks:
            return "low"

        highest_score = chunks[0]["relevance_score"]

        if highest_score >= 0.75:
            return "high"

        if highest_score >= 0.55:
            return "medium"

        return "low"

    @staticmethod
    def chat(
        question: str,
        username: str,
        conversation_id: str | None = None,
    ) -> dict:

        try:
            # Create conversation ID if this is a new conversation.
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            conversation_history = (
                ChatService._conversations.get(
                    conversation_id,
                    [],
                )
            )

            # Retrieve relevant document chunks.
            chunks = RetrievalService.retrieve(
                question=question,
                username=username,
            )

            if not chunks:
                return {
                    "answer": (
                        "I don't have enough information in the "
                        "uploaded documents to answer this question."
                    ),
                    "sources": [],
                    "confidence": "low",
                    "conversation_id": conversation_id,
                }

            # Build context from retrieved chunks.
            context = ChatService._build_context(
                chunks
            )

            # Build messages for Azure OpenAI.
            messages = ChatService._build_prompt(
                question=question,
                context=context,
                conversation_history=conversation_history,
            )

            # Call Azure OpenAI.
            response = chat_client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                temperature=0,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise HTTPException(
                    status_code=HTTP_BAD_GATEWAY,
                    detail="Empty response received from Azure OpenAI",
                )

            # Save conversation history.
            conversation_history.extend(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            )

            ChatService._conversations[
                conversation_id
            ] = conversation_history

            # Prepare sources.
            sources = []

            for chunk in chunks:
                sources.append(
                    {
                        "document": chunk["document"],
                        "page": chunk["page"],
                        "relevance_score": chunk[
                            "relevance_score"
                        ],
                    }
                )

            return {
                "answer": answer,
                "sources": sources,
                "confidence": ChatService._calculate_confidence(
                    chunks
                ),
                "conversation_id": conversation_id,
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=HTTP_BAD_GATEWAY,
                detail="Failed to generate answer",
            ) from exc
                
embedding_service = EmbeddingService()
elasticsearch_service = ElasticsearchService()

chat_client = OpenAI(
    api_key=settings.azure_openai_api_key,
    base_url=(
        f"{settings.azure_openai_endpoint}"
        "/openai/v1/"
    ),
)