INDEX_NAME = "gapblue_documents"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
UPLOAD_DIRECTORY = "uploads"
ALLOWED_FILE_EXTENSION = ".pdf"
MAX_FILE_SIZE_MB = 20
DOCUMENT_STATUS_INDEXED = "indexed"
DOCUMENT_STATUS_PROCESSED = "processed"
ELASTICSEARCH_NUM_CANDIDATES = 50

TOP_K = 5

EMBEDDING_DIMENSIONS = 3072

HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503

BEARE_TOKEN_TYPE = "bearer"

SYSTEM_PROMPT = """
You are a document assistant.

Answer ONLY using the provided context.

Cite the source document and page number whenever possible.

If the answer cannot be found in the provided context,
say that you do not know.

Do not make up information.
"""