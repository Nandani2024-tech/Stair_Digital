### CHUNKING
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 100
MIN_CHUNK_LENGTH_CHARS = 100

### RETRIEVAL
TOP_K_RETRIEVAL = 10          # candidates fetched from Chroma
TOP_K_RERANK = 3              # final chunks passed to LLM after rerank
SIMILARITY_THRESHOLD = 0.55   # Gate 1: Cosine distance cutoff (lower = more similar).
                               # With normalized embeddings, scale is 0 to 2.
                               # Hits with distance > this threshold are blocked.

### LLM
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq model string
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_TOKENS_RESPONSE = 1024
TEMPERATURE = 0.0              # deterministic, no creativity

### CONVERSATION
MAX_HISTORY_TURNS = 6          # keep last 6 turns (3 user + 3 assistant)
SESSION_RESET_ON_NEW_PDF = True

### STORAGE PATHS
CHROMA_DB_DIR = "data/chroma_db"
CHROMA_COLLECTION_NAME = "pdf_chunks"
UPLOAD_DIR = "data/uploads"
LOG_DIR = "data/logs"
LOG_FILE = "data/logs/agent.jsonl"

### CITATION FORMAT
# Every answer MUST render citations as:  [Page 4 | Section 2.1 Inflation Outlook]
# If section title unavailable:           [Page 4]
CITATION_FORMAT = "[Page {page} | Section: {section}]"
CITATION_FORMAT_NO_SECTION = "[Page {page}]"

### GATE BEHAVIOUR
# Gate 1 fires at retrieval layer - before LLM is ever called
# Gate 2 fires at response-parse layer - catches LLM [INSUFFICIENT]
GATE1_REFUSE_MESSAGE_TEMPLATE = (
    "I cannot answer this from the uploaded document. "
    "No section with sufficient relevance was found "
    "(best match distance: {distance}, threshold: {threshold}). "
    "Please ask about content that is explicitly covered in the PDF."
)
GATE2_INSUFFICIENT_TOKEN = "[INSUFFICIENT]"

### PDF HANDLING
MAX_PDF_SIZE_MB = 50
SCANNED_TEXT_MIN_CHARS_PER_PAGE = 30   # below this -> trigger OCR fallback
OCR_DPI = 300
PASSWORD_PROTECTED_MESSAGE = (
    "This PDF is password-protected. "
    "Please provide an unlocked version."
)
CORRUPTED_PDF_MESSAGE = (
    "This file could not be read as a valid PDF. "
    "Please check the file and re-upload."
)
FILE_SIZE_ERROR_MESSAGE = "File too large (max {size}MB)."
