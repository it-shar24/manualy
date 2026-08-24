from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
CHROMA_PATH = CHROMA_DB_DIR  # Alias for backwards compatibility

# Ensure storage directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Model & Engine Configurations
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_VISION_MODEL = "qwen2.5vl:3b"
OLLAMA_TEXT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # general-purpose passage reranker, not domain-specific
TOP_K_CHUNKS = 8

# --- Chunking ---
# Structure-aware chunker targets: pack paragraphs up to CHUNK_TARGET_SIZE,
# never splitting mid-paragraph unless a single paragraph exceeds MAX_HARD_CHUNK_SIZE.
CHUNK_TARGET_SIZE = 1400
CHUNK_OVERLAP = 200
MAX_HARD_CHUNK_SIZE = 2200          # fallback char-split ceiling for oversized paragraphs/tables
HEADER_MAX_LEN = 220                # a paragraph shorter than this AND header-shaped gets glued to the next paragraph

# Generic "scoped range" pattern: matches things like "201-300", "509-550", "REV 2-4".
# This is intentionally domain-agnostic — it just captures hyphenated numeric ranges,
# which show up in serial-number bands, model ranges, batch codes, page ranges, etc.
#
# The negative lookbehind/lookahead (?<![\d-]) / (?![\d-]) is important: without it,
# a multi-segment identifier like "21-61-52-020-052-B" (a subtask/part code, not a
# range) gets misread as several fake "ranges" (21-61, 52-020, etc.), which would
# silently corrupt the range-based retrieval filter. This requires a genuine range to
# stand alone — not be one link in a longer hyphen-chain of numbers.
RANGE_PATTERN = r"(?<![\d-])(\d{2,6})\s*-\s*(\d{2,6})(?![\d-])"

# --- Diagram detection (avoids captioning every page as a "diagram") ---
MIN_VECTOR_DRAWING_COMMANDS = 25    # page.get_drawings() length above this suggests real technical artwork
MIN_RASTER_IMAGE_BYTES = 5000       # embedded raster images smaller than this are usually logos/icons, not diagrams

# --- Vision captioning ---
VISION_MAX_IMAGE_DIM = 1600         # px; must stay legible for dense technical pages
VISION_NUM_PREDICT = 900            # generation ceiling; too low silently truncates dense tables
VISION_NUM_CTX = 4096

# --- Retrieval ---
DENSE_CANDIDATE_POOL = 30           # how many dense-embedding hits to pull before fusion
BM25_CANDIDATE_POOL = 30            # how many BM25 keyword hits to pull before fusion
RRF_K = 60                          # reciprocal rank fusion constant (standard default)
RERANK_CANDIDATE_POOL = 15          # how many fused candidates get passed to the cross-encoder reranker