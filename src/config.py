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
TOP_K_CHUNKS = 8