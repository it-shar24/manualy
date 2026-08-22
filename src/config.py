import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base File Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Local Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2:3b")

# Embedding Model (Runs locally on CPU)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval & Guardrail Thresholds
MAX_DISTANCE_THRESHOLD = 0.55
TOP_K_CHUNKS = 4

# Minimum image dimensions to filter out bullet icons and divider noise (in px)
MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 120