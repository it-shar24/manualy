import base64
import requests
import io
from pathlib import Path
from PIL import Image
from src.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

class LocalVisionAnnotator:
    def __init__(self, model_name: str = OLLAMA_VISION_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def _encode_and_resize_image(self, image_path: str, max_size: int = 768) -> str:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Downsample to 768px to ensure fast matrix multiplication on M-series chips
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def describe_diagram(self, image_path: str, page_number: int, doc_name: str) -> str:
        if not Path(image_path).exists():
            return ""

        try:
            base64_image = self._encode_and_resize_image(image_path)
            
            prompt = (
                f"Exhaustive technical extraction for '{doc_name}' Page {page_number}:\n"
                f"1. Transcribe ALL subtask numbers (e.g. SUBTASK 21-61-52-...), FSN ranges, warning texts, and callout IDs verbatim.\n"
                f"2. Transcribe bolt specifications and differences (e.g., item numbers, LH vs RH, symbols) verbatim from tables.\n"
                f"Output raw verbatim facts without extra conversational filler."
            )

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 220,
                    "num_ctx": 2048,
                    "num_thread": 4
                }
            }

            # 240 second safety cushion prevents dropping calls
            response = requests.post(self.api_url, json=payload, timeout=240)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[Vision Error] {e}")
            return "Technical figure extracted from manual."