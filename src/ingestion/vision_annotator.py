import base64
import requests
from pathlib import Path
from src.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

class LocalVisionAnnotator:
    def __init__(self, model_name: str = OLLAMA_VISION_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def describe_diagram(self, image_path: str, page_number: int, doc_name: str) -> str:
        if not Path(image_path).exists():
            return ""

        base64_image = self._encode_image(image_path)
        
        prompt = (
            f"You are an expert technical document OCR and visual analyst. Analyze this diagram from '{doc_name}' on Page {page_number}.\n"
            f"Transcribe and describe in detail:\n"
            f"1. ALL visible text, warnings, labels, callouts, and notes verbatim.\n"
            f"2. All electrical specifications, wattages, voltages, pinouts, and part numbers.\n"
            f"3. Diagram layout, connected parts, and sequence flow.\n"
            f"Be exhaustive with textual transcriptions from inside the diagram."
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[Vision Error] {e}")
            return "Technical figure extracted from manual."