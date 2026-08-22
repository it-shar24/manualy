import base64
import requests
import os
from pathlib import Path
from typing import Optional
from src.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

class LocalVisionAnnotator:
    def __init__(self, model_name: str = OLLAMA_VISION_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def _encode_image(self, image_path: str) -> str:
        """Converts an image file to a base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def describe_diagram(self, image_path: str, page_number: int, doc_name: str) -> str:
        """
        Sends the diagram to the local Ollama vision model to extract 
        technical labels, part numbers, schematic symbols, and visual layout.
        """
        if not Path(image_path).exists():
            return ""

        base64_image = self._encode_image(image_path)
        
        prompt = (
            f"You are a technical document visual analyst. Analyze this technical diagram/figure extracted from "
            f"'{doc_name}' on Page {page_number}.\n"
            f"Provide a concise, highly specific technical description including:\n"
            f"1. Type of visual (e.g., Exploded View, Wiring Schematic, Parts Diagram, Flowchart, UI Screen).\n"
            f"2. Key labeled components, part numbers, or pinouts visible.\n"
            f"3. Core function or procedure depicted in the diagram.\n"
            f"Keep your output factual, objective, and dense with visible technical terms. Do not hallucinate."
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1  # Low temperature for strict factual accuracy
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"[Vision Error] Could not connect to Ollama at {self.api_url}. Is 'ollama serve' running?")
            return "Visual diagram (description pending local vision server)."
        except Exception as e:
            print(f"[Vision Error] Failed to annotate {image_path}: {e}")
            return "Technical figure extracted from manual."