"""
vision_annotator.py
--------------------
Generates a text description of a diagram page using a local vision model
(via Ollama), so the diagram becomes searchable in the vector store.

KEY FIX vs. previous version: the old settings (768px max image dimension,
400-token output cap) were tuned for running on *every single page* of a
manual, which is expensive, so they were kept small to stay fast. Now that
pdf_parser.py only flags genuine diagram pages (a small fraction of most
manuals), we can afford a much larger image and a much larger output
budget on the pages that actually need it — which directly fixes the
truncated/garbled captions seen on dense technical figures.
"""

import base64
import io
from pathlib import Path

import requests
from PIL import Image

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_VISION_MODEL,
    VISION_MAX_IMAGE_DIM,
    VISION_NUM_PREDICT,
    VISION_NUM_CTX,
)


class LocalVisionAnnotator:
    def __init__(self, model_name: str = OLLAMA_VISION_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def _encode_and_resize_image(self, image_path: str, max_size: int = VISION_MAX_IMAGE_DIM) -> str:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")  # PNG over JPEG: no compression artifacts on fine line-art/small text
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def describe_diagram(self, image_path: str, page_number: int, doc_name: str) -> str:
        if not Path(image_path).exists():
            return ""

        try:
            base64_image = self._encode_and_resize_image(image_path)

            prompt = (
                f"You are looking at a page from a technical manual ('{doc_name}', page {page_number}) "
                f"that has been flagged as containing a diagram, schematic, or illustrated figure.\n\n"
                f"Describe ONLY what is visually present in the image. Follow these rules strictly:\n"
                f"1. If the page contains a diagram, exploded view, wiring/pinout schematic, or symbol "
                f"legend table, describe every labeled part, callout number/letter, and how they connect "
                f"or relate spatially. If a table is embedded in the image (e.g. a symbol-to-part-number "
                f"legend), transcribe it as a clean row-by-row list: 'symbol/label -> value(s)'.\n"
                f"2. Transcribe any text that appears directly inside the diagram itself (labels, warning "
                f"boxes, figure numbers, callout text) VERBATIM — do not paraphrase text you can read.\n"
                f"3. If a value, number, or label is not clearly legible, say '[illegible]' instead of "
                f"guessing a plausible-looking value. Never invent part numbers, thread sizes, or callout "
                f"values that are not visibly printed in the image.\n"
                f"4. If, after inspection, this page turns out to be mostly body text with no meaningful "
                f"diagram content, say so explicitly ('No distinct diagram content on this page') instead "
                f"of describing the text.\n"
                f"Output the description directly, with no preamble."
            )

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": VISION_NUM_PREDICT,
                    "num_ctx": VISION_NUM_CTX,
                    "num_thread": 4,
                },
            }

            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json().get("response", "").strip()

            # Don't index a caption that explicitly says there's nothing here —
            # it adds retrieval noise without adding retrievable information.
            if "no distinct diagram content" in result.lower():
                return ""
            return result
        except Exception as e:
            print(f"[Vision Error] {e}")
            # IMPORTANT: return empty, not a filler string. A previous version
            # returned "Technical figure extracted from manual." on error —
            # that fabricated-sounding placeholder text was itself getting
            # indexed and retrieved as if it were real content.
            return ""
