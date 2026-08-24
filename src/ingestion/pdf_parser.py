"""
pdf_parser.py
-------------
Extracts text and detects real diagram content from any PDF manual.

KEY FIX vs. previous version: the old parser rasterized *every* page and
sent *every* page to the vision model for captioning, even pages that were
100% plain text (a parts table, a paragraph of instructions). That produced
two independent, sometimes-conflicting descriptions of the same content —
one from OCR/native text (usually correct) and one from a small vision
model guessing at a downscaled image of mostly text (often wrong) — and
retrieval had no way to know which one to trust.

This version separates the two concerns:
  1. Text extraction (native, falling back to OCR) always runs on every page.
  2. Diagram detection runs independently, using page structure (embedded
     raster images + vector drawing commands) to decide whether a page
     actually contains a diagram worth describing visually. Only those
     pages get a saved image + get sent to the vision captioner later.

This heuristic is generic — it looks at PDF structure, not manual content —
so it works the same way on an appliance manual, a datasheet, or an
aircraft maintenance manual.
"""

import io
from pathlib import Path
from typing import Any, Dict, List

import pymupdf
import pytesseract
from PIL import Image

from src.config import (
    EXTRACTED_IMAGES_DIR,
    MIN_RASTER_IMAGE_BYTES,
    MIN_VECTOR_DRAWING_COMMANDS,
)


class PDFManualParser:
    def __init__(self, pdf_path: str, min_image_dim: int = 100):
        self.pdf_path = Path(pdf_path)
        self.min_image_dim = min_image_dim
        self.doc_id = self.pdf_path.stem
        self.output_dir = EXTRACTED_IMAGES_DIR / self.doc_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _page_has_diagram(self, page: "pymupdf.Page") -> bool:
        """Generic structural heuristic: does this page contain real
        diagram content, or is it just text (possibly inside table
        borders, which also produce a handful of vector lines)?"""
        # Real embedded raster images above a size floor (filters out
        # tiny logos/icons/decorative bullets).
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                img_info = page.parent.extract_image(xref)
                if len(img_info.get("image", b"")) >= MIN_RASTER_IMAGE_BYTES:
                    return True
            except Exception:
                continue

        # Vector line-art: a technical exploded-view or schematic is drawn
        # with many path/line/curve commands. A table border or a couple of
        # decorative rules produces far fewer. The threshold is a heuristic,
        # not a guarantee — tune MIN_VECTOR_DRAWING_COMMANDS in config.py if
        # a manual's diagrams are simpler or more complex than average.
        drawings = page.get_drawings()
        if len(drawings) >= MIN_VECTOR_DRAWING_COMMANDS:
            return True

        return False

    def extract(self) -> Dict[str, Any]:
        doc = pymupdf.open(str(self.pdf_path))
        pages_data: List[Dict[str, Any]] = []
        total_diagram_pages = 0

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            # 1. Native text extraction
            text = page.get_text("text").strip()

            # 2. OCR fallback if native text is empty or sparse
            if len(text) < 50:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text = ocr_text.strip()
                        print(f"[OCR] Extracted {len(text)} characters from Page {page_num}")
                except Exception as e:
                    print(f"[OCR Warning] Page {page_num} OCR failed: {e}")

            # 3. Diagram detection — only rasterize + save an image if the
            # page actually looks like it contains a diagram. Text-only
            # pages get NO image and therefore never get sent to the
            # (comparatively unreliable) vision captioner.
            has_diagram = self._page_has_diagram(page)
            image_paths: List[str] = []
            if has_diagram:
                pix = page.get_pixmap(dpi=250)  # higher DPI: legibility matters more than every-page speed now
                img_path = self.output_dir / f"page_{page_num}_diagram.png"
                pix.save(str(img_path))
                image_paths.append(str(img_path))
                total_diagram_pages += 1

            pages_data.append({
                "page_number": page_num,
                "text": text,
                "has_diagram": has_diagram,
                "image_paths": image_paths,
            })

        doc.close()

        return {
            "doc_id": self.doc_id,
            "filename": self.pdf_path.name,
            "total_pages": len(pages_data),
            "total_diagram_pages": total_diagram_pages,
            "pages": pages_data,
        }
