import pymupdf as fitz  # PyMuPDF
import os
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import io
from src.config import EXTRACTED_IMAGES_DIR, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT

class PDFManualParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
        self.doc = fitz.open(self.pdf_path)
        self.doc_id = self.pdf_path.stem
        self.image_save_dir = EXTRACTED_IMAGES_DIR / self.doc_id
        self.image_save_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> Dict[str, Any]:
        """
        Parses the PDF and extracts structured page-level text, 
        raster images, and vector diagram snapshots.
        """
        pages_data: List[Dict[str, Any]] = []
        total_extracted_images = 0

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text").strip()
            
            extracted_images: List[str] = []
            image_list = page.get_images(full=True)

            # 1. Extract embedded raster images
            for img_index, img_meta in enumerate(image_list):
                xref = img_meta[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    # Filter out tiny icon noise (arrows, bullet points, divider lines)
                    if pil_img.width < MIN_IMAGE_WIDTH or pil_img.height < MIN_IMAGE_HEIGHT:
                        continue
                except Exception:
                    continue

                img_filename = f"page_{page_num + 1}_fig_{img_index + 1}.{image_ext}"
                img_path = self.image_save_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                extracted_images.append(str(img_path))
                total_extracted_images += 1

            # 2. Check for vector schematics/diagrams on sparse text pages
            # If no raster images exist, but the page contains graphical drawings or exploded views
            if not extracted_images and len(text) < 200:
                pix = page.get_pixmap(dpi=150)
                snapshot_filename = f"page_{page_num + 1}_schematic_snapshot.png"
                snapshot_path = self.image_save_dir / snapshot_filename
                pix.save(str(snapshot_path))
                extracted_images.append(str(snapshot_path))
                total_extracted_images += 1

            pages_data.append({
                "page_number": page_num + 1,
                "text": text,
                "image_paths": extracted_images,
                "has_visuals": len(extracted_images) > 0,
                "source_document": self.pdf_path.name
            })

        return {
            "doc_id": self.doc_id,
            "filename": self.pdf_path.name,
            "total_pages": len(self.doc),
            "total_images_extracted": total_extracted_images,
            "pages": pages_data
        }