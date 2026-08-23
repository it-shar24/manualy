import pymupdf as fitz
from pathlib import Path
from typing import Dict, Any, List
from src.config import EXTRACTED_IMAGES_DIR

class PDFManualParser:
    def __init__(self, pdf_path: str, min_image_dim: int = 100):
        self.pdf_path = Path(pdf_path)
        self.min_image_dim = min_image_dim
        self.doc_id = self.pdf_path.stem
        self.output_dir = EXTRACTED_IMAGES_DIR / self.doc_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> Dict[str, Any]:
        doc = fitz.open(str(self.pdf_path))
        pages_data: List[Dict[str, Any]] = []
        total_images = 0

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            
            # Try raw text extraction first
            text = page.get_text("text").strip()
            
            # If text is empty, check for raw word blocks / layout text
            if not text:
                blocks = page.get_text("blocks")
                text = "\n".join([b[4] for b in blocks if isinstance(b[4], str)]).strip()

            image_paths: List[str] = []

            # 1. Check for embedded raster figures
            image_list = page.get_images(full=True)
            for img_index, img_meta in enumerate(image_list):
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)
                    if w >= self.min_image_dim and h >= self.min_image_dim:
                        ext = base_image.get("ext", "png")
                        img_path = self.output_dir / f"page_{page_num}_fig_{img_index + 1}.{ext}"
                        with open(img_path, "wb") as f:
                            f.write(base_image["image"])
                        image_paths.append(str(img_path))
                        total_images += 1

            # 2. Render visual schematic slice if no images were found
            if not image_paths:
                pix = page.get_pixmap(dpi=150)
                img_path = self.output_dir / f"page_{page_num}_schematic.png"
                pix.save(str(img_path))
                image_paths.append(str(img_path))
                total_images += 1

            pages_data.append({
                "page_number": page_num,
                "text": text,
                "image_paths": image_paths
            })

        doc.close()

        return {
            "doc_id": self.doc_id,
            "filename": self.pdf_path.name,
            "total_pages": len(pages_data),
            "total_images_extracted": total_images,
            "pages": pages_data
        }