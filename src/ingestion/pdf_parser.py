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
            text = page.get_text("text").strip()
            image_paths: List[str] = []

            # 1. Check for raster images
            image_list = page.get_images(full=True)
            
            for img_index, img_meta in enumerate(image_list):
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    if width >= self.min_image_dim and height >= self.min_image_dim:
                        img_ext = base_image.get("ext", "png")
                        image_filename = f"page_{page_num}_fig_{img_index + 1}.{img_ext}"
                        image_path = self.output_dir / image_filename
                        with open(image_path, "wb") as f:
                            f.write(base_image["image"])
                        image_paths.append(str(image_path))
                        total_images += 1

            # 2. Vector Schematic Fallback: If page has drawings/drawings rects but no raster images
            drawings = page.get_drawings()
            if not image_paths and len(drawings) > 10:
                # Render high-res snapshot of page visual content
                pix = page.get_pixmap(dpi=200)
                image_filename = f"page_{page_num}_schematic.png"
                image_path = self.output_dir / image_filename
                pix.save(str(image_path))
                image_paths.append(str(image_path))
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