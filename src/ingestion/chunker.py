from typing import List, Dict, Any
from src.ingestion.vision_annotator import LocalVisionAnnotator

class HybridManualChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_annotator = LocalVisionAnnotator()

    def create_chunks(self, parsed_doc: Dict[str, Any], user_id: str = "demo_user") -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        doc_id = parsed_doc["doc_id"]
        filename = parsed_doc["filename"]

        for page in parsed_doc["pages"]:
            page_num = page["page_number"]
            raw_text = page.get("text", "").strip()

            # 1. Text chunks (if selectable text layer exists)
            if raw_text:
                start = 0
                while start < len(raw_text):
                    end = start + self.chunk_size
                    chunk_text = raw_text[start:end].strip()
                    if chunk_text:
                        chunks.append({
                            "id": f"{doc_id}_p{page_num}_txt_{start}",
                            "content": f"[Document: {filename} | Page {page_num} | Procedural Text]\n{chunk_text}",
                            "metadata": {
                                "doc_id": doc_id,
                                "doc_name": filename,
                                "page_number": page_num,
                                "user_id": user_id,
                                "chunk_type": "text",
                                "image_path": ""
                            }
                        })
                    start += (self.chunk_size - self.chunk_overlap)

            # 2. Vision Diagram & Schematic Chunks
            for img_idx, img_path in enumerate(page["image_paths"]):
                print(f"[Chunker] Generating visual description for Page {page_num} Figure {img_idx+1}...")
                desc = self.vision_annotator.describe_diagram(
                    image_path=img_path,
                    page_number=page_num,
                    doc_name=filename
                )
                chunks.append({
                    "id": f"{doc_id}_p{page_num}_fig_{img_idx+1}",
                    "content": f"[Document: {filename} | Page {page_num} | Visual Figure]\n{desc}",
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": filename,
                        "page_number": page_num,
                        "user_id": user_id,
                        "chunk_type": "diagram",
                        "image_path": img_path
                    }
                })

        return chunks