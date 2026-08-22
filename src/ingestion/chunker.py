from typing import List, Dict, Any
from src.ingestion.vision_annotator import LocalVisionAnnotator

class HybridManualChunker:
    def __init__(self, vision_annotator: LocalVisionAnnotator = None):
        self.vision_annotator = vision_annotator or LocalVisionAnnotator()

    def create_chunks(
        self, 
        parsed_doc: Dict[str, Any], 
        user_id: str = "default_user"
    ) -> List[Dict[str, Any]]:
        """
        Builds hybrid search chunks containing text and diagram descriptions,
        with metadata for citations and multi-tenant isolation.
        """
        chunks = []
        doc_id = parsed_doc["doc_id"]
        doc_name = parsed_doc["filename"]

        for page in parsed_doc["pages"]:
            page_num = page["page_number"]
            page_text = page["text"]
            image_paths = page["image_paths"]

            # 1. Text Chunk (if text is present)
            if page_text:
                chunk_id = f"{doc_id}_p{page_num}_text"
                chunks.append({
                    "id": chunk_id,
                    "content": f"[Document: {doc_name} | Page {page_num}]\n{page_text}",
                    "metadata": {
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "chunk_type": "text",
                        "image_path": image_paths[0] if image_paths else "",
                        "has_visual": len(image_paths) > 0
                    }
                })

            # 2. Visual Chunk (annotating diagrams for semantic search)
            for idx, img_path in enumerate(image_paths):
                print(f"[Chunker] Generating visual description for Page {page_num} Figure {idx+1}...")
                visual_desc = self.vision_annotator.describe_diagram(
                    image_path=img_path,
                    page_number=page_num,
                    doc_name=doc_name
                )

                chunk_id = f"{doc_id}_p{page_num}_img_{idx+1}"
                chunks.append({
                    "id": chunk_id,
                    "content": (
                        f"[Document: {doc_name} | Page {page_num} | Visual Figure]\n"
                        f"{visual_desc}"
                    ),
                    "metadata": {
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "chunk_type": "diagram",
                        "image_path": img_path,
                        "has_visual": True
                    }
                })

        return chunks