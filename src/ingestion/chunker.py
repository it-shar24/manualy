import re
import hashlib
from typing import List, Dict, Any

class HybridManualChunker:
    """
    Strategy 2 Enhanced: Structure-Aware with Contextual Chunk Enrichment.
    Prepends hierarchical breadcrumbs (Document > Section > Context) to prevent
    isolated spec lines and tables from losing semantic relevance.
    """
    def __init__(self, max_section_chars: int = 2500):
        self.max_section_chars = max_section_chars

    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def _extract_section_title(self, text: str) -> str:
        """Extracts the first heading or section identifier from a block of text."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return "General Information"
        
        first_line = lines[0]
        # Match Markdown headers (# Section), numbered headers (1.1 Title), or ALL CAPS headers
        match = re.match(r"^(?:#{1,4}\s+|\d+(?:\.\d+)*\s+)?([A-Za-z0-9\s\-–—/&()]{3,60})", first_line)
        if match:
            return match.group(1).strip()
        return first_line[:50]

    def _split_into_structural_sections(self, text: str) -> List[Dict[str, Any]]:
        # Match markdown headers, numbered sections, or major all-caps section divisions
        header_pattern = r"(?=(?:^#{1,4}\s+|\n(?=\d+\.\d+\s+[A-Z])|\n(?=\d+\s+[A-Z][A-Za-z\s]{3,30}\n)))"
        raw_sections = re.split(header_pattern, text, flags=re.MULTILINE)
        
        structured_blocks = []
        current_section_title = "Overview"

        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            extracted_title = self._extract_section_title(sec_clean)
            if extracted_title:
                current_section_title = extracted_title

            # Preserve markdown tables intact
            is_table = "|" in sec_clean and len([ln for ln in sec_clean.split("\n") if "|" in ln]) >= 2

            if len(sec_clean) > self.max_section_chars and not is_table:
                paragraphs = sec_clean.split("\n\n")
                buffer = ""
                for p in paragraphs:
                    if len(buffer) + len(p) < self.max_section_chars:
                        buffer += ("\n\n" + p if buffer else p)
                    else:
                        if buffer.strip():
                            structured_blocks.append({
                                "section_title": current_section_title,
                                "body": buffer.strip(),
                                "is_table": False
                            })
                        buffer = p
                if buffer.strip():
                    structured_blocks.append({
                        "section_title": current_section_title,
                        "body": buffer.strip(),
                        "is_table": False
                    })
            else:
                structured_blocks.append({
                    "section_title": current_section_title,
                    "body": sec_clean,
                    "is_table": is_table
                })

        return structured_blocks

    def create_chunks(self, parsed_doc: Dict[str, Any], user_id: str = "demo_user") -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        seen_hashes = set()
        doc_id = parsed_doc["doc_id"]
        filename = parsed_doc["filename"]
        doc_display_name = filename.replace(".pdf", "").replace("_", " ")

        for page in parsed_doc["pages"]:
            page_num = page["page_number"]
            raw_text = page.get("text", "").strip()

            if not raw_text:
                continue

            blocks = self._split_into_structural_sections(raw_text)
            for idx, block in enumerate(blocks):
                section_title = block["section_title"]
                body = block["body"]
                content_type = "table" if block["is_table"] else "text"

                # Prepend contextual header breadcrumbs
                enriched_content = (
                    f"[{doc_display_name} > Section: {section_title} | Page {page_num}]\n"
                    f"{body}"
                )

                c_hash = self._hash_content(enriched_content)
                if c_hash not in seen_hashes:
                    seen_hashes.add(c_hash)
                    chunks.append({
                        "id": f"{doc_id}_p{page_num}_struct_{idx}",
                        "content": enriched_content,
                        "metadata": {
                            "doc_id": doc_id,
                            "doc_name": filename,
                            "page_number": page_num,
                            "section_title": section_title,
                            "content_type": content_type,
                            "user_id": user_id,
                            "chunk_type": content_type,
                            "image_path": ""
                        }
                    })

        return chunks