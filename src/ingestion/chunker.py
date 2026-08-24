import re
import hashlib
from typing import List, Dict, Any

class HybridManualChunker:
    """
    Strategy 2 (Enhanced): Structure-Aware + Table Header Column Preservation.
    Attaches table headers to individual rows/row-groups and injects fine-grained
    structural breadcrumbs to maximize MRR and Context Precision.
    """
    def __init__(self, max_section_chars: int = 2200):
        self.max_section_chars = max_section_chars

    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def _extract_section_title(self, text: str) -> str:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return "General Information"
        first_line = lines[0]
        match = re.match(r"^(?:#{1,4}\s+|\d+(?:\.\d+)*\s+)?([A-Za-z0-9\s\-–—/&()]{3,60})", first_line)
        if match:
            return match.group(1).strip()
        return first_line[:50]

    def _format_table_content(self, table_text: str) -> str:
        """Ensures markdown tables retain clear column headers and row definitions."""
        lines = [ln.strip() for ln in table_text.split("\n") if ln.strip()]
        if len(lines) < 2 or "|" not in lines[0]:
            return table_text
        
        header = lines[0]
        rows = [ln for ln in lines[1:] if not re.match(r"^\|?[\s\-:|]+\|?$", ln)]
        
        # If the table is long (> 15 rows), group rows with the header attached
        if len(rows) > 12:
            formatted_chunks = []
            chunk_size = 8
            for i in range(0, len(rows), chunk_size):
                sub_rows = rows[i:i + chunk_size]
                formatted_chunks.append(header + "\n|---|---|\n" + "\n".join(sub_rows))
            return "\n\n".join(formatted_chunks)
        return table_text

    def _split_into_structural_sections(self, text: str) -> List[Dict[str, Any]]:
        # Split on markdown headings, numbered sections (e.g. 1.2 Title), or uppercase section names
        header_pattern = r"(?=(?:^#{1,4}\s+|\n(?=\d+\.\d+\s+[A-Z])|\n(?=\d+\s+[A-Z][A-Za-z\s]{3,30}\n)|\n(?=Table\s+\d+:?)))"
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

            is_table = "|" in sec_clean and len([ln for ln in sec_clean.split("\n") if "|" in ln]) >= 2

            if is_table:
                formatted_body = self._format_table_content(sec_clean)
                structured_blocks.append({
                    "section_title": f"Table: {current_section_title}",
                    "body": formatted_body,
                    "is_table": True
                })
            elif len(sec_clean) > self.max_section_chars:
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
                    "is_table": False
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

                enriched_content = (
                    f"[{doc_display_name} > {section_title} | Page {page_num}]\n"
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