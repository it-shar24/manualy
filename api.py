from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import shutil

from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine
from src.config import UPLOADS_DIR, EXTRACTED_IMAGES_DIR

app = FastAPI(title="Manualy API")

# Enable CORS for Lovable local dev and production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted diagrams directly to the React frontend
app.mount("/images", StaticFiles(directory=str(EXTRACTED_IMAGES_DIR)), name="images")

vector_store = ManualVectorStore()
rag_engine = ManualyRAGEngine(vector_store=vector_store)

class ChatRequest(BaseModel):
    query: str
    user_id: str = "demo_user"

@app.post("/api/upload")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF manuals are supported.")
    
    file_path = UPLOADS_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    parser = PDFManualParser(str(file_path))
    parsed = parser.extract()
    
    chunker = HybridManualChunker()
    chunks = chunker.create_chunks(parsed, user_id="demo_user")
    vector_store.index_chunks(chunks)
    
    # Map image paths to web-accessible URLs
    gallery = []
    for page in parsed["pages"]:
        for img_path in page["image_paths"]:
            p = Path(img_path)
            # URL: /images/<doc_id>/<filename>
            gallery.append({
                "page": page["page_number"],
                "url": f"http://localhost:8000/images/{p.parent.name}/{p.name}"
            })
            
    return {
        "status": "success",
        "doc_id": parsed["doc_id"],
        "filename": parsed["filename"],
        "total_pages": parsed["total_pages"],
        "total_images": parsed["total_images_extracted"],
        "gallery": gallery
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    result = rag_engine.answer_question(query=req.query, user_id=req.user_id)
    
    # Convert local evidence image paths to static URLs
    visual_evidence_urls = []
    for img_path in result.get("visual_evidence", []):
        p = Path(img_path)
        visual_evidence_urls.append(f"http://localhost:8000/images/{p.parent.name}/{p.name}")
        
    return {
        "status": result["status"],
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "visual_evidence": visual_evidence_urls
    }