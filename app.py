import streamlit as st
import os
import shutil
from pathlib import Path
from PIL import Image

from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine
from src.config import UPLOADS_DIR, EXTRACTED_IMAGES_DIR

# ---------------------------------------------------------
# Page Configuration & Whimsical Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Manualy — Visual Manual Workspace",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Whimsical, modern pastel aesthetic)
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        color: #0F172A !important;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 4px;
        border: 1px solid #C7D2FE;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = ManualVectorStore()

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = ManualyRAGEngine(vector_store=st.session_state.vector_store)

if "parsed_doc" not in st.session_state:
    st.session_state.parsed_doc = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# Sidebar: Manual Upload & Summary
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='brand-title'>📖 Manualy</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-tagline'>Turn manuals into interactive visual knowledge.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop your PDF manual here", type=["pdf"])

    if uploaded_file and (st.session_state.parsed_doc is None or st.session_state.parsed_doc.get("filename") != uploaded_file.name):
        save_path = UPLOADS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status("🔮 Reading and indexing visual knowledge...", expanded=True) as status:
            st.write("📄 Extracting text & vector schematics...")
            parser = PDFManualParser(str(save_path))
            parsed = parser.extract()
            
            st.write("🔍 Understanding diagrams via Vision LLM...")
            chunker = HybridManualChunker()
            chunks = chunker.create_chunks(parsed, user_id="demo_user")
            
            st.write("🧠 Storing chunks in ChromaDB...")
            st.session_state.vector_store.index_chunks(chunks)
            st.session_state.parsed_doc = parsed
            
            status.update(label="✨ Document ready to explore!", state="complete", expanded=False)
            st.rerun()

    if st.session_state.parsed_doc:
        st.markdown("---")
        doc = st.session_state.parsed_doc
        st.subheader("📊 Document Overview")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{doc['total_pages']}</div><div class='metric-label'>Pages</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{doc['total_images_extracted']}</div><div class='metric-label'>Figures</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Workspace
# ---------------------------------------------------------
if not st.session_state.parsed_doc:
    st.info("👈 Upload a technical or product manual PDF in the sidebar to start exploring.")
else:
    doc = st.session_state.parsed_doc
    
    # 1. Visual Index Gallery (Horizontal Carousel of Extracted Figures)
    st.markdown("### 🖼️ Visual Document Index")
    st.caption("Extracted diagrams, schematics, and exploded views from your manual:")
    
    all_images = []
    for page in doc["pages"]:
        for img_path in page["image_paths"]:
            all_images.append({
                "page": page["page_number"],
                "path": img_path
            })
            
    if all_images:
        gallery_cols = st.columns(min(len(all_images), 4))
        for idx, item in enumerate(all_images[:8]):
            col = gallery_cols[idx % 4]
            with col:
                if Path(item["path"]).exists():
                    st.image(item["path"], caption=f"Page {item['page']} Figure", use_container_width=True)
    else:
        st.write("No raster diagrams detected. Text-first manual.")

    st.markdown("---")
    
    # 2. Q&A Workspace
    st.markdown("### 💬 Ask Manualy")

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show Visual Evidence & Citations if available
            if msg.get("citations"):
                st.markdown("**Sources:**")
                citation_html = "".join([f"<span class='citation-badge'>📄 Page {c['page']} ({c['type']})</span>" for c in msg["citations"]])
                st.markdown(citation_html, unsafe_allow_html=True)
                
            if msg.get("visual_evidence"):
                st.markdown("**Visual Evidence:**")
                ev_cols = st.columns(len(msg["visual_evidence"]))
                for col_idx, img_p in enumerate(msg["visual_evidence"]):
                    if Path(img_p).exists():
                        ev_cols[col_idx].image(img_p, caption="Referenced Diagram", use_container_width=True)

    # Chat Input Box
    if user_query := st.chat_input("Ask about parts, diagrams, wiring, or instructions..."):
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Guarded Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching manual & schematics..."):
                response = st.session_state.rag_engine.answer_question(
                    query=user_query, 
                    user_id="demo_user"
                )
                
                st.markdown(response["answer"])
                
                # Render Citations & Evidence
                if response.get("citations"):
                    st.markdown("**Sources:**")
                    citation_html = "".join([f"<span class='citation-badge'>📄 Page {c['page']} ({c['type']})</span>" for c in response["citations"]])
                    st.markdown(citation_html, unsafe_allow_html=True)
                    
                if response.get("visual_evidence"):
                    st.markdown("**Visual Evidence:**")
                    ev_cols = st.columns(len(response["visual_evidence"]))
                    for col_idx, img_p in enumerate(response["visual_evidence"]):
                        if Path(img_p).exists():
                            ev_cols[col_idx].image(img_p, caption="Referenced Diagram", use_container_width=True)

        # Save assistant turn to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "citations": response.get("citations", []),
            "visual_evidence": response.get("visual_evidence", [])
        })