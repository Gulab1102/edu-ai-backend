import os, pickle, faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from logger import get_logger

log = get_logger("RAG_BUILDER")

def build_rag(pdf_path, save_dir):
    reader = PdfReader(pdf_path)
    chunks = []

    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text:
            continue
        for i in range(0, len(text), 500):
            chunks.append({
                "page": page_no,
                "content": text[i:i+500]
            })

    log.info(f"Chunks created: {len(chunks)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode([c["content"] for c in chunks])

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, f"{save_dir}/index.faiss")

    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    log.info("RAG build complete")
