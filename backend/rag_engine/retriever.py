import faiss, pickle
from sentence_transformers import SentenceTransformer
from logger import get_logger

log = get_logger("RAG")

class RAGRetriever:
    def __init__(self, db_path):
        self.index = faiss.read_index(f"{db_path}/index.faiss")
        with open(f"{db_path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, top_k=8):
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in idx[0]]
