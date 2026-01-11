import os
from pypdf import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR, "data", "class10", "maths", "quadratic_equations.pdf"
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR, "vector_db", "class10_maths_quadratic_equations"
)

# -----------------------------
# Embedding model
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load PDF text
# -----------------------------
def load_pdf_text():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    reader = PdfReader(PDF_PATH)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise RuntimeError("❌ No text extracted from PDF")

    return text


# -----------------------------
# Build Vector DB
# -----------------------------
def build_index():
    if os.path.exists(VECTOR_DB_PATH):
        print("✅ Vector DB already exists")
        return

    print("📚 Building vector database...")

    text = load_pdf_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            "Method",
            "METHOD",
            "Example",
            "Solution",
            "Steps",
            "STEP"
        ]
    )

    docs = splitter.create_documents([text])

    Chroma.from_documents(
        docs,
        embedding,
        persist_directory=VECTOR_DB_PATH
    )

    print("✅ Vector DB created successfully")


# -----------------------------
# Retrieve relevant context
# -----------------------------
def get_relevant_context(query, k=20):
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding
    )

    docs = vectordb.similarity_search(query, k=k)

    # ✅ FORCE method coverage if methods are asked
    if "method" in query.lower():
        extra_docs = vectordb.similarity_search(
            "methods to solve quadratic equation",
            k=10
        )
        docs = docs + extra_docs

    return "\n\n".join(doc.page_content for doc in docs)
