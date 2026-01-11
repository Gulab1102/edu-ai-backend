from rag_engine.builder import build_rag

if __name__ == "__main__":
    build_rag(
        "data/class10/maths/quadratic_equations.pdf",
        "vector_db/class10_maths_quadratic"
    )
