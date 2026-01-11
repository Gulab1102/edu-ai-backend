def detect_intent(query: str):
    q = query.lower()

    # Explain a specific question
    if "explain" in q and "question" in q:
        return "EXPLAIN_QUESTION"

    # Explain full chapter
    if "explain" in q or "chapter" in q:
        return "EXPLAIN_CHAPTER"

    # Generate questions / paper
    if "question paper" in q or "generate question" in q or "test" in q:
        return "QUESTIONS"

    return "DOUBT"
