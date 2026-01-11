def needs_planning(question: str) -> bool:
    keywords = [
        "generate",
        "questions",
        "overview",
        "explain chapter",
        "teach",
        "all methods",
        "list"
    ]
    q = question.lower()
    return any(k in q for k in keywords)
