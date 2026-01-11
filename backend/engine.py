from rag_engine.retriever import RAGRetriever
from planner.intent import detect_intent
from planner.lesson_plan import lesson_plan
from planner.question_generator import question_plan
from llm import get_llm

class TeachingEngine:
    def __init__(self, vector_db_path):
        self.rag = RAGRetriever(vector_db_path)
        self.llm = get_llm()

    def handle(self, query):
        intent = detect_intent(query)

        # Always retrieve context from PDF
        chunks = self.rag.retrieve(query)
        context = "\n".join([c["content"] for c in chunks])

        # 1️⃣ Explain full chapter (planned teaching)
        if intent == "EXPLAIN_CHAPTER":
            plan = lesson_plan()
            user_prompt = f"""
Explain the chapter strictly from NCERT.

Teaching Plan:
{plan}

Context:
{context}
"""

        # 2️⃣ Explain a specific QUESTION (teacher style)
        elif intent == "EXPLAIN_QUESTION":
            user_prompt = f"""
Explain the following question step-by-step like a teacher.
Use only the NCERT context.

Rules:
- Start with understanding the question
- Mention given data
- Apply correct method
- Solve step by step
- Write final answer clearly

NCERT Context:
{context}

Question:
{query}
"""

        # 3️⃣ Generate question paper
        elif intent == "QUESTIONS":
            qp = question_plan()
            user_prompt = f"""
Generate questions strictly from NCERT.

Difficulty distribution:
{qp}

Context:
{context}
"""

        # 4️⃣ Normal doubt
        else:
            user_prompt = f"""
Answer the question like a teacher.
Use only NCERT context.

Context:
{context}

Question:
{query}
"""

        return self.llm.generate(
            "You are a strict NCERT Class teacher.",
            user_prompt
        )
