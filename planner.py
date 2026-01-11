from llm import call_llm

def create_plan(role, context, question):
    prompt = f"""
You are an expert NCERT Class 10 Maths teacher.

Chapter: Quadratic Equations

Chapter Content:
{context}

User role: {role}
User request:
"{question}"

Before answering, create a clear plan.
If the request involves:
- overview → plan sections
- generating many questions → plan subtopics
- teaching → plan explanation order

Return ONLY bullet points.
Do NOT answer the question.
"""
    return call_llm(prompt, max_tokens=500)
