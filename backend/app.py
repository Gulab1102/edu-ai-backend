from dotenv import load_dotenv
load_dotenv()   # MUST be first

from flask import Flask, request, jsonify
from flask_cors import CORS

from rag import build_index, get_relevant_context
from intent import needs_planning
from planner import create_plan
from llm import call_llm


app = Flask(__name__)
CORS(app)

# Build vector DB once at startup
build_index()


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()
    role = data.get("role", "student")

    if not question:
        return jsonify({"answer": "Please ask a valid question."})

    # -----------------------------
    # RAG: fetch chapter context
    # -----------------------------
    context = get_relevant_context(question)

    # -----------------------------
    # Planning (advanced RAG)
    # -----------------------------
    if needs_planning(question):
        plan = create_plan(role, context, question)
    else:
        plan = "Explain directly from the chapter."

    # -----------------------------
    # FINAL PROMPT (FIXED PROPERLY)
    # -----------------------------
    final_prompt = f"""
You are a Class 10 NCERT Mathematics teacher.

Chapter: Quadratic Equations

STRICT RULES (DO NOT IGNORE):
- Answer ONLY using the provided NCERT chapter content.
- If a specific method is asked, use ONLY that method.
- Do NOT stop in the middle of any solution.
- Complete explanations exactly like classroom teaching.

Chapter Content:
{context}

Teaching Plan:
{plan}

User Question:
{question}

OUTPUT FORMAT RULES:
- Do not include good morning student or something
- Step-by-step explanation
- Clear headings
- Complete answers only
"""

    answer = call_llm(final_prompt)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
