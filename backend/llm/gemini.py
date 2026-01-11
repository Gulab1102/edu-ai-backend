import os
from dotenv import load_dotenv
from google import genai
from llm.base import BaseLLM
from logger import get_logger

load_dotenv()
log = get_logger("GEMINI_LLM")

class GeminiLLM(BaseLLM):
    def __init__(self, model="models/gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        log.info(f"Using Gemini TEXT model: {model}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}"
        )
        return response.text
