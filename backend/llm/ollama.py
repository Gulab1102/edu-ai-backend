import requests
from llm.base import BaseLLM
from logger import get_logger
from config import OLLAMA_MODEL

log = get_logger("OLLAMA")

class OllamaLLM(BaseLLM):
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = OLLAMA_MODEL
        log.info(f"Using Ollama model: {self.model}")

    def generate(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()

        return response.json()["response"]
