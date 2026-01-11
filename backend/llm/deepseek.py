import os
import requests
from dotenv import load_dotenv
from llm.base import BaseLLM
from logger import get_logger

load_dotenv()
log = get_logger("DEEPSEEK_LLM")

class DeepSeekLLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, system_prompt, user_prompt):
        log.info("Calling DeepSeek API")

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()

        log.info("DeepSeek response received")
        return response.json()["choices"][0]["message"]["content"]
