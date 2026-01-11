from config import LLM_PROVIDER

from llm.ollama import OllamaLLM
from llm.deepseek import DeepSeekLLM

try:
    from llm.gemini import GeminiLLM
except ImportError:
    GeminiLLM = None


def get_llm():
    if LLM_PROVIDER == "ollama":
        return OllamaLLM()

    if LLM_PROVIDER == "deepseek":
        return DeepSeekLLM()

    if LLM_PROVIDER == "gemini":
        if GeminiLLM is None:
            raise ImportError("Gemini dependencies not installed")
        return GeminiLLM()

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
