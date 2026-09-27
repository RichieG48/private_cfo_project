"""
LLM Factory: Centralizes the creation of Local and Cloud models.
"""
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    @staticmethod
    def get_local_model(temperature: float = 0.0, format: str | None = None):
        """
        Returns the local Ollama model.
        Used for: routing and question answering over private documents.
        """
        return ChatOllama(
            model=settings.LOCAL_MODEL_NAME,
            temperature=temperature,
            format=format,  # "json" forces structured output
            base_url=settings.OLLAMA_BASE_URL,
        )

    @staticmethod
    def get_cloud_model(temperature: float = 0.7):
        """
        Returns the Gemini cloud model.
        Used for: generic reasoning, coding, financial theory.
        Falls back to the local model if GOOGLE_API_KEY is not set.
        """
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set. Falling back to the local model.")
            return LLMFactory.get_local_model(temperature)

        return ChatGoogleGenerativeAI(
            model=settings.CLOUD_MODEL_NAME,
            temperature=temperature,
            api_key=settings.GOOGLE_API_KEY,
        )


if __name__ == "__main__":
    print("Testing local Ollama connection...")
    try:
        model = LLMFactory.get_local_model()
        response = model.invoke("Say 'System Operational' if you can hear me.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure Ollama is running.")
