"""
LLM Factory: Centralizes the creation of the local Ollama model.
"""
from langchain_ollama import ChatOllama
from src.config import settings

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


if __name__ == "__main__":
    print("Testing local Ollama connection...")
    try:
        model = LLMFactory.get_local_model()
        response = model.invoke("Say 'System Operational' if you can hear me.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure Ollama is running.")
