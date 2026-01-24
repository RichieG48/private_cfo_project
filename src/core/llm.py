"""
LLM Factory: Centralizes the creation of Local and Cloud models.
This allows us to switch providers (e.g., Llama 3 -> Mistral) in one place.
"""
import os
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from src.config import settings

class LLMFactory:
    @staticmethod
    def get_local_model(temperature: float = 0.0, format: str = None):
        """
        Returns the Local LLM Model.
        Used for: Routing, PII scrubbing, Sensitive QA.
        """
        return ChatOllama(
            model=settings.LOCAL_MODEL_NAME, 
            temperature=temperature,
            format=format, # Useful for forcing JSON output
            base_url="http://localhost:11434"
        )

    @staticmethod
    def get_cloud_model(temperature: float = 0.7):
        """
        Returns the Cloud Model (Mistral API).
        Used for: Complex generic reasoning, Coding, Creative writing.
        """
        api_key = settings.GEMINI_API_KEY
        # If no key is found, we fall back to local to prevent crashes
        if not api_key:
            print("⚠️ WARNING: No GEMINI_API_KEY found. Falling back to Local Model.")
            return LLMFactory.get_local_model(temperature)
            
        return ChatGoogleGenerativeAI(
            model=settings.CLOUD_MODEL_NAME,
            temperature=temperature,
            api_key=api_key
        )

# Simple test to verify connection
if __name__ == "__main__":
    print("Testing Local Ollama Connection...")
    try:
        model = LLMFactory.get_local_model()
        response = model.invoke("Say 'System Operational' if you can hear me.")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Ensure Ollama app is running!")