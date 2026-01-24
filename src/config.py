import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # --- Paths ---
    # Dynamically find the project root (one level up from 'src')
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    VECTOR_DB_DIR = DATA_DIR / "vector_store"

    # --- Models ---
    # The local model for routing and sensitive data (Must be pulled in Ollama)
    LOCAL_MODEL_NAME = "mistral" 
    # The powerful cloud model for generic reasoning
    CLOUD_MODEL_NAME = "gemini-2.5-flash"
    
    # --- Vector Store ---
    COLLECTION_NAME = "finance_docs"
    # Size of text chunks (balance between context and retrieval precision)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # --- API Keys ---
    GEMINI_API_KEY = "AIzaSyCjT1iyuqopZkYxBb0DCo5YA87gDsow2uo"

# Create a global config instance
settings = Config()

# Ensure directories exist
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)