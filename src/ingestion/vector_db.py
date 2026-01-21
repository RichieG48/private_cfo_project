import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorDB:
    def __init__(self, reset: bool = False):
        """
        Initialize the Local Vector Store (ChromaDB).
        
        :param reset: If True, deletes the existing database to start fresh.
        """
        if reset:
            self._clear_db()
            
        # 1. The Embedding Model
        # We use 'nomic-embed-text' or 'mistral'. 
        # Nomic is specifically designed for RAG and is very lightweight.
        # Make sure to run 'ollama pull nomic-embed-text' first!
        self.embedding_function = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url="http://localhost:11434"
        )
        
        # 2. The Vector Store Connection
        self.db = Chroma(
            persist_directory=str(settings.VECTOR_DB_DIR),
            embedding_function=self.embedding_function,
            collection_name=settings.COLLECTION_NAME
        )

    def _clear_db(self):
        """Deletes the vector DB folder to force a clean slate."""
        if os.path.exists(settings.VECTOR_DB_DIR):
            shutil.rmtree(settings.VECTOR_DB_DIR)
            logger.warning(f"Cleared Vector DB at {settings.VECTOR_DB_DIR}")

    def ingest_documents(self, docs: List[Document]):
        """
        Takes raw documents, splits them, and stores them in ChromaDB.
        """
        if not docs:
            logger.warning("No documents to ingest.")
            return

        # 1. Splitter Strategy
        # We need chunks that fit into the model's context window.
        # chunk_size=1000 chars is a good balance for financial text.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""] # Try to split by paragraph first
        )
        
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} docs into {len(chunks)} chunks.")

        # 2. Batch Insert (Chroma handles this, but good to know)
        # This converts text -> vectors and saves them to disk.
        self.db.add_documents(chunks)
        logger.info(f"Successfully stored {len(chunks)} chunks in Vector DB.")

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Semantic Search: Finds the 'k' most relevant chunks for a query.
        """
        logger.info(f"Searching for: '{query}'")
        results = self.db.similarity_search(query, k=k)
        return results

# Test Block
if __name__ == "__main__":
    # Note: You must pull the embedding model first:
    # ollama pull nomic-embed-text
    
    from src.ingestion.loader import IngestionManager
    
    # 1. Load Data
    loader = IngestionManager()
    # Replace with your actual file
    file_path = "data/raw/Apple_10K_report.pdf" 
    
    if os.path.exists(file_path):
        raw_docs = loader.load_file(file_path)
        
        # 2. Vectorize
        vdb = VectorDB(reset=True)
        vdb.ingest_documents(raw_docs)
        
        # 3. Test Retrieval
        query = "What is the revenue?"
        results = vdb.search(query)
        
        print("\n--- Retrieval Results ---")
        for i, doc in enumerate(results):
            print(f"Match {i+1}: {doc.page_content[:150]}...")
    else:
        print(f"Please place a PDF at {file_path} to test.")