import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)

class VectorDB:
    def __init__(self, reset: bool = False):
        # 1. SETTINGS
        self.db_path = str(settings.VECTOR_DB_DIR)
        
        # 2. THE RESET (The "Caveman" Fix)
        # We delete the folder BEFORE we initialize the DB class.
        # This prevents "AttributeError" because we aren't closing a DB that doesn't exist yet.
        

        # 3. EMBEDDINGS
        self.embedding_function = OllamaEmbeddings(
            model="nomic-embed-text", 
            base_url="http://localhost:11434"
        )
        
        # 4. INITIALIZE CONNECTION
        # Now that the folder is clean (or exists), we connect.
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function,
            collection_name=settings.COLLECTION_NAME
        )

        if reset :
            self._clear_collection()
            
    def _clear_collection(self):
        """
        Safely clears the database by deleting all documents inside it,
        without destroying the actual database file (avoiding locks).
        """
        try:
            # Get all existing IDs
            existing_data = self.db.get()
            if existing_data['ids']:
                logger.warning(f"Deleting {len(existing_data['ids'])} existing documents...")
                self.db.delete(ids=existing_data['ids'])
                logger.info("Vector DB Cleared (Soft Reset).")
            else:
                logger.info("Vector DB was already empty.")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    def ingest_documents(self, docs):
        """
        Takes raw documents, splits them, and stores them in ChromaDB.
        """
        if not docs:
            return

        # 1. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} docs into {len(chunks)} chunks.")

        # 2. Add to DB
        if chunks:
            self.db.add_documents(chunks)
            logger.info(f"Successfully stored {len(chunks)} chunks.")

    def search(self, query: str, k: int = 4):
        logger.info(f"Searching for: '{query}'")
        return self.db.similarity_search(query, k=k)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test the reset functionality
    print("Testing DB Reset...")
    vdb = VectorDB(reset=True)
    print("DB Initialized.")