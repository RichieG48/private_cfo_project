import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from src.utils.logger import get_logger

# We will set up logging properly later, but this prevents import errors
import logging
logger = logging.getLogger(__name__)

class IngestionManager:
    """
    Unified interface for loading financial documents (PDFs, Excel).
    """
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Detects file type and delegates to the specific loader.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        extension = file_path.split('.')[-1].lower()
        logger.info(f"Ingesting file: {file_path} (Type: {extension})")

        if extension == "pdf":
            return self._load_pdf(file_path)
        elif extension in ["xlsx", "xls"]:
            return self._load_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads PDF. 
        Note: Simple PyPDFLoader splits by page. 
        For complex tables, we might upgrade this later.
        """
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            logger.info(f"Successfully loaded PDF with {len(docs)} pages.")
            
            # Add metadata so we know where this came from later
            for doc in docs:
                doc.metadata["source_type"] = "financial_report"
                
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return []

    def _load_excel(self, file_path: str) -> List[Document]:
        """
        Loads Excel. Treats each row/sheet as a potential document.
        """
        try:
            # UnstructuredExcelLoader creates a text representation of the spreadsheet
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            docs = loader.load()
            logger.info(f"Successfully loaded Excel with {len(docs)} elements.")
            return docs
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            return []

# Simple test block (runs only if you execute this file directly)
if __name__ == "__main__":
    # Create a dummy file to test logic if you don't have one
    ingestor = IngestionManager()
    print("Ingestion Engine Initialized. Ready for files.")