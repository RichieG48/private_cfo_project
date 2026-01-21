from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import LLMFactory
from src.ingestion.vector_db import VectorDB
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LocalRAG:
    def __init__(self):
        # 1. The Brain (Mistral 7B Local)
        self.llm = LLMFactory.get_local_model(temperature=0)
        
        # 2. The Knowledge Base (ChromaDB)
        self.vector_db = VectorDB() 
        # Note: We rely on the DB being already populated by your previous run!
        
        # 3. The Retriever (The search engine)
        self.retriever = self.vector_db.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Fetch top 4 chunks
        )
        
        # 4. The Prompt (The Instructions)
        self.template = """You are a Financial Analyst Assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. 
        Keep the answer concise and professional.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
        # 5. The Chain (LCEL)
        # formatting_func just joins the list of docs into a single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query: str):
        logger.info(f"Running Local RAG for: '{query}'")
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return "I encountered an error processing your request locally."

# Test the Full Loop
if __name__ == "__main__":
    rag = LocalRAG()
    
    # This query matches what you just indexed
    query = "What is the policy for revenue recognition?"
    
    print(f"\nQuestion: {query}")
    print("Thinking...")
    answer = rag.ask(query)
    
    print("\n=== FINAL ANSWER ===")
    print(answer)