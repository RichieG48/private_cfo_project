from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm import LLMFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CloudAgent:
    def __init__(self):
        # This uses the API Key from .env
        self.llm = LLMFactory.get_cloud_model(temperature=0.7)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert Financial Mathematician and Programmer.
        
        User Query: {query}
        
        Provide a detailed, high-level answer. 
        If the user asks for code, provide Python code.
        If the user asks for financial theory, explain it clearly.
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, query: str):
        logger.info(f"Routing to CLOUD Brain: '{query}'")
        try:
            return self.chain.invoke({"query": query})
        except Exception as e:
            logger.error(f"Cloud Error: {e}")
            return "I could not reach the Cloud Brain. Please check your internet or API key."