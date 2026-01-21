from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from src.core.llm import LLMFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 1. Define the Strict Output Structure
# This tells the LLM exactly what keys we need in the JSON.
class RouteDecision(BaseModel):
    classification: Literal["sensitive", "generic"] = Field(
        description="The category of the query. 'sensitive' for financial data/PII. 'generic' for general knowledge/math."
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen."
    )

class SovereignRouter:
    def __init__(self):
        # We use the LOCAL model. 
        # Crucial: We set format="json" to enforce structured output.
        self.llm = LLMFactory.get_local_model(temperature=0, format="json")
        
        # The Instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Data Privacy Officer.
            Analyze the user's query and classify it into one of two JSON categories:
            
            1. 'sensitive': Queries requiring access to private uploaded documents, balance sheets, specific company numbers, or PII.
            2. 'generic': Queries about general economic theory, formulas, Python coding, or public knowledge.
            
            Output ONLY valid JSON matching this schema:
            {{
                "classification": "sensitive" | "generic",
                "reasoning": "your reasoning here"
            }}"""),
            ("human", "Query: {query}")
        ])
        
        # The Chain (using the standard parser since we requested JSON mode)
        self.parser = JsonOutputParser(pydantic_object=RouteDecision)
        self.chain = self.prompt | self.llm | self.parser

    def route_query(self, query: str) -> str:
        """
        Determines if a query should go to the Local RAG or Cloud Brain.
        """
        print(f"Analyzing query: '{query}'...")
        try:
            decision = self.chain.invoke({"query": query})
            
            # Defensive coding: Ensure we got a dictionary back
            if isinstance(decision, dict):
                classification = decision.get("classification", "sensitive")
                reason = decision.get("reasoning", "No reasoning provided")
            else:
                # Fallback
                classification = "sensitive"
                reason = "Invalid output format"

            logger.info(f"Routing Decision: {classification.upper()} | Reason: {reason}")
            
            return classification

        except Exception as e:
            logger.error(f"Router Error: {e}. Defaulting to SENSITIVE (Local).")
            return "sensitive"

# Test Block
if __name__ == "__main__":
    router = SovereignRouter()
    
    # Test 1: Sensitive
    q1 = "What is the salary of the VP of Engineering in the PDF?"
    print(f"\nQuery: {q1}")
    result1 = router.route_query(q1)
    print(f"Result: {result1}")
    
    # Test 2: Generic
    q2 = "Write a Python script to calculate the Fibonacci sequence."
    print(f"\nQuery: {q2}")
    result2 = router.route_query(q2)
    print(f"Result: {result2}")