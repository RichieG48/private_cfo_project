from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
from src.core.llm import LLMFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    classification: Literal["sensitive", "generic"] = Field(
        description="'sensitive' for financial data/PII, 'generic' for general knowledge/math."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this classification was chosen."
    )


class SovereignRouter:
    """
    Classifies queries as 'sensitive' (stay local) or 'generic' (may go to the cloud).
    Fails closed: any error or malformed output is treated as 'sensitive'.
    """

    def __init__(self):
        self.llm = LLMFactory.get_local_model(temperature=0, format="json")

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

        self.chain = self.prompt | self.llm | JsonOutputParser()

    def route_query(self, query: str) -> Literal["sensitive", "generic"]:
        """
        Determines if a query should go to the Local RAG or the Cloud model.
        """
        try:
            decision = RouteDecision.model_validate(self.chain.invoke({"query": query}))
        except ValidationError as e:
            logger.warning(f"Router returned invalid output ({e.error_count()} errors). Defaulting to SENSITIVE.")
            return "sensitive"
        except Exception as e:
            logger.error(f"Router error: {e}. Defaulting to SENSITIVE.")
            return "sensitive"

        logger.info(f"Routing decision: {decision.classification.upper()} | Reason: {decision.reasoning}")
        return decision.classification


if __name__ == "__main__":
    router = SovereignRouter()

    for query in [
        "What is the salary of the VP of Engineering in the PDF?",
        "Write a Python script to calculate the Fibonacci sequence.",
    ]:
        print(f"\nQuery: {query}")
        print(f"Result: {router.route_query(query)}")
