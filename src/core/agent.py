from src.core.router import SovereignRouter
from src.core.rag import LocalRAG
from src.core.cloud_agent import CloudAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PrivateCFO:
    def __init__(self):
        self.router = SovereignRouter()
        self.local_rag = LocalRAG()
        self.cloud_agent = CloudAgent()

    def process_query(self, query: str) -> dict:
        """
        Orchestrates the flow: Router -> Branch -> Answer
        Returns a dictionary with 'answer', 'source', and 'routing_decision'
        """
        # 1. Route
        routing_decision = self.router.route_query(query)
        
        # 2. Dispatch
        if routing_decision == "sensitive":
            answer = self.local_rag.ask(query)
            source = "Local Mistral 7B (Privacy Shield)"
        else:
            answer = self.cloud_agent.ask(query)
            source = "Gemini (Cloud Intelligence)"

        return {
            "answer": answer,
            "source": source,
            "routing_decision": routing_decision
        }

# Quick Test
if __name__ == "__main__":
    cfo = PrivateCFO()
    #print(cfo.process_query("What is the revenue in the PDF?"))
    print(cfo.process_query("Explain the formula for NPV."))