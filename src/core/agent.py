from src.core.router import SovereignRouter
from src.core.rag import LocalRAG
from src.core.cloud_agent import CloudAgent
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PrivateCFO:
    def __init__(self):
        self.router = SovereignRouter()
        self.local_rag = LocalRAG()
        self.cloud_agent = CloudAgent()

    def process_query(self, query: str) -> dict:
        """
        Orchestrates the flow: Router -> Local RAG or Cloud -> Answer.
        Returns a dictionary with 'answer', 'source', and 'routing_decision'
        """
        routing_decision = self.router.route_query(query)

        # Only an explicit "generic" decision may leave the machine.
        if routing_decision == "generic":
            answer = self.cloud_agent.ask(query)
            source = f"Cloud ({settings.CLOUD_MODEL_NAME})"
        else:
            answer = self.local_rag.ask(query)
            source = f"Local ({settings.LOCAL_MODEL_NAME})"

        return {
            "answer": answer,
            "source": source,
            "routing_decision": routing_decision
        }


if __name__ == "__main__":
    cfo = PrivateCFO()
    print(cfo.process_query("Explain the formula for NPV."))