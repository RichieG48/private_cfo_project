import anthropic
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert Financial Mathematician and Programmer.
Provide a detailed, high-level answer.
If the user asks for code, provide Python code.
If the user asks for financial theory, explain it clearly."""


class CloudAgent:
    """Answers generic (non-sensitive) queries with Claude."""

    def __init__(self):
        # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = anthropic.Anthropic()

    def ask(self, query: str) -> str:
        logger.info("Sending query to cloud model.")
        try:
            response = self.client.beta.messages.create(
                model=settings.CLOUD_MODEL_NAME,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
                # Re-run on Anthropic's recommended fallback model if the request is declined.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            logger.error("Cloud authentication failed.")
            return "Cloud model authentication failed. Please check ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            logger.error("Cloud rate limit hit.")
            return "The cloud model is rate limited right now. Please try again shortly."
        except anthropic.APIStatusError as e:
            logger.error(f"Cloud API error {e.status_code}: {e.message}")
            return "The cloud model returned an error. Please try again."
        except anthropic.APIConnectionError as e:
            logger.error(f"Cloud connection error: {e}")
            return "I could not reach the cloud model. Please check your internet connection."

        if response.stop_reason == "refusal":
            logger.warning("Cloud model declined the request.")
            return "The cloud model declined to answer this request."

        return "".join(block.text for block in response.content if block.type == "text")
