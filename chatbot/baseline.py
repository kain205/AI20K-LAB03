from typing import Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

CHATBOT_SYSTEM_PROMPT = """You are a math problem solver.
Solve the given math word problem step by step.
At the very end of your response, write your final numeric answer on its own line in this exact format:
#### <number>
where <number> is the final answer with no units."""


class BaselineChatbot:
    """
    Plain GPT-4o chatbot baseline. No tools, no loop — just one generate() call.

    Stores last_run_metadata in the same shape as ReActAgent so the
    EvaluationRunner can treat both systems uniformly.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.last_run_metadata: Dict[str, Any] = {}

    def run(self, user_input: str) -> str:
        """
        Single-shot generation. Returns the raw LLM content string.
        Populates self.last_run_metadata for the evaluator.
        """
        logger.log_event("CHATBOT_START", {"input_preview": user_input[:100]})

        try:
            result = self.llm.generate(user_input, system_prompt=CHATBOT_SYSTEM_PROMPT)
        except Exception as e:
            logger.log_event("CHATBOT_ERROR", {"error": str(e)})
            self.last_run_metadata = {
                "steps_taken": 1,
                "total_tokens": 0,
                "total_latency_ms": 0.0,
                "failure_type": "timeout",
            }
            return ""

        usage = result.get("usage", {})
        total_tokens = (
            usage.get("total_tokens")
            or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )

        tracker.track_request(
            provider=result.get("provider", "openai"),
            model=self.llm.model_name,
            usage=usage,
            latency_ms=result.get("latency_ms", 0),
        )

        self.last_run_metadata = {
            "steps_taken": 1,
            "total_tokens": total_tokens,
            "total_latency_ms": result.get("latency_ms", 0.0),
            "failure_type": None,
        }

        logger.log_event("CHATBOT_END", self.last_run_metadata)
        return result.get("content", "")
