import re
from typing import Optional
from evaluation.gsm8k_loader import normalize_number

# For ReAct agent: "Final Answer: 42"
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([\d,.\-]+)", re.IGNORECASE)

# For chatbot following the system-prompt instruction: "#### 42"
GSM8K_FORMAT_RE = re.compile(r"####\s*([\d,.\-]+)")

# Fallback: last standalone number in the text (ignores numbers inside words)
LAST_NUMBER_RE = re.compile(r"(?<!\w)([\d,]+(?:\.\d+)?)(?!\w)")


def extract_chatbot_answer(text: str) -> Optional[str]:
    """
    Extracts a numeric answer from free-form chatbot output.

    Priority:
      1. "#### <n>" — chatbot was instructed to end with this format.
      2. Last standalone number in the text.

    Returns a normalized numeric string, or None.
    """
    if not text:
        return None

    m = GSM8K_FORMAT_RE.search(text)
    if m:
        return normalize_number(m.group(1))

    matches = LAST_NUMBER_RE.findall(text)
    if matches:
        return normalize_number(matches[-1])

    return None


def extract_agent_answer(text: str) -> Optional[str]:
    """
    Extracts a numeric answer from ReAct agent output.

    Priority:
      1. "Final Answer: <n>" — standard ReAct format.
      2. Last standalone number in the text — fallback for when agent.run()
         returns the already-extracted numeric string directly.

    Returns a normalized numeric string, or None if the agent timed out.
    """
    if not text:
        return None
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    # Fallback: parse last number (same logic as chatbot extractor).
    # This handles the case where agent.run() returns just the number.
    matches = LAST_NUMBER_RE.findall(text)
    if matches:
        return normalize_number(matches[-1])
    return None


def answers_match(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Compares predicted answer to ground truth after normalizing both.
    Returns True if they are numerically equal.
    """
    if predicted is None:
        return False
    try:
        return float(predicted.replace(",", "")) == float(ground_truth.replace(",", ""))
    except ValueError:
        return predicted.strip() == ground_truth.strip()
