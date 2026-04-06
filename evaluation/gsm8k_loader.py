import re
import random
from typing import List, Dict, Optional

ANSWER_EXTRACTION_RE = re.compile(r"####\s*([\d,.\-]+)")


def load_gsm8k(
    split: str = "test",
    sample: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Loads GSM8K from the HuggingFace datasets library.

    Args:
        split:  "train" (7473 examples) or "test" (1319 examples)
        sample: If provided, randomly samples this many questions.
        seed:   Random seed for reproducibility.

    Returns:
        List of dicts with keys:
            "question"   — the word problem string
            "answer_raw" — full answer string including "#### <n>"
            "answer_num" — normalized numeric string (e.g. "1000", "3.14")
    """
    # Import here so a missing `datasets` package fails at runtime, not import
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset("gsm8k", "main", split=split)

    records = []
    for item in dataset:
        answer_num = extract_gsm8k_answer(item["answer"])
        if answer_num is None:
            continue  # skip malformed entries
        records.append({
            "question": item["question"].strip(),
            "answer_raw": item["answer"].strip(),
            "answer_num": answer_num,
        })

    if sample is not None and sample < len(records):
        random.seed(seed)
        records = random.sample(records, sample)

    return records


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """
    Extracts the ground-truth number from a GSM8K answer string.
    GSM8K answers always end with '#### <number>'.

    Returns a normalized numeric string, or None if the pattern is absent.
    """
    match = ANSWER_EXTRACTION_RE.search(answer_text)
    if not match:
        return None
    return normalize_number(match.group(1))


def normalize_number(s: str) -> str:
    """
    Normalizes a numeric string for comparison:
      - Strips commas:  "1,234"  → "1234"
      - Drops trailing zeros: "3.50" → "3.5"
      - Converts whole-number floats: "42.0" → "42"
    """
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f).rstrip("0").rstrip(".")
    except ValueError:
        return s
