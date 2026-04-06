# Lab 3: Chatbot vs ReAct Agent — GSM8K Results

## Overview

This lab compares a plain LLM Chatbot against a ReAct (Reason + Act) Agent on the [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) math word-problem benchmark. The agent uses a `calculator` tool inside a Thought-Action-Observation loop.

---

## Results (GSM8K, n=20, gpt-4o)

| Metric | Chatbot | Agent v2 |
|--------|---------|----------|
| Accuracy | 90.0% | **95.0%** |
| Correct / Total | 18/20 | **19/20** |
| Avg Steps | 1.0 | 3.15 |
| Avg Tokens | 346.9 | 1,780.4 |

**ReAct Agent v2 beats Chatbot by +5% accuracy**, at the cost of ~5x more tokens and multi-step latency.

---

## Agent v1 — Bug Analysis (0% Accuracy)

Agent v1 achieved **0% accuracy** (timeout on every question). Root causes:

| Bug | Description | Fix |
|-----|-------------|-----|
| **Observation not appended** | Tool result was never written back to scratchpad — LLM repeated the same Action indefinitely | Append `Observation: <result>` after each tool call |
| **Quotes not stripped** | Args like `"430 + 320"` were passed with surrounding quotes, causing `ast` to return a string instead of a number | Strip `"` / `'` from parsed args |
| **Final Answer checked after Action** | Agent looped even when the correct answer was already in the response | Parse `Final Answer:` before `Action:` |

Typical v1 log for one question (5 identical steps, then timeout):
```
AGENT_STEP  step=0  Action: calculator("430 + 320")
AGENT_STEP  step=1  Action: calculator("430 + 320")   # same — no observation fed back
AGENT_STEP  step=2  Action: calculator("430 + 320")
AGENT_STEP  step=3  Action: calculator("430 + 320")
AGENT_STEP  step=4  Action: calculator("430 + 320")
AGENT_END   failure_type=timeout  final_answer="[Timeout: max steps reached]"
```

---

## Agent v2 — Fixes Applied

1. Observation appended to scratchpad after every tool call
2. Quotes stripped from Action arguments before passing to tool
3. `Final Answer:` check runs before `Action:` parse
4. `AGENT_OBSERVATION` log event added
5. LLM output truncated after first `Action:` line (one tool per step)
6. System prompt updated to enforce one Thought + one Action per response

---

## Key Use Case

**Q13 — Agent correct, Chatbot failed (API timeout):**

> "Janet pays $500 for material, $800 for construction, then 10% insurance. Total?"

Agent v2 reasoning:
```
Step 0: calculator(500 + 800)   → 1300
Step 1: calculator(1300 * 0.10) → 130
Step 2: calculator(1300 + 130)  → 1430
Step 3: Final Answer: 1430  ✓
```

---

## Quick Start

```bash
cp .env.example .env        # add your OPENAI_API_KEY
pip install -r requirements.txt

python run.py                    # Agent v2 (default)
python run.py --mode chatbot     # Chatbot baseline
python run.py --version v1       # Agent v1 (buggy, for reference)
```

Logs are written to `logs/` in JSON format for analysis.

---

## Directory Structure

```
src/
  agent/      # ReAct loop implementation
  chatbot/    # Baseline chatbot
  tools/      # calculator and other tools
  llm/        # Provider abstraction (OpenAI / Gemini / Local)
logs/         # Structured JSON telemetry
report/       # Failure analysis and use-case write-ups
```

---

*Full failure analysis: [`report/group_report/agent_v1_failures.md`](report/group_report/agent_v1_failures.md)*  
*Use cases: [`report/group_report/USE_CASES.md`](report/group_report/USE_CASES.md)*
