# Agent v1 Failure Analysis

## Summary

The v1 ReAct agent achieved **0% accuracy** on GSM8K evaluation, timing out on every question. The root cause is a broken observation loop: after calling a tool, the tool's result was never fed back into the scratchpad, causing the LLM to repeat the same Action indefinitely until `max_steps` was reached.

## Bug Description

### Primary Bug: Observation Not Appended to Scratchpad

After executing a tool (e.g., `calculator("430 + 320")`), the agent appended only the LLM's Thought+Action text to the scratchpad but **omitted the `Observation: <result>` line**. On the next iteration, the LLM received the same scratchpad without any tool output, so it had no evidence that the tool had already been called — and generated the identical Action again.

```python
# v1 (buggy) — observation discarded
scratchpad += llm_text + "\n"

# v2 (fixed) — observation preserved
scratchpad += llm_text + "\nObservation: " + observation_text + "\n"
```

### Secondary Bug: Quotes Not Stripped from Action Arguments

The action parser extracted arguments with surrounding quotes intact. For example, `Action: calculator("430 + 320")` produced `args = '"430 + 320"'` (with double quotes). The calculator's AST evaluator parsed this as a string constant rather than an arithmetic expression, returning an `UnsafeExpressionError` — which further confused the LLM.

### Tertiary Bug: Final Answer Checked After Action

If the LLM wrote both an `Action:` line and a `Final Answer:` line in the same response, the Action was parsed first and the Final Answer was ignored. This prevented the loop from terminating even when the LLM had the correct answer.

### Missing: No Observation Logging

No `AGENT_OBSERVATION` log event was emitted after tool execution, making it impossible to verify from JSON logs whether the tool returned a valid result.

## Evidence from Logs

Typical log sequence for a single question:

```
AGENT_START  {"input_preview": "Janet's ducks lay 16 eggs per day..."}
AGENT_STEP   {"step": 0, "output_preview": "Thought: I need to calculate...\nAction: calculator(\"430 + 320\")"}
AGENT_STEP   {"step": 1, "output_preview": "Thought: I need to calculate...\nAction: calculator(\"430 + 320\")"}
AGENT_STEP   {"step": 2, "output_preview": "Thought: I need to calculate...\nAction: calculator(\"430 + 320\")"}
AGENT_STEP   {"step": 3, "output_preview": "Thought: I need to calculate...\nAction: calculator(\"430 + 320\")"}
AGENT_STEP   {"step": 4, "output_preview": "Thought: I need to calculate...\nAction: calculator(\"430 + 320\")"}
AGENT_END    {"steps_taken": 5, "failure_type": "timeout", "final_answer": "[Timeout: max steps reached]"}
```

The same `calculator("430 + 320")` call appears 5 times because the LLM never saw the result `750` in its context.

## Metrics (3-sample GSM8K run)

| Metric                  | Value        |
|-------------------------|--------------|
| Accuracy                | 0%           |
| Timeout rate            | 100%         |
| Avg steps per question  | 5 (max)      |
| Avg tokens per question | ~2,156       |
| Questions answered      | 0 / 3        |

## Fixes Applied in v2

1. **Observation appended** — `scratchpad += llm_text + "\nObservation: " + observation_text + "\n"`
2. **Quotes stripped** — surrounding `"` or `'` removed from parsed Action arguments before passing to tool
3. **Final Answer checked first** — `_parse_final_answer()` runs before `_parse_action()`
4. **Observation logged** — `AGENT_OBSERVATION` event emitted after every tool call
5. **Response truncated** — LLM output cut after first `Action:` line to enforce one-tool-per-step
6. **System prompt updated** — explicit instruction to output exactly one Thought + one Action per response
