"""
ReAct Agent v1 — Buggy baseline.

Known issues:
  1. Observation from tool execution is NOT appended back into the scratchpad,
     so the LLM never sees what the tool returned and repeats the same Action.
  2. Action parsing does not strip quotes from arguments, causing the calculator
     to receive '"4 / 2"' (with quotes) instead of '4 / 2'.
  3. Final Answer is checked AFTER Action parsing — if the LLM writes both
     in one response, the Action wins and the answer is lost.
  4. No observation logging — impossible to verify tool output from JSON logs.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([\d,.\-]+)", re.IGNORECASE)


class ReActAgent:
    """
    ReAct agent v1 — contains known bugs that cause 0% accuracy on GSM8K.
    Kept for comparison purposes.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[str] = []
        self.last_run_metadata: Dict[str, Any] = {}

    def get_system_prompt(self) -> str:
        tool_lines = "\n".join(
            f"  - {t['name']}: {t['description']}" for t in self.tools
        )
        return f"""You are a precise mathematical reasoning assistant. Solve problems step by step.

You have access to these tools:
{tool_lines}

STRICT OUTPUT FORMAT — follow this exactly for every step:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<argument>)

When you have the final numeric answer, output:
Final Answer: <number only, no units, no explanation>

RULES:
1. Always begin with a Thought.
2. Call exactly ONE tool per step.
3. The argument to calculator must be a valid arithmetic expression using only numbers and operators.
4. When you reach the final answer, output "Final Answer: <number>" and stop.
5. Do not invent tools that are not listed above.
"""

    def run(self, user_input: str) -> str:
        scratchpad = ""
        steps_taken = 0
        total_tokens = 0
        total_latency_ms = 0.0
        failure_type: Optional[str] = None
        final_answer: Optional[str] = None

        logger.log_event("AGENT_START", {
            "input_preview": user_input[:100],
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
        })

        while steps_taken < self.max_steps:
            current_prompt = user_input
            if scratchpad:
                current_prompt = user_input + "\n\n" + scratchpad

            try:
                result = self.llm.generate(
                    current_prompt,
                    system_prompt=self.get_system_prompt(),
                )
            except Exception as e:
                logger.log_event("AGENT_LLM_ERROR", {"error": str(e), "step": steps_taken})
                failure_type = "timeout"
                break

            usage = result.get("usage", {})
            step_tokens = (
                usage.get("total_tokens")
                or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
            total_tokens += step_tokens
            total_latency_ms += result.get("latency_ms", 0)

            tracker.track_request(
                provider=result.get("provider", "openai"),
                model=self.llm.model_name,
                usage=usage,
                latency_ms=result.get("latency_ms", 0),
            )

            llm_text = result.get("content", "")

            logger.log_event("AGENT_STEP", {
                "step": steps_taken,
                "output_preview": llm_text[:200],
            })

            # BUG 3: Action is parsed BEFORE Final Answer — if LLM writes both,
            #         Action wins and the final answer is never captured.
            action_tuple = self._parse_action(llm_text)

            if action_tuple is not None:
                tool_name, args_string = action_tuple

                # BUG 2: args_string still has surrounding quotes, e.g. '"4 / 2"'
                observation_text = self._execute_tool(tool_name, args_string)

                # BUG 1: Observation is NOT appended to scratchpad.
                #         The LLM never sees the tool result, so it repeats the same Action.
                scratchpad += llm_text + "\n"
                # BUG 4: No observation logging — can't verify tool output in logs.
                steps_taken += 1
                continue

            final_answer = self._parse_final_answer(llm_text)
            if final_answer is not None:
                scratchpad += llm_text + "\n"
                break

            # No Action and no Final Answer — give a hint
            failure_type = "parse_error"
            scratchpad += llm_text + "\n"
            steps_taken += 1

        if final_answer is None:
            if failure_type is None:
                failure_type = "timeout"
            final_answer = "[Timeout: max steps reached without Final Answer]"

        self.last_run_metadata = {
            "steps_taken": steps_taken,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "failure_type": failure_type,
        }

        logger.log_event("AGENT_END", {
            "steps_taken": steps_taken,
            "failure_type": failure_type,
            "final_answer": final_answer,
        })

        return final_answer

    def _execute_tool(self, tool_name: str, args: str) -> str:
        for tool in self.tools:
            if tool["name"].lower() == tool_name.lower():
                fn = tool.get("fn")
                if fn is None:
                    return f"Error: tool '{tool_name}' has no callable 'fn' key."
                try:
                    return str(fn(args))
                except Exception as e:
                    return f"Error executing {tool_name}: {e}"
        available = [t["name"] for t in self.tools]
        return f"Tool '{tool_name}' not found. Available tools: {available}"

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                action_part = stripped[7:].strip()
                paren_idx = action_part.find("(")
                if paren_idx == -1:
                    continue
                tool_name = action_part[:paren_idx].strip()
                if not tool_name:
                    continue
                last_paren = action_part.rfind(")")
                if last_paren == -1:
                    args = action_part[paren_idx + 1:].strip()
                else:
                    args = action_part[paren_idx + 1:last_paren].strip()
                # BUG 2: quotes are NOT stripped from args
                return tool_name, args
        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = FINAL_ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return None
