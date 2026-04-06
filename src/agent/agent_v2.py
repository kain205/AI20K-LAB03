"""
ReAct Agent v2 — Fixed version.

Fixes over v1:
  1. Observation from tool execution IS appended to scratchpad so the LLM
     sees the result on the next iteration.
  2. Quotes are stripped from Action arguments before passing to the tool.
  3. Final Answer is checked BEFORE Action parsing — if the LLM writes both,
     Final Answer wins and the loop terminates.
  4. AGENT_OBSERVATION log event after every tool call for debuggability.
  5. LLM response is truncated to the first Action line — prevents the model
     from pre-generating multiple Thought/Action/Observation cycles.
  6. System prompt explicitly instructs one Thought + one Action per response.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*([\d,.\-]+)", re.IGNORECASE)


class ReActAgent:
    """
    ReAct agent v2 — production-quality Thought-Action-Observation loop.
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

IMPORTANT: Output exactly ONE Thought and ONE Action per response. Stop immediately after the Action line. Do not write the Observation yourself. Do not chain multiple Thought/Action pairs in a single response.
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

            # FIX 5: Truncate to first Action line — discard any content after it
            # so the LLM cannot pre-generate multiple cycles in one response.
            llm_text = self._truncate_to_first_action(llm_text)

            logger.log_event("AGENT_STEP", {
                "step": steps_taken,
                "output_preview": llm_text[:200],
            })

            # FIX 3: Check Final Answer BEFORE Action parsing.
            final_answer = self._parse_final_answer(llm_text)
            if final_answer is not None:
                scratchpad += llm_text + "\n"
                break

            action_tuple = self._parse_action(llm_text)

            if action_tuple is None:
                failure_type = "parse_error"
                recovery = (
                    "Observation: I could not parse your action. "
                    "Please use the exact format: Action: tool_name(argument)\n"
                )
                scratchpad += llm_text + "\n" + recovery
                steps_taken += 1
                continue

            tool_name, args_string = action_tuple

            observation_text = self._execute_tool(tool_name, args_string)

            # FIX 4: Log the observation for debuggability.
            logger.log_event("AGENT_OBSERVATION", {
                "step": steps_taken,
                "tool": tool_name,
                "args": args_string,
                "observation": observation_text[:200],
            })

            if "not found" in observation_text.lower():
                if failure_type is None:
                    failure_type = "hallucination"

            # FIX 1: Append the full Thought+Action AND Observation to scratchpad
            # so the LLM sees what the tool returned on the next iteration.
            scratchpad += llm_text + "\nObservation: " + observation_text + "\n"
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

    def _truncate_to_first_action(self, text: str) -> str:
        """
        Keep everything up to and including the first Action: tool(...) line.
        Discard anything after it so the loop processes one tool call at a time.
        """
        lines = text.split("\n")
        result_lines = []
        for line in lines:
            result_lines.append(line)
            stripped = line.strip()
            if stripped.lower().startswith("action:") and "(" in stripped:
                break
        return "\n".join(result_lines)

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
                # FIX 2: Strip surrounding quotes (single or double) from args.
                if len(args) >= 2 and args[0] in ('"', "'") and args[-1] == args[0]:
                    args = args[1:-1]
                return tool_name, args
        return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = FINAL_ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return None
