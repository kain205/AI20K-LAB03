import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from src.core.llm_provider import LLMProvider
from src.tools import DEFAULT_TOOLS
from chatbot.baseline import BaselineChatbot
from evaluation.gsm8k_loader import load_gsm8k
from evaluation.answer_extractor import (
    extract_chatbot_answer,
    extract_agent_answer,
    answers_match,
)
from src.telemetry.logger import logger


class EvaluationRunner:
    """
    Orchestrates evaluation of BaselineChatbot and ReActAgent on GSM8K.

    Both systems share the same LLM instance and are evaluated sequentially
    on the same set of questions, making the comparison fair.
    """

    def __init__(
        self,
        llm: LLMProvider,
        sample_size: int = 50,
        max_agent_steps: int = 5,
        results_dir: str = "results",
        gsm8k_split: str = "test",
        seed: int = 42,
        request_delay: float = 0.0,
        agent_version: str = "v2",
    ) -> None:
        self.llm = llm
        self.sample_size = sample_size
        self.max_agent_steps = max_agent_steps
        self.results_dir = results_dir
        self.gsm8k_split = gsm8k_split
        self.seed = seed
        self.request_delay = request_delay  # seconds to sleep between questions
        self.agent_version = agent_version

        os.makedirs(results_dir, exist_ok=True)

        self.chatbot = BaselineChatbot(llm=llm)

        if agent_version == "v1":
            from src.agent.agent_v1 import ReActAgent
        else:
            from src.agent.agent_v2 import ReActAgent

        self.agent = ReActAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_steps=max_agent_steps,
        )

    def run(self) -> Dict[str, Any]:
        """
        Loads dataset, evaluates both systems, saves results, prints table.
        Returns the full results dict.
        """
        questions = load_gsm8k(
            split=self.gsm8k_split,
            sample=self.sample_size,
            seed=self.seed,
        )

        logger.log_event("EVAL_START", {
            "sample_size": len(questions),
            "split": self.gsm8k_split,
            "model": self.llm.model_name,
        })

        chatbot_results: List[Dict] = []
        agent_results: List[Dict] = []

        for i, q in enumerate(tqdm(questions, desc="Evaluating")):
            question = q["question"]
            ground_truth = q["answer_num"]

            cb_record = self._evaluate_chatbot(i, question, ground_truth)
            chatbot_results.append(cb_record)

            if self.request_delay > 0:
                time.sleep(self.request_delay)

            ag_record = self._evaluate_agent(i, question, ground_truth)
            agent_results.append(ag_record)

            if self.request_delay > 0:
                time.sleep(self.request_delay)

        results = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "model": self.llm.model_name,
                "sample_size": len(questions),
                "gsm8k_split": self.gsm8k_split,
                "max_agent_steps": self.max_agent_steps,
                "seed": self.seed,
            },
            "chatbot": {
                "per_question": chatbot_results,
                "aggregate": self._aggregate(chatbot_results),
            },
            "agent": {
                "per_question": agent_results,
                "aggregate": self._aggregate(agent_results),
            },
        }

        self._save_results(results)
        self._print_comparison_table(results)

        logger.log_event("EVAL_END", {
            "chatbot_accuracy": results["chatbot"]["aggregate"]["accuracy_pct"],
            "agent_accuracy": results["agent"]["aggregate"]["accuracy_pct"],
        })

        return results

    # ------------------------------------------------------------------
    # Per-question evaluation
    # ------------------------------------------------------------------

    def _evaluate_chatbot(self, idx: int, question: str, ground_truth: str) -> Dict:
        output = self.chatbot.run(question)
        meta = self.chatbot.last_run_metadata
        predicted = extract_chatbot_answer(output)
        correct = answers_match(predicted, ground_truth)

        failure_type = meta.get("failure_type")
        if not correct and failure_type is None:
            failure_type = "wrong_answer"

        record = {
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "steps_taken": meta["steps_taken"],
            "total_tokens": meta["total_tokens"],
            "latency_ms": meta["total_latency_ms"],
            "failure_type": failure_type,
            "raw_output": output,
        }
        logger.log_event(
            "CHATBOT_RESULT",
            {k: v for k, v in record.items() if k != "raw_output"},
        )
        return record

    def _evaluate_agent(self, idx: int, question: str, ground_truth: str) -> Dict:
        output = self.agent.run(question)
        meta = self.agent.last_run_metadata
        predicted = extract_agent_answer(output)
        correct = answers_match(predicted, ground_truth)

        failure_type = meta.get("failure_type")
        if not correct and failure_type is None:
            failure_type = "wrong_answer"

        record = {
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "steps_taken": meta["steps_taken"],
            "total_tokens": meta["total_tokens"],
            "latency_ms": meta["total_latency_ms"],
            "failure_type": failure_type,
            "raw_output": output,
        }
        logger.log_event(
            "AGENT_RESULT",
            {k: v for k, v in record.items() if k != "raw_output"},
        )
        return record

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, records: List[Dict]) -> Dict:
        n = len(records)
        if n == 0:
            return {}

        correct_count = sum(1 for r in records if r["correct"])
        failure_counts: Dict[str, int] = {}
        for r in records:
            ft = r["failure_type"] or "none"
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

        return {
            "total_questions": n,
            "correct": correct_count,
            "accuracy_pct": round(correct_count / n * 100, 2),
            "avg_steps": round(sum(r["steps_taken"] for r in records) / n, 2),
            "avg_tokens": round(sum(r["total_tokens"] for r in records) / n, 1),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 1),
            "failure_breakdown": failure_counts,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_results(self, results: Dict) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.results_dir, f"gsm8k_results_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {path}")

    def _print_comparison_table(self, results: Dict) -> None:
        cb = results["chatbot"]["aggregate"]
        ag = results["agent"]["aggregate"]
        meta = results["metadata"]

        sep = "+" + "-" * 28 + "+" + "-" * 22 + "+" + "-" * 22 + "+"
        row = "| {:<26} | {:^20} | {:^20} |"

        print("\n")
        print("=" * 76)
        print(
            f"  GSM8K BENCHMARK RESULTS  |  Model: {meta['model']}  |  n={meta['sample_size']}"
        )
        print("=" * 76)
        print(sep)
        print(row.format("Metric", "Chatbot (Baseline)", "ReAct Agent"))
        print(sep)
        print(row.format("Accuracy (%)", f"{cb['accuracy_pct']}%", f"{ag['accuracy_pct']}%"))
        print(row.format(
            "Correct / Total",
            f"{cb['correct']} / {cb['total_questions']}",
            f"{ag['correct']} / {ag['total_questions']}",
        ))
        print(row.format("Avg Steps / Question", str(cb["avg_steps"]), str(ag["avg_steps"])))
        print(row.format("Avg Tokens / Question", str(cb["avg_tokens"]), str(ag["avg_tokens"])))
        print(row.format("Avg Latency (ms)", str(cb["avg_latency_ms"]), str(ag["avg_latency_ms"])))
        print(sep)

        # Failure breakdown sub-table
        all_failure_types = sorted(
            set(cb["failure_breakdown"]) | set(ag["failure_breakdown"])
        )
        for ft in all_failure_types:
            cb_count = cb["failure_breakdown"].get(ft, 0)
            ag_count = ag["failure_breakdown"].get(ft, 0)
            print(row.format(f"  Failure: {ft}", str(cb_count), str(ag_count)))

        print(sep)

        # Verdict
        diff = ag["accuracy_pct"] - cb["accuracy_pct"]
        if diff > 0:
            verdict = f"ReAct Agent wins by +{diff:.2f}%"
        elif diff < 0:
            verdict = f"Chatbot wins by +{-diff:.2f}%"
        else:
            verdict = "Draw"

        print(f"\n  VERDICT: {verdict}")
        print("=" * 76)
