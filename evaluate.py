"""
GSM8K Benchmark: Chatbot (baseline) vs ReAct Agent

Usage:
    python evaluate.py --sample 50 --provider openai
    python evaluate.py --sample 100 --model gpt-4o --steps 7
    python evaluate.py --sample 50 --split train --seed 99
    python evaluate.py --sample 50 --delay 0.5   # 0.5s between questions (rate limiting)
"""
import argparse
import os
import sys

# Make src/ importable without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def build_provider(provider_name: str, model: str):
    """Factory — returns a configured LLMProvider instance."""
    if provider_name == "openai":
        from src.core.openai_provider import OpenAIProvider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment / .env")
        return OpenAIProvider(model_name=model, api_key=api_key)
    elif provider_name == "google":
        from src.core.gemini_provider import GeminiProvider
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment / .env")
        return GeminiProvider(model_name=model, api_key=api_key)
    elif provider_name == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)
    else:
        raise ValueError(f"Unknown provider '{provider_name}'. Choose: openai | google | local")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GSM8K Benchmark: plain chatbot vs ReAct agent"
    )
    parser.add_argument(
        "--sample", type=int, default=50,
        help="Number of GSM8K questions to evaluate (default: 50)",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "google", "local"],
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name override. Defaults: openai→gpt-4o, google→gemini-1.5-flash",
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Max ReAct loop steps per question (default: 5)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "test"],
        help="GSM8K dataset split (default: test)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save JSON results (default: results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset sampling (default: 42)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to sleep between questions — use if hitting rate limits (default: 0)",
    )
    parser.add_argument(
        "--version", type=str, default="v2", choices=["v1", "v2"],
        help="Agent version to evaluate: v1 (buggy baseline) or v2 (fixed). Default: v2",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    default_models = {
        "openai": "gpt-4o",
        "google": "gemini-1.5-flash",
        "local": "local",
    }
    model_name = args.model or default_models[args.provider]

    print(f"Provider : {args.provider}")
    print(f"Model    : {model_name}")
    print(f"Agent    : {args.version}")
    print(f"Sample   : {args.sample} questions from GSM8K ({args.split} split)")
    print(f"Steps    : up to {args.steps} ReAct steps per question")
    print()

    llm = build_provider(args.provider, model_name)

    from evaluation.runner import EvaluationRunner
    runner = EvaluationRunner(
        llm=llm,
        sample_size=args.sample,
        max_agent_steps=args.steps,
        results_dir=args.results_dir,
        gsm8k_split=args.split,
        seed=args.seed,
        request_delay=args.delay,
        agent_version=args.version,
    )
    runner.run()


if __name__ == "__main__":
    main()
