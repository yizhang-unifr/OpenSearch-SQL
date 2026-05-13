"""Quick connectivity test for configured LLM providers.

Usage:
    python eval/test_llm_providers.py                  # test all providers
    python eval/test_llm_providers.py --provider openai
    python eval/test_llm_providers.py --provider swiss_ai
    python eval/test_llm_providers.py --provider bedrock
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env from project root (two levels up from OpenSearch-SQL/)
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=False)

# Make config/ importable (project root is parents[3] relative to this file)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "config"))

import yaml
from llm_factory import LLMFactory

_PROMPT = "Reply with exactly one word: working"

_PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": os.environ.get("MODEL", "gpt-4o-mini"),
        "temperature": 0.0,
    },
    "bedrock": {
        "provider": "bedrock",
        "model": os.environ.get("AWS_MODEL_ID", "qwen.qwen3-next-80b-a3b"),
        "temperature": 0.0,
    },
    "swiss_ai": {
        "provider": "swiss_ai",
        "model": os.environ.get("SWISS_AI_MODEL", "Qwen/Qwen3.5-27B"),
        "temperature": 0.0,
    },
}


def _test_provider(name: str, config: dict) -> bool:
    model_label = config["model"]
    print(f"\n{'─'*60}")
    print(f"Provider : {name}")
    print(f"Model    : {model_label}")

    try:
        llm = LLMFactory.from_config_dict(config)
        from langchain_core.messages import HumanMessage
        t0 = time.time()
        resp = llm.invoke([HumanMessage(content=_PROMPT)])
        elapsed = round((time.time() - t0) * 1000)
        text = resp.content if hasattr(resp, "content") else str(resp)
        print(f"Status   : OK  ({elapsed} ms)")
        print(f"Response : {text[:120]}")
        return True
    except Exception as exc:
        print(f"Status   : FAIL")
        print(f"Error    : {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(_PROVIDERS) + ["all"], default="all")
    args = ap.parse_args()

    targets = list(_PROVIDERS.items()) if args.provider == "all" else [(args.provider, _PROVIDERS[args.provider])]

    results = {}
    for name, config in targets:
        results[name] = _test_provider(name, config)

    print(f"\n{'═'*60}")
    print("Summary:")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
