"""
LLM model adapter for OpenSearch-SQL using LLMFactory.

Replaces the original model.py to use the project's LLMFactory (config/llm_factory.py),
which supports openai, bedrock, scayle, and ollama providers.
All pipeline nodes continue to use the same interface:
    chat_model = model_chose(step, engine)
    response = chat_model.get_ans(prompt, temperature, ...)
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (for .env loading; project packages are installed via pyproject.toml)
# ---------------------------------------------------------------------------
_OPENSEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _OPENSEARCH_ROOT.parent

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from config.llm_factory import LLMFactory, _load_yaml
from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse

# ---------------------------------------------------------------------------
# Global LLM config path – override via set_llm_config_path() or
# the LLM_CONFIG_PATH environment variable.
# ---------------------------------------------------------------------------
_LLM_CONFIG_PATH: Path = Path(
    os.environ.get("LLM_CONFIG_PATH", str(_PROJECT_ROOT / "config" / "models.yaml"))
)


def set_llm_config_path(path: str | Path) -> None:
    """Override the LLM config file path at runtime."""
    global _LLM_CONFIG_PATH
    _LLM_CONFIG_PATH = Path(path)


def _get_base_config() -> dict:
    """Load the raw YAML config dict (cached per path)."""
    return _load_yaml(_LLM_CONFIG_PATH)


# ---------------------------------------------------------------------------
# model_chose – drop-in replacement for the original function.
# Every pipeline node calls:  chat_model = model_chose(node_name, config["engine"])
# We ignore the *engine* string and use the YAML-configured provider instead.
# ---------------------------------------------------------------------------

def model_chose(step: str, model: str = "gpt-4 32K"):
    """Create an LLM adapter using LLMFactory.

    Args:
        step:  Pipeline node name (used for logging).
        model: Engine name from pipeline_setup (ignored – the model is
               determined by config/models.yaml).

    Returns:
        An ``LLMFactoryAdapter`` instance with the familiar ``get_ans()`` API.
    """
    return LLMFactoryAdapter(step)


# ---------------------------------------------------------------------------
# Base class – keeps fewshot_parse / convert_table / log_record identical
# to the original so nothing downstream breaks.
# ---------------------------------------------------------------------------

class req:
    """Minimal base retained for interface compatibility."""

    def __init__(self, step: str, model: str = "") -> None:
        self.Cost = 0
        self.model = model
        self.step = step

    def log_record(self, prompt_text, output):
        try:
            logger = Logger()
            logger.log_conversation(prompt_text, "Human", self.step)
            logger.log_conversation(output, "AI", self.step)
        except (ValueError, AttributeError):
            # Logger singleton not yet initialised (standalone usage) – skip
            pass

    def fewshot_parse(self, question, evidence, sql):
        s = prompts_fewshot_parse().parse_fewshot.format(question=question, sql=sql)
        ext = self.get_ans(s)
        ext = ext.replace("```", "").strip()
        ext = ext.split("#SQL:")[0]
        ans = self.convert_table(ext, sql)
        return ans

    def convert_table(self, s, sql):
        l = re.findall(r" ([^ ]*) +AS +([^ ]*)", sql)
        x, v = s.split("#values:")
        t, s = x.split("#SELECT:")
        for li in l:
            s = s.replace(f"{li[1]}.", f"{li[0]}.")
        return t + "#SELECT:" + s + "#values:" + v


# ---------------------------------------------------------------------------
# The adapter – wraps a LangChain LLM and exposes get_ans()
# ---------------------------------------------------------------------------

class LLMFactoryAdapter(req):
    """Wraps a LangChain LLM from ``LLMFactory`` so every OpenSearch-SQL
    pipeline node can call ``get_ans()`` unchanged.
    """

    def __init__(self, step: str) -> None:
        super().__init__(step, model="llm_factory")
        self._base_config: dict = _get_base_config()
        # Cache LLM instances keyed by temperature for reuse
        self._llm_cache: dict[float, object] = {}

    # -- internal helpers ---------------------------------------------------

    def _get_llm(self, temperature: float | None = None):
        """Return a (possibly cached) LLM for the given temperature."""
        config = dict(self._base_config)
        if temperature is not None:
            config["temperature"] = temperature
        temp_key = config.get("temperature", 0.0)

        if temp_key not in self._llm_cache:
            self._llm_cache[temp_key] = LLMFactory.from_config_dict(config)
        return self._llm_cache[temp_key]

    @staticmethod
    def _build_messages(prompt_text: str):
        """Build LangChain message objects from a plain-text prompt."""
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(
                content="You are an SQL expert, skilled in handling various SQL-related issues."
            ),
            HumanMessage(content=prompt_text),
        ]

    # -- public API (same signature the rest of the codebase expects) -------

    def get_ans(
        self,
        messages: str,
        temperature: float = 0.0,
        top_p: float | None = None,
        n: int = 1,
        single: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        """Generate a response from the LLM.

        Args:
            messages:    Prompt string.
            temperature: Sampling temperature (per-call override).
            top_p:       Top-p / nucleus sampling (best-effort, provider-dependent).
            n:           Number of completions to generate.
            single:      When *True* (and n==1) return a plain ``str``.
                         When *False* return ``[{"message": {"content": ...}}, ...]``.
            debug:       Print the prompt if *True*.
            **kwargs:    Extra arguments (silently ignored for compatibility).

        Returns:
            ``str`` when *single* and *n == 1*, otherwise a list of choice dicts.
        """
        if debug:
            print(messages)

        llm = self._get_llm(temperature=temperature)
        chat_msgs = self._build_messages(messages)

        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                if n == 1 or single:
                    result = llm.invoke(chat_msgs)
                    response_clean = result.content
                    if self.step != "prepare_train_queries":
                        self.log_record(messages, response_clean)
                    return response_clean
                else:
                    # Multiple completions – invoke concurrently for speed
                    choices = self._generate_n(llm, chat_msgs, n)
                    if self.step != "prepare_train_queries":
                        self.log_record(messages, str([c["message"]["content"][:80] for c in choices]))
                    return choices

            except Exception as e:
                last_error = e
                wait = min(2 * attempt, 5)
                print(f"LLM error (attempt {attempt}/{max_retries}): {e}")
                time.sleep(wait)

        # Exhausted retries
        print(f"LLM failed after {max_retries} attempts. Last error: {last_error}")
        if n == 1 or single:
            return ""
        return [{"message": {"content": ""}} for _ in range(n)]

    def _generate_n(self, llm, chat_msgs, n: int):
        """Generate *n* completions concurrently."""

        def _invoke(_):
            result = llm.invoke(chat_msgs)
            return {"message": {"content": result.content}}

        # Use threads for I/O-bound LLM calls
        workers = min(n, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_invoke, i) for i in range(n)]
            choices = [f.result() for f in futures]
        return choices
