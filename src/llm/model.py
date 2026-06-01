"""
LLM model adapter for OpenSearch-SQL using LLMFactory.

Replaces the original model.py to use the project's LLMFactory (config/llm_factory.py),
which supports openai, bedrock, scayle, and ollama providers.
All pipeline nodes continue to use the same interface:
    chat_model = model_chose(step, engine)
    response = chat_model.get_ans(prompt, temperature, ...)
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (for .env loading; project packages are installed via pyproject.toml)
# ---------------------------------------------------------------------------
_OPENSEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _OPENSEARCH_ROOT.parent.parent

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# Add project root to sys.path so config can be imported
import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional external factory. If unavailable, use the local fallback below.
try:
    from config.llm_factory import LLMFactory, _load_yaml  # type: ignore
except (ModuleNotFoundError, ImportError) as e:
    print(
        f"⚠️  config.llm_factory import failed ({e}), using fallback", file=sys.stderr
    )
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover
        yaml = None

    def _load_yaml(path: Path) -> dict:
        """Load YAML config or return a safe default."""
        if path.exists() and yaml is not None:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
        env_model = (
            os.environ.get("MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        return {
            "provider": os.environ.get("LLM_PROVIDER", "openai"),
            "model": env_model,
            "temperature": float(os.environ.get("LLM_TEMPERATURE", "0")),
        }

    class LLMFactory:
        """Fallback factory used when `config.llm_factory` is unavailable."""

        @staticmethod
        def from_config_dict(config: dict[str, Any]):
            provider = str(config.get("provider", "openai")).lower()
            model = (
                config.get("model")
                or os.environ.get("MODEL")
                or os.environ.get("AWS_MODEL_ID")
                or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LLM_MODEL")
                or "gpt-4o-mini"
            )
            temperature = float(config.get("temperature", 0))

            if provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(model=model, temperature=temperature)
            elif provider == "bedrock":
                from langchain_aws import ChatBedrock

                region_name = config.get("region_name") or os.environ.get(
                    "AWS_REGION", "eu-west-1"
                )
                return ChatBedrock(
                    model_id=model,
                    region_name=region_name,
                    temperature=temperature,
                )
            elif provider == "ollama":
                from langchain_ollama import ChatOllama

                base_url = config.get("base_url") or os.environ.get(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                )
                return ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Use one of: openai, bedrock, ollama, anthropic"
                )


from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse


# ---------------------------------------------------------------------------
# Token-usage extraction + per-phase stats helpers
# ---------------------------------------------------------------------------


def _extract_usage(msg: Any) -> dict:
    """Best-effort token counts from a LangChain response message.

    Returns ``{"input_tokens", "output_tokens", "total_tokens"}`` with ``None``
    for any value the backend did not provide. Handles the langchain standard
    ``usage_metadata`` (bedrock/openai/anthropic) plus OpenAI ``token_usage`` and
    Ollama ``prompt_eval_count``/``eval_count`` fallbacks.
    """
    out = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict) and um:
        out["input_tokens"] = um.get("input_tokens")
        out["output_tokens"] = um.get("output_tokens")
        out["total_tokens"] = um.get("total_tokens")

    if out["input_tokens"] is None or out["output_tokens"] is None:
        rm = getattr(msg, "response_metadata", None) or {}
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if tu:
            out["input_tokens"] = out["input_tokens"] or tu.get(
                "prompt_tokens", tu.get("input_tokens")
            )
            out["output_tokens"] = out["output_tokens"] or tu.get(
                "completion_tokens", tu.get("output_tokens")
            )
            out["total_tokens"] = out["total_tokens"] or tu.get("total_tokens")
        # Ollama-style counts live directly on response_metadata
        if out["input_tokens"] is None and "prompt_eval_count" in rm:
            out["input_tokens"] = rm.get("prompt_eval_count")
        if out["output_tokens"] is None and "eval_count" in rm:
            out["output_tokens"] = rm.get("eval_count")

    if (
        out["total_tokens"] is None
        and out["input_tokens"] is not None
        and out["output_tokens"] is not None
    ):
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


def _build_stats(per_call: list[dict], total_duration_ms: float) -> dict:
    """Aggregate per-call dicts into one generation-phase stats blob.

    Each ``per_call`` dict carries ``duration_ms``/``input_tokens``/``output_tokens``.
    """
    durs = [c.get("duration_ms") for c in per_call if c.get("duration_ms") is not None]
    ins = [c.get("input_tokens") for c in per_call]
    outs = [c.get("output_tokens") for c in per_call]
    return {
        "n_calls": len(per_call),
        "total_duration_ms": total_duration_ms,
        "per_call_duration_ms": [c.get("duration_ms") for c in per_call],
        "min_duration_ms": min(durs) if durs else None,
        "max_duration_ms": max(durs) if durs else None,
        "avg_duration_ms": round(sum(durs) / len(durs), 1) if durs else None,
        "input_tokens": ins,
        "output_tokens": outs,
        "total_input_tokens": sum(t for t in ins if t is not None),
        "total_output_tokens": sum(t for t in outs if t is not None),
    }

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

    def log_record(self, prompt_text, output, duration_ms=None, model_info=None):
        try:
            logger = Logger()
            logger.log_conversation(prompt_text, "Human", self.step)
            logger.log_conversation(output, "AI", self.step)
            logger.log_llm_call(self.step, prompt_text, output, duration_ms=duration_ms, model_info=model_info)
        except (ValueError, AttributeError):
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

    @staticmethod
    def _parse_thinking(text: str) -> tuple[str, str]:
        """Strip <think>...</think> reasoning from model output.

        Several models emit chain-of-thought inside <think> tags before their
        actual answer (Qwen3, DeepSeek-R1, QwQ, and others). This function
        splits the two so the pipeline always receives clean structured output
        while the reasoning block is preserved in the call log for inspection.

        Returns (thinking_content, final_answer) as stripped strings.
        If no <think> block is present the function is a no-op:
        returns ("", original_text).
        """
        import re
        match = re.search(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "", text

    def _get_llm(self, temperature: float | None = None):
        """Return a (possibly cached) LLM for the given temperature."""
        config = dict(self._base_config)
        if temperature is not None:
            config["temperature"] = temperature
        temp_key = config.get("temperature", 0.0)

        if temp_key not in self._llm_cache:
            self._llm_cache[temp_key] = LLMFactory.from_config_dict(config)
        return self._llm_cache[temp_key]

    def _build_messages(self, prompt_text: str):
        """Build LangChain message objects from a plain-text prompt.

        When enable_thinking is false (the default), appends /no_think to the
        system message. This is the reliable cross-backend control token for
        models that support chain-of-thought reasoning (Qwen3, QwQ, etc.),
        and is a no-op for models that don't recognise it.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        system_base = "You are an SQL expert, skilled in handling various SQL-related issues."
        enable_thinking = self._base_config.get("enable_thinking", False)
        system_content = system_base if enable_thinking else f"{system_base} /no_think"

        return [
            SystemMessage(content=system_content),
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
        return_stats: bool = False,
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
            return_stats: When *True*, return ``(payload, stats)`` where *stats* is the
                         per-phase timing/token blob (see ``_build_stats``). The same
                         blob is always stored on ``self._last_stats`` regardless.
            **kwargs:    Extra arguments (silently ignored for compatibility).

        Returns:
            ``str`` when *single* and *n == 1*, otherwise a list of choice dicts.
            If *return_stats* is set, a ``(payload, stats)`` tuple instead.
        """
        if debug:
            print(messages)

        llm = self._get_llm(temperature=temperature)
        chat_msgs = self._build_messages(messages)
        model_info = {
            "model": getattr(self, "_base_config", {}).get("model", "unknown"),
            "temperature": temperature,
            "n": n,
        }

        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
        last_error = None

        def _ret(payload, stats):
            self._last_stats = stats
            return (payload, stats) if return_stats else payload

        for attempt in range(1, max_retries + 1):
            try:
                _t0 = time.time()
                if n == 1 or single:
                    result = llm.invoke(chat_msgs)
                    duration_ms = round((time.time() - _t0) * 1000, 1)
                    usage = _extract_usage(result)
                    thinking, response_clean = self._parse_thinking(result.content)
                    call_info = {**model_info, **usage}
                    if thinking:
                        call_info["thinking"] = thinking
                    if self.step != "prepare_train_queries":
                        self.log_record(
                            messages,
                            response_clean,
                            duration_ms=duration_ms,
                            model_info=call_info,
                        )
                    stats = _build_stats(
                        [{"duration_ms": duration_ms, **usage}], duration_ms
                    )
                    return _ret(response_clean, stats)
                else:
                    choices = self._generate_n(llm, chat_msgs, n)
                    total_ms = round((time.time() - _t0) * 1000, 1)
                    for c in choices:
                        thinking, answer = self._parse_thinking(c["message"]["content"])
                        c["message"]["content"] = answer
                        if thinking:
                            c["thinking"] = thinking
                    if self.step != "prepare_train_queries":
                        # one call-log file per choice (per-call duration + tokens)
                        for c in choices:
                            ci = {
                                **model_info,
                                "input_tokens": c.get("input_tokens"),
                                "output_tokens": c.get("output_tokens"),
                                "total_tokens": c.get("total_tokens"),
                            }
                            if c.get("thinking"):
                                ci["thinking"] = c["thinking"]
                            try:
                                Logger().log_llm_call(
                                    self.step,
                                    messages,
                                    c["message"]["content"],
                                    duration_ms=c.get("duration_ms"),
                                    model_info=ci,
                                )
                            except (ValueError, AttributeError):
                                pass
                        # conversation .log written once (combined)
                        try:
                            logger = Logger()
                            logger.log_conversation(messages, "Human", self.step)
                            logger.log_conversation(
                                [c["message"]["content"] for c in choices],
                                "AI",
                                self.step,
                            )
                        except (ValueError, AttributeError):
                            pass
                    stats = _build_stats(choices, total_ms)
                    return _ret(choices, stats)

            except Exception as e:
                last_error = e
                wait = min(2 * attempt, 5)
                print(f"LLM error (attempt {attempt}/{max_retries}): {e}")
                time.sleep(wait)

        # Exhausted retries
        print(f"LLM failed after {max_retries} attempts. Last error: {last_error}")
        empty_stats = _build_stats([], 0.0)
        if n == 1 or single:
            return _ret("", empty_stats)
        return _ret([{"message": {"content": ""}} for _ in range(n)], empty_stats)

    def _generate_n(self, llm, chat_msgs, n: int):
        """Generate *n* completions concurrently, timing each call individually.

        Each returned choice carries its own ``duration_ms`` and token counts so the
        caller can record per-call stats alongside the concurrent wall-clock total.
        """

        def _invoke(_):
            _t0 = time.time()
            result = llm.invoke(chat_msgs)
            duration_ms = round((time.time() - _t0) * 1000, 1)
            usage = _extract_usage(result)
            return {
                "message": {"content": result.content},
                "duration_ms": duration_ms,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
            }

        # Use threads for I/O-bound LLM calls
        workers = min(n, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_invoke, i) for i in range(n)]
            choices = [f.result() for f in futures]
        return choices
