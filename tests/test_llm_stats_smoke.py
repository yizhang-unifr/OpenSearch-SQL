"""DB-free smoke test for per-call timing + token instrumentation.

Run: uv run python src/OpenSearch-SQL/tests/test_llm_stats_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm.model import _extract_usage, _build_stats, LLMFactoryAdapter  # noqa: E402
from runner.logger import Logger  # noqa: E402


class _FakeMsg:
    def __init__(self, content, usage_metadata=None, response_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class _FakeLLM:
    """Returns a message carrying usage_metadata, like ChatBedrock."""
    def __init__(self):
        self._i = 0

    def invoke(self, _msgs):
        self._i += 1
        return _FakeMsg(
            f"SELECT {self._i};",
            usage_metadata={"input_tokens": 100 + self._i, "output_tokens": 5 + self._i,
                            "total_tokens": 105 + 2 * self._i},
        )


def test_extract_usage():
    # langchain standard
    u = _extract_usage(_FakeMsg("x", usage_metadata={"input_tokens": 10, "output_tokens": 3,
                                                     "total_tokens": 13}))
    assert u == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}, u
    # OpenAI token_usage fallback + derived total
    u = _extract_usage(_FakeMsg("x", response_metadata={"token_usage": {"prompt_tokens": 7,
                                                                        "completion_tokens": 2}}))
    assert u == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}, u
    # Ollama-style
    u = _extract_usage(_FakeMsg("x", response_metadata={"prompt_eval_count": 4, "eval_count": 6}))
    assert u == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}, u
    # missing
    u = _extract_usage(_FakeMsg("x"))
    assert u == {"input_tokens": None, "output_tokens": None, "total_tokens": None}, u
    print("✓ _extract_usage")


def test_build_stats():
    per_call = [
        {"duration_ms": 100.0, "input_tokens": 10, "output_tokens": 2},
        {"duration_ms": 300.0, "input_tokens": 10, "output_tokens": 4},
        {"duration_ms": 200.0, "input_tokens": 10, "output_tokens": 3},
    ]
    s = _build_stats(per_call, total_duration_ms=350.0)
    assert s["n_calls"] == 3
    assert s["total_duration_ms"] == 350.0
    assert s["min_duration_ms"] == 100.0
    assert s["max_duration_ms"] == 300.0
    assert s["avg_duration_ms"] == 200.0
    assert s["total_input_tokens"] == 30
    assert s["total_output_tokens"] == 9
    # empty
    e = _build_stats([], 0.0)
    assert e["n_calls"] == 0 and e["min_duration_ms"] is None
    print("✓ _build_stats")


def test_get_ans_n_logs_per_call(tmp: Path):
    Logger("meteo", "0", str(tmp))  # init singleton -> writes under tmp/logs/0_meteo/<node>
    adapter = LLMFactoryAdapter("candidate_generate")
    adapter._llm_cache[0.7] = _FakeLLM()  # bypass real backend

    choices, stats = adapter.get_ans("the prompt", temperature=0.7, n=3, single=False,
                                     return_stats=True)
    assert len(choices) == 3, choices
    assert stats["n_calls"] == 3, stats
    assert stats["total_input_tokens"] > 0 and stats["total_output_tokens"] > 0, stats
    assert len(stats["per_call_duration_ms"]) == 3
    assert adapter._last_stats == stats

    call_dir = tmp / "logs" / "0_meteo" / "candidate_generate"
    files = sorted(call_dir.glob("call_*.json"))
    assert len(files) == 3, [f.name for f in files]
    payload = json.loads(files[0].read_text())
    assert payload["input_tokens"] is not None and payload["output_tokens"] is not None, payload
    assert payload["duration_ms"] is not None, payload
    print("✓ get_ans(n=3) -> 3 call files w/ tokens + stats")


def test_get_ans_single(tmp: Path):
    Logger("meteo", "1", str(tmp))
    adapter = LLMFactoryAdapter("candidate_generate")
    adapter._llm_cache[0.0] = _FakeLLM()
    out, stats = adapter.get_ans("p", temperature=0.0, return_stats=True)
    assert isinstance(out, str) and out.startswith("SELECT"), out
    assert stats["n_calls"] == 1 and stats["min_duration_ms"] == stats["max_duration_ms"]
    assert stats["total_input_tokens"] > 0
    # default return_stats=False -> plain str
    out2 = adapter.get_ans("p", temperature=0.0)
    assert isinstance(out2, str), out2
    print("✓ get_ans single path + back-compat str return")


if __name__ == "__main__":
    test_extract_usage()
    test_build_stats()
    with tempfile.TemporaryDirectory() as d:
        test_get_ans_n_logs_per_call(Path(d))
        test_get_ans_single(Path(d))
    print("\nALL SMOKE TESTS PASSED")
