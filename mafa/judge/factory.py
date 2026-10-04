"""Build a ready-to-use judge: backend wrapped in rate-limiting then caching.

Composition order (outermost first): ``CachingJudge( RateLimitedJudge( backend ) )``
so cache hits never touch the rate limiter or the network.
"""

from __future__ import annotations

from pathlib import Path

from mafa.judge.backends import GeminiJudge, GroqJudge, OllamaJudge
from mafa.judge.base import Judge
from mafa.judge.cache import CachingJudge
from mafa.judge.ratelimit import RateLimitedJudge

# Sensible default free-tier request budgets (requests/min). Conservative on
# purpose; adjust once you know your account's real limits.
_DEFAULT_RPM = {"groq": 25, "gemini": 14, "ollama": 0}  # 0 = no throttle locally

_DEFAULT_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
}


def _backend(kind: str, model: str | None) -> Judge:
    model = model or _DEFAULT_MODEL[kind]
    if kind == "groq":
        return GroqJudge(model)
    if kind == "gemini":
        return GeminiJudge(model)
    if kind == "ollama":
        return OllamaJudge(model)
    raise ValueError(f"unknown judge kind: {kind!r} (expected groq|gemini|ollama)")


def build_judge(
    kind: str = "ollama",
    model: str | None = None,
    cache_db: str | Path = ".cache/judge.sqlite",
    rpm: int | None = None,
) -> Judge:
    """Construct a cached, rate-limited judge of the given ``kind``.

    ``kind`` is one of ``"groq"``, ``"gemini"``, ``"ollama"``.
    """
    kind = kind.lower()
    rpm = _DEFAULT_RPM.get(kind, 0) if rpm is None else rpm
    backend = _backend(kind, model)
    limited = RateLimitedJudge(backend, rpm=rpm)
    return CachingJudge(limited, db_path=cache_db)
