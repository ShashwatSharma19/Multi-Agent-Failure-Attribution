"""The judge interface and shared data types.

Every backend and every wrapper implements :class:`Judge`. Baselines depend only
on this interface, so swapping Groq <-> Gemini <-> Ollama is a one-line config
change and never touches attribution logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class JudgeError(RuntimeError):
    """Raised when a backend cannot produce a completion.

    ``retryable`` tells the rate limiter whether a retry could help: transient
    failures (HTTP 429, network errors) are retryable; permanent ones (a bad model
    id, auth failure, malformed request → 4xx) are not, so they fail fast.
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class Judge(ABC):
    """An LLM that answers a single (system, user) prompt with text.

    ``model_id`` must uniquely identify the underlying model; the caching wrapper
    keys on it, so two different models never share a cache entry.
    """

    model_id: str

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text response. Raises :class:`JudgeError` on failure."""
        raise NotImplementedError
