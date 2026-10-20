"""Concrete judge backends: Ollama (local), Groq and Gemini (free cloud tiers).

All three talk HTTP directly via ``requests`` to keep dependencies light — no
heavy vendor SDKs. Each raises :class:`JudgeError` on any failure so the
rate-limit wrapper can decide whether to retry.
"""

from __future__ import annotations

import os

import requests

from mafa.judge.base import Judge, JudgeError

_DEFAULT_TIMEOUT = 180  # seconds; long logs + a local model can be slow


class OllamaJudge(Judge):
    """Local model served by Ollama (default: the already-installed llama3.2:3B).

    Fully offline and free. A weak judge on long logs — intended as the
    reproducible fallback and as an ablation row, not the headline backend.
    """

    def __init__(self, model_id: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model_id = model_id
        self._url = f"{host.rstrip('/')}/api/chat"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = requests.post(self._url, json=payload, timeout=_DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise JudgeError(f"Ollama call failed: {exc}") from exc


class _OpenAICompatJudge(Judge):
    """Shared logic for OpenAI-compatible chat-completions endpoints (Groq)."""

    _endpoint: str
    _api_key: str

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # gpt-oss are reasoning models: keep reasoning short so the formatted answer
        # reliably fits in the token budget (and to cut cost/latency).
        if "gpt-oss" in self.model_id:
            payload["reasoning_effort"] = "low"
        try:
            resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=_DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            raise JudgeError(f"request failed: {exc}", retryable=True) from exc
        if resp.status_code == 429:
            raise JudgeError("rate limited (HTTP 429)", retryable=True)
        if resp.status_code >= 400:
            raise JudgeError(f"HTTP {resp.status_code}: {resp.text[:200]}", retryable=False)
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise JudgeError(f"bad response shape: {exc}", retryable=False) from exc


class GroqJudge(_OpenAICompatJudge):
    """Groq free tier (OpenAI-compatible). Default headline judge."""

    def __init__(self, model_id: str = "openai/gpt-oss-120b", api_key: str | None = None):
        self.model_id = model_id
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise JudgeError("GROQ_API_KEY not set (export it or put it in .env)", retryable=False)
        self._api_key = key
        self._endpoint = "https://api.groq.com/openai/v1/chat/completions"


class GeminiJudge(Judge):
    """Google Gemini free tier (generativeLanguage REST API)."""

    def __init__(self, model_id: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model_id = model_id
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise JudgeError("GEMINI_API_KEY not set (export it or put it in .env)", retryable=False)
        self._api_key = key
        self._base = "https://generativelanguage.googleapis.com/v1beta/models"

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        url = f"{self._base}/{self.model_id}:generateContent?key={self._api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        try:
            resp = requests.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            raise JudgeError(f"request failed: {exc}", retryable=True) from exc
        if resp.status_code == 429:
            raise JudgeError("rate limited (HTTP 429)", retryable=True)
        if resp.status_code >= 400:
            raise JudgeError(f"HTTP {resp.status_code}: {resp.text[:200]}", retryable=False)
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise JudgeError(f"bad response shape: {exc}", retryable=False) from exc
