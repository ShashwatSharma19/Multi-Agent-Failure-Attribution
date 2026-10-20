"""A rate-limiting + retry decorator over any :class:`Judge`.

Handles the free-tier edge cases from the design:

* caps requests to ``rpm`` per minute (simple spacing between calls);
* retries :class:`JudgeError` (e.g. HTTP 429) with exponential backoff, so a
  transient limit pauses instead of crashing the run.

Place this *under* :class:`CachingJudge` so cache hits never consume the budget.
"""

from __future__ import annotations

import time

from mafa.judge.base import Judge, JudgeError


class RateLimitedJudge(Judge):
    def __init__(
        self,
        inner: Judge,
        rpm: int = 30,
        max_retries: int = 5,
        base_delay: float = 2.0,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.inner = inner
        self.model_id = inner.model_id
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = self._clock() - self._last_call
        wait = self._min_interval - elapsed
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                return self.inner.complete(system, user, max_tokens, temperature)
            except JudgeError as exc:
                if not exc.retryable or attempt == self._max_retries:
                    raise  # permanent error, or out of attempts — don't keep backing off
                self._sleep(self._base_delay * (2**attempt))
