"""Cost accounting over any :class:`Judge`.

Records how many calls a run makes and an estimate of the tokens consumed, so we
can report a metric like "same accuracy at a third of the calls" when the
scope-delineation pass is enabled vs. not.

Placed **outermost** in the judge stack so it counts every logical call the
attribution method issues (cache hit or not) — that is the method's intrinsic
cost, which is what the ablation compares. Token counts are a deliberate
estimate (``len(text) // 4``): exact per-backend usage differs, but the estimator
is identical with and without scope, so the *relative* reduction is meaningful.
"""

from __future__ import annotations

from mafa.judge.base import Judge


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4) if text else 0


class AccountingJudge(Judge):
    def __init__(self, inner: Judge):
        self.inner = inner
        self.model_id = inner.model_id
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        self.calls += 1
        self.prompt_tokens += estimate_tokens(system) + estimate_tokens(user)
        out = self.inner.complete(system, user, max_tokens, temperature)
        self.completion_tokens += estimate_tokens(out)
        return out

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


def unwrap(judge: Judge, cls: type) -> Judge | None:
    """Find the wrapper of type ``cls`` in a chain of ``.inner`` delegates."""
    cur: Judge | None = judge
    for _ in range(16):
        if cur is None:
            return None
        if isinstance(cur, cls):
            return cur
        cur = getattr(cur, "inner", None)
    return None
