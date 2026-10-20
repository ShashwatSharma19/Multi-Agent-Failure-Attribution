"""Scope-delineation: one cheap judge call picks the window holding the failure.

The two-stage idea behind the strongest published step-accuracy gains: before
running expensive per-step attribution, split the log into a few contiguous
windows and ask the judge which single window contains the decisive error. The
expensive pass then runs only inside that window — fewer calls per log, and
shorter prompts, which is also what keeps us under the free-tier token limits.

``delineate`` returns an absolute ``(start, end)`` step range (inclusive). Step
indices are preserved end-to-end, so downstream predictions and scoring stay in
the original indexing.
"""

from __future__ import annotations

import math
import re

from mafa.io import Trace
from mafa.judge.base import Judge, JudgeError

SCOPE_SYSTEM = "You localize the region of a multi-agent conversation that contains the decisive error."


def windows(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into up to ``k`` near-equal contiguous windows."""
    if n <= 0:
        return []
    size = math.ceil(n / k)
    return [(i, min(i + size - 1, n - 1)) for i in range(0, n, size)]


def _prompt(trace: Trace, wins: list[tuple[int, int]], include_ground_truth: bool) -> str:
    segments = []
    for i, (s, e) in enumerate(wins, 1):
        # shortcut: truncate each step to ~200 chars — the judge only needs the gist to
        # pick a region, and this keeps the whole-transcript scope prompt small on long logs.
        body = "\n".join(
            f"Step {st.index} - {st.agent}: {st.content[:200]}" for st in trace.steps[s : e + 1]
        )
        segments.append(f"Segment {i} (steps {s}-{e}):\n{body}")
    answer = f"The correct answer to the problem is: {trace.ground_truth}\n" if include_ground_truth else ""
    return (
        "A multi-agent conversation tried and failed to solve a problem.\n"
        f"The problem is: {trace.question}\n"
        + answer
        + "The conversation is split into the numbered segments below. Exactly one segment "
        "contains the step where the decisive error that caused the failure first occurs.\n\n"
        + "\n\n".join(segments)
        + "\n\nWhich segment contains the decisive error? Respond with ONLY the segment number."
    )


def delineate(
    judge: Judge,
    trace: Trace,
    n_windows: int = 3,
    max_tokens: int = 256,  # headroom for reasoning models whose answer follows a reasoning pass
    include_ground_truth: bool = True,
) -> tuple[int, int]:
    """Return the ``(start, end)`` window likely to contain the failure.

    Short logs (<= ``n_windows + 1`` steps) are returned whole — there is nothing
    worth narrowing. On a judge error or an unparseable answer, falls back to the
    last window (failures skew late) rather than crashing the run.
    """
    n = len(trace.steps)
    if n <= 2 or n <= n_windows + 1:
        return (0, max(n - 1, 0))

    wins = windows(n, n_windows)
    try:
        answer = judge.complete(SCOPE_SYSTEM, _prompt(trace, wins, include_ground_truth), max_tokens=max_tokens)
    except JudgeError:
        return (0, n - 1)  # fail open: do not narrow if the pass itself failed

    match = re.search(r"\d+", answer or "")
    if not match:
        return wins[-1]
    idx = int(match.group()) - 1
    if 0 <= idx < len(wins):
        return wins[idx]
    return wins[-1]
