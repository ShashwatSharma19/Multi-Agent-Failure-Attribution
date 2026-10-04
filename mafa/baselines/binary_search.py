"""Binary search: recursively ask which half holds the decisive error.

Matches the reference recursion (``mid = start + (end - start)//2``; "upper half"
keeps ``[start, mid]``, "lower half" keeps ``[mid+1, end]``). On an ambiguous
response the reference picks a half at random; we use a seeded RNG (default seed
0) so that rare tie-break is reproducible. Everything else is identical.
"""

from __future__ import annotations

import random

from mafa.baselines.prompts import BINARY_SYSTEM, binary_prompt
from mafa.eval.scorer import Prediction
from mafa.io import Trace
from mafa.judge.base import Judge, JudgeError


def _segment_text(trace: Trace, start: int, end: int) -> str:
    return "\n".join(f"{s.agent}: {s.content}" for s in trace.steps[start : end + 1])


def _find(judge, trace, start, end, max_tokens, include_gt, rng) -> int:
    if start >= end:
        return max(start, 0)

    mid = start + (end - start) // 2
    prompt = binary_prompt(
        trace.question,
        trace.ground_truth,
        _segment_text(trace, start, end),
        f"from step {start} to step {end}",
        f"from step {start} to step {mid}",
        f"from step {mid + 1} to step {end}",
        include_gt,
    )
    try:
        result = judge.complete(BINARY_SYSTEM, prompt, max_tokens=max_tokens).lower()
    except JudgeError:
        return start  # API failure: report the segment start, as a reasonable fallback

    if "upper half" in result:
        return _find(judge, trace, start, mid, max_tokens, include_gt, rng)
    if "lower half" in result:
        return _find(judge, trace, min(mid + 1, end), end, max_tokens, include_gt, rng)
    # Ambiguous response -> random half (reproducible via seeded rng).
    if rng.randint(0, 1) == 0:
        return _find(judge, trace, start, mid, max_tokens, include_gt, rng)
    return _find(judge, trace, min(mid + 1, end), end, max_tokens, include_gt, rng)


def run(judge: Judge, trace: Trace, max_tokens: int = 1024, include_ground_truth: bool = True, seed: int = 0) -> Prediction:
    if not trace.steps:
        return Prediction("", "", raw="empty trace")
    rng = random.Random(seed)
    step_idx = _find(judge, trace, 0, len(trace.steps) - 1, max_tokens, include_ground_truth, rng)
    agent = trace.steps[step_idx].agent
    return Prediction(agent, str(step_idx), raw=f"binary search -> step {step_idx}")
