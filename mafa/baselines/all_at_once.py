"""All-at-once: feed the whole transcript, parse (agent, step) from one response."""

from __future__ import annotations

from mafa.baselines.prompts import ALL_AT_ONCE_SYSTEM, all_at_once_prompt
from mafa.eval.scorer import Prediction, parse_prediction
from mafa.io import Trace
from mafa.judge.base import Judge, JudgeError


def run(
    judge: Judge,
    trace: Trace,
    max_tokens: int = 1024,
    include_ground_truth: bool = True,
    scope: tuple[int, int] | None = None,  # accepted for a uniform interface; ignored (one-call method, keeps reference prompt + global step numbering intact)
) -> Prediction:
    prompt = all_at_once_prompt(
        trace.question, trace.ground_truth, trace.transcript, include_ground_truth
    )
    try:
        text = judge.complete(ALL_AT_ONCE_SYSTEM, prompt, max_tokens=max_tokens)
    except JudgeError:
        return Prediction("", "", raw="")
    return parse_prediction(text)
