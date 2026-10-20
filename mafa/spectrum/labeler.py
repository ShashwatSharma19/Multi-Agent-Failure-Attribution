"""Per-step error labeling: the spectrum's raw signal.

Reuses the reference step prompt, but — unlike ``step_by_step`` — does **not** stop
at the first "yes". It labels every step in scope error / not-error, which is what
turns a single failing log into a spectrum of pass/fail "tests" over agents.

Cost is O(steps in scope) judge calls per log; combine with ``--scope`` to keep it
cheap on long logs.
"""

from __future__ import annotations

from dataclasses import dataclass

from mafa.baselines.prompts import STEP_SYSTEM, step_prompt
from mafa.io import Trace
from mafa.judge.base import Judge, JudgeError


@dataclass(frozen=True)
class StepLabel:
    index: int
    agent: str
    is_error: bool
    reason: str = ""


def label_steps(
    judge: Judge,
    trace: Trace,
    scope: tuple[int, int] | None = None,
    include_ground_truth: bool = True,
    max_tokens: int = 256,
) -> list[StepLabel]:
    """Label each step in ``scope`` (or all steps) as error / not-error."""
    lo, hi = scope if scope is not None else (0, len(trace.steps) - 1)
    history = ""
    labels: list[StepLabel] = []
    for step in trace.steps:
        if step.index < lo:
            continue
        if step.index > hi:
            break
        # shortcut: prompt history is the scope window only, not the whole transcript,
        # so calls stay O(window) on long logs. Upgrade if judging a step needs earlier context.
        history += f"Step {step.index} - {step.agent}: {step.content}\n"
        prompt = step_prompt(
            trace.question, trace.ground_truth, history, step.index, step.agent, include_ground_truth
        )
        try:
            answer = judge.complete(STEP_SYSTEM, prompt, max_tokens=max_tokens)
        except JudgeError:
            answer = ""
        is_error = answer.lower().strip().startswith("1. yes")
        reason = answer.split("Reason:", 1)[-1].strip() if is_error and "Reason:" in answer else ""
        labels.append(StepLabel(step.index, step.agent, is_error, reason))
    return labels
