"""Step-by-step: walk the transcript, stop at the first step judged an error.

Matches the reference: the running history is prefixed ``Step {idx} - {agent}: ...``,
and the first response starting with ``1. yes`` fixes the predicted (agent, step).
If no step is flagged, there is no prediction (counts as incorrect), exactly as
the reference's "no decisive errors found" path.
"""

from __future__ import annotations

from mafa.baselines.prompts import STEP_SYSTEM, step_prompt
from mafa.eval.scorer import Prediction
from mafa.io import Trace
from mafa.judge.base import Judge, JudgeError


def run(judge: Judge, trace: Trace, max_tokens: int = 1024, include_ground_truth: bool = True) -> Prediction:
    history_so_far = ""
    for step in trace.steps:
        history_so_far += f"Step {step.index} - {step.agent}: {step.content}\n"
        prompt = step_prompt(
            trace.question, trace.ground_truth, history_so_far,
            step.index, step.agent, include_ground_truth,
        )
        try:
            answer = judge.complete(STEP_SYSTEM, prompt, max_tokens=max_tokens)
        except JudgeError:
            return Prediction("", "", raw="")  # reference treats API error as unable to proceed

        if answer.lower().strip().startswith("1. yes"):
            reason = answer.split("Reason:", 1)[-1].strip()
            return Prediction(step.agent, str(step.index), raw=reason)

    return Prediction("", "", raw="no decisive error found")
