"""Scoring that replicates the Who&When reference ``evaluate.py`` exactly.

Two reference semantics matter for an honest "reproduced" claim and are kept
deliberately:

1. **Substring match, not equality.** The reference counts an agent correct when
   ``actual_agent in predicted_agent`` and a step correct when
   ``actual_step in predicted_step``. We keep that. (It is looser than exact
   match — e.g. actual step ``"2"`` matches predicted ``"12"`` — but changing it
   would mean we are no longer reproducing the paper's number.)
2. **Denominator is the total number of logs**, not the number of parsed
   predictions. A log whose prediction failed to parse counts as wrong, never as
   absent. So accuracy = correct / total_traces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Mirrors evaluate.py: re.search(r"Agent Name:\s*([\w_]+)", ..., IGNORECASE)
_AGENT_RE = re.compile(r"Agent Name:\s*([\w_]+)", re.IGNORECASE)
_STEP_RE = re.compile(r"Step Number:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Prediction:
    """A model's attribution for one trace.

    ``predicted_agent`` / ``predicted_step`` are empty strings when the model
    produced nothing parseable — which the scorer treats as incorrect.
    """

    predicted_agent: str
    predicted_step: str
    raw: str = ""


@dataclass(frozen=True)
class Scores:
    agent_accuracy: float  # percent
    step_accuracy: float  # percent
    correct_agent: int
    correct_step: int
    total: int


def parse_prediction(text: str) -> Prediction:
    """Extract (agent, step) from a free-text ``all_at_once`` response.

    Uses the reference regexes. Missing fields become empty strings.
    """
    text = text or ""
    agent_match = _AGENT_RE.search(text)
    step_match = _STEP_RE.search(text)
    return Prediction(
        predicted_agent=agent_match.group(1) if agent_match else "",
        predicted_step=step_match.group(1) if step_match else "",
        raw=text,
    )


def _agent_correct(actual: str | None, predicted: str) -> bool:
    return bool(actual) and bool(predicted) and actual in predicted


def _step_correct(actual: str | None, predicted: str) -> bool:
    return bool(actual) and bool(predicted) and actual in predicted


def score(
    predictions: dict[str, Prediction],
    truth: dict[str, tuple[str | None, str | None]],
    total: int,
) -> Scores:
    """Score predictions against ground truth.

    ``predictions`` and ``truth`` are keyed by the same id (the source filename).
    ``truth`` maps id -> (mistake_agent, mistake_step). ``total`` is the number of
    logs in the dataset subset (the accuracy denominator).
    """
    correct_agent = 0
    correct_step = 0
    for key, pred in predictions.items():
        actual_agent, actual_step = truth.get(key, (None, None))
        if _agent_correct(actual_agent, pred.predicted_agent):
            correct_agent += 1
        if _step_correct(actual_step, pred.predicted_step):
            correct_step += 1

    denom = total if total > 0 else 1
    return Scores(
        agent_accuracy=correct_agent / denom * 100.0,
        step_accuracy=correct_step / denom * 100.0,
        correct_agent=correct_agent,
        correct_step=correct_step,
        total=total,
    )
