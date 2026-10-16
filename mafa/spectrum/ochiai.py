"""Ochiai suspiciousness over step labels, plus a cross-log role prior.

Ochiai for a component ``c``:  ``ef / sqrt((ef + nf) * (ef + ep))`` where

* ``ef`` = failing tests covering ``c``  -> error steps authored by the agent,
* ``ep`` = passing tests covering ``c``  -> non-error steps authored by the agent,
* ``nf`` = failing tests NOT covering ``c`` -> error steps by other agents.

Here ``ef + nf`` is the total number of error steps and ``ef + ep`` is the total
steps authored by the agent, so the score simplifies to
``ef / sqrt(total_error * steps_by_agent)``.
"""

from __future__ import annotations

import math
from collections import defaultdict

from mafa.eval.scorer import Prediction
from mafa.spectrum.labeler import StepLabel


def ochiai_scores(labels: list[StepLabel]) -> dict[str, float]:
    """Per-agent Ochiai suspiciousness from one log's step labels."""
    total_error = sum(1 for lab in labels if lab.is_error)
    err_by_agent: dict[str, int] = defaultdict(int)
    total_by_agent: dict[str, int] = defaultdict(int)
    for lab in labels:
        total_by_agent[lab.agent] += 1
        if lab.is_error:
            err_by_agent[lab.agent] += 1

    scores: dict[str, float] = {}
    for agent, total in total_by_agent.items():
        ef = err_by_agent[agent]
        denom = math.sqrt(total_error * total) if total_error > 0 and total > 0 else 0.0
        scores[agent] = ef / denom if denom > 0 else 0.0
    return scores


def spectrum_prediction(labels: list[StepLabel], scores: dict[str, float]) -> Prediction:
    """Pick the most suspicious agent and its earliest flagged step."""
    if not scores or max(scores.values(), default=0.0) <= 0.0:
        return Prediction("", "", raw="no error signal in spectrum")
    agent = max(scores, key=lambda a: scores[a])
    flagged = [lab.index for lab in labels if lab.agent == agent and lab.is_error]
    if not flagged:
        return Prediction(agent, "", raw="agent chosen but no flagged step")
    return Prediction(agent, str(min(flagged)), raw=f"spectrum suspiciousness={scores[agent]:.3f}")


class RolePrior:
    """Accumulates the step-level spectrum per role *across logs*.

    This supplies the "many observations of the same component" that a single
    all-failures log cannot, by pooling the error/coverage counts for each role
    (agent name) seen so far.
    """

    def __init__(self) -> None:
        self._err: dict[str, int] = defaultdict(int)
        self._total: dict[str, int] = defaultdict(int)
        self._total_err = 0

    def update(self, labels: list[StepLabel]) -> None:
        for lab in labels:
            self._total[lab.agent] += 1
            if lab.is_error:
                self._err[lab.agent] += 1
                self._total_err += 1

    def score(self, role: str) -> float:
        total = self._total.get(role, 0)
        if self._total_err <= 0 or total <= 0:
            return 0.0
        return self._err.get(role, 0) / math.sqrt(self._total_err * total)

    def scores(self) -> dict[str, float]:
        return {role: self.score(role) for role in self._total}
