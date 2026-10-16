"""Weighted fusion of the LLM vote and the spectrum score.

``score(agent) = weight * llm_vote(agent) + (1 - weight) * spectrum_norm(agent)``

The LLM vote is a hard indicator (1.0 for the agent the base method named, else 0).
The spectrum is max-normalized to [0, 1] so the two signals are comparable. We
start with this single-weight scheme rather than a learned ranker: 184 logs cannot
train a ranker and still hold a clean test split.
"""

from __future__ import annotations

from mafa.eval.scorer import Prediction
from mafa.spectrum.labeler import StepLabel


def normalize(scores: dict[str, float]) -> dict[str, float]:
    """Max-normalize to [0, 1]; all-zero (or empty) maps to zeros."""
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / top for k, v in scores.items()}


def blend(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    """Convex blend of two score maps: ``alpha*a + (1-alpha)*b`` over the key union."""
    keys = set(a) | set(b)
    return {k: alpha * a.get(k, 0.0) + (1 - alpha) * b.get(k, 0.0) for k in keys}


def fuse(
    llm_pred: Prediction,
    spectrum_scores: dict[str, float],
    labels: list[StepLabel],
    weight: float = 0.5,
) -> Prediction:
    """Fuse an LLM prediction with spectrum suspiciousness into one prediction."""
    spec = normalize(spectrum_scores)
    candidates = set(spec)
    if llm_pred.predicted_agent:
        candidates.add(llm_pred.predicted_agent)
    if not candidates:
        return Prediction("", "", raw="no candidates")

    def final(agent: str) -> float:
        llm = 1.0 if agent and agent == llm_pred.predicted_agent else 0.0
        return weight * llm + (1 - weight) * spec.get(agent, 0.0)

    agent = max(candidates, key=final)

    if agent == llm_pred.predicted_agent and llm_pred.predicted_step:
        step = llm_pred.predicted_step
    else:
        flagged = [lab.index for lab in labels if lab.agent == agent and lab.is_error]
        step = str(min(flagged)) if flagged else llm_pred.predicted_step
    return Prediction(agent, step, raw=f"fused(weight={weight})")
