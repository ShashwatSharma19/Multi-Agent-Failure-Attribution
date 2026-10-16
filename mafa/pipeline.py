"""Orchestration: gather judge-derived evidence once, then finalize a prediction.

Splitting "gather" (makes judge calls) from "finalize" (pure) lets us tune the
fusion weight and build the role prior *after* all the expensive calls are cached,
without re-querying the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from mafa.baselines import METHODS as _BASELINES
from mafa.eval.scorer import Prediction
from mafa.fuse.fusion import blend, fuse
from mafa.io import Trace
from mafa.judge.base import Judge
from mafa.scope import delineate
from mafa.spectrum import RolePrior, StepLabel, label_steps, ochiai_scores, spectrum_prediction

ALL_METHODS = list(_BASELINES) + ["spectrum", "fused"]


@dataclass
class Evidence:
    source_file: str
    truth: tuple[str | None, str | None]
    scope: tuple[int, int] | None
    baseline_pred: Prediction | None  # set for the three baseline methods
    llm_pred: Prediction | None  # the base LLM vote, for fused
    labels: list[StepLabel] | None  # for spectrum/fused
    scores: dict[str, float] | None  # raw Ochiai, for spectrum/fused


def gather_evidence(
    method: str,
    judge: Judge,
    trace: Trace,
    include_ground_truth: bool = True,
    use_scope: bool = False,
    windows: int = 3,
    base_method: str = "all_at_once",
) -> Evidence:
    """Run the judge-dependent work for one trace and return its evidence."""
    scope = (
        delineate(judge, trace, n_windows=windows, include_ground_truth=include_ground_truth)
        if use_scope
        else None
    )
    truth = (trace.mistake_agent, trace.mistake_step)

    if method in _BASELINES:
        pred = _BASELINES[method](judge, trace, include_ground_truth=include_ground_truth, scope=scope)
        return Evidence(trace.source_file, truth, scope, pred, None, None, None)

    # spectrum / fused both need per-step labels and Ochiai scores
    labels = label_steps(judge, trace, scope=scope, include_ground_truth=include_ground_truth)
    scores = ochiai_scores(labels)
    llm_pred = None
    if method == "fused":
        llm_pred = _BASELINES[base_method](
            judge, trace, include_ground_truth=include_ground_truth, scope=scope
        )
    return Evidence(trace.source_file, truth, scope, None, llm_pred, labels, scores)


def effective_scores(ev: Evidence, role_prior: RolePrior | None, role_weight: float) -> dict[str, float]:
    """Ochiai scores, optionally blended with the cross-log role prior."""
    scores = ev.scores or {}
    if role_prior is None:
        return scores
    return blend(scores, role_prior.scores(), role_weight)


def finalize(method: str, ev: Evidence, scores: dict[str, float], weight: float) -> Prediction:
    """Turn evidence (+ effective scores) into the final prediction. Pure."""
    if ev.baseline_pred is not None:
        return ev.baseline_pred
    if method == "spectrum":
        return spectrum_prediction(ev.labels or [], scores)
    return fuse(ev.llm_pred or Prediction("", ""), scores, ev.labels or [], weight=weight)
