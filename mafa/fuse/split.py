"""Dev/test split and weight tuning — so fusion weights never see the test set.

With only 184 logs, tuning the fusion weight on the same logs we report on would be
tuning on the test set. ``split_traces`` carves out a held-out dev set (reproducible
via ``seed``); ``tune_weight`` grid-searches the weight on dev only.
"""

from __future__ import annotations

import random

from mafa.eval.scorer import Prediction
from mafa.fuse.fusion import fuse
from mafa.io import Trace
from mafa.spectrum.labeler import StepLabel

# One precomputed item per dev trace: the LLM vote, the spectrum, the labels, and truth.
DevItem = tuple[Prediction, dict[str, float], list[StepLabel], tuple[str | None, str | None]]

_DEFAULT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def split_traces(
    traces: list[Trace], dev_frac: float = 0.3, seed: int = 0
) -> tuple[list[Trace], list[Trace]]:
    """Return ``(dev, test)``. Order within each split follows the input order."""
    idx = list(range(len(traces)))
    random.Random(seed).shuffle(idx)
    k = int(len(traces) * dev_frac)
    dev_files = {traces[i].source_file for i in idx[:k]}
    dev = [t for t in traces if t.source_file in dev_files]
    test = [t for t in traces if t.source_file not in dev_files]
    return dev, test


def _agent_correct(actual: str | None, predicted: str) -> bool:
    return bool(actual) and bool(predicted) and actual in predicted


def _step_correct(actual: str | None, predicted: str) -> bool:
    return bool(actual) and bool(predicted) and actual in predicted


def tune_weight(dev_items: list[DevItem], grid: tuple[float, ...] = _DEFAULT_GRID) -> float:
    """Pick the weight maximizing (agent_correct + step_correct) on the dev set.

    Ties break toward the smaller weight (lean on the spectrum). Empty dev → 0.5.
    """
    if not dev_items:
        return 0.5
    best_w, best_score = _DEFAULT_GRID[len(_DEFAULT_GRID) // 2], -1
    for w in grid:
        score = 0
        for llm_pred, spec, labels, (ta, ts) in dev_items:
            pred = fuse(llm_pred, spec, labels, weight=w)
            score += _agent_correct(ta, pred.predicted_agent)
            score += _step_correct(ts, pred.predicted_step)
        if score > best_score:
            best_score, best_w = score, w
    return best_w
