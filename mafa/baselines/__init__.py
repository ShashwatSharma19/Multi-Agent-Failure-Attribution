"""The three Who&When baseline attribution strategies.

Each ``run(judge, trace)`` returns a :class:`~mafa.eval.scorer.Prediction`.
Prompts are copied from the reference ``Automated_FA/Lib/utils.py`` so our numbers
are a faithful reproduction, not a reinvention.
"""

from mafa.baselines.all_at_once import run as all_at_once
from mafa.baselines.binary_search import run as binary_search
from mafa.baselines.step_by_step import run as step_by_step

METHODS = {
    "all_at_once": all_at_once,
    "step_by_step": step_by_step,
    "binary_search": binary_search,
}

__all__ = ["all_at_once", "step_by_step", "binary_search", "METHODS"]
