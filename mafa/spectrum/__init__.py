"""Phase 3: spectrum-based fault localization adapted to an all-failures benchmark.

Who&When has no passing runs, so a whole-run spectrum is impossible. Instead we
build a **step-level** spectrum: each step is a "test case" labeled error/not by a
cheap per-step probe, and each agent is a "component" that covers the steps it
authored. Ochiai suspiciousness then ranks agents; a cross-log ``RolePrior``
accumulates the same signal per recurring role.
"""

from mafa.spectrum.labeler import StepLabel, label_steps
from mafa.spectrum.ochiai import RolePrior, ochiai_scores, spectrum_prediction

__all__ = ["StepLabel", "label_steps", "ochiai_scores", "spectrum_prediction", "RolePrior"]
