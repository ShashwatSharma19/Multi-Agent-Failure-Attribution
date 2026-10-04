"""Evaluation: parse predictions and score them against ground truth."""

from mafa.eval.scorer import Prediction, Scores, parse_prediction, score

__all__ = ["Prediction", "Scores", "parse_prediction", "score"]
