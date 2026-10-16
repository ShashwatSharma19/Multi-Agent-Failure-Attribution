import math

from mafa.spectrum import RolePrior, StepLabel, label_steps, ochiai_scores, spectrum_prediction
from tests.conftest import MockJudge


def test_label_steps_marks_errors_and_respects_scope(sample_trace):
    # Yes for every step, but scope restricts which steps are evaluated.
    judge = MockJudge(["1. Yes\n2. Reason: bad"])
    labels = label_steps(judge, sample_trace, scope=(1, 2))
    assert [lab.index for lab in labels] == [1, 2]  # step 0 skipped
    assert all(lab.is_error for lab in labels)
    assert len(judge.calls) == 2


def test_ochiai_formula():
    # 3 steps by A (1 error), 1 step by B (1 error). total_error=2.
    labels = [
        StepLabel(0, "A", True),
        StepLabel(1, "A", False),
        StepLabel(2, "A", False),
        StepLabel(3, "B", True),
    ]
    scores = ochiai_scores(labels)
    assert math.isclose(scores["A"], 1 / math.sqrt(2 * 3))
    assert math.isclose(scores["B"], 1 / math.sqrt(2 * 1))
    assert scores["B"] > scores["A"]  # B is more concentrated -> more suspicious


def test_ochiai_all_clean_is_zero():
    labels = [StepLabel(0, "A", False), StepLabel(1, "B", False)]
    assert ochiai_scores(labels) == {"A": 0.0, "B": 0.0}


def test_spectrum_prediction_picks_top_agent_earliest_step():
    labels = [StepLabel(0, "A", True), StepLabel(1, "A", True), StepLabel(2, "B", True)]
    scores = {"A": 0.9, "B": 0.3}
    pred = spectrum_prediction(labels, scores)
    assert pred.predicted_agent == "A"
    assert pred.predicted_step == "0"  # earliest flagged step by A


def test_spectrum_prediction_no_signal_is_empty():
    labels = [StepLabel(0, "A", False)]
    assert spectrum_prediction(labels, {"A": 0.0}).predicted_agent == ""


def test_role_prior_accumulates_across_logs():
    prior = RolePrior()
    prior.update([StepLabel(0, "WebSurfer", True), StepLabel(1, "Planner", False)])
    prior.update([StepLabel(0, "WebSurfer", True), StepLabel(1, "WebSurfer", False)])
    # WebSurfer: 2 errors of 3 steps; total_error=2 -> 2/sqrt(2*3)
    assert math.isclose(prior.score("WebSurfer"), 2 / math.sqrt(2 * 3))
    assert prior.score("Planner") == 0.0
    assert prior.score("Unseen") == 0.0
