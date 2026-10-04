from mafa.baselines import all_at_once, binary_search, step_by_step
from tests.conftest import MockJudge


def test_all_at_once_parses_single_response(sample_trace):
    judge = MockJudge(["Agent Name: Math_Expert\n Step Number: 1\n Reason: wrong"])
    pred = all_at_once(judge, sample_trace)
    assert pred.predicted_agent == "Math_Expert"
    assert pred.predicted_step == "1"
    assert len(judge.calls) == 1  # one call for the whole transcript


def test_step_by_step_stops_at_first_yes(sample_trace):
    # No for step 0, Yes for step 1 -> predicts agent/step of step 1.
    judge = MockJudge(["1. No\n2. Reason: fine", "1. Yes\n2. Reason: 2+2!=5"])
    pred = step_by_step(judge, sample_trace)
    assert pred.predicted_agent == "Math_Expert"
    assert pred.predicted_step == "1"
    assert len(judge.calls) == 2  # stopped early, did not evaluate step 2


def test_step_by_step_no_error_found_is_empty(sample_trace):
    judge = MockJudge(["1. No\n2. Reason: fine"])  # repeats No for every step
    pred = step_by_step(judge, sample_trace)
    assert pred.predicted_agent == ""


def test_binary_search_converges_to_a_step(sample_trace):
    # Always "lower half" -> should converge to the last step (index 2).
    judge = MockJudge(["lower half"])  # repeated
    pred = binary_search(judge, sample_trace)
    assert pred.predicted_step == "2"
    assert pred.predicted_agent == "Checker"


def test_binary_search_upper_half_goes_left(sample_trace):
    judge = MockJudge(["upper half"])  # repeated -> converge to step 0
    pred = binary_search(judge, sample_trace)
    assert pred.predicted_step == "0"
