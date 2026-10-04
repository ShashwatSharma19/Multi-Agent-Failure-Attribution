from mafa.eval.scorer import Prediction, parse_prediction, score


def test_parse_prediction_standard_format():
    text = "Agent Name: Math_Expert\n Step Number: 1\n Reason for Mistake: bad math"
    pred = parse_prediction(text)
    assert pred.predicted_agent == "Math_Expert"
    assert pred.predicted_step == "1"


def test_parse_prediction_missing_fields_are_empty():
    pred = parse_prediction("I could not decide.")
    assert pred.predicted_agent == ""
    assert pred.predicted_step == ""


def test_score_substring_match_semantics():
    # Reference uses `actual in predicted`: ground-truth agent is a substring of prediction.
    preds = {"1.json": Prediction("Math_Expert_2", "1")}
    truth = {"1.json": ("Math_Expert", "1")}
    s = score(preds, truth, total=1)
    assert s.correct_agent == 1
    assert s.correct_step == 1


def test_score_denominator_is_total_not_predictions():
    # 1 correct out of 2 total logs -> 50%, even if only 1 prediction was made.
    preds = {"1.json": Prediction("Math_Expert", "1")}
    truth = {"1.json": ("Math_Expert", "1"), "2.json": ("Planner", "0")}
    s = score(preds, truth, total=2)
    assert s.agent_accuracy == 50.0
    assert s.step_accuracy == 50.0


def test_empty_prediction_counts_as_wrong():
    preds = {"1.json": Prediction("", "")}
    truth = {"1.json": ("Math_Expert", "1")}
    s = score(preds, truth, total=1)
    assert s.correct_agent == 0
    assert s.correct_step == 0
