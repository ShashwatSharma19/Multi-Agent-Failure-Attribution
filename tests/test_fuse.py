from mafa.eval.scorer import Prediction
from mafa.fuse import blend, fuse, normalize, split_traces, tune_weight
from mafa.spectrum import StepLabel


def test_normalize_scales_to_unit_max():
    assert normalize({"a": 2.0, "b": 1.0}) == {"a": 1.0, "b": 0.5}
    assert normalize({"a": 0.0}) == {"a": 0.0}
    assert normalize({}) == {}


def test_blend_convex_combination():
    out = blend({"a": 1.0}, {"a": 0.0, "b": 1.0}, alpha=0.75)
    assert out["a"] == 0.75
    assert out["b"] == 0.25


def test_fuse_weight_one_follows_llm():
    llm = Prediction("Planner", "0")
    spec = {"Math_Expert": 1.0}
    labels = [StepLabel(1, "Math_Expert", True)]
    pred = fuse(llm, spec, labels, weight=1.0)
    assert pred.predicted_agent == "Planner"
    assert pred.predicted_step == "0"


def test_fuse_weight_zero_follows_spectrum():
    llm = Prediction("Planner", "0")
    spec = {"Math_Expert": 1.0}
    labels = [StepLabel(1, "Math_Expert", True)]
    pred = fuse(llm, spec, labels, weight=0.0)
    assert pred.predicted_agent == "Math_Expert"
    assert pred.predicted_step == "1"  # earliest flagged step of the chosen agent


def test_split_traces_disjoint_and_reproducible(sample_trace):
    import dataclasses

    traces = [dataclasses.replace(sample_trace, source_file=f"{i}.json") for i in range(10)]
    dev, test = split_traces(traces, dev_frac=0.3, seed=0)
    assert len(dev) == 3 and len(test) == 7
    assert set(t.source_file for t in dev).isdisjoint(t.source_file for t in test)
    dev2, _ = split_traces(traces, dev_frac=0.3, seed=0)
    assert [t.source_file for t in dev] == [t.source_file for t in dev2]  # deterministic


def test_tune_weight_prefers_spectrum_when_it_is_right():
    # LLM always wrong, spectrum always right -> tuning should favor low weight.
    labels = [StepLabel(1, "Right", True)]
    items = [
        (Prediction("Wrong", "9"), {"Right": 1.0}, labels, ("Right", "1"))
        for _ in range(5)
    ]
    assert tune_weight(items) == 0.0
