from mafa.io import load_dataset, load_trace


def test_algo_log_uses_name_field(algo_log):
    trace = load_trace(algo_log, is_handcrafted=False)
    assert [s.agent for s in trace.steps] == ["Planner", "Math_Expert"]
    assert trace.mistake_agent == "Math_Expert"
    assert trace.mistake_step == "1"  # kept as string, 0-indexed
    assert trace.ground_truth == "4"
    assert trace.is_correct is False


def test_hand_log_uses_role_field(hand_log):
    trace = load_trace(hand_log, is_handcrafted=True)
    assert [s.agent for s in trace.steps] == ["human", "WebSurfer"]
    assert trace.mistake_agent == "WebSurfer"


def test_transcript_format(sample_trace):
    assert sample_trace.transcript.splitlines()[1] == "Math_Expert: 2+2=5"


def test_steps_are_zero_indexed(sample_trace):
    assert [s.index for s in sample_trace.steps] == [0, 1, 2]


def test_load_dataset_numeric_sort_and_limit(tmp_path):
    import json

    for n in (1, 2, 10):
        (tmp_path / f"{n}.json").write_text(
            json.dumps({"history": [], "question": f"q{n}", "mistake_agent": "a", "mistake_step": "0"}),
            encoding="utf-8",
        )
    traces = load_dataset(tmp_path, is_handcrafted=False)
    assert [t.source_file for t in traces] == ["1.json", "2.json", "10.json"]
    assert len(load_dataset(tmp_path, is_handcrafted=False, limit=2)) == 2
