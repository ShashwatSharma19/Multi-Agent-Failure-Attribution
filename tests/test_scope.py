from mafa.io import Step, Trace
from mafa.scope import delineate, windows
from tests.conftest import MockJudge


def _trace(n: int) -> Trace:
    return Trace(
        question_id="q",
        question="solve",
        ground_truth="42",
        steps=[Step(i, f"Agent{i}", f"content {i}") for i in range(n)],
        mistake_agent="Agent3",
        mistake_step="3",
        mistake_reason="",
        is_correct=False,
        source_file="1.json",
        is_handcrafted=False,
    )


def test_windows_split_even():
    assert windows(6, 3) == [(0, 1), (2, 3), (4, 5)]


def test_windows_split_uneven():
    assert windows(7, 3) == [(0, 2), (3, 5), (6, 6)]


def test_short_trace_returns_full_range_without_calling_judge():
    judge = MockJudge(["SHOULD NOT BE CALLED"])
    assert delineate(judge, _trace(3), n_windows=3) == (0, 2)
    assert judge.calls == []  # too short to narrow -> no judge call


def test_delineate_selects_window_from_answer():
    # 6 steps, 3 windows -> segments 1:(0,1) 2:(2,3) 3:(4,5). Judge says "2".
    judge = MockJudge(["Segment 2"])
    assert delineate(judge, _trace(6), n_windows=3) == (2, 3)
    assert len(judge.calls) == 1  # exactly one cheap call


def test_delineate_unparseable_answer_falls_back_to_last_window():
    judge = MockJudge(["I am not sure"])
    assert delineate(judge, _trace(6), n_windows=3) == (4, 5)


def test_delineate_out_of_range_index_falls_back_to_last_window():
    judge = MockJudge(["Segment 9"])
    assert delineate(judge, _trace(6), n_windows=3) == (4, 5)
