"""Shared test fixtures: tiny synthetic traces and a scriptable mock judge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mafa.io import Step, Trace
from mafa.judge.base import Judge, JudgeError


class MockJudge(Judge):
    """A judge that returns scripted responses, in order, and records prompts.

    If ``responses`` is a list it is consumed one per call; if it is a dict it is
    matched by substring against the user prompt (first match wins).
    """

    def __init__(self, responses, model_id="mock-1"):
        self.model_id = model_id
        self._responses = responses
        self._i = 0
        self.calls: list[tuple[str, str]] = []
        self.fail_times = 0  # raise JudgeError on the first N calls

    def complete(self, system, user, max_tokens=1024, temperature=0.0) -> str:
        self.calls.append((system, user))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise JudgeError("scripted failure")
        if isinstance(self._responses, dict):
            for needle, resp in self._responses.items():
                if needle in user:
                    return resp
            return "Agent Name: nobody\n Step Number: 0\n"
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return resp


@pytest.fixture
def mock_judge():
    return MockJudge


@pytest.fixture
def sample_trace():
    return Trace(
        question_id="q1",
        question="What is 2+2?",
        ground_truth="4",
        steps=[
            Step(0, "Planner", "Let's compute."),
            Step(1, "Math_Expert", "2+2=5"),
            Step(2, "Checker", "Looks right."),
        ],
        mistake_agent="Math_Expert",
        mistake_step="1",
        mistake_reason="arithmetic error",
        is_correct=False,
        source_file="1.json",
        is_handcrafted=False,
    )


@pytest.fixture
def algo_log(tmp_path: Path) -> Path:
    """Write a realistic Algorithm-Generated-style log (agent id in ``name``)."""
    data = {
        "is_correct": False,
        "question": "What is 2+2?",
        "question_ID": "q1",
        "ground_truth": "4",
        "history": [
            {"role": "assistant", "name": "Planner", "content": "Let's compute."},
            {"role": "assistant", "name": "Math_Expert", "content": "2+2=5"},
        ],
        "mistake_agent": "Math_Expert",
        "mistake_step": "1",
        "mistake_reason": "arithmetic error",
        "system_prompt": "solve it",
    }
    p = tmp_path / "1.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def hand_log(tmp_path: Path) -> Path:
    """Write a Hand-Crafted-style log (agent id in ``role``, no ``name``)."""
    data = {
        "is_correct": False,
        "question": "Find X.",
        "question_ID": "h1",
        "ground_truth": "42",
        "history": [
            {"role": "human", "content": "Find X."},
            {"role": "WebSurfer", "content": "X is 7."},
        ],
        "mistake_agent": "WebSurfer",
        "mistake_step": "1",
        "mistake_reason": "wrong lookup",
        "system_prompt": "",
    }
    p = tmp_path / "1.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p
