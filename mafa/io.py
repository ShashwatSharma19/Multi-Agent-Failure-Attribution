"""Load Who&When logs into a normalized ``Trace`` object.

A Who&When JSON file has this shape::

    {
      "is_correct": false,
      "question": "...",
      "question_ID": "...",
      "ground_truth": "...",
      "history": [{"content": "...", "role": "...", "name": "..."}, ...],
      "mistake_agent": "Excel_Expert",
      "mistake_step": "0",
      "mistake_reason": "...",
      "system_prompt": "..."
    }

The agent identity lives in a different field per subset: ``name`` for the
Algorithm-Generated logs, ``role`` for the Hand-Crafted logs (this matches the
reference implementation's ``index_agent = "role" if is_handcrafted else "name"``).
Steps are 0-indexed, matching ``mistake_step``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """One turn in a multi-agent conversation."""

    index: int
    agent: str
    content: str


@dataclass(frozen=True)
class Trace:
    """A normalized, failed multi-agent run with ground-truth attribution."""

    question_id: str
    question: str
    ground_truth: str
    steps: list[Step]
    mistake_agent: str | None
    mistake_step: str | None
    mistake_reason: str | None
    is_correct: bool
    source_file: str
    is_handcrafted: bool

    @property
    def transcript(self) -> str:
        """Render steps as ``agent: content`` lines (reference format)."""
        return "\n".join(f"{s.agent}: {s.content}" for s in self.steps)


def _agent_field(is_handcrafted: bool) -> str:
    return "role" if is_handcrafted else "name"


def load_trace(path: str | Path, is_handcrafted: bool) -> Trace:
    """Parse a single Who&When JSON file into a :class:`Trace`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    field = _agent_field(is_handcrafted)
    steps = [
        Step(
            index=i,
            agent=str(entry.get(field, "Unknown Agent")),
            content=str(entry.get("content", "")),
        )
        for i, entry in enumerate(data.get("history", []))
    ]

    def _opt_str(key: str) -> str | None:
        val = data.get(key)
        return None if val is None else str(val)

    return Trace(
        question_id=str(data.get("question_ID", path.stem)),
        question=str(data.get("question", "")),
        ground_truth=str(data.get("ground_truth", "")),
        steps=steps,
        mistake_agent=_opt_str("mistake_agent"),
        mistake_step=_opt_str("mistake_step"),
        mistake_reason=_opt_str("mistake_reason"),
        is_correct=bool(data.get("is_correct", False)),
        source_file=path.name,
        is_handcrafted=is_handcrafted,
    )


def _sorted_json_files(directory: Path) -> list[Path]:
    """JSON files sorted numerically (``1.json`` < ``2.json`` < ``10.json``)."""
    files = [p for p in directory.iterdir() if p.suffix == ".json"]
    return sorted(files, key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))


def load_dataset(
    directory: str | Path,
    is_handcrafted: bool,
    limit: int | None = None,
) -> list[Trace]:
    """Load every log in a Who&When subset directory, in numeric filename order.

    ``limit`` truncates to the first N logs — use a small value for a cheap
    development loop so full 184-log runs stay rare.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    files = _sorted_json_files(directory)
    if limit is not None:
        files = files[:limit]
    return [load_trace(p, is_handcrafted) for p in files]
