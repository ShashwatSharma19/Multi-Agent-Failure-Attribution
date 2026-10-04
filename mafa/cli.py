"""Run a baseline over a Who&When subset and report accuracy vs. the paper.

Example::

    python -m mafa.cli --method all_at_once --judge ollama --subset algo --limit 20

The run prints our agent/step accuracy next to the published baseline so the
comparison is always honest: we report (a) *our* reproduction with *our* judge and
(b) the paper's number produced with GPT-4o — they are not the same judge, so the
real claim later is "fused method vs. our own reproduced baseline".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mafa.baselines import METHODS
from mafa.eval.scorer import score
from mafa.io import load_dataset
from mafa.judge.factory import build_judge

# Published baselines (Zhang et al., ICML 2025), for reference only.
PAPER_BASELINE = {"agent": 53.5, "step": 14.2}

_SUBSET_DIR = {
    "algo": ("data/whoandwhen/Who&When/Algorithm-Generated", False),
    "hand": ("data/whoandwhen/Who&When/Hand-Crafted", True),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MAFA baseline runner")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--judge", default="ollama", choices=["ollama", "groq", "gemini"])
    p.add_argument("--model", default=None, help="override the judge model id")
    p.add_argument("--subset", default="algo", choices=list(_SUBSET_DIR))
    p.add_argument("--limit", type=int, default=None, help="only the first N logs (dev loop)")
    p.add_argument("--no-ground-truth", action="store_true", help="answer-free ablation")
    p.add_argument("--out", default="outputs", help="where to write predictions json")
    args = p.parse_args(argv)

    rel_dir, is_handcrafted = _SUBSET_DIR[args.subset]
    traces = load_dataset(rel_dir, is_handcrafted, limit=args.limit)
    if not traces:
        print(f"No traces found in {rel_dir}")
        return 1

    judge = build_judge(args.judge, model=args.model)
    method = METHODS[args.method]
    include_gt = not args.no_ground_truth

    predictions = {}
    truth = {}
    for i, trace in enumerate(traces, 1):
        pred = method(judge, trace, include_ground_truth=include_gt)
        predictions[trace.source_file] = pred
        truth[trace.source_file] = (trace.mistake_agent, trace.mistake_step)
        print(f"[{i}/{len(traces)}] {trace.source_file}: "
              f"agent={pred.predicted_agent!r} step={pred.predicted_step!r}")

    scores = score(predictions, truth, total=len(traces))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}_{args.judge}_{args.subset}.json"
    out_path.write_text(
        json.dumps(
            {
                "method": args.method,
                "judge": args.judge,
                "model": judge.model_id,
                "subset": args.subset,
                "n": len(traces),
                "agent_accuracy": scores.agent_accuracy,
                "step_accuracy": scores.step_accuracy,
                "predictions": {
                    k: {"agent": v.predicted_agent, "step": v.predicted_step}
                    for k, v in predictions.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Results ===")
    print(f"method={args.method} judge={args.judge} model={judge.model_id} "
          f"subset={args.subset} n={len(traces)}")
    print(f"Agent accuracy: {scores.agent_accuracy:5.1f}%  "
          f"({scores.correct_agent}/{scores.total})   "
          f"[paper GPT-4o baseline: {PAPER_BASELINE['agent']}%]")
    print(f"Step  accuracy: {scores.step_accuracy:5.1f}%  "
          f"({scores.correct_step}/{scores.total})   "
          f"[paper GPT-4o baseline: {PAPER_BASELINE['step']}%]")
    cache_hits = getattr(judge, "hits", None)
    if cache_hits is not None:
        print(f"(cache hits={judge.hits} misses={judge.misses})")
    print(f"predictions written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
