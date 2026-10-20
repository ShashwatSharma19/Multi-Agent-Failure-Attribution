"""Run an attribution method over a Who&When subset and report accuracy + cost.

Examples::

    python -m mafa.cli --method all_at_once --judge ollama --subset algo --limit 20
    python -m mafa.cli --method fused --judge groq --subset hand --scope --tune

Honesty note: we report (a) *our* reproduction with *our* judge and (b) the paper's
GPT-4o number — different judges, so the real claim is "fused vs. our own baseline".
When ``--tune`` is set the fusion weight is chosen on a held-out dev split and the
reported numbers are on the untouched test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from mafa.eval.scorer import score
from mafa.fuse.split import split_traces, tune_weight
from mafa.io import load_dataset
from mafa.judge import AccountingJudge, CachingJudge, build_judge, unwrap
from mafa.pipeline import ALL_METHODS, effective_scores, finalize, gather_evidence
from mafa.spectrum import RolePrior

# Published baselines (Zhang et al., ICML 2025), for reference only.
PAPER_BASELINE = {"agent": 53.5, "step": 14.2}

_SUBSET_DIR = {
    "algo": ("data/whoandwhen/Who&When/Algorithm-Generated", False),
    "hand": ("data/whoandwhen/Who&When/Hand-Crafted", True),
}


def _parse_args(argv):
    p = argparse.ArgumentParser(description="MAFA attribution runner")
    p.add_argument("--method", required=True, choices=ALL_METHODS)
    p.add_argument("--judge", default="ollama", choices=["ollama", "groq", "gemini"])
    p.add_argument("--model", default=None, help="override the judge model id")
    p.add_argument("--subset", default="algo", choices=list(_SUBSET_DIR))
    p.add_argument("--limit", type=int, default=None, help="only the first N logs (dev loop)")
    p.add_argument("--no-ground-truth", action="store_true", help="answer-free ablation")
    p.add_argument("--scope", action="store_true", help="enable scope-delineation (Phase 2)")
    p.add_argument("--windows", type=int, default=3, help="number of scope windows")
    p.add_argument("--base-method", default="all_at_once", choices=["all_at_once", "step_by_step", "binary_search"],
                   help="LLM vote used inside fused")
    p.add_argument("--fuse-weight", type=float, default=0.5, help="fusion weight (ignored if --tune)")
    p.add_argument("--tune", action="store_true", help="tune fusion weight on a dev split, report on test")
    p.add_argument("--dev-frac", type=float, default=0.3, help="held-out dev fraction for --tune")
    p.add_argument("--role-prior", action="store_true", help="blend a cross-log role prior into the spectrum")
    p.add_argument("--role-weight", type=float, default=0.5, help="weight on per-log Ochiai vs role prior")
    p.add_argument("--out", default="outputs", help="where to write predictions json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # read GROQ_API_KEY / GEMINI_API_KEY from a .env in the project root
    args = _parse_args(argv)
    rel_dir, is_handcrafted = _SUBSET_DIR[args.subset]
    traces = load_dataset(rel_dir, is_handcrafted, limit=args.limit)
    if not traces:
        print(f"No traces found in {rel_dir}")
        return 1

    judge = build_judge(args.judge, model=args.model)
    accounting = unwrap(judge, AccountingJudge)
    cache = unwrap(judge, CachingJudge)
    include_gt = not args.no_ground_truth

    if args.scope and args.method == "all_at_once":
        print("note: --scope has no effect on all_at_once; it narrows the per-step methods.")

    # --- Pass 1: gather judge-derived evidence once per trace (records cost) ---
    evidence = {}
    cost = {}
    for i, trace in enumerate(traces, 1):
        if accounting is not None:
            accounting.reset()
        ev = gather_evidence(
            args.method, judge, trace,
            include_ground_truth=include_gt, use_scope=args.scope,
            windows=args.windows, base_method=args.base_method,
        )
        evidence[trace.source_file] = ev
        cost[trace.source_file] = (
            accounting.calls if accounting else 0,
            accounting.tokens if accounting else 0,
        )
        print(f"[{i}/{len(traces)}] {trace.source_file}: gathered "
              f"(calls={cost[trace.source_file][0]} ~tokens={cost[trace.source_file][1]}"
              f"{f' scope={ev.scope}' if ev.scope else ''})")

    # --- Decide reporting set, role prior, and fusion weight ---
    is_fused = args.method == "fused"
    if is_fused and args.tune:
        dev_traces, test_traces = split_traces(traces, dev_frac=args.dev_frac)
    else:
        dev_traces, test_traces = [], traces
    report_traces = test_traces

    role_prior = None
    if args.role_prior:
        role_prior = RolePrior()
        prior_source = dev_traces if (is_fused and args.tune) else traces
        for t in prior_source:
            lab = evidence[t.source_file].labels
            if lab:
                role_prior.update(lab)

    weight = args.fuse_weight
    if is_fused and args.tune:
        dev_items = []
        for t in dev_traces:
            ev = evidence[t.source_file]
            dev_items.append((ev.llm_pred, effective_scores(ev, role_prior, args.role_weight), ev.labels or [], ev.truth))
        weight = tune_weight(dev_items)
        print(f"\ntuned fusion weight on {len(dev_traces)} dev logs -> weight={weight}")

    # --- Pass 2: finalize predictions on the reporting set (pure, no judge calls) ---
    predictions = {}
    truth = {}
    total_calls = total_tokens = 0
    for t in report_traces:
        ev = evidence[t.source_file]
        scores = effective_scores(ev, role_prior, args.role_weight)
        predictions[t.source_file] = finalize(args.method, ev, scores, weight)
        truth[t.source_file] = ev.truth
        c, tok = cost[t.source_file]
        total_calls += c
        total_tokens += tok

    scores_out = score(predictions, truth, total=len(report_traces))
    n = len(report_traces)
    avg_calls = total_calls / n if n else 0
    avg_tokens = total_tokens / n if n else 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_scope" if args.scope else ""
    suffix += "_tuned" if (is_fused and args.tune) else ""
    out_path = out_dir / f"{args.method}_{args.judge}_{args.subset}_n{n}{suffix}.json"
    out_path.write_text(
        json.dumps(
            {
                "method": args.method, "judge": args.judge, "model": judge.model_id,
                "subset": args.subset, "scope": args.scope, "windows": args.windows if args.scope else None,
                "fused": is_fused, "fuse_weight": weight if is_fused else None,
                "tuned": is_fused and args.tune, "role_prior": args.role_prior,
                "n_reported": n, "n_total": len(traces),
                "agent_accuracy": scores_out.agent_accuracy, "step_accuracy": scores_out.step_accuracy,
                "avg_calls_per_log": avg_calls, "avg_est_tokens_per_log": avg_tokens,
                "predictions": {k: {"agent": v.predicted_agent, "step": v.predicted_step} for k, v in predictions.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Results ===")
    tag = " (test split)" if (is_fused and args.tune) else ""
    print(f"method={args.method} judge={args.judge} model={judge.model_id} subset={args.subset} "
          f"n={n}{tag} scope={args.scope}" + (f" weight={weight}" if is_fused else ""))
    print(f"Agent accuracy: {scores_out.agent_accuracy:5.1f}%  ({scores_out.correct_agent}/{scores_out.total})   "
          f"[paper GPT-4o baseline: {PAPER_BASELINE['agent']}%]")
    print(f"Step  accuracy: {scores_out.step_accuracy:5.1f}%  ({scores_out.correct_step}/{scores_out.total})   "
          f"[paper GPT-4o baseline: {PAPER_BASELINE['step']}%]")
    print(f"Cost: {avg_calls:.1f} judge calls/log, ~{avg_tokens:.0f} est tokens/log")
    if cache is not None:
        print(f"(cache hits={cache.hits} misses={cache.misses})")
    print(f"predictions written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
