# Multi-Agent Failure Attribution Engine — Design Spec

**Date:** 2026-10-03
**Status:** Approved design, pre-implementation

## 1. Goal

Build a tool that, given a log of a failed multi-agent run, identifies **which agent**
caused the failure and **which step** was decisive — and measurably beats the published
baseline on the **Who&When** benchmark.

**Headline number (the whole pitch):** report *agent-accuracy* and *step-accuracy* vs. the
paper's baseline of **53.5% / 14.2%** (Zhang et al., ICML 2025), on the same benchmark.
The README leads with that one comparison line.

Non-goals: not a general observability platform; not a new benchmark; no training of
custom models.

## 2. Constraints

- **Hardware:** laptop — i7-12650H, RTX 3050 **4 GB VRAM** (~3.8 GB free), 16 GB RAM.
- **Cost:** zero paid subscriptions.
- **Judge inference** is the only model-heavy component; it must work within free-tier
  limits and/or on the local 4 GB GPU.

## 3. Judge layer (keystone decision)

A single interface with swappable backends, wrapped for caching and rate limiting.

```
Judge.ask(prompt: str) -> str          # the interface

Backends:
  GroqJudge      # free tier, Llama-3.3-70B — strong, fast; DEFAULT for headline numbers
  GeminiJudge    # free tier, Gemini Flash — alternate cloud, high daily request cap
  OllamaJudge    # local llama3.2:3B (2 GB, already installed) — offline/reproducible fallback + ablation row

Wrappers (decorator pattern, compose over any backend):
  CachingJudge      # disk cache keyed by hash(model + prompt) -> response
  RateLimitedJudge  # token/request budget + exponential backoff on 429
```

Rationale and tradeoffs (cloud vs local) are settled:
- **Cloud default (Groq):** strong judge comparable to the paper's model class; produces the
  headline number. Exposed to free-tier rate limits — mitigated below.
- **Local `llama3.2:3B`:** fits in 4 GB VRAM, runs today, fully offline. A *weak* judge
  (likely below baseline alone) — its role is (a) reproducible/offline fallback when cloud
  limits lock us out, and (b) an ablation row: "same pipeline, local 3B, no internet: X/Y".
- Optional stronger local judge (`qwen2.5:7b` / `llama3.1:8b`, Q4 ≈4.7 GB, partial CPU
  offload, ~minutes/log) is a one-command upgrade — noted, not required.

### Rate-limit edge cases (designed-in mitigations)

- **Tokens-per-minute is the binding limit**, not request count: one long "all-at-once"
  prompt can exceed TPM alone → the Phase-2 scope pass trims tokens per request.
- **Step-by-step multiplies requests** into daily caps → cache + checkpoint per log so a
  lockout resumes instead of restarting; runs can span sessions without recompute.
- **Splitting a run across days risks two different judge versions** → pin model id in run
  metadata; cache makes a run deterministic once computed.
- **Context-window overflow is a hard failure, not a 429** → detect and route via scope pass.

## 4. Architecture

```
mafa/
  io/        Who&When log -> normalized Trace (agents, ordered steps, final pass/fail, ground truth)
  judge/     Judge interface + Groq/Gemini/Ollama backends + Caching/RateLimited wrappers
  baselines/ paper's 3 strategies: all_at_once, step_by_step, binary_search
  scope/     Phase-2 scope-delineation: narrow a long log to the span likely containing the failure
  spectrum/  Phase-3 SBFL scorer (Ochiai) -> suspiciousness per (agent|step) across many runs
  fuse/      combine judge score + spectrum score (weighted, or simple learned re-ranker)
  eval/      predictions vs ground truth -> agent-acc, step-acc, tokens/cost per log
  store/     SQLite: traces, predictions, cached judge calls, run metadata
  harness/   Phase-4: emit Who&When-style logs from a LangGraph/Verity-OS pipeline; induce failures
  viewer/    Phase-5: Streamlit trace explorer (timeline + suspiciousness + NL explanation)
```

Each module has one purpose, a defined interface, and is testable in isolation.

## 5. Data flow

```
Who&When log
  -> Trace (io)
  -> [scope narrows span]                     (Phase 2+)
  -> baseline/judge predicts (agent, step)
  -> spectrum suspiciousness across runs       (Phase 3+)
  -> fuse (judge + spectrum)                   (Phase 3+)
  -> eval vs ground truth
  -> store (SQLite) + viewer (Streamlit)
```

## 6. Phase → commit mapping

One commit per phase, pushed to the user's GitHub repo. The user runs all git commands;
commit messages carry **no Claude/Anthropic attribution trailers**. Repo URL supplied by
the user after Phase 1.

| Phase | Deliverable | Commit contains |
|---|---|---|
| 0 setup | repo skeleton, `io`, `store`, judge interface + cache/rate-limit, tests, dataset download check | scaffolding + loader |
| 1 | reproduce 3 baselines; eval harness reproduces paper numbers on a subset | `baselines/`, `eval/` |
| 2 | scope-delineation pass + token/cost tracking | `scope/` |
| 3 | SBFL scorer + fusion; **full benchmark vs 53.5%/14.2%** | `spectrum/`, `fuse/` |
| 4 | own LangGraph/Verity-OS logs, 15–20 induced labeled failures, validation | `harness/` + labeled logs |
| 5 | Streamlit viewer + Docker-compose + README tradeoff table | `viewer/`, `docker/`, `README` |

## 7. Testing

- TDD where it pays: `io` (parse sample log), `eval` (scoring math), `spectrum` (Ochiai on a
  toy matrix), `fuse` (combination logic).
- Judge backends **mocked** in tests — no live API calls in the suite.
- Harness validity gate: reproduce the paper's baseline on a subset *before* building the
  novel parts (Phase 1 acceptance criterion).

## 8. Tech stack

Python 3.13 · requests/httpx (cloud judge) · ollama (local) · numpy/pandas (spectrum, eval) ·
sqlite3 (stdlib) · pytest · Streamlit (viewer) · Docker Compose (Phase 5).

## 9. Key risks

- **Dataset access** is the Phase-1 critical path → Phase 0 verifies the
  `ag2ai/Agents_Failure_Attribution` repo + Who&When data download before anything else.
- **Free-tier limits** → cache + rate limiter + per-log checkpoint + scope pass.
- **Reproducibility** → deterministic cache; pinned model ids in run metadata.
- **Weak local judge** → positioned as fallback/ablation, not headline.

## 10. References

- Zhang et al., *Which Agent Causes Task Failures and When?* (ICML 2025) — Who&When benchmark, 53.5%/14.2% baseline.
- *Who is Introducing the Failure?* — FAMAS (2026) — SBFL basis for Phase 3.
- *Where Did It All Go Wrong?* — Hierarchical Error Attribution (2026) — ablation comparison.
- Who&When Pro (ICML 2026) — larger successor benchmark (stretch validation).
- `ag2ai/Agents_Failure_Attribution` — official repo/leaderboard, dataset source.
