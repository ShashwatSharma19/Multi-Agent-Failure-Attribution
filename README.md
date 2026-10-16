# Multi-Agent Failure Attribution Engine

**Given a log of a failed multi-agent run, find which agent — and which step — caused the failure.**

Measured on the [Who&When](https://github.com/ag2ai/Agents_Failure_Attribution) benchmark
(Zhang et al., ICML 2025), whose published baseline localizes the responsible **agent**
~53.5% of the time and the decisive **step** ~14.2% of the time. This project reproduces
those baselines and then adds a scope-delineation pass and a spectrum-based (SBFL) scorer
to beat them.

> **Status:** Phase 3 — baseline harness + scope pass + spectrum/fusion scorer. Awaiting a cloud-judge run for the headline number.

---

## Why this matters (real-world applications)

Teams are shipping **agentic systems** (several LLM agents + tools chained together) into
production. When one produces a wrong answer or a bad action, there is no "stack trace" —
engineers reread long transcripts and guess which agent broke it. This project is the
diagnostic that turns *"something went wrong somewhere"* into *"agent #3 failed at step 7,
here's why."*

| Setting | How failure attribution is used |
|---|---|
| **AI agent-observability platforms** (LangSmith, Arize, Braintrust, Langfuse) | A core feature they sell — this is a credible mini-version of their attribution/debugging layer. |
| **Customer-support agent swarms** | A mishandled ticket is localized to the agent/step that caused it, cutting mean-time-to-resolution. |
| **RAG / multi-step research assistants** | A wrong answer is traced to retrieval vs. fact-check vs. synthesis, so engineers fix the right stage. |
| **Coding agents / automated PR bots** | An auto-PR that breaks CI is attributed to the sub-agent/step that introduced the bug. |
| **Regulated workflows** (finance, insurance, healthcare) | Audit & accountability: *which* component failed when an automated decision goes wrong. |

**Two operating modes:**
- **Post-hoc (primary):** run after a failure — debugging, incident postmortems, and CI
  regression testing (*"which agent regressed after we changed the prompt/model?"*). This is
  where the LLM-as-judge lives.
- **Near-real-time (extension):** the cheap **spectrum/SBFL score** (pure statistics, no LLM
  call) can run *inline* during a live run to flag the likely-failing step the moment things
  go wrong — triggering **self-healing**: retry just that one agent/step, or escalate to a
  human, instead of re-running the whole expensive pipeline. A direct cost + reliability win.

---

## How it works

```
Who&When log
  -> Trace (mafa.io)                      normalize agents, steps, ground truth
  -> baseline attribution (mafa.baselines) all_at_once | step_by_step | binary_search
     via a pluggable LLM judge (mafa.judge)
  -> score vs ground truth (mafa.eval)     agent-accuracy, step-accuracy
```

### The judge is swappable (and the free tiers actually work)

`mafa.judge` exposes one interface with three backends behind caching + rate-limiting:

| Backend | Role |
|---|---|
| **Groq** (`llama-3.3-70b-versatile`, free tier) | **Default headline judge** — strong, fast, comparable to the paper's model class. |
| **Gemini** (`gemini-2.0-flash`, free tier) | Alternate cloud judge with a high daily request cap. |
| **Ollama** (`llama3.2:3B`, local) | Offline/reproducible fallback + an ablation row ("runs fully on a 4 GB laptop"). |

Two wrappers make the free tiers viable for a 184-log benchmark:
- **`CachingJudge`** — SQLite cache keyed on `hash(model + prompt)`. Re-runs cost zero quota
  and are perfectly reproducible; a rate-limit lockout mid-run resumes from the cache.
- **`RateLimitedJudge`** — spaces requests to a per-minute budget and retries HTTP 429 with
  exponential backoff instead of crashing.

---

## Quickstart

```bash
# 1. Get the benchmark (not committed to this repo)
git clone --depth 1 https://github.com/ag2ai/Agents_Failure_Attribution.git data/whoandwhen

# 2. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt

# 3. Run a baseline locally (no API key needed) on a small subset
python -m mafa.cli --method all_at_once --judge ollama --subset algo --limit 10

# 4. Headline run on the free Groq tier (needs GROQ_API_KEY in .env)
python -m mafa.cli --method all_at_once --judge groq --subset algo

# 5. Phase 2: scope-delineation (narrows long logs; fewer calls per log)
python -m mafa.cli --method step_by_step --judge groq --subset hand --scope --windows 3
```

### Scope-delineation (Phase 2)

A cheap one-call pass splits a log into `--windows` contiguous segments and asks
the judge which segment holds the decisive error; the expensive per-step
attribution then runs only inside that window. This reduces judge **calls** and
**tokens** per log (reported as `Cost: N judge calls/log, ~T est tokens/log`) and
is what keeps long logs under the free-tier limits. It applies to `step_by_step`
and `binary_search`; `all_at_once` is a single call and is left unchanged.

### Spectrum + fusion (Phase 3)

Who&When logs are **all failures** (no passing runs), so a classic whole-run
spectrum is impossible. Instead we build a **step-level** spectrum: a per-step probe
labels each step error / not-error, each agent "covers" the steps it authored, and
**Ochiai** suspiciousness ranks the agents. A cross-log `RolePrior` pools the same
signal per recurring role. The `fused` method combines the direct LLM vote with the
spectrum via a weighted score; the weight is tuned on a **held-out dev split** so the
reported test numbers are never tuned on.

```bash
# spectrum-only attribution
python -m mafa.cli --method spectrum --judge groq --subset hand --scope

# fused (LLM vote + spectrum), weight tuned on a dev split, reported on test
python -m mafa.cli --method fused --judge groq --subset hand --scope --tune

# optionally blend the cross-log role prior
python -m mafa.cli --method fused --judge groq --subset algo --role-prior
```

This yields the Phase 3 comparison: **baseline vs. spectrum-only vs. fused**.

Subsets: `algo` (126 Algorithm-Generated logs) and `hand` (58 Hand-Crafted logs).
Methods: `all_at_once`, `step_by_step`, `binary_search`.

## Tests

```bash
python -m pytest
```

Logic is tested with a scripted mock judge — the suite makes **no live API calls**.

---

## Honest-comparison note

The paper's 53.5% / 14.2% were produced with **GPT-4o**. Our reproduction uses a different
(free) judge, so our baseline will not land exactly on those numbers. The real claim this
project builds toward is **fused method vs. our own reproduced baseline on the identical
harness** — with the paper's number shown alongside for context, never as a cross-judge
apples-to-apples result.

## References

- Zhang et al., *Which Agent Causes Task Failures and When?*, ICML 2025 — Who&When benchmark.
- `ag2ai/Agents_Failure_Attribution` — official dataset, baselines, and leaderboard.
