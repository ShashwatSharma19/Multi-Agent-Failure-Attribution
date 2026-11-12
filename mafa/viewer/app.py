"""Read-only web viewer for attribution runs.

Routes:
  /                     list runs found in outputs/*.json
  /run?f=<file>         one run: metric cards + per-log predicted-vs-truth table
  /log?f=<file>&id=<x>  one log: transcript timeline with predicted/true step marked,
                        plus per-agent suspiciousness (spectrum/fused runs)

No judge calls, no writes — it only reads outputs/ and the dataset, so it is safe to
run live. Start with: uvicorn mafa.viewer.app:app --port 8000
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from mafa.cli import PAPER_BASELINE, _SUBSET_DIR
from mafa.io import load_trace

app = FastAPI(title="MAFA viewer")
OUTPUTS = Path("outputs")

_CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--line:#2a2f3a;--fg:#e6e8ec;--mut:#9aa3b2;
--ok:#3fb950;--bad:#f85149;--pred:#d29922;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:960px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.mut{color:var(--mut)}.crumb{margin-bottom:16px;font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;min-width:150px}.card .big{font-size:26px;font-weight:700}
.card .lbl{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;margin-top:8px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);font-size:14px}
th{color:var(--mut);font-weight:600}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.step{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);
border-radius:8px;padding:10px 14px;margin:8px 0}
.step.pred{border-left-color:var(--pred)}.step.truth{border-left-color:var(--ok)}
.step .hd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.agent{font-weight:700}.badge{font-size:11px;padding:2px 7px;border-radius:999px;font-weight:700}
.b-pred{background:rgba(210,153,34,.18);color:var(--pred)}
.b-truth{background:rgba(63,185,80,.18);color:var(--ok)}
.b-err{background:rgba(248,81,73,.15);color:var(--bad)}
pre{white-space:pre-wrap;word-break:break-word;margin:0;color:#c9d1d9;font-size:13px}
.bar{height:8px;background:var(--line);border-radius:999px;overflow:hidden;min-width:120px}
.bar>i{display:block;height:100%;background:var(--accent)}
.row{display:flex;align-items:center;gap:10px;margin:4px 0}
.result{background:rgba(88,166,255,.06);border:1px solid var(--accent);border-left:4px solid var(--accent);
border-radius:12px;padding:16px 18px;margin:14px 0}
.result .rlabel{color:var(--mut);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.result .rbig{font-size:20px;margin:6px 0}
.reason{font-style:italic;color:#c9d1d9;border-left:3px solid var(--line);padding-left:10px;margin-top:6px}
.callout{font-size:12px;font-weight:700;margin-bottom:4px}
.callout.pred{color:var(--pred)}.callout.truth{color:var(--ok)}
.why{background:rgba(210,153,34,.12);border-left:3px solid var(--pred);border-radius:6px;
padding:8px 10px;margin:2px 0;font-size:13px;font-style:italic;color:#e8d9b0}
.why-ans{font-size:13px;color:var(--ok);margin:4px 0 6px}
"""


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )


def _runs() -> list[Path]:
    if not OUTPUTS.is_dir():
        return []
    return sorted(OUTPUTS.glob("*.json"))


def _hit(actual: str | None, predicted: str) -> bool:
    return bool(actual) and bool(predicted) and actual in predicted  # reference substring rule


def _pct(x) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) else "—"


@app.get("/", response_class=HTMLResponse)
def home():
    runs = _runs()
    if not runs:
        return _page("MAFA viewer", "<h1>MAFA viewer</h1><p class='mut'>No runs in <code>outputs/</code> yet. "
                     "Run e.g. <code>python -m mafa.cli --method all_at_once --judge groq --subset algo --limit 15</code>.</p>")
    rows = []
    for p in runs:
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            f"<tr><td><a href='/run?f={html.escape(p.name)}'>{html.escape(p.name)}</a></td>"
            f"<td>{html.escape(d.get('method',''))}</td><td>{html.escape(str(d.get('judge','')))}</td>"
            f"<td>{html.escape(str(d.get('subset','')))}</td><td>{d.get('n_reported','?')}</td>"
            f"<td><b>{_pct(d.get('agent_accuracy'))}</b></td><td><b>{_pct(d.get('step_accuracy'))}</b></td></tr>"
        )
    body = (
        "<h1>Multi-Agent Failure Attribution</h1>"
        "<p class='mut'>Runs recorded in <code>outputs/</code>. Click one to inspect per-log predictions.</p>"
        "<table><tr><th>run</th><th>method</th><th>judge</th><th>subset</th><th>n</th>"
        "<th>agent acc</th><th>step acc</th></tr>" + "".join(rows) + "</table>"
    )
    return _page("MAFA viewer", body)


@app.get("/run", response_class=HTMLResponse)
def run_view(f: str):
    p = OUTPUTS / Path(f).name  # defend against path traversal
    if not p.is_file():
        return _page("not found", "<p>Run not found. <a href='/'>back</a></p>")
    d = json.loads(p.read_text(encoding="utf-8"))
    subset = d.get("subset", "algo")
    rel_dir, is_hand = _SUBSET_DIR.get(subset, _SUBSET_DIR["algo"])

    cards = [
        ("agent accuracy", _pct(d.get("agent_accuracy")), f"paper {PAPER_BASELINE['agent']}%"),
        ("step accuracy", _pct(d.get("step_accuracy")), f"paper {PAPER_BASELINE['step']}%"),
        ("logs", str(d.get("n_reported", "?")), subset),
        ("cost / log", f"{d.get('avg_calls_per_log',0):.1f} calls", f"~{d.get('avg_est_tokens_per_log',0):.0f} tok"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='lbl'>{html.escape(l)}</div><div class='big'>{html.escape(v)}</div>"
        f"<div class='mut'>{html.escape(s)}</div></div>" for l, v, s in cards
    )

    rows = []
    for logid, pred in d.get("predictions", {}).items():
        try:
            tr = load_trace(Path(rel_dir) / logid, is_hand)
            ta, ts = tr.mistake_agent, tr.mistake_step
        except Exception:
            ta = ts = None
        a_ok = _hit(ta, pred.get("agent", ""))
        s_ok = _hit(ts, pred.get("step", ""))
        mark = lambda ok: ("<span class='badge b-truth'>&#10003; correct</span>" if ok
                           else "<span class='badge b-err'>&#10007; wrong</span>")
        rows.append(
            f"<tr><td><a href='/log?f={html.escape(p.name)}&id={html.escape(logid)}'>{html.escape(logid)}</a></td>"
            f"<td>{html.escape(pred.get('agent','') or '—')}</td><td>{html.escape(str(ta or '—'))}</td><td>{mark(a_ok)}</td>"
            f"<td>{html.escape(str(pred.get('step','') or '—'))}</td><td>{html.escape(str(ts or '—'))}</td><td>{mark(s_ok)}</td></tr>"
        )

    meta = (f"method <b>{html.escape(d.get('method',''))}</b> · judge {html.escape(str(d.get('judge','')))} "
            f"({html.escape(str(d.get('model','')))}) · scope {d.get('scope')}"
            + (f" · fused weight {d.get('fuse_weight')}" if d.get("fused") else ""))
    body = (
        f"<div class='crumb'><a href='/'>&larr; runs</a></div>"
        f"<h1>{html.escape(p.name)}</h1><p class='mut'>{meta}</p>"
        f"<div class='cards'>{card_html}</div>"
        "<h2>Per-log predictions</h2>"
        "<p class='mut' style='font-size:13px'>Each row is one failed run. <b>Agent</b> = which "
        "agent is blamed; <b>step</b> = which 0-indexed turn the first mistake happened. "
        "<b>pred</b> = the engine's guess, <b>true</b> = the dataset's known answer. "
        "<span class='ok'>&#10003; correct</span> = they match; <span class='bad'>&#10007; wrong</span> = they don't.</p>"
        "<table><tr><th>log</th>"
        "<th>pred agent</th><th>true agent</th><th>agent?</th>"
        "<th>pred step</th><th>true step</th><th>step?</th></tr>" + "".join(rows) + "</table>"
    )
    return _page(p.name, body)


@app.get("/log", response_class=HTMLResponse)
def log_view(f: str, id: str):
    p = OUTPUTS / Path(f).name
    if not p.is_file():
        return _page("not found", "<p>Run not found. <a href='/'>back</a></p>")
    d = json.loads(p.read_text(encoding="utf-8"))
    subset = d.get("subset", "algo")
    rel_dir, is_hand = _SUBSET_DIR.get(subset, _SUBSET_DIR["algo"])
    pred = d.get("predictions", {}).get(id, {})
    logid = Path(id).name
    try:
        tr = load_trace(Path(rel_dir) / logid, is_hand)
    except Exception as exc:
        return _page("error", f"<p>Could not load {html.escape(logid)}: {html.escape(str(exc))}</p>")

    pred_step = str(pred.get("step", ""))
    true_step = str(tr.mistake_step)
    err_steps = {lb["index"] for lb in pred.get("labels", []) if lb.get("is_error")}

    reason = pred.get("reason", "")
    gt = html.escape(tr.ground_truth[:300]) or "—"
    steps_html = []
    for s in tr.steps:
        cls = "step"; callout = ""
        if str(s.index) == pred_step:
            cls += " pred"
            callout += "<div class='callout pred'>&#9888; engine flagged the failure here &mdash; why:</div>"
            if reason:
                callout += f"<div class='why'>{html.escape(reason)}</div>"
            callout += f"<div class='why-ans'>Expected (correct) answer: <b>{gt}</b> &nbsp;&middot;&nbsp; the agent's actual output is shown below.</div>"
        if str(s.index) == true_step:
            cls += " truth"
            callout += "<div class='callout truth'>&#10003; actual first mistake (dataset label)</div>"
        err = " <span class='badge b-err'>flagged error</span>" if s.index in err_steps else ""
        content = s.content if len(s.content) <= 800 else s.content[:800] + " …"
        steps_html.append(
            f"<div id='step{s.index}' class='{cls}'>{callout}"
            f"<div class='hd'><span class='mut'>step {s.index}</span>"
            f"<span class='agent'>{html.escape(s.agent)}</span>{err}</div>"
            f"<pre>{html.escape(content)}</pre></div>"
        )

    # suspiciousness bars (spectrum/fused runs only)
    susp = ""
    scores = pred.get("scores") or {}
    if scores:
        top = max(scores.values()) or 1.0
        bars = "".join(
            f"<div class='row'><div style='min-width:160px'>{html.escape(a)}</div>"
            f"<div class='bar'><i style='width:{(v/top*100):.0f}%'></i></div>"
            f"<div class='mut'>{v:.3f}</div></div>"
            for a, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        )
        susp = f"<h2>Agent suspiciousness (spectrum)</h2>{bars}"

    a_ok = _hit(tr.mistake_agent, pred.get("agent", ""))
    s_ok = _hit(tr.mistake_step, pred_step)
    pa, psv = html.escape(pred.get("agent", "") or "—"), html.escape(pred_step or "—")

    def chip(ok, label):
        return (f"<span class='badge b-truth'>&#10003; {label} correct</span>" if ok
                else f"<span class='badge b-err'>&#10007; {label} wrong</span>")

    result = (
        "<div class='result'><div class='rlabel'>Attribution result — what the engine concluded</div>"
        f"<div class='rbig'>&#9888; Blames <b>{pa}</b> at <b>step {psv}</b></div>"
        + (f"<div class='reason'>&ldquo;{html.escape(reason)}&rdquo;</div>" if reason else "")
        + f"<div class='mut' style='margin-top:10px'>Expected (correct) answer: <b>{gt}</b></div>"
        + f"<div class='mut' style='margin-top:6px'>Dataset's labelled culprit: <b>{html.escape(str(tr.mistake_agent))}</b> "
          f"@ step <b>{html.escape(true_step)}</b> &nbsp; {chip(a_ok,'agent')} {chip(s_ok,'step')}</div>"
        + (f"<div style='margin-top:8px'><a href='#step{html.escape(pred_step)}'>jump to the flagged step &darr;</a></div>"
           if pred_step else "")
        + "</div>"
    )
    body = (
        f"<div class='crumb'><a href='/'>runs</a> / <a href='/run?f={html.escape(p.name)}'>{html.escape(p.name)}</a> / {html.escape(logid)}</div>"
        f"<h1>{html.escape(logid)}</h1>"
        f"<p class='mut'>Task the agents tried to solve: {html.escape(tr.question[:300])}</p>"
        f"{result}{susp}"
        f"<h2>Transcript ({len(tr.steps)} steps) — the failed run the engine analyzed</h2>"
        "<p class='mut' style='font-size:13px'>This is the <b>input</b>: agents doing their work. The mistake is "
        "<b>not labeled</b> in the log — the engine infers it. "
        "<span style='color:var(--pred)'>Amber = engine's pick</span>, "
        "<span style='color:var(--ok)'>green = dataset's answer</span>.</p>"
        + "".join(steps_html)
    )
    return _page(logid, body)
