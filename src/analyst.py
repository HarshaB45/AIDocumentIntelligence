"""
Analyst Agent (Risk Scoring, Trends, Pure NLP Summarization)
-----------------------------------------------------------
- Input : outputs/validate/*.json   (from Validator)
- Output:
  * outputs/analyze/per_doc.csv     (document-level metrics)
  * outputs/analyze/per_doc.json
  * outputs/analyze/corpus_kpis.json
  * outputs/analyze/corpus_summary.md  (human-readable, extractive NLP; no LLM)

What it does:
1) Compute risk score per document (rule-based weights on _issues + numeric drift).
2) KPIs & trends:
   - Net days stats & distribution
   - Governing law frequency, top 5
   - Party-level average risk
   - Monthly volume & median net days (by effective_date_normalized)
3) Summarization (pure NLP):
   - Build a corpus from risky clause snippets (payment_terms, termination_clause, issue snippets)
   - Sentence split, TF-IDF over sentences, re-rank with risk-keyword boost
   - Select top-K sentences as concise corpus summary (no LLM)

Deps:
  pip install pandas scikit-learn numpy python-dotenv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from statistics import median
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Paths & Config ----------------

ROOT = Path(__file__).resolve().parents[1]
VALIDATE_DIR = ROOT / "outputs" / "validate"
AN_CFG_FILE  = ROOT / "config" / "analyst.config.json"
VAL_RULES    = ROOT / "config" / "validator.rules.json"   # to reuse standards if present

def load_cfg() -> Dict[str, Any]:
    if AN_CFG_FILE.exists():
        cfg = json.loads(AN_CFG_FILE.read_text(encoding="utf-8"))
    else:
        cfg = {}
    # sane defaults
    cfg.setdefault("weights", {
        "MISSING": 3.0, "AMBIGUOUS": 2.0, "BAD_DATE": 1.0,
        "NET_DAYS_OVER_MAX": 3.5, "NET_DAYS_OUTLIER": 2.5,
        "NET_DAYS_DRIFT_VS_PARTY": 2.0, "GOVERNING_LAW_INCONSISTENT": 2.0,
        "PAYMENT_AMOUNT_HIGH": 2.5
    })
    cfg.setdefault("risk_buckets", {"low": 0, "medium": 5, "high": 10})
    cfg.setdefault("top_k_summary_sentences", 10)
    cfg.setdefault("risk_keywords", [
        "termination","indemnify","liability","jurisdiction","payment",
        "net","late fee","penalty","warranty","force majeure",
        "assignment","notice","confidential","arbitration","governing law"
    ])
    cfg.setdefault("outputs", {
        "analyze_dir": "outputs/analyze",
        "per_doc_csv": "outputs/analyze/per_doc.csv",
        "per_doc_json": "outputs/analyze/per_doc.json",
        "kpis_json": "outputs/analyze/corpus_kpis.json",
        "summary_md": "outputs/analyze/corpus_summary.md"
    })
    return cfg

def load_validator_rules() -> Dict[str, Any]:
    if VAL_RULES.exists():
        return json.loads(VAL_RULES.read_text(encoding="utf-8"))
    return {
        "standards": {"net_days_standard": 30, "max_net_days": 120, "amount_upper_warn": 1_000_000.0}
    }

# ---------------- Utilities ----------------

SENT_SPLIT_RX = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9(])')

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', (text or '').strip())
    if not text:
        return []
    sents = SENT_SPLIT_RX.split(text)
    # clean & dedupe short noise
    out = []
    seen = set()
    for s in sents:
        s = s.strip()
        if 30 <= len(s) <= 400:  # keep meaningful length
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out

def norm_party_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r,.;:-()[]{}\"'")
    return s.upper()

def party_key(parties: Any) -> Optional[str]:
    if not isinstance(parties, list) or len(parties) == 0:
        return None
    names = [norm_party_name(x) for x in parties if isinstance(x, str) and x.strip()]
    if not names:
        return None
    key_names = sorted(names[:2])
    return " | ".join(key_names)

# ---------------- Risk Scoring ----------------

def score_document(doc: Dict[str, Any], weights: Dict[str, float], rules: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_score, component_breakdown)
    """
    breakdown: Dict[str, float] = Counter()
    issues = doc.get("_issues", []) or []

    # issue weights
    for iss in issues:
        t = iss.get("type")
        if not t:
            continue
        w = float(weights.get(t, 0.0))
        if w:
            breakdown[t] += w

    # numeric drifts (optional)
    nd = doc.get("payment_net_days")
    std = rules.get("standards", {}).get("net_days_standard", 30)
    if isinstance(nd, int) and isinstance(std, (int, float)):
        if nd > std:
            # scale: + (nd - std) / 30, capped
            breakdown["NET_DAYS_DRIFT_NUMERIC"] += min(3.0, max(0.0, (nd - std) / 30.0))

    amt = doc.get("payment_amount_normalized")
    warn_amt = rules.get("standards", {}).get("amount_upper_warn", 1_000_000.0)
    if isinstance(amt, (int, float)) and amt > warn_amt:
        breakdown["AMOUNT_OVER_WARN_NUMERIC"] += min(2.0, (amt - warn_amt) / (warn_amt or 1.0))

    total = float(sum(breakdown.values()))
    return total, dict(breakdown)

def bucketize(score: float, buckets: Dict[str, float]) -> str:
    # buckets: low<=medium<high thresholds
    medium = float(buckets.get("medium", 5))
    high = float(buckets.get("high", 10))
    if score >= high: return "high"
    if score >= medium: return "medium"
    return "low"

# ---------------- Data Assembly ----------------

def load_validated_docs() -> List[Dict[str, Any]]:
    files = sorted(VALIDATE_DIR.glob("*.json"))
    docs = []
    for p in files:
        if p.name.startswith("_"):
            continue
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    return docs

def build_per_doc_table(docs: List[Dict[str, Any]], cfg: Dict[str, Any], val_rules: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for d in docs:
        score, parts = score_document(d, cfg["weights"], val_rules)
        rows.append({
            "doc_id": d.get("_doc_id"),
            "party_key": party_key(d.get("parties")),
            "governing_law": d.get("governing_law"),
            "effective_date": d.get("effective_date_normalized"),
            "payment_net_days": d.get("payment_net_days"),
            "payment_amount": d.get("payment_amount_normalized"),
            "risk_score": round(score, 3),
            "risk_bucket": bucketize(score, cfg["risk_buckets"]),
            "issues_count": len(d.get("_issues", []) or []),
            "issue_types": ",".join(sorted(set([i.get("type","") for i in (d.get("_issues", []) or []) if i.get("type")]))),
        })
    df = pd.DataFrame(rows).fillna("")
    return df

# ---------------- Trends / KPIs ----------------

def date_to_month(s: str) -> Optional[str]:
    if not s: return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m")
    except Exception:
        return None

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    if df.empty:
        return kpis

    # Net days stats
    nd = pd.to_numeric(df["payment_net_days"], errors="coerce").dropna()
    if not nd.empty:
        kpis["net_days_median"] = float(nd.median())
        kpis["net_days_p90"] = float(nd.quantile(0.90))
        kpis["net_days_min"] = int(nd.min())
        kpis["net_days_max"] = int(nd.max())

    # Risk distribution
    rb = df["risk_bucket"].value_counts().to_dict()
    kpis["risk_bucket_counts"] = rb
    kpis["avg_risk_score"] = float(pd.to_numeric(df["risk_score"], errors="coerce").mean())

    # Governing law top 5
    gl_counts = df["governing_law"].replace("", np.nan).dropna().value_counts().head(5).to_dict()
    kpis["top_governing_law"] = gl_counts

    # Party average risk (top 10 by volume)
    party_group = df[df["party_key"] != ""].groupby("party_key")["risk_score"].mean().sort_values(ascending=False)
    kpis["party_avg_risk_top10"] = {k: float(v) for k,v in party_group.head(10).items()}

    # Monthly trends
    months = df["effective_date"].apply(date_to_month)
    dfm = df.copy()
    dfm["month"] = months
    monthly = dfm[dfm["month"].notna()].groupby("month").agg(
        contracts=("doc_id", "count"),
        median_net_days=("payment_net_days", lambda x: float(pd.to_numeric(x, errors="coerce").median()))
    ).reset_index().to_dict(orient="records")
    kpis["monthly"] = monthly

    return kpis

# ---------------- Pure NLP Summarization ----------------

def collect_risky_snippets(docs: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for d in docs:
        for fld in ("payment_terms","termination_clause"):
            v = d.get(fld)
            if isinstance(v, dict) and v.get("text"):
                out.append(v["text"])
        # append any spans implicated by issues (if present)
        for iss in d.get("_issues", []) or []:
            fld = iss.get("field")
            v = d.get(fld) if fld else None
            if isinstance(v, dict) and v.get("text"):
                out.append(v["text"])
    return out

def summarize_corpus(sentences: List[str], risk_keywords: List[str], top_k: int = 10) -> List[str]:
    if not sentences:
        return []

    # TF-IDF over sentences; score sentences by sum of tf-idf weights of their terms
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    # keyword boost
    if risk_keywords:
        kw = [k.lower() for k in risk_keywords]
        bonus = np.zeros_like(scores, dtype=float)
        for i, s in enumerate(sentences):
            ls = s.lower()
            bonus[i] = sum(ls.count(k) for k in kw) * 0.25  # small boost per hit
        scores = scores + bonus

    # pick top K distinct sentences
    order = np.argsort(scores)[::-1]
    picked = []
    seen = set()
    for idx in order:
        s = sentences[idx].strip()
        # avoid near-duplicates by simple hash of first 80 chars
        key = s[:80].lower()
        if key in seen:
            continue
        picked.append(s)
        seen.add(key)
        if len(picked) >= top_k:
            break
    return picked

# ---------------- Writer helpers ----------------

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_md_summary(path: Path, kpis: Dict[str, Any], top_sents: List[str]):
    lines = ["# Portfolio Summary (Analyst — Pure NLP)\n"]
    lines.append("## Key Metrics\n")
    if kpis:
        lines.append(f"- **Average risk score:** {kpis.get('avg_risk_score','n/a')}")
        rb = kpis.get("risk_bucket_counts", {})
        if rb: lines.append(f"- **Risk buckets:** {rb}")
        if "net_days_median" in kpis:
            lines.append(f"- **Net days (median / p90):** {kpis['net_days_median']} / {kpis.get('net_days_p90','n/a')}")
        gl = kpis.get("top_governing_law", {})
        if gl: lines.append(f"- **Top governing law:** {gl}")
        pr = kpis.get("party_avg_risk_top10", {})
        if pr: lines.append(f"- **Highest avg risk (top 5 parties):** {dict(list(pr.items())[:5])}")
        lines.append("")
    else:
        lines.append("- No KPIs available.\n")

    lines.append("## Concise Extractive Summary (no LLM)\n")
    if top_sents:
        for s in top_sents:
            one = s.replace("\n", " ").strip()
            lines.append(f"- {one}")
    else:
        lines.append("_No salient sentences found._")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------- Main ----------------

def run_analyst():
    cfg = load_cfg()
    val_rules = load_validator_rules()

    docs = load_validated_docs()
    if not docs:
        raise SystemExit(f"No validated JSONs found in {VALIDATE_DIR}")

    # Per-doc table
    df = build_per_doc_table(docs, cfg, val_rules)

    # KPIs / trends
    kpis = compute_kpis(df)

    # Pure NLP extractive summary
    snippets = collect_risky_snippets(docs)
    # sentence list from all snippets
    sentences: List[str] = []
    for snip in snippets:
        sentences.extend(split_sentences(snip))
    top_sents = summarize_corpus(
        sentences,
        risk_keywords=cfg.get("risk_keywords", []),
        top_k=int(cfg.get("top_k_summary_sentences", 10))
    )

    # Outputs
    out_dir = ROOT / cfg["outputs"]["analyze_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_doc_csv  = ROOT / cfg["outputs"]["per_doc_csv"]
    per_doc_json = ROOT / cfg["outputs"]["per_doc_json"]
    kpis_json    = ROOT / cfg["outputs"]["kpis_json"]
    summary_md   = ROOT / cfg["outputs"]["summary_md"]

    df.to_csv(per_doc_csv, index=False, encoding="utf-8")
    write_json(per_doc_json, json.loads(df.to_json(orient="records")))
    write_json(kpis_json, kpis)
    write_md_summary(summary_md, kpis, top_sents)

    print(f"✓ Analyzed {len(docs)} docs")
    print(f"✓ Wrote per-doc CSV : {per_doc_csv.relative_to(ROOT)}")
    print(f"✓ Wrote per-doc JSON: {per_doc_json.relative_to(ROOT)}")
    print(f"✓ Wrote KPIs JSON   : {kpis_json.relative_to(ROOT)}")
    print(f"✓ Wrote summary MD  : {summary_md.relative_to(ROOT)}")

if __name__ == "__main__":
    run_analyst()
