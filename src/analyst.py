"""
Analyst Agent (Strategist)
--------------------------
Reads validated JSONs from outputs/validate/, computes:
- Per-document risk score (weighted by issue types)
- Portfolio insights (counts, distributions, medians, outliers)
Writes:
- outputs/analyze/per_doc/*.json (risk + tags)
- outputs/analyze/portfolio.json (global KPIs)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from statistics import median, mean
from collections import Counter, defaultdict

# ---- Paths ----
ROOT_DIR      = Path(__file__).resolve().parents[1]
VALIDATE_DIR  = ROOT_DIR / "outputs" / "validate"
ANALYZE_DIR   = ROOT_DIR / "outputs" / "analyze"
PER_DOC_DIR   = ANALYZE_DIR / "per_doc"
ANALYSIS_RULES_FILE = ROOT_DIR / "config" / "analystrules.json"

ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
PER_DOC_DIR.mkdir(parents=True, exist_ok=True)

# ---- Loaders ----
def load_rules() -> Dict[str, Any]:
    with open(ANALYSIS_RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_validated() -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for p in sorted(VALIDATE_DIR.glob("*.json")):
        if p.name.startswith("_"):  # skip corpus summary
            continue
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    if not recs:
        raise SystemExit(f"No validated records found in {VALIDATE_DIR}")
    return recs

# ---- Scoring ----
def score_issues(issues: List[Dict[str,str]], weights: Dict[str,int], cap: int) -> Tuple[int, Dict[str,int]]:
    by_type = Counter([i.get("type","UNKNOWN") for i in issues])
    raw = sum(weights.get(t, 0) * c for t, c in by_type.items())
    return min(raw, cap), dict(by_type)

def classify_bucket(score: int, buckets: Dict[str,int]) -> str:
    # buckets: {low:0, medium:4, high:8}
    if score >= buckets.get("high", 8):
        return "high"
    if score >= buckets.get("medium", 4):
        return "medium"
    return "low"

def collect_tags(rec: Dict[str,Any]) -> List[str]:
    tags: List[str] = []
    # governance tag
    law = (rec.get("governing_law") or "").strip()
    if law:
        tags.append(f"law:{law}")
    # payment speed tag
    net = rec.get("payment_net_days")
    if isinstance(net, int):
        if net <= 30: tags.append("net:fast")
        elif net <= 60: tags.append("net:moderate")
        else: tags.append("net:slow")
    # completeness tag
    if any(i.get("type") == "MISSING" for i in rec.get("_issues", [])):
        tags.append("incomplete")
    return tags

# ---- Portfolio KPIs ----
def compute_portfolio_kpis(records: List[Dict[str,Any]], weights: Dict[str,int], caps: Dict[str,int], bucket_cfg: Dict[str,int], kpi_cfg: Dict[str,Any]) -> Dict[str,Any]:
    results: List[Dict[str,Any]] = []
    ambiguous_terms = Counter()
    net_values = []
    laws = Counter()
    bucket_counts = Counter()
    issue_type_counts = Counter()
    doc_scores = []

    for r in records:
        score, by_type = score_issues(r.get("_issues", []), weights, caps.get("max_risk_score_per_doc", 20))
        bucket = classify_bucket(score, bucket_cfg)
        tags = collect_tags(r)

        # aggregate per-portfolio
        issue_type_counts.update(by_type)
        if isinstance(r.get("payment_net_days"), int):
            net_values.append(r["payment_net_days"])
        if r.get("governing_law"):
            laws.update([r["governing_law"]])
        bucket_counts.update([bucket])
        doc_scores.append(score)

        # collect ambiguous terms mentioned in details
        for i in r.get("_issues", []):
            if i.get("type") == "AMBIGUOUS":
                # attempt to parse the terms from detail if present like "Ambiguous language found: term1, term2"
                detail = (i.get("detail") or "").lower()
                if "found:" in detail:
                    part = detail.split("found:", 1)[1]
                    for t in [x.strip() for x in part.split(",") if x.strip()]:
                        ambiguous_terms.update([t])

        # write per-doc file enriched
        per_doc = {
            "_doc_id": r.get("_doc_id"),
            "_char_count": r.get("_char_count"),
            "_issues": r.get("_issues", []),
            "risk_score": score,
            "risk_bucket": bucket,
            "tags": tags,
            "governing_law": r.get("governing_law"),
            "payment_net_days": r.get("payment_net_days"),
            "payment_amounts": r.get("payment_amounts", []),
            "effective_date": r.get("effective_date")
        }
        (PER_DOC_DIR / f"{r['_doc_id']}.json").write_text(json.dumps(per_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    portfolio: Dict[str,Any] = {}
    portfolio["count_documents"] = len(records)
    portfolio["avg_risk_score"] = round(mean(doc_scores), 2) if doc_scores else 0
    portfolio["median_risk_score"] = int(median(doc_scores)) if doc_scores else 0
    portfolio["risk_distribution"] = dict(bucket_counts)
    portfolio["top_issue_types"] = dict(issue_type_counts.most_common())
    portfolio["governing_law_distribution"] = dict(laws.most_common())
    portfolio["median_net_days"] = int(median(net_values)) if net_values else None
    portfolio["top_ambiguous_terms"] = [t for t, _ in ambiguous_terms.most_common(kpi_cfg.get("top_n_ambiguous_terms", 10))]

    return portfolio

# ---- Run ----
def run_analyst():
    rules = load_rules()
    weights = rules.get("weights", {})
    caps = rules.get("caps", {"max_risk_score_per_doc": 20})
    buckets = rules.get("risk_buckets", {"low":0, "medium":4, "high":8})
    kpi_cfg = rules.get("kpis", {"top_n_ambiguous_terms": 10})

    validated = load_validated()
    portfolio = compute_portfolio_kpis(validated, weights, caps, buckets, kpi_cfg)

    # write portfolio summary
    (ANALYZE_DIR / "portfolio.json").write_text(json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✓ Analyst complete -> outputs/analyze/portfolio.json")
    print("✓ Per-doc risk written -> outputs/analyze/per_doc/*.json")

if __name__ == "__main__":
    run_analyst()
