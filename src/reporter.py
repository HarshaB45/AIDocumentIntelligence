"""
Reporter Agent
--------------
Inputs:
  - Validated docs: outputs/validate/*.json        (from Validator)
  - Per-doc metrics: outputs/analyze/per_doc.json  (from Analyst)
  - KPIs: outputs/analyze/corpus_kpis.json         (from Analyst)

Outputs:
  - findings.csv  : issue-level structured data for easy import
  - findings.json : same as JSON
  - report.md     : comprehensive, human-readable report with direct quotes

What it includes:
  - Portfolio KPIs and risk distribution
  - Top risky documents (table)
  - Flagged inconsistencies and anomalies (issue-level) with direct quotes
  - Missing fields & ambiguous clauses lists
  - Outlier net terms / amount warnings with evidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CFG_FILE = ROOT / "config" / "reporter.config.json"

def load_cfg() -> Dict[str, Any]:
    if CFG_FILE.exists():
        return json.loads(CFG_FILE.read_text(encoding="utf-8"))
    # Safe defaults if config is missing
    return {
        "inputs": {
            "validated_dir": "outputs/validate",
            "per_doc_json": "outputs/analyze/per_doc.json",
            "kpis_json": "outputs/analyze/corpus_kpis.json"
        },
        "outputs": {
            "report_dir": "outputs/report",
            "findings_csv": "outputs/report/findings.csv",
            "findings_json": "outputs/report/findings.json",
            "report_md": "outputs/report/report.md"
        },
        "report": {"top_risky_docs": 25, "max_quote_chars": 400}
    }

# ---------------- Helpers ----------------

def clamp_quote(text: str, max_chars: int) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    return t[:max_chars] + ("…" if len(t) > max_chars else "")

def get_field_text(doc: Dict[str, Any], field: Optional[str]) -> Optional[str]:
    """Return the clause text for a field if it exists as a span dict."""
    if not field:
        return None
    v = doc.get(field)
    if isinstance(v, dict):
        return v.get("text")
    if isinstance(v, str):  # in case any field is plain string
        return v
    return None

def load_validated_docs(validated_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(validated_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    return docs

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_md(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------- Findings assembly ----------------

ISSUE_COLUMNS = [
    "doc_id",
    "issue_type",
    "field",
    "governing_law",
    "payment_net_days",
    "payment_amount",
    "details",
    "quote"
]

def build_issue_rows(validated_docs: List[Dict[str, Any]], max_quote_chars: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in validated_docs:
        doc_id = d.get("_doc_id")
        glaw = d.get("governing_law")
        nd = d.get("payment_net_days")
        amt = d.get("payment_amount_normalized")

        issues = d.get("_issues", []) or []
        for iss in issues:
            itype = iss.get("type", "")
            field = iss.get("field")
            details = {k: v for k, v in iss.items() if k not in ("type", "field")}

            # direct quote: prefer the field's span text
            quote = get_field_text(d, field)
            quote = clamp_quote(quote or "", max_quote_chars)

            rows.append({
                "doc_id": doc_id,
                "issue_type": itype,
                "field": field or "",
                "governing_law": glaw or "",
                "payment_net_days": nd if isinstance(nd, (int, float)) else "",
                "payment_amount": amt if isinstance(amt, (int, float)) else "",
                "details": json.dumps(details, ensure_ascii=False),
                "quote": quote
            })
    return rows

# ---------------- Markdown report ----------------

def md_table_from_df(df: pd.DataFrame, max_rows: int = 25) -> List[str]:
    if df.empty:
        return ["_none_"]
    df2 = df.head(max_rows)
    # Build a simple Markdown table
    cols = list(df2.columns)
    lines = ["|" + "|".join(cols) + "|",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df2.iterrows():
        vals = [str(r[c]) if r[c] != "" else "-" for c in cols]
        lines.append("|" + "|".join(vals) + "|")
    return lines

def build_markdown_report(
    kpis: Dict[str, Any],
    per_doc_df: pd.DataFrame,
    findings_df: pd.DataFrame,
    top_risky_docs: int
) -> List[str]:
    lines: List[str] = []
    lines.append("# Contract Analysis Report (Reporter)\n")

    # KPIs
    lines.append("## Portfolio KPIs\n")
    if kpis:
        if "avg_risk_score" in kpis:
            lines.append(f"- **Average Risk Score:** {kpis.get('avg_risk_score')}")
        rb = kpis.get("risk_bucket_counts", {})
        if rb: lines.append(f"- **Risk Buckets:** {rb}")
        if "net_days_median" in kpis:
            lines.append(f"- **Net Days (Median / P90):** {kpis['net_days_median']} / {kpis.get('net_days_p90','n/a')}")
        gl = kpis.get("top_governing_law", {})
        if gl: lines.append(f"- **Top Governing Law:** {gl}")
        party = kpis.get("party_avg_risk_top10", {})
        if party: lines.append(f"- **Highest Avg Risk (Top Parties):** {dict(list(party.items())[:5])}")
        lines.append("")
    else:
        lines.append("_No KPIs available._\n")

    # Top risky docs table
    lines.append(f"## Top {top_risky_docs} Risky Documents\n")
    if not per_doc_df.empty:
        top_df = per_doc_df.sort_values("risk_score", ascending=False)[
            ["doc_id", "risk_score", "risk_bucket", "issues_count", "payment_net_days", "governing_law"]
        ]
        lines.extend(md_table_from_df(top_df, max_rows=top_risky_docs))
    else:
        lines.append("_No documents._")
    lines.append("")

    # Issue sections
    def section(title: str, filter_func):
        lines.append(f"## {title}\n")
        sub = findings_df[findings_df.apply(filter_func, axis=1)]
        if sub.empty:
            lines.append("_none_\n")
            return
        # brief table of first 10
        cols = ["doc_id", "issue_type", "field", "payment_net_days", "payment_amount"]
        lines.extend(md_table_from_df(sub[cols], max_rows=10))
        lines.append("")
        # quotes
        lines.append("**Representative Quotes:**")
        for _, row in sub.head(10).iterrows():
            q = str(row.get("quote", "")).replace("\n", " ").strip()
            if q:
                lines.append(f"> {q}")
        lines.append("")

    section("Missing Fields", lambda r: r["issue_type"] == "MISSING")
    section("Ambiguous Clauses", lambda r: r["issue_type"] == "AMBIGUOUS")
    section("Net Days Outliers", lambda r: r["issue_type"] in ("NET_DAYS_OUTLIER", "NET_DAYS_OVER_MAX", "NET_DAYS_DRIFT_VS_PARTY"))
    section("Governing Law Inconsistencies", lambda r: r["issue_type"] == "GOVERNING_LAW_INCONSISTENT")
    section("High Payment Amount Warnings", lambda r: r["issue_type"] == "PAYMENT_AMOUNT_HIGH")

    return lines

# ---------------- Main ----------------

def run_reporter():
    cfg = load_cfg()

    # Inputs
    validated_dir = ROOT / cfg["inputs"]["validated_dir"]
    per_doc_json  = ROOT / cfg["inputs"]["per_doc_json"]
    kpis_json     = ROOT / cfg["inputs"]["kpis_json"]

    # Outputs
    report_dir    = ROOT / cfg["outputs"]["report_dir"]
    findings_csv  = ROOT / cfg["outputs"]["findings_csv"]
    findings_json = ROOT / cfg["outputs"]["findings_json"]
    report_md     = ROOT / cfg["outputs"]["report_md"]

    report_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    docs = load_validated_docs(validated_dir)
    per_doc = load_json(per_doc_json)
    kpis    = load_json(kpis_json)

    if per_doc is None or kpis is None:
        raise SystemExit("Analyst artifacts missing. Run analyst.py first.")

    per_doc_df = pd.DataFrame(per_doc)
    issue_rows = build_issue_rows(docs, max_quote_chars=int(cfg["report"]["max_quote_chars"]))
    findings_df = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)

    # Write structured outputs
    findings_df.to_csv(findings_csv, index=False, encoding="utf-8")
    write_json(findings_json, json.loads(findings_df.to_json(orient="records", force_ascii=False)))

    # Write Markdown
    lines = build_markdown_report(
        kpis=kpis,
        per_doc_df=per_doc_df,
        findings_df=findings_df,
        top_risky_docs=int(cfg["report"]["top_risky_docs"])
    )
    write_md(report_md, lines)

    print(f"✓ Findings CSV  : {findings_csv.relative_to(ROOT)}")
    print(f"✓ Findings JSON : {findings_json.relative_to(ROOT)}")
    print(f"✓ Report MD     : {report_md.relative_to(ROOT)}")

if __name__ == "__main__":
    run_reporter()
