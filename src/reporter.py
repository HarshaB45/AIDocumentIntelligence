"""
Reporter Agent (Communicator)
-----------------------------
Reads Analyst outputs and produces:
- CSV: outputs/reports/contracts.csv  (one row per contract, key fields + risk)
- Markdown: outputs/reports/summary.md (portfolio KPIs + per-contract highlights)

No external dependencies required.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# ---- Paths ----
ROOT_DIR       = Path(__file__).resolve().parents[1]
ANALYZE_DIR    = ROOT_DIR / "outputs" / "analyze"
PER_DOC_DIR    = ANALYZE_DIR / "per_doc"
REPORTS_DIR    = ROOT_DIR / "outputs" / "reports"

PORTFOLIO_JSON = ANALYZE_DIR / "portfolio.json"
CSV_PATH       = REPORTS_DIR / "contracts.csv"
MD_PATH        = REPORTS_DIR / "summary.md"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_portfolio() -> Dict[str, Any]:
    if not PORTFOLIO_JSON.exists():
        raise SystemExit(f"Missing portfolio.json at {PORTFOLIO_JSON}. Run analyst.py first.")
    return json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))


def load_per_doc() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in sorted(PER_DOC_DIR.glob("*.json")):
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    if not docs:
        raise SystemExit(f"No per-doc analysis files in {PER_DOC_DIR}. Run analyst.py first.")
    return docs


def write_csv(per_docs: List[Dict[str, Any]]) -> None:
    # Choose compact, useful columns for business reviewers
    cols = [
        "_doc_id",
        "risk_score",
        "risk_bucket",
        "governing_law",
        "payment_net_days",
        "effective_date",
        "_issues_count",
        "tags",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_docs:
            issues = r.get("_issues", [])
            row = {
                "_doc_id": r.get("_doc_id"),
                "risk_score": r.get("risk_score"),
                "risk_bucket": r.get("risk_bucket"),
                "governing_law": r.get("governing_law"),
                "payment_net_days": r.get("payment_net_days"),
                "effective_date": r.get("effective_date"),
                "_issues_count": len(issues),
                "tags": ";".join(r.get("tags", [])),
            }
            w.writerow(row)


def write_markdown(portfolio: Dict[str, Any], per_docs: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Contract Portfolio Summary\n")

    # Portfolio KPIs
    lines.append("## Portfolio KPIs\n")
    lines.append(f"- **Documents:** {portfolio.get('count_documents', 0)}")
    lines.append(f"- **Avg Risk Score:** {portfolio.get('avg_risk_score', 0)}")
    lines.append(f"- **Median Risk Score:** {portfolio.get('median_risk_score', 0)}")
    dist = portfolio.get("risk_distribution", {})
    lines.append(f"- **Risk Distribution:** low={dist.get('low',0)}, medium={dist.get('medium',0)}, high={dist.get('high',0)}")
    lines.append(f"- **Median Net Days:** {portfolio.get('median_net_days')}")
    laws = portfolio.get("governing_law_distribution", {})
    if laws:
        top_laws = ", ".join([f"{k}({v})" for k, v in list(laws.items())[:10]])
        lines.append(f"- **Top Governing Laws:** {top_laws}")
    amb = portfolio.get("top_ambiguous_terms", [])
    if amb:
        lines.append(f"- **Top Ambiguous Terms:** {', '.join(amb)}")
    lines.append("")

    # Top Issue Types
    lines.append("### Top Issue Types\n")
    if portfolio.get("top_issue_types"):
        for t, c in portfolio["top_issue_types"].items():
            lines.append(f"- **{t}**: {c}")
    else:
        lines.append("- (none detected)")
    lines.append("")

    # Table: Per-document snapshot (sorted by risk desc)
    lines.append("## Contracts (sorted by risk)\n")
    header = ["Doc ID", "Risk", "Bucket", "Law", "Net Days", "Issues", "Tags"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    per_docs_sorted = sorted(per_docs, key=lambda r: r.get("risk_score", 0), reverse=True)
    for r in per_docs_sorted:
        row = [
            r.get("_doc_id", ""),
            str(r.get("risk_score", "")),
            r.get("risk_bucket", ""),
            r.get("governing_law", "") or "—",
            str(r.get("payment_net_days", "—")),
            str(len(r.get("_issues", []))),
            ", ".join(r.get("tags", [])) or "—",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Optional: list top-N high-risk contracts with brief issue bullets
    lines.append("\n### High-Risk Highlights\n")
    topN = per_docs_sorted[:10]
    if not topN:
        lines.append("- (none)")
    else:
        for r in topN:
            lines.append(f"- **{r.get('_doc_id','')}** — Risk {r.get('risk_score','?')} ({r.get('risk_bucket','')})")
            issues = r.get("_issues", [])
            for i in issues[:5]:  # limit bullets
                lines.append(f"  - {i.get('field','?')}: {i.get('type','?')} — {i.get('detail','')}")
            if len(issues) > 5:
                lines.append(f"  - (+{len(issues)-5} more issues)")
    lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_reporter():
    portfolio = load_portfolio()
    per_docs = load_per_doc()
    write_csv(per_docs)
    write_markdown(portfolio, per_docs)
    print(f"✓ Reporter wrote CSV -> {CSV_PATH}")
    print(f"✓ Reporter wrote Markdown -> {MD_PATH}")


if __name__ == "__main__":
    run_reporter()
