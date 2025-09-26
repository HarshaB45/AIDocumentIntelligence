"""
Validator Agent (Normalize + Cross-Reference + Anomalies)
---------------------------------------------------------
- Input : outputs/extract/*.json  (from Extractor)
- Output: outputs/validate/*.json (per-doc normalized & issues)
         + outputs/validate/_crossref_report.{json,md}

What it does (no LLM, no summarization):
1) Normalize:
   - Dates (YYYY-MM-DD)
   - Payment amounts (float) and payment net days (int)
   - Light ambiguity flags (keyword list)
2) Cross-Reference:
   - Corpus medians & outliers (e.g., net days far from median)
   - Per-party consistency checks:
       * Governing law consistent across the party’s contracts
       * Net-days deviations vs party median (tolerance)
3) Anomalies:
   - Missing required fields
   - Net-days over standards / outliers
   - Amount unusually high (warn threshold)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from statistics import median
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "outputs" / "extract"
OUT_DIR = ROOT / "outputs" / "validate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RULES_FILE = ROOT / "config" / "validationrules.json"

# -------------------- Load rules --------------------

def load_rules() -> Dict[str, Any]:
    if not RULES_FILE.exists():
        raise FileNotFoundError(f"Rules file not found: {RULES_FILE}")
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))

# -------------------- Normalization helpers --------------------

DATE_PATTERNS = [
    ("%b %d, %Y", r"[A-Za-z]{3}\s+\d{1,2},\s+\d{4}"),
    ("%B %d, %Y", r"[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}"),
    ("%m/%d/%Y",  r"\d{1,2}/\d{1,2}/\d{4}"),
    ("%m/%d/%y",  r"\d{1,2}/\d{1,2}/\d{2}")
]

CURRENCY_RX = r"(?:USD|US\$|\$)\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|\d+(?:\.\d{2})?)"
NET_DAYS_RX = r"net\s*(\d{1,3})"

def normalize_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt, _ in DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    for fmt, rx in DATE_PATTERNS:
        m = re.search(rx, s)
        if m:
            try:
                dt = datetime.strptime(m.group(0), fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return None

def normalize_amount(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(CURRENCY_RX, text, flags=re.IGNORECASE)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None

def extract_net_days(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(NET_DAYS_RX, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def looks_ambiguous(text: str, terms: List[str]) -> bool:
    low = (text or "").lower()
    return any(t.lower() in low for t in terms)

def norm_party_name(s: str) -> str:
    if not s:
        return ""
    # Simple normalization: collapse whitespace, strip punctuation edges
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r,.;:-()[]{}\"'")
    return s.upper()

def party_key(parties: Any) -> Optional[str]:
    """
    Generate a deterministic key for party pair (if available).
    Uses first two names after normalization; sorts for stability.
    """
    if not isinstance(parties, list) or len(parties) == 0:
        return None
    names = [norm_party_name(x) for x in parties if isinstance(x, str) and x.strip()]
    if not names:
        return None
    key_names = sorted(names[:2])
    return " | ".join(key_names)

# -------------------- First pass: normalize & per-doc issues --------------------

def validate_doc_first_pass(doc: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    issues = out.get("_issues", []) or []

    # Effective date normalization
    eff = out.get("effective_date")
    if eff:
        norm = normalize_date(eff if isinstance(eff, str) else str(eff))
        if norm:
            out["effective_date_normalized"] = norm
        else:
            issues.append({"type": "BAD_DATE", "field": "effective_date", "value": eff})

    # Payment terms normalization
    pt = out.get("payment_terms")
    if isinstance(pt, dict) and pt.get("text"):
        amt = normalize_amount(pt["text"])
        if amt is not None:
            out["payment_amount_normalized"] = amt
        nd = extract_net_days(pt["text"])
        if nd is not None:
            out["payment_net_days"] = nd
        if looks_ambiguous(pt["text"], rules.get("ambiguity_terms", [])):
            issues.append({
                "type": "AMBIGUOUS",
                "field": "payment_terms",
                "reason": "Ambiguous language detected",
                "span": {"start": pt.get("start"), "end": pt.get("end")}
            })

    # Termination ambiguity (no normalization, just flag)
    tc = out.get("termination_clause")
    if isinstance(tc, dict) and tc.get("text"):
        if looks_ambiguous(tc["text"], rules.get("ambiguity_terms", [])):
            issues.append({
                "type": "AMBIGUOUS",
                "field": "termination_clause",
                "reason": "Ambiguous language detected",
                "span": {"start": tc.get("start"), "end": tc.get("end")}
            })

    # Required fields present?
    for f in rules.get("required_fields", []):
        if not out.get(f):
            issues.append({"type": "MISSING", "field": f})

    out["_issues"] = issues
    return out

# -------------------- Collect corpus stats --------------------

def collect_corpus_rows(validated_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for d in validated_docs:
        rows.append({
            "doc_id": d.get("_doc_id"),
            "party_key": party_key(d.get("parties")),
            "governing_law": d.get("governing_law"),
            "payment_net_days": d.get("payment_net_days"),
            "amount": d.get("payment_amount_normalized"),
        })
    return rows

def corpus_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    net_days_vals = [r["payment_net_days"] for r in rows if isinstance(r.get("payment_net_days"), int)]
    net_days_median = median(net_days_vals) if net_days_vals else None

    gov_law_counts = Counter([r["governing_law"] for r in rows if r.get("governing_law")])
    top_gov_law = gov_law_counts.most_common(1)[0][0] if gov_law_counts else None

    # Per-party aggregations
    per_party = defaultdict(list)
    for r in rows:
        if r.get("party_key"):
            per_party[r["party_key"]].append(r)

    per_party_net_median = {}
    per_party_gov_modes = {}
    for pk, lst in per_party.items():
        nds = [x["payment_net_days"] for x in lst if isinstance(x.get("payment_net_days"), int)]
        per_party_net_median[pk] = median(nds) if nds else None

        laws = [x["governing_law"] for x in lst if x.get("governing_law")]
        per_party_gov_modes[pk] = Counter(laws).most_common(1)[0][0] if laws else None

    return {
        "net_days_median": net_days_median,
        "top_governing_law": top_gov_law,
        "per_party_net_median": per_party_net_median,
        "per_party_gov_modes": per_party_gov_modes,
        "gov_law_counts": gov_law_counts
    }

# -------------------- Second pass: cross-reference checks --------------------

def apply_cross_reference(validated_docs: List[Dict[str, Any]], rows: List[Dict[str, Any]], rules: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats = corpus_stats(rows)
    updated = []

    # For report: collect inconsistencies
    inconsistent_governing_law = []  # per-party mismatches
    net_days_outliers = []            # per-doc outliers
    net_days_party_drifts = []        # drift vs party median
    amount_high = []                  # unusually high payments

    net_median = stats.get("net_days_median")
    pct_from_med = rules.get("corpus_outliers", {}).get("net_days_pct_from_median", 0.5)
    max_net_days = rules.get("standards", {}).get("max_net_days", 120)
    warn_amount = rules.get("standards", {}).get("amount_upper_warn", 1_000_000.0)
    per_party_tol = rules.get("cross_reference", {}).get("per_party_net_days_tolerance", 15)
    check_gov_consistency = rules.get("cross_reference", {}).get("check_per_party_governing_law_consistency", True)

    # Build quick maps for second pass
    party_to_gov_mode = stats.get("per_party_gov_modes", {})
    party_to_net_median = stats.get("per_party_net_median", {})

    doc_map = {d.get("_doc_id"): d for d in validated_docs}

    for r in rows:
        doc = doc_map.get(r["doc_id"])
        if not doc:
            continue
        issues = doc.get("_issues", []) or []
        pk = r.get("party_key")
        nd = r.get("payment_net_days")
        glaw = r.get("governing_law")
        amt = r.get("amount")

        # 1) Corpus outliers for net days
        if isinstance(nd, int):
            if nd > max_net_days:
                issues.append({"type":"NET_DAYS_OVER_MAX","value": nd, "max": max_net_days})
                net_days_outliers.append({"doc_id": r["doc_id"], "payment_net_days": nd, "reason": "over_max"})

            if net_median is not None and net_median > 0:
                # absolute pct deviation from median
                pct_dev = abs(nd - net_median) / net_median
                if pct_dev >= pct_from_med:
                    issues.append({"type":"NET_DAYS_OUTLIER","value": nd, "median": net_median, "pct_dev": round(pct_dev,2)})
                    net_days_outliers.append({"doc_id": r["doc_id"], "payment_net_days": nd, "reason": "far_from_median", "median": net_median})

        # 2) Per-party governing law consistency
        if check_gov_consistency and pk:
            mode_law = party_to_gov_mode.get(pk)
            if glaw and mode_law and glaw != mode_law:
                issues.append({"type":"GOVERNING_LAW_INCONSISTENT","party_key": pk, "value": glaw, "party_mode": mode_law})
                inconsistent_governing_law.append({"doc_id": r["doc_id"], "party_key": pk, "value": glaw, "party_mode": mode_law})

        # 3) Per-party net-days drift
        if pk and isinstance(nd, int):
            pmed = party_to_net_median.get(pk)
            if isinstance(pmed, (int, float)):
                if abs(nd - pmed) > per_party_tol:
                    issues.append({"type":"NET_DAYS_DRIFT_VS_PARTY","party_key": pk, "value": nd, "party_median": pmed, "tolerance": per_party_tol})
                    net_days_party_drifts.append({"doc_id": r["doc_id"], "party_key": pk, "value": nd, "party_median": pmed})

        # 4) Amount unusually high (warn)
        if isinstance(amt, (int, float)) and amt > warn_amount:
            issues.append({"type":"PAYMENT_AMOUNT_HIGH","value": amt, "warn_above": warn_amount})
            amount_high.append({"doc_id": r["doc_id"], "amount": amt})

        doc["_issues"] = issues
        updated.append(doc)

    # Build cross-ref report
    report = {
        "doc_count": len(rows),
        "net_days_median": stats.get("net_days_median"),
        "governing_law_top": stats.get("top_governing_law"),
        "governing_law_counts": stats.get("gov_law_counts").most_common() if stats.get("gov_law_counts") else [],
        "per_party_contract_counts": {pk: len(lst) for pk, lst in _group_by(rows, "party_key").items()},
        "inconsistent_governing_law": inconsistent_governing_law,
        "net_days_outliers": net_days_outliers,
        "net_days_party_drifts": net_days_party_drifts,
        "amount_high_warnings": amount_high
    }
    return updated, report

def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in rows:
        k = r.get(key)
        if k:
            out[k].append(r)
    return out

# -------------------- I/O helpers --------------------

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_md_report(path: Path, report: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Cross-Reference Report (Validator)\n")
    lines.append(f"- Documents: **{report.get('doc_count', 0)}**")
    lines.append(f"- Median Net Days: **{report.get('net_days_median', 'n/a')}**")
    lines.append(f"- Top Governing Law: **{report.get('governing_law_top', 'n/a')}**\n")

    lines.append("## Governing Law Distribution\n")
    gl_counts = report.get("governing_law_counts", [])
    if gl_counts:
        for law, cnt in gl_counts:
            lines.append(f"- {law}: {cnt}")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## Per-Party Contract Counts\n")
    pcounts = report.get("per_party_contract_counts", {})
    if pcounts:
        for pk, cnt in sorted(pcounts.items(), key=lambda x: -x[1]):
            lines.append(f"- {pk}: {cnt}")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## Inconsistent Governing Law (by Party)\n")
    inc = report.get("inconsistent_governing_law", [])
    if inc:
        for item in inc[:200]:
            lines.append(f"- {item['doc_id']}: {item['party_key']} — value={item['value']} vs party_mode={item['party_mode']}")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## Net Days Outliers\n")
    outliers = report.get("net_days_outliers", [])
    if outliers:
        for item in outliers[:200]:
            reason = item.get("reason")
            med = item.get("median")
            if reason == "far_from_median" and med is not None:
                lines.append(f"- {item['doc_id']}: {item['payment_net_days']} (median {med})")
            else:
                lines.append(f"- {item['doc_id']}: {item['payment_net_days']} ({reason})")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## Net Days Drift vs Party Median\n")
    drifts = report.get("net_days_party_drifts", [])
    if drifts:
        for item in drifts[:200]:
            lines.append(f"- {item['doc_id']}: {item['party_key']} — {item['value']} vs party_median {item['party_median']}")
    else:
        lines.append("_none_")
    lines.append("")

    lines.append("## High Payment Amount Warnings\n")
    highs = report.get("amount_high_warnings", [])
    if highs:
        for item in highs[:200]:
            lines.append(f"- {item['doc_id']}: amount={item['amount']}")
    else:
        lines.append("_none_")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

# -------------------- Runner --------------------

def run_validator():
    rules = load_rules()

    # 1) First pass: normalize + per-doc issues
    src_files = sorted(EXTRACT_DIR.glob("*.json"))
    if not src_files:
        raise SystemExit(f"No extractor JSONs found in {EXTRACT_DIR}")

    validated_docs: List[Dict[str, Any]] = []
    for p in src_files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        vdoc = validate_doc_first_pass(doc, rules)
        validated_docs.append(vdoc)

    # Write first-pass outputs (so you can inspect even before cross-ref)
    for d in validated_docs:
        outp = OUT_DIR / d.get("_doc_id", "unknown.json")
        write_json(outp, d)

    # 2) Cross-reference pass (corpus + per-party consistency)
    rows = collect_corpus_rows(validated_docs)
    updated_docs, report = apply_cross_reference(validated_docs, rows, rules)

    # Write updated per-doc outputs (with cross-ref issues appended)
    for d in updated_docs:
        outp = OUT_DIR / d.get("_doc_id", "unknown.json")
        write_json(outp, d)

    # 3) Write cross-reference reports
    report_json = OUT_DIR / "_crossref_report.json"
    report_md   = OUT_DIR / "_crossref_report.md"
    write_json(report_json, report)
    write_md_report(report_md, report)

    print(f"✓ Validated {len(updated_docs)} docs")
    print(f"✓ Wrote per-doc JSONs     -> {OUT_DIR.relative_to(ROOT)}")
    print(f"✓ Cross-ref report (JSON) -> {report_json.relative_to(ROOT)}")
    print(f"✓ Cross-ref report (MD)   -> {report_md.relative_to(ROOT)}")

if __name__ == "__main__":
    run_validator()
