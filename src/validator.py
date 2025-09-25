"""
Validator Agent (Auditor)
-------------------------
Reads extracted JSON files from outputs/extract/, normalizes values,
flags anomalies, and performs cross-document consistency checks.

Writes per-document validated JSONs to outputs/validate/
and a corpus summary to outputs/validate/_corpus_issues.json
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from statistics import median

# -------- Paths --------
ROOT_DIR      = Path(__file__).resolve().parents[1]
EXTRACT_DIR   = ROOT_DIR / "outputs" / "extract"
VALIDATE_DIR  = ROOT_DIR / "outputs" / "validate"
RULES_FILE    = ROOT_DIR / "config" / "validationrules.json"

VALIDATE_DIR.mkdir(parents=True, exist_ok=True)

# -------- Rule Loading --------
def load_rules() -> Dict[str, Any]:
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# -------- Normalization Helpers --------
def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Convert common date formats to ISO YYYY-MM-DD if possible."""
    if not raw:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw  # leave unchanged if no known format matches

NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100
}
def normalize_word_numbers(text: str) -> str:
    """Replace single-number words (e.g., 'thirty') with digits; simple heuristic."""
    def repl(m):
        w = m.group(0).lower()
        return str(NUM_WORDS.get(w, w))
    return re.sub(r"\b(" + "|".join(map(re.escape, NUM_WORDS.keys())) + r")\b", repl, text, flags=re.IGNORECASE)

CURRENCY_MAP = {
    "$":"USD","usd":"USD","€":"EUR","eur":"EUR","£":"GBP","gbp":"GBP","₹":"INR","inr":"INR"
}
def extract_currency_amounts(text: str) -> List[Dict[str, Any]]:
    """Extract currency amounts like $12,345.67, INR 5000, €500"""
    hits: List[Dict[str, Any]] = []
    for cur_sym, cur_code in CURRENCY_MAP.items():
        # Symbol form (e.g., $12,345.67)
        sym_rx = re.escape(cur_sym) + r"\s*([0-9][0-9,]*\.?[0-9]*)"
        for m in re.finditer(sym_rx, text, flags=re.IGNORECASE):
            try:
                amt = float(m.group(1).replace(",", ""))
                hits.append({"currency": cur_code, "amount": amt})
            except ValueError:
                pass
        # Code form (e.g., USD 12000)
        code_rx = r"\b" + re.escape(cur_code) + r"\s*([0-9][0-9,]*\.?[0-9]*)\b"
        for m in re.finditer(code_rx, text, flags=re.IGNORECASE):
            try:
                amt = float(m.group(1).replace(",", ""))
                hits.append({"currency": cur_code, "amount": amt})
            except ValueError:
                pass
    return hits

def extract_net_days(payment_terms: Optional[Dict[str, Any]]) -> Optional[int]:
    """Find Net X inside payment terms."""
    if not payment_terms or "text" not in payment_terms:
        return None
    txt = payment_terms["text"]
    # Normalize number words to digits before searching
    txt_norm = normalize_word_numbers(txt)
    m = re.search(r"\bnet\s*(\d{1,3})\b", txt_norm, re.IGNORECASE)
    return int(m.group(1)) if m else None

# -------- Record Validation --------
def validate_record(record: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize values and flag anomalies against company rules.
    Returns an enriched record with _issues and normalized fields.
    """
    issues: List[Dict[str, str]] = []

    # 0) Missing required fields
    if rules.get("flag_missing_as_issue", False):
        for field in rules.get("required_fields", []):
            val = record.get(field)
            if val is None:
                issues.append({"field": field, "type": "MISSING", "detail": f"Required field '{field}' is missing"})

    # 1) Normalize dates
    record["effective_date"] = normalize_date(record.get("effective_date"))

    # 2) Normalize payment terms text and detect Net days
    if record.get("payment_terms") and isinstance(record["payment_terms"], dict):
        normalized_text = normalize_word_numbers(record["payment_terms"].get("text", ""))
        record["payment_terms"]["text_normalized"] = normalized_text

    net_days = extract_net_days(record.get("payment_terms") or {})
    record["payment_net_days"] = net_days

    std = rules.get("standard_payment_days", 30)
    max_net = rules.get("max_net_days", 120)
    if isinstance(net_days, int):
        if net_days > std:
            issues.append({"field": "payment_terms", "type": "ANOMALY",
                           "detail": f"Net {net_days} exceeds standard {std}"})
        if net_days > max_net:
            issues.append({"field": "payment_terms", "type": "EXTREME",
                           "detail": f"Net {net_days} beyond acceptable max {max_net}"})

    # Currency normalization (collect amounts present)
    amounts = extract_currency_amounts((record.get("payment_terms") or {}).get("text", ""))
    if amounts:
        record["payment_amounts"] = amounts

    # 3) Governing law policy
    law = (record.get("governing_law") or "").strip()
    allowed_laws = rules.get("allowed_governing_law", [])
    if law and allowed_laws and (law not in allowed_laws):
        issues.append({"field": "governing_law", "type": "NOTICE",
                       "detail": f"Non-standard governing law: {law}"})

    # 4) Redactions / ambiguous language
    # Redactions in parties
    if rules.get("flag_redactions_in_parties") and record.get("parties"):
        joined = " ".join([p for p in record["parties"] if isinstance(p, str)])
        if "[***]" in joined or "***" in joined:
            issues.append({"field": "parties", "type": "AMBIGUOUS",
                           "detail": "Party names appear redacted"})

    # Ambiguous terms in key spans
    amb_terms = [t.lower() for t in rules.get("ambiguous_terms", [])]
    for field in ("termination_clause", "payment_terms"):
        span = record.get(field)
        if isinstance(span, dict):
            text_lower = span.get("text", "").lower()
            hits = sorted({t for t in amb_terms if t in text_lower})
            if hits:
                issues.append({"field": field, "type": "AMBIGUOUS",
                               "detail": f"Ambiguous language found: {', '.join(hits)}"})

    record["_issues"] = issues
    return record

# -------- Corpus-Level Cross-Document Check --------
def corpus_cross_check(validated_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes corpus baselines (e.g., median Net days) and flags outliers.
    Returns a summary: {"baseline_net_days": X, "issues": [...]}
    """
    net_vals = [r.get("payment_net_days") for r in validated_records if isinstance(r.get("payment_net_days"), int)]
    baseline_net = int(median(net_vals)) if net_vals else None

    issues: List[Dict[str, Any]] = []
    if baseline_net is not None:
        for r in validated_records:
            net = r.get("payment_net_days")
            # Deviation threshold: 15 days difference from median (configurable if desired)
            if isinstance(net, int) and abs(net - baseline_net) >= 15:
                issues.append({
                    "_doc_id": r.get("_doc_id"),
                    "field": "payment_terms",
                    "type": "CROSS_DOC_DEVIATION",
                    "detail": f"Net {net} deviates from corpus median {baseline_net}"
                })

    return {"baseline_net_days": baseline_net, "issues": issues}

# -------- Runner --------
def run_validator():
    # Load rules
    rules = load_rules()

    # Load extractor outputs
    files = sorted(EXTRACT_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No extractor outputs found in {EXTRACT_DIR}")

    validated_records: List[Dict[str, Any]] = []

    # Per-document validation
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        validated = validate_record(rec, rules)
        validated_records.append(validated)

        out_path = VALIDATE_DIR / f.name
        out_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Validated {f.name} -> {out_path.name}")

    # Cross-document consistency summary
    corpus_summary = corpus_cross_check(validated_records)
    (VALIDATE_DIR / "_corpus_issues.json").write_text(json.dumps(corpus_summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(f"✓ Corpus cross-check -> outputs/validate/_corpus_issues.json")

if __name__ == "__main__":
    run_validator()
