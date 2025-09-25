"""
Extractor Agent (Hybrid: rules + spaCy + LangChain LLM)
-------------------------------------------------------
- Reads schema from config/schema.contract.json
- Regex/heuristics first (fast, cheap)
- spaCy NER enrichment (optional via config)
- LangChain LLM fallback for missing/low-confidence fields (optional via config)
- Writes one JSON per contract to outputs/extract/
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Load OPENAI_API_KEY from .env (for LangChain/OpenAI)
from dotenv import load_dotenv
load_dotenv()

# Local helpers
from extractor_utils import make_chunks, rank_chunks_for_field
from extractor_nlp import nlp_parties, nlp_governing_law
from lc_extractor import lc_llm_extract_field


# -------- Paths --------
ROOT_DIR     = Path(__file__).resolve().parents[1]
TXT_DIR      = ROOT_DIR / "full_contract_txt"
SCHEMA_FILE  = ROOT_DIR / "config" / "schema.json"
LLM_CFG_FILE = ROOT_DIR / "config" / "extractor.llm.json"
OUT_DIR      = ROOT_DIR / "outputs" / "extract"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- Loaders --------
def load_schema() -> Dict[str, Any]:
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_FILE}")
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

def load_llm_cfg() -> Dict[str, Any]:
    if LLM_CFG_FILE.exists():
        return json.loads(LLM_CFG_FILE.read_text(encoding="utf-8"))
    # sensible defaults if config missing
    return {
        "use_spacy": False,
        "use_llm": False,
        "llm_model": "gpt-4o-mini",
        "llm_top_k_chunks": 2,
        "regex_confidence": 0.7,
        "min_span_len": 60
    }


# -------- Regex heuristics --------
DATE_RX  = r"(?:effective|commencement)\s*date[:\s]*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})"
LAW_RX   = r"governed\s+by\s+the\s+laws?\s+of\s+([A-Za-z\s,]+?)(?:[,.\n]|$)"
PARTY_RX = r"\bbetween\s+(.*?)\s+and\s+(.*?)(?:[,.\n]|$)"

def find_span(text: str, pattern: str, window: int = 600) -> Optional[Dict[str, Any]]:
    """
    Find a regex match and return a generous snippet with start/end char positions.
    Adds a baseline confidence for regex hits.
    """
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not m:
        return None
    s, e = m.span()
    e = min(len(text), max(e, s + window))
    return {"text": text[s:e].strip(), "start": s, "end": e, "confidence": 0.72, "source": "regex"}

def extract_rules(text: str) -> Dict[str, Any]:
    """
    First-pass extraction using regex/heuristics only.
    You can expand/replace with more patterns as needed.
    """
    rec: Dict[str, Any] = {}

    # Parties (simple heuristic)
    m = re.search(PARTY_RX, text, re.IGNORECASE | re.DOTALL)
    if m:
        rec["parties"] = [p.strip(" .,\n") for p in m.groups()]

    # Effective Date
    m = re.search(DATE_RX, text, re.IGNORECASE)
    if m:
        rec["effective_date"] = m.group(1).strip()

    # Governing Law (scalar)
    m = re.search(LAW_RX, text, re.IGNORECASE)
    if m:
        rec["governing_law"] = m.group(1).strip(" .,\n")

    # Clause spans
    rec["payment_terms"]      = find_span(text, r"\bpayment\s+terms?\b.{0,500}")
    rec["termination_clause"] = find_span(text, r"\btermination\b.{0,800}")

    return rec


# -------- Confidence check for deciding on LLM fallback --------
def confident(val: Any, min_conf: float, min_len: int) -> bool:
    """
    Return True if a field value looks 'good enough' to skip LLM fallback.
    - For dict/span fields: require confidence >= min_conf and text length >= min_len
    - For scalar fields: present value is considered sufficient
    """
    if val is None:
        return False
    if isinstance(val, dict):
        c = float(val.get("confidence", 0.0))
        L = len(val.get("text", "") or "")
        return (c >= min_conf) and (L >= min_len)
    return True  # scalar fields (e.g., strings) considered present


# -------- Main runner --------
def run_extractor() -> None:
    schema = load_schema()
    cfg = load_llm_cfg()

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {TXT_DIR}")

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        chunks = make_chunks(text)  # [(start,end,text), ...]
        record = extract_rules(text)
        provenance: Dict[str, Any] = {}

        # ---- spaCy enrichment (optional) ----
        if cfg.get("use_spacy", False):
            # Parties
            if not record.get("parties"):
                parties = nlp_parties(text)
                if parties:
                    record["parties"] = parties
                    provenance["parties"] = {"source": "spacy"}
            # Governing law (scalar)
            if not record.get("governing_law"):
                gl = nlp_governing_law(text)
                if gl:
                    record["governing_law"] = gl
                    provenance["governing_law"] = {"source": "spacy"}

        # ---- LangChain LLM fallback (optional) ----
        if cfg.get("use_llm", False):
            topk     = int(cfg.get("llm_top_k_chunks", 2))
            min_conf = float(cfg.get("regex_confidence", 0.7))
            min_len  = int(cfg.get("min_span_len", 60))
            model    = cfg.get("llm_model", "gpt-4o-mini")

            # Only consider fields that benefit from LLM extraction
            fields_to_try = [
                f for f in schema.keys()
                if f in ("parties", "effective_date", "governing_law", "payment_terms", "termination_clause")
            ]

            for field in fields_to_try:
                v = record.get(field)
                if confident(v, min_conf, min_len):
                    continue  # already good enough

                # Rank chunks by relevance hints and try top-K
                ranked = rank_chunks_for_field(field, chunks)[:topk]
                best = None
                for ch in ranked:
                    cand = lc_llm_extract_field(
                        txt_path.name, field, ch.start, ch.text, model_name=model
                    )
                    if not cand:
                        continue
                    if cand.get("text"):
                        if (not best) or (cand.get("confidence", 0) > best.get("confidence", 0)):
                            best = cand

                if best:
                    best["source"] = "llm"
                    record[field] = best
                    provenance[field] = {"source": "llm", "confidence": best.get("confidence", 0)}

        # ---- Ensure all schema keys exist & add metadata ----
        out: Dict[str, Any] = {k: record.get(k) for k in schema.keys()}
        out["_doc_id"] = txt_path.name
        out["_char_count"] = len(text)
        out["_provenance"] = provenance

        # Save JSON (overwrite if exists)
        out_path = OUT_DIR / f"{txt_path.stem}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Extracted {txt_path.name} -> {out_path.name}")


if __name__ == "__main__":
    run_extractor()
