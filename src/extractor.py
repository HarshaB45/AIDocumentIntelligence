"""
Extractor Agent (Optimized)
- Regex/heuristics → optional spaCy → optional LangChain LLM fallback
- Faster chunking, keyword-prefiltered ranking
- Threaded per-file processing + LLM concurrency throttle
- Persistent LangChain cache to avoid repeat calls
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load OPENAI_API_KEY from .env
from dotenv import load_dotenv
load_dotenv()

# LangChain cache (persists across runs)
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

# Local helpers
from extractor_utils import make_chunks, rank_chunks_for_field
from extractor_nlp import nlp_parties, nlp_governing_law
from lc_extractor import lc_llm_extract_field

# -------- Paths --------
ROOT_DIR     = Path(__file__).resolve().parents[1]
TXT_DIR      = ROOT_DIR / "full_contract2_txt"
SCHEMA_FILE  = ROOT_DIR / "config" / "schema.json"
LLM_CFG_FILE = ROOT_DIR / "config" / "extractor.llm.json"
OUT_DIR      = ROOT_DIR / "outputs" / "extract"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Enable LC cache
set_llm_cache(SQLiteCache(str(ROOT_DIR / ".lc_cache.db")))

# -------- Regex heuristics --------
DATE_RX  = r"(?:effective|commencement)\s*date[:\s]*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})"
LAW_RX   = r"governed\s+by\s+the\s+laws?\s+of\s+([A-Za-z\s,]+?)(?:[,.\n]|$)"
PARTY_RX = r"\bbetween\s+(.*?)\s+and\s+(.*?)(?:[,.\n]|$)"

def load_schema() -> Dict[str, Any]:
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_FILE}")
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

def load_cfg() -> Dict[str, Any]:
    if LLM_CFG_FILE.exists():
        return json.loads(LLM_CFG_FILE.read_text(encoding="utf-8"))
    # sensible defaults
    return {
        "use_spacy": True,
        "use_llm": True,
        "llm_model": "gpt-4o-mini",
        "llm_top_k_chunks": 1,
        "regex_confidence": 0.6,
        "min_span_len": 20,
        "max_workers": 8,
        "llm_concurrency": 3,
        "chunk_size": 2000,
        "chunk_overlap": 150
    }

def find_span(text: str, pattern: str, window: int = 600) -> Optional[Dict[str, Any]]:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not m:
        return None
    s, e = m.span()
    e = min(len(text), max(e, s + window))
    return {"text": text[s:e].strip(), "start": s, "end": e, "confidence": 0.72, "source": "regex"}

def extract_rules(text: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    m = re.search(PARTY_RX, text, re.IGNORECASE | re.DOTALL)
    if m:
        rec["parties"] = [p.strip(" .,\n") for p in m.groups()]
    m = re.search(DATE_RX, text, re.IGNORECASE)
    if m:
        rec["effective_date"] = m.group(1).strip()
    m = re.search(LAW_RX, text, re.IGNORECASE)
    if m:
        rec["governing_law"] = m.group(1).strip(" .,\n")
    rec["payment_terms"]      = find_span(text, r"\bpayment\s+terms?\b.{0,500}")
    rec["termination_clause"] = find_span(text, r"\btermination\b.{0,800}")
    return rec

def confident(val: Any, min_conf: float, min_len: int) -> bool:
    if val is None:
        return False
    if isinstance(val, dict):
        c = float(val.get("confidence", 0.0))
        L = len(val.get("text", "") or "")
        return (c >= min_conf) and (L >= min_len)
    return True

# ----- LLM concurrency guard -----
_cfg = load_cfg()
_LLM_SEM = threading.Semaphore(int(_cfg.get("llm_concurrency", 3)))

def guarded_llm_extract(doc_id: str, field: str, chunk_start: int, chunk_text: str, model_name: str):
    with _LLM_SEM:
        return lc_llm_extract_field(doc_id, field, chunk_start, chunk_text, model_name=model_name)

# ----- Per-file processing -----
def process_file(txt_path: Path, schema: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, str]:
    t0 = time.perf_counter()

    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    # Respect config chunk size/overlap
    chunks = make_chunks(text, size=int(cfg.get("chunk_size", 2000)), overlap=int(cfg.get("chunk_overlap", 150)))
    record = extract_rules(text)
    provenance: Dict[str, Any] = {}

    # spaCy (optional)
    if cfg.get("use_spacy", True):
        if not record.get("parties"):
            parties = nlp_parties(text)
            if parties:
                record["parties"] = parties
                provenance["parties"] = {"source": "spacy"}
        if not record.get("governing_law"):
            gl = nlp_governing_law(text)
            if gl:
                record["governing_law"] = gl
                provenance["governing_law"] = {"source": "spacy"}

    # LLM fallback (optional and throttled)
    if cfg.get("use_llm", True):
        topk     = int(cfg.get("llm_top_k_chunks", 1))
        min_conf = float(cfg.get("regex_confidence", 0.6))
        min_len  = int(cfg.get("min_span_len", 20))
        model    = cfg.get("llm_model", "gpt-4o-mini")

        fields_to_try = [
            f for f in schema.keys()
            if f in ("parties", "effective_date", "governing_law", "payment_terms", "termination_clause")
        ]

        for field in fields_to_try:
            v = record.get(field)
            if confident(v, min_conf, min_len):
                continue
            ranked = rank_chunks_for_field(field, chunks)[:topk]
            best = None
            for ch in ranked:
                t_llm0 = time.perf_counter()
                cand = guarded_llm_extract(txt_path.name, field, ch.start, ch.text, model_name=model)
                t_llm1 = time.perf_counter()
                # print(f"[LLM] {field} {txt_path.name} took {t_llm1 - t_llm0:.2f}s")  # optional log
                if not cand:
                    continue
                if cand.get("text"):
                    if (not best) or (cand.get("confidence", 0) > best.get("confidence", 0)):
                        best = cand
            if best:
                best["source"] = "llm"
                record[field] = best
                provenance[field] = {"source": "llm", "confidence": best.get("confidence", 0)}

    # Output
    out: Dict[str, Any] = {k: record.get(k) for k in schema.keys()}
    out["_doc_id"] = txt_path.name
    out["_char_count"] = len(text)
    out["_provenance"] = provenance

    out_path = OUT_DIR / f"{txt_path.stem}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    t1 = time.perf_counter()
    # print(f"[DONE] {txt_path.name} in {t1 - t0:.2f}s")  # optional log
    return txt_path.name, out_path.name

# ----- Runner with thread pool -----
def run_extractor() -> None:
    schema = load_schema()
    cfg = load_cfg()

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {TXT_DIR}")

    max_workers = int(cfg.get("max_workers", 8))
    print(f"Processing {len(txt_files)} docs with {max_workers} workers; LLM concurrency {int(cfg.get('llm_concurrency', 3))}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_file, p, schema, cfg) for p in txt_files]
        for fut in as_completed(futures):
            src, dst = fut.result()
            print(f"✓ Extracted {src} -> {dst}")

if __name__ == "__main__":
    run_extractor()

