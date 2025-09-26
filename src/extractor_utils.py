from dataclasses import dataclass
from typing import List
from rapidfuzz import fuzz

@dataclass
class Chunk:
    start: int
    end: int
    text: str

def make_chunks(text: str, size: int = 2000, overlap: int = 150) -> List[Chunk]:
    """Split text into overlapping chunks with global offsets."""
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append(Chunk(i, j, text[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

FIELD_HINTS = {
    "governing_law": ["governed by the laws of", "governing law"],
    "payment_terms": ["payment terms", "invoice", "net", "payment", "due"],
    "termination_clause": ["termination", "term and termination", "terminate"],
    "parties": ["between", "made by and between", "provider", "recipient", "party"],
    "effective_date": ["effective date", "commencement date", "effective as of"]
}

def _keyword_score(txt: str, hints: List[str]) -> int:
    t = txt.lower()
    return sum(1 for h in hints if h in t)

def rank_chunks_for_field(field: str, chunks: List[Chunk]) -> List[Chunk]:
    """
    Cheap keyword prefilter to top ~5 chunks, then fuzzy-rank those.
    This avoids fuzzy-scoring every chunk.
    """
    hints = FIELD_HINTS.get(field, [])
    if not hints:
        return chunks
    prelim = sorted(
        chunks, key=lambda c: _keyword_score(c.text, hints), reverse=True
    )[:5]  # keep only the most promising few
    def score(c: Chunk) -> int:
        return max(fuzz.partial_ratio(h, c.text.lower()) for h in hints)
    return sorted(prelim, key=score, reverse=True)
