from dataclasses import dataclass
from typing import List
from rapidfuzz import fuzz

@dataclass
class Chunk:
    start: int
    end: int
    text: str

def make_chunks(text: str, size: int = 3500, overlap: int = 300) -> List[Chunk]:
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append(Chunk(i, j, text[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

FIELD_HINTS = {
    "governing_law": ["governed by the laws of", "governing law"],
    "payment_terms": ["payment terms", "invoice", "net", "payment"],
    "termination_clause": ["termination", "term and termination"],
    "parties": ["between", "made by and between", "provider", "recipient"],
    "effective_date": ["effective date", "commencement date"]
}

def rank_chunks_for_field(field: str, chunks: List[Chunk]) -> List[Chunk]:
    hints = FIELD_HINTS.get(field, [])
    if not hints:
        return chunks
    def score(c: Chunk) -> int:
        return max(fuzz.partial_ratio(h, c.text.lower()) for h in hints)
    return sorted(chunks, key=score, reverse=True)
