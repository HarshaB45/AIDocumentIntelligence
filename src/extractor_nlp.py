from typing import List, Optional
import re, spacy

_nlp = None
def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

US_STATES = {"Delaware","California","New York","Texas","Florida","Washington","Massachusetts","Illinois","Georgia","Virginia"}
COUNTRIES = {"United States","Singapore","India","United Kingdom","Canada","Germany","France"}

def nlp_parties(text: str) -> Optional[List[str]]:
    # Limit to the first 3000 chars (faster; parties usually up front)
    doc = nlp()(text[:3000])
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG","PERSON")]
    uniq = []
    for x in orgs:
        if len(x) < 3:
            continue
        if x not in uniq:
            uniq.append(x)
    return uniq[:4] or None

def nlp_governing_law(text: str) -> Optional[str]:
    M = re.search(r"governed\s+by\s+the\s+laws?\s+of\s+([A-Za-z ,]+)", text, re.IGNORECASE)
    if M:
        cand = M.group(1).strip(" .,\n")
        for s in sorted(US_STATES | COUNTRIES, key=len, reverse=True):
            if s.lower() in cand.lower():
                return s
        return cand
    # fallback: NER (scan is cheap; small model)
    doc = nlp()(text[:4000])
    for ent in doc.ents:
        if ent.label_ in ("GPE","LOC") and ent.text in (US_STATES | COUNTRIES):
            return ent.text
    return None
