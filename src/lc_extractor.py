from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class FieldExtraction(BaseModel):
    field: str = Field(..., description="Field name")
    text: str = Field(..., description="Extracted span text")
    start: int = Field(..., description="GLOBAL char start in full doc")
    end: int = Field(..., description="GLOBAL char end in full doc")
    confidence: float = Field(..., ge=0.0, le=1.0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a contract extractor. Return only the JSON object matching the schema."),
    ("user", """Schema:
- field: string
- text: string
- start: int  (GLOBAL char offset in full document)
- end: int    (GLOBAL char offset in full document)
- confidence: float in [0,1]

Field to extract: {field}

Definitions:
- governing_law: jurisdiction whose laws govern the agreement.
- payment_terms: invoicing, due dates, Net days, etc.
- termination_clause: termination rights, notice, for/without cause.

Global offset of this excerpt: {chunk_start}
Excerpt:
{chunk_text}

Rules:
- Return one JSON object ONLY.
- If not found in this excerpt, set confidence=0 and empty text.
""")
])

def lc_llm_extract_field(doc_id: str, field: str, chunk_start: int, chunk_text: str,
                         model_name: str = "gpt-4o-mini") -> Optional[dict]:
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = PROMPT | llm.with_structured_output(FieldExtraction)
    try:
        result: FieldExtraction = chain.invoke({
            "field": field,
            "chunk_start": chunk_start,
            "chunk_text": chunk_text
        })
        data = result.model_dump()
        # If the model returned local offsets by mistake, adjust to global
        if data["start"] < chunk_start or data["end"] < chunk_start:
            data["start"] = chunk_start + max(0, data["start"])
            data["end"]   = chunk_start + max(0, data["end"])
        return data
    except Exception:
        return None
