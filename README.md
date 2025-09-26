# AIDocumentIntelligence

Explainable, policy-driven document intelligence for contracts and business docs.
Deterministic rules (no black-box ML) extract key fields, validate against policy, score risk, and produce decision-ready summaries and exports.

# Problem Statement
Businesses run on documents—contracts, financial reports, compliance forms, and invoices. A single large company can manage thousands of active contracts at any given time. Manually reviewing these documents to extract critical data, check for consistency, and assess risk is a monumental task. It's slow, expensive, and dangerously prone to human error. A missed clause in a single contract could cost millions.

Ingests a collection of complex documents, and transforms them into actionable insights and structured data.

# Pipeline

## Extractor Agent
Input - Folder of PDFs + User Defined Schema  

Task - Accurately locate specific clauses and terms 

## Validator Agent
Input - Extracted Data from Extractor Agent 

Task - Standardize dates, currencies and terminology

Compare key data points across documents to flag inconsistencies

Identify missing clauses and ambiguous language

## Analyst Agent
Input - Validated data from Validator Agent
Tasks - Assign risk score, summarize and identify trends

## Reported Agent
Input - Insights and Analysis from Analyst Agent

Outputs - Structured CSV, Markdown Report with findings, risks and flagged inconsistencies

# Repo Layout
```
AIDOC/
|-- .venv/
|-- config/
|-- full_contract_pdf/
|   |-- Part_I/
|   |-- Part_II/
|   `-- Part_III/
|-- full_contract_txt/
|-- full_contract2_txt/
|-- outputs/
|-- analyze/
|-- extract/
|-- report/
|-- reports/
|-- validate/
|-- src/
|   |-- __pycache__/
|   |-- analyst.py
|   |-- extractor_nlp.py
|   |-- extractor_utils.py
|   |-- extractor.py
|   |-- ingest.py
|   |-- lc_extractor.py
|   |-- reporter.py
|   `-- validator.py
|-- .env
|-- .gitignore
|-- .lc_cache.db
|-- app.py
|-- README.md
|-- requirements.txt
`-- test_openai_key.py

```

# How to run this
# clone & enter

```
git clone https://github.com/HarshaB45/AIDocumentIntelligence
cd AIDOC
```
# create & activate venv
```
python -m venv .venv
```
# Windows
```
.venv\Scripts\activate
```
# macOS/Linux
```
source .venv/bin/activate
```

# Libraries Used

Streamlit — simple web UI (tabs, upload, tables, charts)

LangChain — light orchestration for the Extractor → Validator → Analyst → Reporter flow

Pandas — tables, KPIs, CSV/JSON exports

NumPy — quick stats (median, P90) and scoring math

RapidFuzz — fuzzy matching for party names / clauses

spaCy — lightweight NER/linguistic helpers

python-dotenv — load config/secrets from .env
# Results & KPIs (example placeholders)

Risk buckets: Low / Medium counts; outlier docs queued for review

Average risk: mean on 0–10 scale

Net Days: Median / P90 (e.g., 30 / 30 ⇒ strong Net-30 adherence)

Governing law: Top jurisdictions (e.g., Delaware, CA, NY)

Next actions: review outliers; expand rule pack (liability cap, auto-renew, notice windows)

Replace with your latest run’s numbers when you present.

# Security & privacy

Deterministic, rule-based pipeline; no external LLM calls required

Local processing; optional on-prem / air-gapped deployment

Audit trail: each flag includes rule ID and trigger context

# Roadmap

More rule packs (liability caps, auto-renew windows, indemnity/insurance thresholds)

Queryable knowledge base (ask questions over the normalized corpus)

Schema auto-suggest from document sampling

Connectors (SharePoint/Box/Drive, CLM export hooks)

On-prem bundle (Docker + SSO/SIEM hooks)

# Troubleshooting

Odd dates/currency: check locale and normalization regex/maps

Too many flags: calibrate thresholds; add “approved exceptions” list

Performance: limit batch size; avoid heavy OCR unless needed


