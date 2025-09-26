# app.py
from __future__ import annotations
import json, sys, subprocess, time, re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent

# ---- Artifact paths ----
PER_DOC_CSV    = ROOT / "outputs" / "analyze" / "per_doc.csv"
FINDINGS_CSV   = ROOT / "outputs" / "report" / "findings.csv"
KPIS_JSON      = ROOT / "outputs" / "analyze" / "corpus_kpis.json"
REPORT_MD      = ROOT / "outputs" / "report" / "report.md"

# Optional scripts
INGEST_SCRIPT   = ROOT / "src" / "ingest_pdf_part1.py"
EXTRACT_SCRIPT  = ROOT / "src" / "extractor.py"
VALIDATE_SCRIPT = ROOT / "src" / "validator.py"
ANALYST_SCRIPT  = ROOT / "src" / "analyst.py"
REPORTER_SCRIPT = ROOT / "src" / "reporter.py"

st.set_page_config(page_title="AI Document Intelligence", layout="wide")

# ------------- Helpers -------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None

def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return ""

def run_step(pyfile: Path, label: str) -> tuple[bool,str,float]:
    """Run a Python script as a subprocess."""
    if not pyfile.exists():
        return False, f"Script not found: {pyfile}", 0.0
    cmd = [sys.executable, str(pyfile)]
    start = time.perf_counter()
    try:
        with st.spinner(f"Running {label}…"):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        dt = time.perf_counter() - start
        ok = proc.returncode == 0
        tail = (proc.stdout or "") + ("\n" + (proc.stderr or ""))
        return ok, tail.strip()[-4000:], dt
    except Exception as e:
        dt = time.perf_counter() - start
        return False, f"{e}", dt

def to_month(s: str):
    try:
        return pd.to_datetime(s, format="%Y-%m-%d").strftime("%Y-%m")
    except Exception:
        return None

# ------------- Sidebar: Pipeline Controls -------------
st.sidebar.title("Pipeline")
st.sidebar.caption("Run agents from here. Steps are idempotent (safe to re-run).")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    do_ingest = st.button("Ingest PDFs → TXT", use_container_width=True)
with col_b:
    do_extract = st.button("Run Extractor", use_container_width=True)

col_c, col_d = st.sidebar.columns(2)
with col_c:
    do_validate = st.button("Run Validator", use_container_width=True)
with col_d:
    do_analyst = st.button("Run Analyst", use_container_width=True)

col_e, col_f = st.sidebar.columns(2)
with col_e:
    do_report = st.button("Run Reporter", use_container_width=True)
with col_f:
    do_refresh = st.button("Reload Data", use_container_width=True)

log_box = st.sidebar.empty()
timing_box = st.sidebar.empty()

def show_result(ok: bool, name: str, log: str, dt: float):
    if ok:
        log_box.success(f"✓ {name} done in {dt:.2f}s")
    else:
        log_box.error(f"✗ {name} failed in {dt:.2f}s")
    with st.sidebar.expander(f"Logs: {name}", expanded=not ok):
        st.code(log or "(no output)")

# ------------- Execute pipeline steps -------------
if do_ingest:
    ok, out, dt = run_step(INGEST_SCRIPT, "Ingest PDFs")
    show_result(ok, "Ingest PDFs", out, dt)

if do_extract:
    ok, out, dt = run_step(EXTRACT_SCRIPT, "Extractor")
    show_result(ok, "Extractor", out, dt)

if do_validate:
    ok, out, dt = run_step(VALIDATE_SCRIPT, "Validator")
    show_result(ok, "Validator", out, dt)

if do_analyst:
    ok, out, dt = run_step(ANALYST_SCRIPT, "Analyst")
    show_result(ok, "Analyst", out, dt)

if do_report:
    ok, out, dt = run_step(REPORTER_SCRIPT, "Reporter")
    show_result(ok, "Reporter", out, dt)

# ------------- Data Loading -------------
@st.cache_data(show_spinner=False)
def load_all():
    df_docs = load_csv(PER_DOC_CSV)
    df_find = load_csv(FINDINGS_CSV)
    kpis = load_json(KPIS_JSON) or {}
    report_md = load_text(REPORT_MD)
    return df_docs, df_find, kpis, report_md

if do_refresh or any([do_ingest, do_extract, do_validate, do_analyst, do_report]):
    load_all.clear()

df_docs, df_find, kpis, report_md = load_all()

# ------------- Top KPIs -------------
st.title("AI Document Intelligence – Interactive Dashboard")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total docs", len(df_docs) if not df_docs.empty else 0)
m2.metric("Avg risk score", f"{(pd.to_numeric(df_docs.get('risk_score', pd.Series([])), errors='coerce').mean() if not df_docs.empty else 0):.2f}")
if kpis:
    m3.metric("Median net days", f"{kpis.get('net_days_median','—')}")
    rb = kpis.get("risk_bucket_counts", {})
    m4.metric("High-risk docs", rb.get("high", 0))
else:
    m3.metric("Median net days", "—")
    m4.metric("High-risk docs", 0)

st.divider()

# ------------- Filters -------------
st.subheader("Filters")
risk_opts  = sorted(df_docs["risk_bucket"].dropna().unique()) if not df_docs.empty and "risk_bucket" in df_docs else []
gl_opts    = sorted(df_docs["governing_law"].dropna().unique()) if not df_docs.empty and "governing_law" in df_docs else []
party_opts = sorted(df_docs["party_key"].dropna().unique()) if not df_docs.empty and "party_key" in df_docs else []

c1, c2, c3 = st.columns(3)
sel_risk  = c1.multiselect("Risk bucket", risk_opts, default=risk_opts)
sel_glaw  = c2.multiselect("Governing law", gl_opts, default=gl_opts)
sel_party = c3.multiselect("Party (normalized)", party_opts, default=party_opts)

# ------------- Net days range slider -------------
nd_min, nd_max = 0, 200
if not df_docs.empty and "payment_net_days" in df_docs.columns:
    nd_col = pd.to_numeric(df_docs["payment_net_days"], errors="coerce").dropna()
    if not nd_col.empty:
        nd_min, nd_max = int(nd_col.min()), int(nd_col.max())
        if nd_max - nd_min < 10:
            nd_max = nd_min + 10

nd_range = st.slider(
    "Net days range",
    min_value=nd_min,
    max_value=nd_max,
    value=(nd_min, nd_max),
    step=1
)

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if sel_risk:
        out = out[out["risk_bucket"].isin(sel_risk)]
    if sel_glaw:
        out = out[out["governing_law"].isin(sel_glaw)]
    if sel_party:
        out = out[out["party_key"].isin(sel_party)]
    if "payment_net_days" in out.columns:
        nd = pd.to_numeric(out["payment_net_days"], errors="coerce")
        out = out[(nd >= nd_range[0]) & (nd <= nd_range[1]) | nd.isna()]
    return out

df_filt = apply_filters(df_docs)

# ------------- Charts: Risk distribution -------------
st.subheader("Risk distribution")
if not df_filt.empty and "risk_bucket" in df_filt.columns:
    fig = px.histogram(df_filt, x="risk_bucket")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No risk_bucket column found or no data.")

st.divider()

# ------------- Top risky docs -------------
st.subheader("Top risky documents")
if not df_filt.empty:
    top_df = df_filt.sort_values("risk_score", ascending=False).head(100)
    st.dataframe(top_df[["doc_id","risk_score","risk_bucket","issues_count","payment_net_days","governing_law","party_key"]], use_container_width=True)
else:
    st.info("No documents match current filters.")

st.divider()

# ------------- Issues with quotes -------------
st.subheader("Flagged inconsistencies & quotes")
if not df_find.empty:
    docs_keep = set(df_filt["doc_id"]) if not df_filt.empty else set(df_docs["doc_id"])
    sub = df_find[df_find["doc_id"].isin(docs_keep)].copy()
    issue_types = sorted(sub["issue_type"].dropna().unique())
    chosen = st.multiselect("Issue types to show", issue_types, default=issue_types)
    if chosen:
        sub = sub[sub["issue_type"].isin(chosen)]
    st.dataframe(sub[["doc_id","issue_type","field","payment_net_days","payment_amount","quote","details"]], use_container_width=True)
else:
    st.info("No findings.csv found. Run reporter.")

st.divider()

# ------------- Full Markdown report -------------
st.subheader("Comprehensive report (Markdown)")
if REPORT_MD.exists():
    st.markdown(load_text(REPORT_MD))
else:
    st.info("Report not found. Run reporter.")

st.divider()

# ------------- Queryable Knowledge Base -------------
st.subheader("Queryable Knowledge Base")
query_input = st.text_input(
    "Ask a question about your contracts (e.g., 'Which contracts are governed by California law?')"
)

if query_input:
    if df_docs.empty:
        st.info("No document data loaded.")
    else:
        filtered = df_docs.copy()
        q_lower = query_input.lower()

        # Simple rule-based NL parsing
        if "california" in q_lower and "governed by" in q_lower:
            if "governing_law" in filtered.columns:
                filtered = filtered[filtered["governing_law"].str.contains("California", case=False, na=False)]
        elif "termination" in q_lower and "less than" in q_lower:
            match = re.search(r"less than (\d+)", q_lower)
            if match and "termination_notice_days" in filtered.columns:
                days = int(match.group(1))
                filtered = filtered[pd.to_numeric(filtered["termination_notice_days"], errors="coerce") < days]

        st.write(f"Showing {len(filtered)} matching contracts")
        st.dataframe(filtered.head(100), use_container_width=True)

st.divider()

# ------------- Schema Auto-Suggestion -------------
st.subheader("Schema Auto-Suggestion")
if st.button("Suggest Extraction Schema from Sample Docs"):
    if df_docs.empty:
        st.info("No documents loaded to suggest schema.")
    else:
        sample = df_docs.head(5)
        suggested_schema = {col: "text" for col in sample.columns}

        st.markdown("Suggested schema (you can edit):")
        edited_schema = {}
        for k, v in suggested_schema.items():
            edited_schema[k] = st.text_input(f"Field: {k}", value=v)

        if st.button("Approve Schema"):
            schema_path = ROOT / "outputs" / "analyze" / "approved_schema.json"
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            schema_path.write_text(json.dumps(edited_schema, indent=2), encoding="utf-8")
            st.success(f"Schema saved to {schema_path}")

st.divider()

# ------------- Downloads -------------
st.subheader("Download artifacts")
cA, cB, cC = st.columns(3)
with cA:
    if PER_DOC_CSV.exists():
        st.download_button("per_doc.csv", data=PER_DOC_CSV.read_bytes(), file_name="per_doc.csv", mime="text/csv")
with cB:
    if FINDINGS_CSV.exists():
        st.download_button("findings.csv", data=FINDINGS_CSV.read_bytes(), file_name="findings.csv", mime="text/csv")
with cC:
    if REPORT_MD.exists():
        st.download_button("report.md", data=REPORT_MD.read_bytes(), file_name="report.md", mime="text/markdown")
