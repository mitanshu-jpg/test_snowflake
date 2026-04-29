"""
IDSA – Intelligent Data Strategy Advisor
==========================================
Streamlit in Snowflake application.

Deploy this file as a Streamlit app inside your Snowflake account.
The app uses:
  • Snowflake Cortex Search  – semantic context retrieval
  • Gemini 1.5 Pro API       – advanced LLM reasoning
  • Cortex COMPLETE (fallback)– mistral-large when Gemini unavailable
  • FPDF2                     – PDF report generation (installed via packages.txt)

packages.txt (Streamlit in Snowflake):
    fpdf2
    snowflake-ml-python
    requests
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import json
import uuid
import textwrap
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
import streamlit as st
import pandas as pd
from fpdf import FPDF

# ── Snowflake ─────────────────────────────────────────────────────────────────
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete as CortexComplete
import snowflake.snowpark.functions as F

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE        = "Intelligent Data Strategy Advisor"
CORTEX_MODEL     = "mistral-large"
GEMINI_MODEL     = "gemini-1.5-pro"
SEARCH_SERVICE   = "IDSA_DB.IDSA_SCHEMA.IDSA_SEARCH_SERVICE"
SEARCH_LIMIT     = 8          # top-k context chunks
MAX_CTX_CHARS    = 6000       # truncate context fed to LLM
SESSION_TABLE    = "IDSA_DB.IDSA_SCHEMA.USER_SESSIONS"

DATA_SOURCE_OPTIONS = [
    "Salesforce (CRM)",
    "MySQL / PostgreSQL",
    "SQL Server",
    "Oracle DB",
    "MongoDB",
    "Website Clickstream / Logs",
    "REST APIs",
    "CSV / Excel Files",
    "AWS S3",
    "Azure Blob Storage",
    "Google Cloud Storage",
    "Google Analytics",
    "SAP ERP",
    "HubSpot",
    "Stripe / Payment Systems",
    "Custom / Other",
]

ETL_TOOLS = [
    "Fivetran", "Airbyte (OSS)", "AWS Glue",
    "Azure Data Factory", "dbt", "Informatica",
    "Talend", "Stitch", "Matillion",
    "Snowflake Snowpipe", "Python (custom)",
]


# ══════════════════════════════════════════════════════════════════════════════
# ── Session state bootstrap ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "session_id":   str(uuid.uuid4()),
        "messages":     [],          # [{role, content}]
        "user_config":  {},          # data sources, priorities, etc.
        "last_report":  "",
        "gemini_key":   "",
        "use_gemini":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# ── Snowflake helpers ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_session():
    """Return the active Snowpark session (cached)."""
    return get_active_session()


def cortex_search(query: str) -> str:
    """
    Query Cortex Search Service and return concatenated context chunks.
    Falls back gracefully if the service is unavailable.
    """
    session = get_session()
    try:
        result = (
            session.sql(f"""
                SELECT PARSE_JSON(
                    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                        '{SEARCH_SERVICE}',
                        OBJECT_CONSTRUCT(
                            'query',     '{query.replace("'", "''")}',
                            'columns',   ARRAY_CONSTRUCT('HEADING','CONTENT','SOURCE_SITE','CATEGORY'),
                            'limit',     {SEARCH_LIMIT}
                        )::VARCHAR
                    )
                ) AS RESULTS
            """)
            .collect()
        )
        if not result:
            return ""
        raw = result[0]["RESULTS"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        chunks = raw.get("results", [])
        parts = []
        for c in chunks:
            site = c.get("SOURCE_SITE", "")
            heading = c.get("HEADING", "")
            content = c.get("CONTENT", "")
            parts.append(f"[{site}] {heading}\n{content}")
        ctx = "\n\n---\n\n".join(parts)
        return ctx[:MAX_CTX_CHARS]
    except Exception as e:
        log.warning("Cortex Search failed: %s", e)
        return ""


def cortex_llm(prompt: str) -> str:
    """Call Snowflake Cortex COMPLETE as fallback LLM."""
    try:
        result = CortexComplete(CORTEX_MODEL, prompt)
        return result.strip()
    except Exception as e:
        return f"[Cortex LLM error: {e}]"


def save_session(session_id: str, config: dict, messages: list, report: str):
    """Persist session data to Snowflake."""
    session = get_session()
    try:
        session.sql(f"""
            MERGE INTO {SESSION_TABLE} AS tgt
            USING (SELECT '{session_id}' AS SESSION_ID) AS src
            ON tgt.SESSION_ID = src.SESSION_ID
            WHEN MATCHED THEN UPDATE SET
                USER_INPUTS  = PARSE_JSON('{json.dumps(config).replace("'","''")}'),
                CHAT_HISTORY = PARSE_JSON('{json.dumps(messages).replace("'","''")}'),
                LAST_REPORT  = '{report.replace("'","''")}',
                UPDATED_AT   = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
                (SESSION_ID, USER_INPUTS, CHAT_HISTORY, LAST_REPORT)
            VALUES
                ('{session_id}',
                 PARSE_JSON('{json.dumps(config).replace("'","''")}'),
                 PARSE_JSON('{json.dumps(messages).replace("'","''")}'),
                 '{report.replace("'","''")}')
        """).collect()
    except Exception as e:
        log.warning("Session save failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# ── Gemini API ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def gemini_generate(prompt: str, api_key: str) -> str:
    """Call Google Gemini 1.5 Pro via REST API."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            return candidates[0]["content"]["parts"][0]["text"].strip()
        return "[Gemini returned no candidates]"
    except requests.RequestException as e:
        return f"[Gemini API error: {e}]"


def generate_response(user_query: str, config: dict, api_key: str, use_gemini: bool) -> str:
    """
    Full RAG pipeline:
      1. Retrieve context from Cortex Search
      2. Optionally fetch real-time pricing snippets
      3. Build structured prompt
      4. Call Gemini or Cortex LLM
    """
    # Step 1: Retrieve context
    ctx = cortex_search(user_query)

    # Step 2: Summarise user configuration
    sources_str = ", ".join(config.get("sources", ["Not specified"]))
    volume_str  = config.get("data_volume", "Not specified")
    latency_str = config.get("latency", "Not specified")
    budget_str  = config.get("budget", "Not specified")

    # Step 3: Build prompt
    prompt = f"""You are the Intelligent Data Strategy Advisor (IDSA), a senior data engineering expert.
Your role is to recommend ETL/ELT pipelines, tools, and architecture for moving data into Snowflake.

## User Profile
- Data Sources: {sources_str}
- Data Volume: {volume_str}
- Latency Requirement: {latency_str}
- Budget Tier: {budget_str}

## Retrieved Knowledge Base Context
{ctx if ctx else "(No additional context retrieved; rely on your expertise)"}

## User Question
{user_query}

## Instructions
Provide a structured, expert recommendation addressing the user's question.
Cover: tool recommendations, architecture design, connector mapping, pricing insights,
and cost optimization tips. Be specific and actionable. Use clear sections.
If asked for a full pipeline design, include a step-by-step data flow.
Always mention trade-offs between recommended options.
Format with markdown headers and bullet points for clarity.
"""

    if use_gemini and api_key:
        return gemini_generate(prompt, api_key)
    else:
        return cortex_llm(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# ── PDF Report Generator ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class IDSAReportPDF(FPDF):
    """Custom FPDF subclass for IDSA branded reports."""

    BRAND_PRIMARY   = (13, 71, 161)    # deep blue
    BRAND_ACCENT    = (30, 136, 229)   # bright blue
    BRAND_DARK      = (21, 21, 21)
    BRAND_LIGHT     = (245, 248, 255)

    def header(self):
        self.set_fill_color(*self.BRAND_PRIMARY)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.cell(0, 10, "IDSA  –  Intelligent Data Strategy Advisor", align="C")
        self.ln(14)
        self.set_text_color(*self.BRAND_DARK)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        ts = datetime.now().strftime("%d %b %Y  %H:%M")
        self.cell(0, 8, f"Generated by IDSA  ·  {ts}  ·  Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_fill_color(*self.BRAND_ACCENT)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {title}", fill=True, ln=True)
        self.ln(2)
        self.set_text_color(*self.BRAND_DARK)

    def body_text(self, text: str, font_size: int = 10):
        self.set_font("Helvetica", size=font_size)
        # Wrap and write line by line (handles unicode via latin-1 fallback)
        for line in text.split("\n"):
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            self.multi_cell(0, 5, safe)
        self.ln(1)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(55, 6, key + ":", ln=False)
        self.set_font("Helvetica", size=10)
        safe_val = str(value).encode("latin-1", errors="replace").decode("latin-1")
        self.multi_cell(0, 6, safe_val)


def build_pdf(config: dict, messages: list, report_text: str) -> bytes:
    """Construct and return a PDF report as bytes."""
    pdf = IDSAReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Cover block ──────────────────────────────────────────────────────────
    pdf.set_fill_color(*IDSAReportPDF.BRAND_LIGHT)
    pdf.rect(10, 22, 190, 38, "F")
    pdf.set_xy(15, 25)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*IDSAReportPDF.BRAND_PRIMARY)
    pdf.cell(0, 10, "Data Pipeline Strategy Report", ln=True)
    pdf.set_xy(15, 36)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(80, 80, 80)
    ts = datetime.now().strftime("%A, %d %B %Y  %H:%M")
    pdf.cell(0, 6, f"Generated: {ts}", ln=True)
    pdf.set_xy(15, 44)
    pdf.cell(0, 6, f"Session ID: {st.session_state.get('session_id','N/A')}", ln=True)
    pdf.ln(22)

    # ── Section 1: User Configuration ────────────────────────────────────────
    pdf.section_title("1. User Configuration")
    pdf.kv_row("Data Sources",  ", ".join(config.get("sources", [])) or "Not specified")
    pdf.kv_row("Data Volume",   config.get("data_volume", "Not specified"))
    pdf.kv_row("Latency",       config.get("latency", "Not specified"))
    pdf.kv_row("Budget Tier",   config.get("budget", "Not specified"))
    pdf.kv_row("Preferred Tools", ", ".join(config.get("preferred_tools", [])) or "No preference")
    pdf.kv_row("Special Notes",  config.get("notes", "None"))
    pdf.ln(4)

    # ── Section 2: Conversation Summary ──────────────────────────────────────
    pdf.section_title("2. Conversation Summary")
    for msg in messages:
        role  = "You" if msg["role"] == "user" else "IDSA"
        label = f"[{role}]  "
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, label, ln=True)
        pdf.body_text(msg["content"], font_size=9)
        pdf.ln(1)

    # ── Section 3: Final Strategy Report ────────────────────────────────────
    if report_text:
        pdf.add_page()
        pdf.section_title("3. Final Strategy & Recommendations")
        # Strip markdown for cleaner PDF output
        clean = (
            report_text
            .replace("**", "")
            .replace("##", "")
            .replace("###", "")
            .replace("# ", "")
        )
        pdf.body_text(clean, font_size=10)

    # ── Section 4: Pricing Reference ────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("4. ETL Tool Pricing Reference (2024)")
    pricing_data = [
        ("Fivetran",           "MAR-based (Monthly Active Rows)",  "$1–$2 / 1K MAR",       "Managed, no-code"),
        ("Airbyte OSS",        "Self-hosted, free",                "$0 (infra cost only)",  "Open source, flexible"),
        ("Airbyte Cloud",      "Credit-based",                     "~$10 / credit",         "Managed cloud"),
        ("AWS Glue",           "DPU-hours",                        "$0.44 / DPU-hour",      "Serverless"),
        ("Azure Data Factory", "Pipeline runs + data movement",    "$1 / 1K runs",          "Azure-native"),
        ("dbt Cloud",          "Developer seats",                  "$50 / seat / month",    "Transformations only"),
        ("Informatica IICS",   "IPU credits",                      "Custom pricing",        "Enterprise"),
        ("Stitch",             "Rows replicated",                  "From $100 / month",     "Simple, fast setup"),
        ("Matillion",          "Instance + runtime",               "From $2 / hour",        "Cloud-native ETL"),
        ("Snowpipe",           "Credits + storage",                "~$2.50 / credit",       "Native to Snowflake"),
    ]
    pdf.set_font("Helvetica", "B", 8)
    col_w = [42, 46, 40, 52]
    for h, w in zip(["Tool", "Pricing Model", "Approx. Cost", "Best For"], col_w):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", size=8)
    fill = False
    for row in pricing_data:
        if fill:
            pdf.set_fill_color(235, 242, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        for val, w in zip(row, col_w):
            pdf.cell(w, 5, str(val), border=1, fill=True)
        pdf.ln()
        fill = not fill
    pdf.ln(4)

    # ── Section 5: Architecture Guidance ────────────────────────────────────
    pdf.section_title("5. Reference Architecture (Source → Snowflake)")
    arch_text = """
Recommended data flow for most enterprise setups:

  [Source Systems]  →  [Ingestion Layer]  →  [Raw Zone]  →  [Transform]  →  [Serving Layer]

  1. SOURCE SYSTEMS
     CRM, databases, files, cloud storage, APIs, web logs

  2. INGESTION LAYER (ETL/ELT Tool)
     Options: Fivetran / Airbyte / AWS Glue / Snowpipe
     Delivers raw data into Snowflake RAW schema

  3. SNOWFLAKE RAW ZONE
     Schema: RAW_{SOURCE_NAME}
     Tables: Immutable, append-only landing tables

  4. TRANSFORMATION LAYER (dbt)
     Staging models: clean & standardise
     Intermediate models: business logic
     Mart models: final aggregated tables

  5. SERVING LAYER
     BI tools: Tableau, Power BI, Looker
     APIs / operational apps
     Snowflake Cortex AI (analytics + LLM features)

  COST OPTIMISATION TIPS:
  • Cluster tables only when > 500 GB and query patterns benefit
  • Use Snowflake AUTO_SUSPEND (60s for dev, 300s for prod)
  • Prefer COPY INTO over Snowpipe for batch loads > 100 MB
  • Use RESULT_CACHE aggressively (auto-enabled)
  • Partition large tables by DATE column to prune micro-partitions
  • Use Snowflake's Resource Monitors to cap credit spend
"""
    pdf.body_text(arch_text, font_size=9)

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# ── UI: Sidebar ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar configuration form. Returns config dict."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/f/ff/Snowflake_Logo.svg",
            width=140,
        )
        st.markdown("### ⚙️ Configure Your Environment")

        sources = st.multiselect(
            "📦 Data Sources",
            options=DATA_SOURCE_OPTIONS,
            default=st.session_state.user_config.get("sources", []),
            help="Select all systems you want to move data FROM",
        )

        data_volume = st.selectbox(
            "📊 Data Volume / Day",
            ["< 1 GB", "1 – 10 GB", "10 – 100 GB", "100 GB – 1 TB", "> 1 TB"],
            index=["< 1 GB","1 – 10 GB","10 – 100 GB","100 GB – 1 TB","> 1 TB"].index(
                st.session_state.user_config.get("data_volume","1 – 10 GB")
            ) if st.session_state.user_config.get("data_volume") else 1,
        )

        latency = st.selectbox(
            "⏱️ Latency Requirement",
            ["Batch (daily)", "Near-real-time (hourly)", "Real-time (< 5 min)", "Streaming (< 30s)"],
            index=0,
        )

        budget = st.selectbox(
            "💰 Budget Tier",
            ["Startup (< $500/mo)", "SMB ($500–$2K/mo)", "Mid-market ($2K–$10K/mo)", "Enterprise ($10K+/mo)"],
            index=0,
        )

        preferred_tools = st.multiselect(
            "🔧 Preferred Tools (optional)",
            options=ETL_TOOLS,
            default=st.session_state.user_config.get("preferred_tools", []),
        )

        notes = st.text_area(
            "📝 Special Requirements",
            value=st.session_state.user_config.get("notes", ""),
            placeholder="HIPAA compliance, on-prem sources, real-time CDC...",
            height=90,
        )

        st.divider()
        st.markdown("### 🤖 LLM Settings")

        gemini_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=st.session_state.gemini_key,
            help="Leave blank to use Snowflake Cortex (mistral-large)",
        )
        use_gemini = bool(gemini_key)
        lbl = "✅ Gemini 1.5 Pro active" if use_gemini else "⚡ Using Cortex mistral-large"
        st.caption(lbl)
        st.session_state.gemini_key = gemini_key
        st.session_state.use_gemini = use_gemini

        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.last_report = ""
            st.rerun()

        config = {
            "sources":        sources,
            "data_volume":    data_volume,
            "latency":        latency,
            "budget":         budget,
            "preferred_tools": preferred_tools,
            "notes":          notes,
        }
        st.session_state.user_config = config
        return config


# ══════════════════════════════════════════════════════════════════════════════
# ── UI: Main page ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def render_chat(config: dict):
    """Render the main chat interface."""

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg,#0d47a1 0%,#1565c0 60%,#1976d2 100%);
            padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(13,71,161,0.35);
        ">
            <h1 style="color:#fff;margin:0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;">
                🧠 Intelligent Data Strategy Advisor
            </h1>
            <p style="color:#bbdefb;margin:8px 0 0;font-size:1rem;">
                Powered by Snowflake Cortex Search · Gemini 1.5 Pro · Real-time ETL Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick-start prompts ──────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        starter_cols = st.columns(3)
        starters = [
            ("🔗 Salesforce → Snowflake", "How do I move Salesforce CRM data to Snowflake? Compare Fivetran vs Airbyte."),
            ("🌐 Web Logs Pipeline",       "Design a pipeline to collect website click logs and load them into Snowflake in near-real-time."),
            ("💸 Cost Comparison",         "Compare pricing for Fivetran, Airbyte OSS and AWS Glue for a 10 GB/day workload."),
        ]
        for col, (label, query) in zip(starter_cols, starters):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.rerun()

    # ── Message history ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about ETL tools, pipelines, architecture, pricing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base & generating insights..."):
                response = generate_response(
                    user_query=prompt,
                    config=config,
                    api_key=st.session_state.gemini_key,
                    use_gemini=st.session_state.use_gemini,
                )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_report = response

            # Persist to Snowflake
            save_session(
                st.session_state.session_id,
                config,
                st.session_state.messages,
                response,
            )


def render_report_panel(config: dict):
    """Render the report generation and download panel."""
    with st.expander("📄 Generate & Download Full Report (PDF)", expanded=False):
        st.markdown(
            "Export your entire conversation, configuration, recommendations, "
            "pricing tables, and architecture guide as a branded PDF report."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.session_state.last_report:
                st.success("✅ Latest IDSA recommendation ready for export")
                st.markdown(st.session_state.last_report[:800] + "...")
            else:
                st.info("Ask a question first to generate recommendations.")

        with col2:
            if st.session_state.messages:
                if st.button("📥 Build PDF Report", type="primary", use_container_width=True):
                    with st.spinner("Building PDF..."):
                        pdf_bytes = build_pdf(
                            config=config,
                            messages=st.session_state.messages,
                            report_text=st.session_state.last_report,
                        )
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"IDSA_Report_{datetime.now():%Y%m%d_%H%M}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )


def render_metrics():
    """Render live knowledge base stats."""
    try:
        session = get_session()
        result = session.sql(
            "SELECT COUNT(*) AS CNT, COUNT(DISTINCT SOURCE_SITE) AS SITES "
            "FROM IDSA_DB.IDSA_SCHEMA.RAW_ETL_KNOWLEDGE"
        ).collect()
        cnt   = result[0]["CNT"]
        sites = result[0]["SITES"]
    except Exception:
        cnt, sites = "–", "–"

    c1, c2, c3 = st.columns(3)
    c1.metric("📚 Knowledge Chunks", cnt)
    c2.metric("🌐 Data Sources Indexed", sites)
    c3.metric("🤖 LLM Engine", "Gemini 1.5 Pro" if st.session_state.use_gemini else "Cortex mistral-large")


# ══════════════════════════════════════════════════════════════════════════════
# ── Entry point ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject minimal CSS refinements
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] { border-radius:10px; padding:6px 0; }
        [data-testid="stSidebar"]     { background:#f0f4ff; }
        .stButton > button            { border-radius:8px; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session()
    config = render_sidebar()

    render_metrics()
    st.divider()
    render_chat(config)
    render_report_panel(config)


if __name__ == "__main__":
    main()
