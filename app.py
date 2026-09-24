"""
app.py - OpsPulse AI Streamlit Application
SRE Log Triage & Automated Root Cause Analysis
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

# Import local modules
import rag_store
import agent_engine

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="OpsPulse AI - SRE Incident Triage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #0e1726;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f3f4f6;
        word-wrap: break-word;
    }
    .severity-badge-critical {
        background-color: #ef4444;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-badge-high {
        background-color: #f97316;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-badge-medium {
        background-color: #eab308;
        color: #111827;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .severity-badge-low {
        background-color: #10b981;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .service-tag {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .runbook-card {
        border-left: 4px solid #38bdf8;
        background: #0f172a;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Default Sample Log Presets
SAMPLE_LOGS = {
    "PostgreSQL Pool Saturation + Ingress 504 (Full Cascade)": """2026-09-25T00:52:11.104Z [ingress-nginx] 10.244.2.1 - - [25/Sep/2026:00:52:11 +0000] "POST /api/v2/orders/checkout HTTP/1.1" 504 167 "-" "Mozilla/5.0" 60.002 [production-order-service-8080] 10.244.3.42:8080 0 60.001 504
2026-09-25T00:52:11.109Z [ingress-nginx] [error] 31#31: *89412 upstream timed out (110: Connection timed out) while reading response header from upstream, client: 198.51.100.24, server: api.production.company.com, request: "POST /api/v2/orders/checkout HTTP/1.1", upstream: "http://10.244.3.42:8080/api/v2/orders/checkout"
2026-09-25T00:52:12.330Z [order-service-7f49b846-9qk4z] WARN  c.z.h.p.HikariPool [pool-postgres-rw] - HikariPool-1 - Connection is not available, request timed out after 30005ms (active=98, idle=0, waiting=24).
2026-09-25T00:52:12.332Z [order-service-7f49b846-9qk4z] ERROR o.s.b.w.s.s.ErrorPageFilter - Forwarding to error page from request [/api/v2/orders/checkout] due to exception [org.springframework.dao.CannotAcquireLockException: Connection pool exhausted; nested exception is org.postgresql.util.PSQLException: FATAL: remaining connection slots are reserved for non-replication superuser connections]
2026-09-25T00:52:14.015Z [postgresql-primary-0] [3-1] LOG:  could not receive data from client: Connection reset by peer
2026-09-25T00:52:14.020Z [postgresql-primary-0] [4-1] FATAL:  remaining connection slots are reserved for non-replication superuser connections
2026-09-25T00:52:14.022Z [auth-service-5c6d467-f23z9] ERROR c.c.a.AuthFilter - Database heartbeat failure: connection slot acquisition failure. Returning HTTP 500.""",

    "Kubernetes Ingress 504 Gateway Timeout": """2026-09-25T01:12:04.221Z [ingress-nginx] [error] 42#42: *109282 upstream timed out (110: Connection timed out) while reading response header from upstream, client: 203.0.113.88, server: api.production.company.com, request: "GET /api/v1/search?q=catalog HTTP/1.1", upstream: "http://10.244.1.88:8080/api/v1/search?q=catalog"
2026-09-25T01:12:05.100Z [search-service-5d6b8c9-j4k2l] WARN  o.e.t.TransportService - Worker threads saturated (pool_size=64, active=64, queue_depth=1200)
2026-09-25T01:12:06.410Z [ingress-nginx] 10.244.0.1 - - [25/Sep/2026:01:12:06 +0000] "GET /api/v1/search HTTP/1.1" 504 167 "-" "Mozilla/5.0" 60.005 [production-search-service-8080] 10.244.1.88:8080 0 60.005 504""",

    "PostgreSQL Thread Pool Exhaustion": """2026-09-25T01:05:01.002Z [postgresql-0] [postgres] [pid=41022] FATAL: remaining connection slots are reserved for non-replication superuser connections
2026-09-25T01:05:01.120Z [billing-service-9c7b-xk8m] ERROR o.h.e.j.s.SqlExceptionHelper - org.postgresql.util.PSQLException: FATAL: sorry, too many clients already
2026-09-25T01:05:02.040Z [postgresql-0] [postgres] [pid=38192] LOG: process 38192 still waiting for ExclusiveLock on relation 16422 of database 16384 after 120000.124 ms"""
}


# Initialize Session State
if "indexed" not in st.session_state:
    st.session_state.indexed = False
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.metadata = []

if "triage_results" not in st.session_state:
    st.session_state.triage_results = None


# Cached Vector Index Loader
@st.cache_resource(show_spinner=False)
def load_cached_index():
    runbooks_dir = os.path.join(os.path.dirname(__file__), "runbooks")
    return rag_store.build_runbook_index(runbooks_dir)


# Load Vector Store
try:
    if not st.session_state.indexed:
        idx, chs, meta = load_cached_index()
        st.session_state.index = idx
        st.session_state.chunks = chs
        st.session_state.metadata = meta
        st.session_state.indexed = True
except Exception as e:
    st.error(f"Error loading runbook vector index: {e}")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚡ OpsPulse AI")
    st.caption("Autonomous SRE Log Triage & Root Cause Analysis")
    st.divider()

    st.subheader("🔑 Groq LLM Settings")
    env_groq_key = os.getenv("GROQ_API_KEY", "")
    if env_groq_key == "your_groq_api_key_here":
        env_groq_key = ""

    user_groq_key = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        help="Enter your Groq API key (starts with gsk_). If blank, high-fidelity offline simulation mode runs."
    )

    selected_model = st.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="Defaults to Llama 3.3 70B Versatile"
    )

    # Active Mode Indicator
    groq_client = agent_engine.get_groq_client(user_groq_key)
    if groq_client:
        st.success("🟢 Groq Live API Connected", icon="✅")
    else:
        st.info("🟡 Simulation & Heuristic Mode Active\n(Provide Groq key for live LLM inference)", icon="ℹ️")

    st.divider()
    st.subheader("📚 Incident Runbooks Knowledge Base")
    st.write(f"**Indexed Chunks**: `{len(st.session_state.chunks)}` chunks")
    st.write(f"**Knowledge Source**: `sample_k8s_runbook.pdf`")
    
    if st.button("🔄 Re-index Runbooks", use_container_width=True):
        st.cache_resource.clear()
        idx, chs, meta = rag_store.build_runbook_index(os.path.join(os.path.dirname(__file__), "runbooks"))
        st.session_state.index = idx
        st.session_state.chunks = chs
        st.session_state.metadata = meta
        st.session_state.indexed = True
        st.toast("Runbooks successfully re-indexed!", icon="📖")

    st.divider()
    st.caption("OpsPulse AI • Powered by Groq & FAISS")


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown("## 🛡️ SRE Incident Diagnostic Console")
st.markdown("Analyze production logs, pinpoint cascading failures, and generate verified hotfixes with RAG-grounded multi-agent reasoning.")

# Presets Selector
col_preset, col_clear = st.columns([4, 1])
with col_preset:
    preset_choice = st.selectbox(
        "Load Sample Incident Log Preset:",
        options=list(SAMPLE_LOGS.keys()),
        index=0
    )
with col_clear:
    st.write("")
    st.write("")
    if st.button("🧹 Clear Input", use_container_width=True):
        st.session_state.raw_logs_input = ""
        st.rerun()

# Text Area for Logs
log_text = st.text_area(
    "Paste Raw System / Pod / Ingress Logs:",
    value=SAMPLE_LOGS[preset_choice],
    height=200,
    help="Paste raw unstructured log lines, stack traces, or Ingress access logs."
)

# Execution Action Button
triage_button = st.button("🚀 Triage & Diagnose Incident", type="primary", use_container_width=True)

# Run Agents Pipeline
if triage_button and log_text.strip():
    with st.status("⚡ OpsPulse Agents Executing...", expanded=True) as status:
        # Step 1: Log Parser Agent
        status.update(label="1/3 [Log Parser Agent] Extracting telemetry and error signature...", state="running")
        parsed = agent_engine.parse_logs(log_text, client=groq_client, model=selected_model)
        
        # Step 2: RAG Runbook Search
        status.update(label="2/3 [Runbook Vector Store] Querying FAISS for matched SRE procedures...", state="running")
        rag_query = f"{parsed.get('error_signature', '')} {parsed.get('summary', '')}"
        matched_runbooks = rag_store.search_runbooks(
            query=rag_query,
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            metadata=st.session_state.metadata,
            top_k=3
        )
        runbook_context = rag_store.format_runbook_context_for_prompt(matched_runbooks)

        # Step 3: Diagnostic Agent (RCA)
        status.update(label="3/3 [Diagnostic & Remediation Agents] Synthesizing RCA & Hotfix Script...", state="running")
        rca_report = agent_engine.diagnose_root_cause(
            parsed_logs=parsed,
            raw_logs=log_text,
            runbook_context=runbook_context,
            client=groq_client,
            model=selected_model
        )

        # Step 4: Remediation Agent
        remediation = agent_engine.generate_remediation(
            parsed_logs=parsed,
            rca_report=rca_report,
            runbook_context=runbook_context,
            client=groq_client,
            model=selected_model
        )

        # Save to session
        st.session_state.triage_results = {
            "parsed": parsed,
            "matched_runbooks": matched_runbooks,
            "rca_report": rca_report,
            "remediation": remediation
        }
        status.update(label="✅ Triage & RCA Complete!", state="complete", expanded=False)


# ==========================================
# METRICS & RESULTS DISPLAY
# ==========================================
if st.session_state.triage_results:
    results = st.session_state.triage_results
    parsed = results["parsed"]
    matched_runbooks = results["matched_runbooks"]
    rca_report = results["rca_report"]
    remediation = results["remediation"]

    st.markdown("### 📊 Incident Telemetry Cards")

    # Side-by-side metric cards
    c1, c2, c3 = st.columns([1, 1.2, 2.2])

    severity = parsed.get("severity", "UNKNOWN").upper()
    severity_class = f"severity-badge-{severity.lower()}" if severity.lower() in ("critical", "high", "medium", "low") else "severity-badge-high"

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Severity Level</div>
            <div class="metric-value"><span class="{severity_class}">{severity}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        services = parsed.get("affected_services", ["unknown"])
        service_tags_html = "".join([f'<span class="service-tag">{s}</span>' for s in services])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Affected Services</div>
            <div class="metric-value" style="padding-top: 4px;">{service_tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        signature = parsed.get("error_signature", "Unidentified Error")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Identified Error Signature</div>
            <div class="metric-value" style="font-size: 1.05rem; line-height: 1.4;">{signature}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Render Outputs in 3 Tabs
    tab_rca, tab_script, tab_runbooks = st.tabs([
        "🔍 Root Cause Analysis (RCA)",
        "🛠️ Generated Fix Script",
        "📖 Retrieved Runbooks"
    ])

    # Tab 1: RCA
    with tab_rca:
        st.markdown("#### 🔬 Detailed Root Cause Analysis")
        st.markdown(f"> **Incident Summary**: {parsed.get('summary', '')}")
        st.markdown(rca_report)

    # Tab 2: Generated Fix Script
    with tab_script:
        st.markdown("#### 🚀 Actionable Mitigation & Hotfix Script")
        st.markdown(remediation.get("instructions", ""))

        script_code = remediation.get("script", "")
        script_lang = remediation.get("script_lang", "bash")
        
        st.markdown(f"**Executable Hotfix Script (`{script_lang}`)**:")
        st.code(script_code, language=script_lang)

        # Download button
        file_ext = "sh" if script_lang == "bash" else "py"
        st.download_button(
            label=f"💾 Download Hotfix Script (hotfix.{file_ext})",
            data=script_code,
            file_name=f"hotfix.{file_ext}",
            mime="text/plain",
            use_container_width=True
        )

    # Tab 3: Retrieved Runbooks
    with tab_runbooks:
        st.markdown("#### 📚 Matched Runbook Procedures from Vector Store")
        if not matched_runbooks:
            st.info("No matching runbook chunks retrieved from vector store.")
        else:
            for item in matched_runbooks:
                relevance_pct = int(item["score"] * 100)
                st.markdown(f"""
                <div class="runbook-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: #38bdf8;">📄 {item['source']} (Page {item['page']})</span>
                        <span style="background: #1e293b; color: #a7f3d0; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600;">
                            Match Score: {relevance_pct}%
                        </span>
                    </div>
                    <div style="font-size: 0.92rem; color: #cbd5e1; white-space: pre-wrap; font-family: monospace;">{item['text']}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    # Initial state guidance
    st.info("👆 Click **'Triage & Diagnose Incident'** to parse logs, retrieve runbooks, and perform automated RCA.")
