"""
Streamlit UI for AI Query System
Quick chat interface with lineage visibility
"""

import streamlit as st
import sys
import json
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main_pipeline import AIQuerySystem
from layers.layer6_storyteller import QueryResponse, LineageTrace
from document_processor import classify_file
from datetime import datetime, timezone
import tempfile
import os

from dotenv import load_dotenv
load_dotenv()

import pymongo
import bcrypt


@st.cache_resource
def get_db():
    mongo_uri = os.environ.get("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client["nexus_intelligence"]

db = get_db()
users_collection = db["users"]
chats_collection = db["chat_histories"]

# Page configuration
st.set_page_config(
    page_title="Nexus Intelligence",
    page_icon=":material/database:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_chat_sessions():
    """Serialize, OPTIMIZE, and save chat sessions to MongoDB."""
    user_email = st.session_state.get("user_email")
    if not user_email:
        return

    MAX_SESSIONS = 10
    try:
        session_keys = list(st.session_state.chat_sessions.keys())
        if len(session_keys) > MAX_SESSIONS:
            for old_key in session_keys[:-MAX_SESSIONS]:
                del st.session_state.chat_sessions[old_key]

        serializable_sessions = {}
        for session_id, messages in st.session_state.chat_sessions.items():
            serializable_messages = []
            for msg in messages:
                ser_msg = {"role": msg["role"], "content": msg["content"]}
                if "feedback" in msg:
                    ser_msg["feedback"] = msg["feedback"]
                if "raw_docs" in msg:
                    ser_msg["raw_docs"] = [{"id": "System optimized: Context hidden in history", "content": "..."}]
                if "lineage" in msg and msg["lineage"]:
                    lin = msg["lineage"]
                    if hasattr(lin, 'to_dict'): lin_dict = lin.to_dict()
                    else: lin_dict = lin
                    ser_msg["lineage"] = {
                        "query": lin_dict.get("query", ""),
                        "route": lin_dict.get("route", ""),
                        "sql_run": lin_dict.get("sql_run", ""),
                        "cache_hit": lin_dict.get("cache_hit", False),
                        "execution_time_ms": lin_dict.get("execution_time_ms", 0)
                    }
                serializable_messages.append(ser_msg)
            serializable_sessions[session_id] = serializable_messages

        chats_collection.update_one(
            {"email": user_email},
            {"$set": {
                "chat_sessions": serializable_sessions,
                "session_counter": st.session_state.session_counter
            }},
            upsert=True
        )
    except Exception as e:
        st.error(f"Background Save failed: {e}")

def load_chat_sessions():
    user_email = st.session_state.get("user_email")
    if not user_email:
        _reset_local_session()
        return

    try:
        user_data = chats_collection.find_one({"email": user_email})
        if user_data and "chat_sessions" in user_data:
            sessions = user_data["chat_sessions"]
            st.session_state.session_counter = user_data.get("session_counter", 1)

            for session_id, messages in sessions.items():
                for msg in messages:
                    if "lineage" in msg and msg["lineage"]:
                        lin_data = msg["lineage"]
                        msg["lineage"] = LineageTrace(
                            query=lin_data.get("query", ""),
                            route=lin_data.get("route", ""),
                            sql_run=lin_data.get("sql_run", None),
                            tables_used=[],
                            schemas_retrieved=[],
                            documents_retrieved=[],
                            cache_hit=lin_data.get("cache_hit", False),
                            cache_similarity=None,
                            execution_time_ms=lin_data.get("execution_time_ms", 0),
                            timestamp=""
                        )

            st.session_state.chat_sessions = sessions
            if sessions:
                st.session_state.current_session_id = list(sessions.keys())[-1]
            else:
                _reset_local_session()
        else:
            _reset_local_session()
    except Exception as e:
        st.error(f"Failed to load chat sessions: {e}")
        _reset_local_session()

def _reset_local_session():
    st.session_state.chat_sessions = {"Session 1": []}
    st.session_state.current_session_id = "Session 1"
    st.session_state.session_counter = 1

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position:  200% 0; }
        }

        .stApp {
            background: #f8fafc !important;
            color: #0f172a !important;
        }

        [data-testid="stHeader"], header[data-testid="stHeader"] {
            background-color: #f8fafc !important;
        }
        .stApp p, .stApp label, .stMarkdown p,
        .stTextInput label, .stTextArea label,
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', system-ui, sans-serif !important;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 4rem !important;
            max-width: 860px !important;
        }

        h1 {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.6px !important;
            background: linear-gradient(135deg, #0f172a 0%, #334155 60%, #64748b 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-align: center !important;
            margin-bottom: 0 !important;
        }

        [data-testid="stSidebar"] {
            background: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] header {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            background: none !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #475569 !important;
            font-size: 0.85rem !important;
        }

        [data-testid="stSidebar"] button[kind="tertiary"] {
            justify-content: flex-start !important;
            padding: 0.45rem 0.6rem !important;
            font-size: 0.84rem !important;
            color: #475569 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            border: none !important;
        }
        [data-testid="stSidebar"] button[kind="tertiary"]:hover {
            color: #0f172a !important;
            background: rgba(15,23,42,0.07) !important;
            transform: translateX(2px) !important;
        }

        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] p,
        [data-testid="stSidebar"] button[kind="primary"] span {
            background: #0f172a !important;
            border: none !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            font-size: 0.88rem !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background: #1e293b !important;
            box-shadow: 0 4px 12px rgba(15,23,42,0.2) !important;
        }

        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] svg {
            display: none !important;
        }
        [data-testid="stSidebarCollapsedControl"] button::before,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::before {
            content: "☰";
            font-size: 1.2rem;
            color: #334155;
        }

        [data-testid="stChatMessage"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.6rem !important;
            animation: fadeInUp 0.3s ease-out !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
            transition: box-shadow 0.2s ease !important;
        }
        [data-testid="stChatMessage"]:hover {
            box-shadow: 0 3px 10px rgba(0,0,0,0.07) !important;
        }
        [data-testid="stChatMessage"] p {
            color: #1e293b !important;
            font-size: 0.94rem !important;
            line-height: 1.8 !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #0f172a !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        .stButton > button[kind="primary"] {
            background: #0f172a !important;
            border: none !important;
            color: #f8fafc !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            padding: 0.5rem 1.2rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #1e293b !important;
            box-shadow: 0 4px 12px rgba(15,23,42,0.2) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="secondary"],
        .stButton > button:not([kind]) {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: #334155 !important;
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="secondary"]:hover,
        .stButton > button:not([kind]):hover {
            border-color: #94a3b8 !important;
            background: #f8fafc !important;
            transform: translateY(-1px) !important;
        }

        div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, .stTextInput input, .stTextArea textarea {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: #0f172a !important;
            border-radius: 10px !important;
            font-size: 0.9rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-baseweb="input"]:focus-within > div, div[data-baseweb="textarea"]:focus-within > div, .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #334155 !important;
            box-shadow: 0 0 0 3px rgba(51,65,85,0.08) !important;
            background-color: #ffffff !important;
        }
        .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label, div[data-testid="stFileUploader"] label {
            color: #475569 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }

        [data-testid="stChatInput"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        }

        div[data-testid="stPopover"] > button {
            border-radius: 14px !important;
            border: 1px solid #e2e8f0 !important;
            background-color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            height: 48px !important;
            padding: 0 !important;
            color: #334155 !important;
            transition: border-color 0.2s ease !important;
        }
        div[data-testid="stPopover"] > button:hover {
            border-color: #334155 !important;
        }
        div[data-testid="stPopover"] > button svg[data-testid="stIconMaterial"]:last-of-type,
        div[data-testid="stPopover"] > button div > svg {
            display: none !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #334155 !important;
            box-shadow: 0 0 0 3px rgba(51,65,85,0.08) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #0f172a !important;
            font-size: 0.93rem !important;
        }

        .streamlit-expanderHeader {
            background: #f8fafc !important;
            border-radius: 10px !important;
            color: #475569 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .streamlit-expanderHeader:hover {
            color: #0f172a !important;
            background: #f1f5f9 !important;
        }

        code, .stCode, pre {
            font-family: 'JetBrains Mono', monospace !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            font-size: 0.82rem !important;
            color: #1e293b !important;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 0.83rem !important;
            font-weight: 500 !important;
            color: #64748b !important;
            padding: 0.4rem 0.8rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: #0f172a !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background: #0f172a !important;
            height: 2px !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            background: #e2e8f0 !important;
        }

        .stAlert {
            border-radius: 10px !important;
            font-size: 0.88rem !important;
            border-width: 1px !important;
        }

        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #cbd5e1 !important;
            border-radius: 12px !important;
            background: #f8fafc !important;
            transition: border-color 0.2s ease !important;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #94a3b8 !important;
        }

        hr { border-color: #e2e8f0 !important; margin: 0.75rem 0 !important; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        .stMultiSelect, .stSelectbox { font-size: 0.88rem !important; }
        [data-testid="stPopover"] summary > svg:last-of-type { display: none !important; }

        [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] > div {
            transition: transform 0.2s ease !important;
        }
        </style>
    """, unsafe_allow_html=True)


def chunk_text(text: str, chunk_size: int = 300) -> list:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def render_loading_screen():
    loading_html = """
    <div style="
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 70vh; animation: fadeInUp 0.5s ease-out;
        font-family: 'Inter', sans-serif;
    ">
        <div style="
            width: 48px; height: 48px; border-radius: 50%;
            border: 3px solid #e2e8f0; border-top-color: #334155;
            animation: spin 0.8s linear infinite;
            margin-bottom: 2rem;
        "></div>
        <h2 style="
            font-size: 1.5rem; font-weight: 600; color: #1e293b;
            margin-bottom: 0.5rem; letter-spacing: -0.3px;
        ">Nexus Intelligence</h2>
        <p style="color: #64748b; font-size: 0.92rem; margin-bottom: 1.5rem;">Initializing AI Query Engine</p>
        <div style="
            width: 220px; height: 4px; background: #e2e8f0; border-radius: 4px; overflow: hidden;
        ">
            <div style="
                width: 100%; height: 100%;
                background: linear-gradient(90deg, transparent, #475569, transparent);
                background-size: 200% 100%;
                animation: shimmer 1.5s infinite;
            "></div>
        </div>
        <p style="color: #94a3b8; font-size: 0.75rem; margin-top: 1.2rem; letter-spacing: 0.5px;">
            Connecting to databases, loading models & cache layers
        </p>
    </div>
    <style>
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
    </style>
    """
    st.markdown(loading_html, unsafe_allow_html=True)


def initialize_session_state():
    if "query_system" not in st.session_state:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            render_loading_screen()
        try:
            st.session_state.query_system = AIQuerySystem(load_sample_schemas=False)
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.session_state.query_system = None
        loading_placeholder.empty()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "chat_sessions" not in st.session_state:
        load_chat_sessions()
        st.session_state.active_filters = [st.session_state.current_session_id]

    if "target_source" not in st.session_state:
        st.session_state.target_source = None

    st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_session_id]


def display_lineage(lineage):
    import json
    with st.expander("Developer Trace & SQL Details", icon=":material/data_object:", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "SQL Executed", "RAG Sources", "Raw JSON"])

        with tab1:
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Route", lineage.route.upper() if lineage.route else "UNKNOWN")
            with colB:
                cache_status = "Hit" if lineage.cache_hit else "Miss"
                st.metric("Cache Status", cache_status)
            with colC:
                st.metric("Execution Time", f"{lineage.execution_time_ms:.0f} ms")
            if lineage.cache_similarity:
                st.info(f"Query was resolved from semantic cache with {lineage.cache_similarity:.1%} confidence.", icon=":material/bolt:")

        with tab2:
            if lineage.sql_run:
                st.markdown("**Generated SQL:**")
                st.code(lineage.sql_run, language="sql")
            else:
                st.info("No SQL executed for this query.", icon=":material/info:")

        with tab3:
            st.markdown("### Retrieved Context")
            st.write(f"- **Tables Used:** {', '.join(lineage.tables_used) or 'None'}")
            st.write(f"- **Schemas Retrieved:** {', '.join(lineage.schemas_retrieved) or 'None'}")
            st.write(f"- **Documents Retrieved:** {', '.join(lineage.documents_retrieved) or 'None'}")

        with tab4:
            st.json(json.loads(lineage.to_json()))

def parse_and_add_documents(uploaded_files):
    if not st.session_state.query_system:
        st.error("System not initialized.", icon=":material/error:")
        return

    system = st.session_state.query_system
    results = {"structured": [], "unstructured": [], "failed": []}

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=Path(uploaded_file.name).stem + "_"
        ) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # result = system.upload_file(tmp_path, original_file_name=uploaded_file.name)
            
            result = system.upload_file(
                tmp_path,
                original_file_name=uploaded_file.name,
                user_email=st.session_state.get("user_email"),
                session_id=st.session_state.get("current_session_id"),
                upload_ts=datetime.now(timezone.utc).isoformat(),
            )
            result["file_name"] = uploaded_file.name

            if result["success"]:
                if result["file_type"] == "structured":
                    results["structured"].append(result)
                else:
                    results["unstructured"].append(result)
            else:
                results["failed"].append(result)
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass

    total = len(uploaded_files)
    success_count = len(results["structured"]) + len(results["unstructured"])
    total_unstructured_chunks = sum(int(r.get("chunk_count", 0)) for r in results["unstructured"])

    # Update MongoDB explicitly with new documents so visibility restricts properly
    user_email = st.session_state.get("user_email")
    if user_email:
        new_docs = []
        for r in results["unstructured"]: new_docs.append(str(r["file_name"]))
        for r in results["structured"]: new_docs.append(str(r["file_name"]))

        if new_docs:
            try:
                users_collection.update_one(
                    {"email": user_email},
                    {"$addToSet": {"documents": {"$each": new_docs}}}
                )
            except Exception as e:
                st.error(f"Failed to secure document access bindings: {e}")

    if success_count == total:
        st.toast(f"Successfully ingested {total} file(s)", icon=":material/check_circle:")
    elif success_count > 0:
        st.toast(f"{success_count}/{total} files ingested", icon=":material/warning:")
    else:
        st.toast("All files failed to ingest", icon=":material/error:")

    # Explicit ingestion confirmation so users can verify chunk indexing succeeded.
    if results["unstructured"]:
        per_file = []
        for r in results["unstructured"]:
            name = r.get("file_name", "unknown")
            chunks = int(r.get("chunk_count", 0))
            per_file.append(f"{name}: {chunks} chunks")
        st.success(
            f"RAG chunks indexed: {total_unstructured_chunks} total | " + " | ".join(per_file),
            icon=":material/dataset:",
        )

    if results["structured"]:
        per_table = []
        for r in results["structured"]:
            name = r.get("file_name", "unknown")
            rows = int(r.get("row_count", 0))
            per_table.append(f"{name}: {rows} rows")
        st.info(
            "Structured files loaded: " + " | ".join(per_table),
            icon=":material/table_chart:",
        )

    if results["failed"]:
        failed_summary = " | ".join(
            f"{r.get('file_name', 'unknown')}: {r.get('message', 'Failed')}"
            for r in results["failed"]
        )
        st.error(f"Ingest failures: {failed_summary}", icon=":material/error:")

    # Auto-shift context to the newest successfully ingested file.
    successful_files = [r.get("file_name") for r in (results["structured"] + results["unstructured"]) if r.get("file_name")]
    if successful_files:
        st.session_state.target_source = successful_files[-1]
        st.toast(f"Context switched to latest file: {successful_files[-1]}", icon=":material/description:")

def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="padding:0.3rem 0 1rem 0; border-bottom:1px solid #e2e8f0; margin-bottom:1rem;">
                <p style="font-size:0.7rem; font-weight:700; letter-spacing:2px;
                          text-transform:uppercase; color:#94a3b8; margin:0 0 0.1rem 0;">
                    Enterprise
                </p>
                <span style="font-family:'Inter',sans-serif; font-weight:700; font-size:1rem;
                             color:#0f172a; letter-spacing:-0.3px;">
                    Nexus Intelligence
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <p style="font-size:0.7rem; font-weight:600; letter-spacing:1.5px;
                      text-transform:uppercase; color:#94a3b8; margin:0 0 0.5rem 0.1rem;">
                Conversations
            </p>
        """, unsafe_allow_html=True)

        if st.button("New Chat", icon=":material/add:", type="primary", use_container_width=True):
            st.session_state.session_counter += 1
            new_id = f"Session {st.session_state.session_counter}"
            st.session_state.chat_sessions[new_id] = []
            st.session_state.current_session_id = new_id
            save_chat_sessions()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Context Filter")

        # Get user's authorized docs for the dropdown
        user_email = st.session_state.get("user_email")
        user_record = users_collection.find_one({"email": user_email}) if user_email else None
        user_docs = user_record.get("documents", []) if user_record else []

        # Clean up doc names for the UI, preserving upload/insertion order
        raw_docs = [d.get("file_name", "") if isinstance(d, dict) else str(d) for d in user_docs]
        clean_docs = []
        for doc_name in raw_docs:
            doc_name = str(doc_name).strip()
            if doc_name and doc_name != "unknown" and doc_name not in clean_docs:
                clean_docs.append(doc_name)

        options = ["Select a file"] + clean_docs
        current_target = st.session_state.get("target_source")
        if current_target in clean_docs:
            default_index = options.index(current_target)
        elif clean_docs:
            # If no explicit selection exists, default to newest known file.
            default_index = len(options) - 1
            st.session_state.target_source = clean_docs[-1]
        else:
            default_index = 0
            st.session_state.target_source = None

        selected_source = st.selectbox(
            "Target specific document:",
            options,
            index=default_index,
            help="Force the AI to only read from this specific file.",
            label_visibility="collapsed"
        )

        # Save to session state
        st.session_state.target_source = selected_source if selected_source != "Select a file" else None

        if not clean_docs:
            st.caption("No files available. Upload a file to enable querying.")
        st.divider()
        # ---------------------------------------------
        st.header("Chat History")

        for session_id in reversed(list(st.session_state.chat_sessions.keys())):
            title = session_id
            session_messages = st.session_state.chat_sessions[session_id]
            if len(session_messages) > 0 and session_messages[0]["role"] == "user":
                title_text = session_messages[0]["content"][:25] + ("..." if len(session_messages[0]["content"]) > 25 else "")
                title = f"{title_text}"

            icon = ":material/chat_bubble:" if session_id == st.session_state.current_session_id else ":material/chat_bubble_outline:"

            if st.button(title, key=f"btn_{session_id}", icon=icon, type="tertiary", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()

        st.markdown("<br>" * 3, unsafe_allow_html=True)

        with st.expander("Settings & Resources", icon=":material/settings:", expanded=False):
            st.caption("System Resources")
            if st.session_state.query_system:
                stats = st.session_state.query_system.get_stats()
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Cache", stats['cache_stats']['total_entries'])
                with c2: st.metric("Docs", stats['tag_collections']['documents'])
                with c3: st.metric("Schemas", stats['tag_collections']['schemas'])

                uploads = st.session_state.query_system.list_uploads()

                # Fetch only user owned documents for sidebar display visibility
                user_email = st.session_state.get("user_email")
                user_record = users_collection.find_one({"email": user_email}) if user_email else None
                user_docs = user_record.get("documents", []) if user_record else []

                normalized_user_docs = set()
                for item in user_docs:
                    if isinstance(item, dict):
                        name = str(item.get("file_name", "")).strip()
                    else:
                        name = str(item).strip()
                    if name:
                        normalized_user_docs.add(name)

                filtered_docs = [
                    d for d in uploads.get("documents", [])
                    if d.get("file_name", d["id"]) in normalized_user_docs
                ]

                if uploads["schemas"] or filtered_docs:
                    with st.expander("Loaded Data Sources", icon=":material/database:"):
                        if uploads["schemas"]:
                            st.caption("SQL Tables")
                            for s in uploads["schemas"]:
                                st.markdown(f"- `{s}`")
                        if filtered_docs:
                            st.caption("RAG Documents")
                            seen_files = set()
                            for d in filtered_docs:
                                fname = d.get("file_name", d["id"])
                                if fname not in seen_files:
                                    seen_files.add(fname)
                                    st.markdown(f"- {fname}")

                if st.button("Clear Cache", icon=":material/mop:", use_container_width=True):
                    count = st.session_state.query_system.cache.clear()
                    st.toast(f"Cleared {count} cache entries", icon=":material/check:")
                if st.button("Clear Current Chat", icon=":material/clear_all:", use_container_width=True):
                    st.session_state.chat_sessions[st.session_state.current_session_id] = []
                    save_chat_sessions()
                    st.rerun()

                st.divider()
                if st.button("Logout", icon=":material/logout:", use_container_width=True):
                    st.session_state.authenticated = False
                    st.session_state.user_email = None
                    st.session_state.user_name = None
                    st.session_state.chat_sessions = {}
                    st.session_state.messages = []
                    _reset_local_session()
                    st.rerun()
            else:
                st.warning("System Offline")

def render_auth_screen():
    st.markdown("""
        <div style="text-align: center; padding-top: 4rem; animation: fadeInUp 0.7s ease-out;">
            <div style="display: inline-block; padding: 0.6rem 1.4rem; border: 1px solid rgba(20,184,166,0.3); border-radius: 50px; margin-bottom: 1.5rem; font-size: 0.78rem; color: #2dd4bf; letter-spacing: 1.5px; text-transform: uppercase; font-family: 'Inter', sans-serif;">Secure Access Portal</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>Nexus Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6e7681; margin-bottom: 2.5rem; font-family: Inter, sans-serif; font-size: 1rem;'>Enterprise Knowledge & Semantic RAG Engine</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2.0, 1])
    with col2:
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Log In", "Sign Up"])

            with tab1:
                st.markdown("### Welcome Back")
                st.caption("Please authenticate to access your Nexus Data Warehouse.")
                login_email = st.text_input("Work Email", placeholder="name@company.com", key="login_email")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                st.write("")
                if st.button("Secure Login", icon=":material/login:", use_container_width=True, type="primary"):
                    if not login_email or not login_pass:
                        st.error("Please fill in both fields.")
                    else:
                        user = users_collection.find_one({"email": login_email})
                        if user and bcrypt.checkpw(login_pass.encode('utf-8'), user["password"]):
                            st.session_state.authenticated = True
                            st.session_state.user_email = login_email
                            st.session_state.user_name = user.get("name", "User")
                            load_chat_sessions()
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")

            with tab2:
                st.markdown("### Create Account")
                st.caption("Your data remains isolated under Enterprise standards.")
                signup_name = st.text_input("Full Name", placeholder="Jane Doe")
                signup_email = st.text_input("Work Email", placeholder="name@company.com", key="signup_email")
                signup_pass = st.text_input("Password", type="password", key="signup_pass")
                st.write("")
                if st.button("Register Account", icon=":material/person_add:", use_container_width=True, type="primary"):
                    if not signup_name or not signup_email or not signup_pass:
                        st.error("Please fill in all fields.")
                    elif users_collection.find_one({"email": signup_email}):
                        st.error("An account with this email already exists.")
                    elif len(signup_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        hashed_pw = bcrypt.hashpw(signup_pass.encode('utf-8'), bcrypt.gensalt())
                        users_collection.insert_one({
                            "name": signup_name,
                            "email": signup_email,
                            "password": hashed_pw
                        })
                        st.toast("Registration Successful! Logging you in...", icon=":material/check_circle:")
                        st.session_state.authenticated = True
                        st.session_state.user_email = signup_email
                        st.session_state.user_name = signup_name
                        load_chat_sessions()
                        st.rerun()

def render_welcome_screen():
    user_name = st.session_state.get("user_name", "")
    greeting = f"Hello, {user_name.split()[0]}." if user_name else "Hello."

    st.markdown(f"""
        <div style="text-align:center; margin-top:3.5rem; margin-bottom:2rem; animation: fadeInUp 0.5s ease-out;">
            <p style="font-size:0.75rem; font-weight:600; letter-spacing:2px; text-transform:uppercase;
                      color:#94a3b8; margin-bottom:0.5rem;">Nexus Intelligence</p>
            <h2 style="font-size:2rem; font-weight:700; color:#0f172a;
                       letter-spacing:-0.5px; margin-bottom:0.5rem;">{greeting}</h2>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Ask anything about your data — structured or unstructured.
            </p>
        </div>
    """, unsafe_allow_html=True)

    examples = [
        ("Customer Insights",   ":material/people:",      "How many customers do we have?"),
        ("Financial Overview",  ":material/payments:",    "What is the total generated revenue?"),
        ("Operations Update",   ":material/inventory_2:", "Show me the 5 most recent orders"),
    ]

    recent_searches, seen = [], set()
    base_queries = {e[2] for e in examples}
    for s_id, msgs in reversed(st.session_state.chat_sessions.items()):
        for m in reversed(msgs):
            if m["role"] == "user" and m["content"] not in seen and m["content"] not in base_queries:
                seen.add(m["content"])
                recent_searches.append((s_id, m["content"]))

    if len(recent_searches) >= 1:
        examples[0] = (f"From {recent_searches[0][0]}", ":material/history:", recent_searches[0][1])
    if len(recent_searches) >= 2:
        examples[1] = (f"From {recent_searches[1][0]}", ":material/history:", recent_searches[1][1])
    if len(recent_searches) >= 3:
        examples[2] = (f"From {recent_searches[2][0]}", ":material/history:", recent_searches[2][1])

    col1, col2, col3 = st.columns(3)

    # 1. ADD 'enumerate' to get a unique index 'i' for each item
    for i, ((name, icon, query), col) in enumerate(zip(examples, [col1, col2, col3])):
        with col:
            with st.container(border=True):
                st.markdown(f"""
                    <p style="font-size:0.75rem; font-weight:600; letter-spacing:1px;
                              text-transform:uppercase; color:#94a3b8; margin-bottom:0.2rem;">
                        {name}
                    </p>
                    <p style="font-size:0.88rem; color:#334155; margin:0; line-height:1.5;">
                        "{query[:55]}{'...' if len(query)>55 else ''}"
                    </p>
                """, unsafe_allow_html=True)
                st.write("")

                # 2. FIX: Use 'i' in the key so it is always 100% unique (ex_btn_0, ex_btn_1, etc.)
                if st.button("Run query", icon=icon, key=f"ex_btn_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    save_chat_sessions()
                    st.rerun()

# import streamlit.components.v1 as components

# def inject_mentions_js(schemas):
#     """Injects a highly customized JS popover for @mentions in st.chat_input."""
#     js_schemas = json.dumps(schemas)
#     js_code = f"""
#     <script>
#     (function() {{
#         const schemas_str = JSON.stringify({js_schemas});
#         const parentDoc = window.parent.document;

#         // Dynamically update schemas on the parent scope so closures stay fresh
#         parentDoc.mentionSchemas = JSON.parse(schemas_str);

#         function initMentionPopup() {{
#             let popup = parentDoc.getElementById("mention-popup");
#             if (!popup) {{
#                 popup = parentDoc.createElement("div");
#                 popup.id = "mention-popup";
#                 popup.style.cssText = "display: none; position: absolute; z-index: 999999; background: white; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-height: 200px; overflow-y: auto; min-width: 200px;";
#                 parentDoc.body.appendChild(popup);
#             }}

#             if (parentDoc.body.dataset.mentionsBound === "true") return;
#             parentDoc.body.dataset.mentionsBound = "true";

#             parentDoc.body.addEventListener('input', function(e) {{
#                 if (e.target.tagName !== 'TEXTAREA') return;

#                 let textarea = e.target;
#                 let val = textarea.value;

#                 let chatInputContainer = e.target.closest('div[data-testid="stChatInput"]');
#                 if (!chatInputContainer) {{
#                    chatInputContainer = textarea.parentElement;
#                 }}

#                 let cursorStart = textarea.selectionStart;
#                 let textBeforeCursor = val.substring(0, cursorStart);

#                 let lastAt = textBeforeCursor.lastIndexOf('@');
#                 if (lastAt !== -1) {{
#                     let searchStr = textBeforeCursor.substring(lastAt + 1);
#                     if (!searchStr.includes(' ')) {{
#                         let currentSchemas = parentDoc.mentionSchemas || [];
#                         let matches = currentSchemas.filter(s => s.toLowerCase().startsWith(searchStr.toLowerCase()));

#                         if (matches.length > 0) {{
#                             popup.innerHTML = "";
#                             matches.forEach(m => {{
#                                 let div = parentDoc.createElement("div");
#                                 div.innerText = "@" + m;
#                                 div.style.cssText = "padding: 8px 12px; cursor: pointer; font-family: Inter, sans-serif; font-size: 14px; color: #1e293b; border-bottom: 1px solid #f1f5f9; background: white;";
#                                 div.onmouseover = () => div.style.backgroundColor = "#f1f5f9";
#                                 div.onmouseout = () => div.style.backgroundColor = "white";
#                                 div.onclick = () => {{
#                                     let beforeAt = val.substring(0, lastAt);
#                                     let afterCursor = val.substring(cursorStart);
#                                     let newVal = beforeAt + "@" + m + " " + afterCursor;

#                                     const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
#                                     nativeInputValueSetter.call(textarea, newVal);
#                                     textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));

#                                     popup.style.display = "none";
#                                     textarea.focus();

#                                     setTimeout(() => {{
#                                         textarea.selectionStart = textarea.selectionEnd = beforeAt.length + m.length + 2;
#                                     }}, 20);
#                                 }};
#                                 popup.appendChild(div);
#                             }});

#                             let rect = chatInputContainer.getBoundingClientRect();
#                             if (rect.top === 0 && rect.left === 0) {{
#                                 rect = textarea.getBoundingClientRect();
#                             }}

#                             popup.style.left = Math.max(0, rect.left) + "px";
#                             let topPos = rect.top + parentDoc.defaultView.scrollY - Math.min(matches.length * 37, 200) - 10;
#                             popup.style.top = Math.max(0, topPos) + "px";
#                             popup.style.display = "block";
#                             return;
#                         }}
#                     }}
#                 }}
#                 popup.style.display = "none";
#             }});

#             parentDoc.addEventListener('click', function(e) {{
#                if (!popup.contains(e.target)) {{
#                    popup.style.display = "none";
#                }}
#             }});
#         }}

#         initMentionPopup();
#     }})();
#     </script>
#     """
#     components.html(js_code, height=0, width=0)

import time
import streamlit.components.v1 as components

def inject_mentions_js(schemas):
    """Injects a highly customized JS popover for @mentions in st.chat_input."""
    import json
    js_schemas = json.dumps(schemas)

    js_code = f"""
    <div data-timestamp="{time.time()}" style="display:none;"></div>
    <script>
    (function() {{
        const schemas_str = JSON.stringify({js_schemas});
        const parentDoc = window.parent.document;
        const parentWin = window.parent;

        parentWin.mentionSchemas = JSON.parse(schemas_str);

        if (parentWin.mentionsBound) return;
        parentWin.mentionsBound = true;

        let popup = parentDoc.getElementById("nexus-mention-popup");
        if (!popup) {{
            popup = parentDoc.createElement("div");
            popup.id = "nexus-mention-popup";
            // FIXED POSITIONING: Breaks out of Streamlit scroll containers
            popup.style.cssText = "display: none; position: fixed; z-index: 9999999; background: white; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 -4px 20px rgba(0,0,0,0.15); max-height: 250px; overflow-y: auto; min-width: 300px; padding: 4px;";
            parentDoc.body.appendChild(popup);
        }}

        parentDoc.body.addEventListener('input', function(e) {{
            if (e.target.tagName !== 'TEXTAREA') return;

            let textarea = e.target;
            let val = textarea.value;
            let cursorStart = textarea.selectionStart;
            let textBeforeCursor = val.substring(0, cursorStart);

            let lastAt = textBeforeCursor.lastIndexOf('@');
            if (lastAt !== -1) {{
                let searchStr = textBeforeCursor.substring(lastAt + 1);

                if (!searchStr.includes(' ')) {{
                    let currentSchemas = parentWin.mentionSchemas || [];

                    // CRITICAL FIX: Ensure 's' is actually a string before calling toLowerCase!
                    let matches = currentSchemas.filter(s => {{
                        if (typeof s !== 'string') return false;
                        return s.toLowerCase().includes(searchStr.toLowerCase());
                    }});

                    if (matches.length > 0) {{
                        popup.innerHTML = "";

                        let header = parentDoc.createElement("div");
                        header.innerText = "Select Context";
                        header.style.cssText = "padding: 6px 10px; font-size: 11px; font-weight: bold; color: #94a3b8; text-transform: uppercase;";
                        popup.appendChild(header);

                        matches.forEach(m => {{
                            let div = parentDoc.createElement("div");
                            div.innerText = "📄 " + m;
                            div.style.cssText = "padding: 10px 12px; cursor: pointer; font-family: Inter, sans-serif; font-size: 14px; color: #0f172a; border-radius: 6px; margin-bottom: 2px; transition: background 0.1s;";

                            div.onmouseover = () => div.style.backgroundColor = "#f1f5f9";
                            div.onmouseout = () => div.style.backgroundColor = "transparent";

                            div.onclick = () => {{
                                let beforeAt = val.substring(0, lastAt);
                                let afterCursor = val.substring(cursorStart);
                                let newVal = beforeAt + "@" + m + " " + afterCursor;

                                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
                                nativeInputValueSetter.call(textarea, newVal);
                                textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));

                                popup.style.display = "none";
                                textarea.focus();

                                setTimeout(() => {{
                                    textarea.selectionStart = textarea.selectionEnd = beforeAt.length + m.length + 2;
                                }}, 10);
                            }};
                            popup.appendChild(div);
                        }});

                        let chatInputContainer = textarea.closest('div[data-testid="stChatInput"]');
                        if (!chatInputContainer) chatInputContainer = textarea.parentElement;

                        let rect = chatInputContainer.getBoundingClientRect();
                        let bottomOffset = parentWin.innerHeight - rect.top + 5;

                        popup.style.bottom = bottomOffset + "px";
                        popup.style.left = rect.left + "px";
                        popup.style.width = Math.max(rect.width, 300) + "px";
                        popup.style.top = "auto";

                        popup.style.display = "block";
                        return;
                    }}
                }}
            }}
            popup.style.display = "none";
        }});

        parentDoc.addEventListener('click', function(e) {{
           if (popup && !popup.contains(e.target) && e.target.tagName !== 'TEXTAREA') {{
               popup.style.display = "none";
           }}
        }});
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

def main():
    """Main Streamlit application."""
    inject_custom_css()
    initialize_session_state()

    safe_schemas = []

    if st.session_state.get("authenticated", False) and st.session_state.get("query_system"):
        try:
            uploads = st.session_state.query_system.list_uploads()

            # 1. Safely extract core schemas as strings
            for s in uploads.get("schemas", []):
                if isinstance(s, str):
                    safe_schemas.append(s)

            # 2. Fetch strictly the files owned by THIS specific user
            user_email = st.session_state.get("user_email")
            user_record = users_collection.find_one({"email": user_email}) if user_email else None
            user_docs = user_record.get("documents", []) if user_record else []

            # 3. Aggressively sanitize MongoDB RAG documents
            for doc_item in user_docs:
                # If a legacy document was saved as a dictionary, extract the string name safely
                doc_name = doc_item.get("file_name", "") if isinstance(doc_item, dict) else str(doc_item)

                if doc_name and doc_name.strip() and doc_name != "unknown":
                    safe_schemas.append(doc_name)

                    # Safely create a stem (no extension) for typing speed
                    if "." in doc_name:
                        stem = doc_name.rsplit('.', 1)[0]
                        if stem:
                            safe_schemas.append(stem)

            # Deduplicate the final safe list
            safe_schemas = list(set(safe_schemas))

        except Exception as e:
            st.error(f"Failed to load mentions context safely: {e}")

    # Inject the UI observer with our clean string array
    inject_mentions_js(safe_schemas)

    # Gateway Security Intercept (UI Only Mock)
    if not st.session_state.get("authenticated", False):
        render_auth_screen()
        return

    # Authorized Content Below
    st.title("Nexus Intelligence")
    st.markdown("""
        <div style="text-align:center; padding:0.5rem 0 0.25rem 0;">
            <p style="font-size:0.7rem; font-weight:700; letter-spacing:2.5px;
                      text-transform:uppercase; color:#94a3b8; margin-bottom:0.3rem;">
                Enterprise · AI-Powered
            </p>
            <p style="font-size:0.82rem; color:#94a3b8; margin-top:0.2rem; letter-spacing:0.5px;">
                Natural language · Semantic search · SQL generation
            </p>
        </div>
        <hr style="margin:0.75rem 0 1rem 0; border-color:#e2e8f0;">
    """, unsafe_allow_html=True)

    render_sidebar()
    if len(st.session_state.messages) == 0:
        render_welcome_screen()
    else:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    st.write(message["content"])

                    fb = st.feedback("thumbs", key=f"feed_{st.session_state.current_session_id}_{i}")
                    if fb is not None and message.get("feedback") != fb:
                        message["feedback"] = fb
                        save_chat_sessions()

                    raw_docs = message.get("raw_docs")
                    if raw_docs:
                        with st.expander("📄 Sourced Knowledge Contexts", expanded=True):
                            for idx, doc in enumerate(raw_docs):
                                doc_id = doc.get("id", "Document snippet")
                                clean_id = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
                                st.markdown(f"**{clean_id}** (Context {idx+1})")
                                st.info(doc.get("content", "No content snippet"), icon=":material/format_quote:")

                    if "lineage" in message:
                        display_lineage(message["lineage"])
            else:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    st.write("")

    prompt = None
    input_col, attach_col = st.columns([1.2, 15], vertical_alignment="bottom")
    with input_col:
        with st.popover("", icon=":material/attach_file:", use_container_width=True):
            st.markdown("**Knowledge & Context Management**")
            st.caption("Documents uploaded here will ONLY be readable within this specific Chat Thread to prevent confusion.")

            uploaded_files = st.file_uploader(
                "Upload local files",
                accept_multiple_files=True,
                type=['csv', 'pdf', 'json', 'xlsx', 'xls', 'txt', 'docx', 'md'],
                label_visibility="collapsed",
                help="Structured (CSV, Excel, JSON) → SQL queryable | Unstructured (PDF, TXT, DOCX, MD) → RAG knowledge base"
            )
            if uploaded_files and st.button("Ingest Files", icon=":material/upload_file:", use_container_width=True):
                with st.spinner("Processing semantics..."):
                    parse_and_add_documents(uploaded_files)
                # CRUCIAL: Immediately rerun to refresh sidebar visibility and Mentions JS context
                st.rerun()

            st.divider()

            st.caption("Cross-Session Context Sharing")
            all_sessions = reversed(list(st.session_state.chat_sessions.keys()))
            st.session_state.active_filters = st.multiselect(
                "Include Knowledge From:",
                options=list(all_sessions),
                default=[st.session_state.current_session_id],
                help="By default, AI only sees documents uploaded in the current chat. Add older sessions here to pull their documents into this chat's brain."
            )

    with attach_col:
        prompt = st.chat_input("Enter a prompt here")

    if prompt:
        if not st.session_state.get("target_source"):
            st.toast("Please add a file in the sidebar and select it from Context Filter.", icon=":material/warning:")
            return
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_sessions()
        st.rerun()

    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]

        if st.session_state.query_system:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                with st.spinner("Processing your request..."):
                    try:
                        selected_sessions = st.session_state.get("active_filters", [st.session_state.current_session_id])
                        context_filter = {"session_id": {"$in": selected_sessions}}

                        user_record = users_collection.find_one({"email": st.session_state.user_email}) if st.session_state.get("user_email") else None
                        authorized_docs = user_record.get("documents", []) if user_record else []

                        # response = st.session_state.query_system.run_pipeline(
                        #     user_query=user_prompt,
                        #     context_filter=context_filter,
                        #     authorized_docs=authorized_docs,
                        #     target_source=st.session_state.get("target_source")
                        # )
                        response = st.session_state.query_system.run_pipeline(
                            user_query=user_prompt,
                            context_filter=context_filter,
                            authorized_docs=authorized_docs,
                            target_source=st.session_state.get("target_source"),
                            user_email=st.session_state.get("user_email"),
                        )
                        docs_count = len(getattr(response, "raw_docs", []) or [])
                        selected_target = st.session_state.get("target_source") or "All Documents"
                        st.caption(
                            f"Route: {response.lineage.route.upper()} | Target: {selected_target} | Retrieved Chunks: {docs_count}"
                            )
                        st.write(response.answer)
                        if response.lineage.cache_hit:
                            st.toast("Answered from Cache", icon=":material/bolt:")
                        display_lineage(response.lineage)
                        if response.lineage.route in ["rag", "both"] and not (getattr(response, "raw_docs", None) or []):
                            st.warning(
                            "No document chunks matched current filters. Try 'All Documents' or adjust session filters.",
                            icon=":material/warning:",
                            )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "lineage": response.lineage,
                            "raw_docs": getattr(response, "raw_docs", None),
                            "feedback": None
                        })
                        save_chat_sessions()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Query generation failed: {str(e)}", icon=":material/error:")
                        st.session_state.messages.pop()
        else:
            st.error("System is offline or failed to initialize.", icon=":material/error:")

if __name__ == "__main__":
    main()
