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

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Nexus Intelligence",
    page_icon=":material/database:",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHAT_HISTORY_FILE = "./data/chat_history.json"

def save_chat_sessions():
    """Serialize and save chat sessions securely to disk to prevent refresh loss."""
    os.makedirs("./data", exist_ok=True)
    try:
        serializable_sessions = {}
        for session_id, messages in st.session_state.chat_sessions.items():
            serializable_messages = []
            for msg in messages:
                ser_msg = {"role": msg["role"], "content": msg["content"]}
                if "lineage" in msg and msg["lineage"]:
                    if hasattr(msg["lineage"], 'to_dict'):
                        ser_msg["lineage"] = msg["lineage"].to_dict()
                    else:
                        ser_msg["lineage"] = msg["lineage"]
                serializable_messages.append(ser_msg)
            serializable_sessions[session_id] = serializable_messages
            
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump({
                "chat_sessions": serializable_sessions,
                "session_counter": st.session_state.session_counter
            }, f, indent=2)
    except Exception as e:
        self.logger.error(f"Background Save failed: {e}")

def load_chat_sessions():
    """Load safely persistent chat sessions from disk."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                data = json.load(f)
                
            sessions = data.get("chat_sessions", {})
            st.session_state.session_counter = data.get("session_counter", 1)
            
            # Rehydrate LineageTrace Dataclasses!
            for session_id, messages in sessions.items():
                for msg in messages:
                    if "lineage" in msg and msg["lineage"]:
                        if isinstance(msg["lineage"], dict):
                            valid_keys = LineageTrace.__dataclass_fields__.keys()
                            clean_kwargs = {k: v for k, v in msg["lineage"].items() if k in valid_keys}
                            msg["lineage"] = LineageTrace(**clean_kwargs)
                            
            st.session_state.chat_sessions = sessions
            if sessions:
                st.session_state.current_session_id = list(sessions.keys())[-1]
            else:
                st.session_state.chat_sessions = {"Session 1": []}
                st.session_state.current_session_id = "Session 1"
        except Exception as e:
            st.session_state.chat_sessions = {"Session 1": []}
            st.session_state.current_session_id = "Session 1"
            st.session_state.session_counter = 1
    else:
        st.session_state.chat_sessions = {"Session 1": []}
        st.session_state.current_session_id = "Session 1"
        st.session_state.session_counter = 1

def inject_custom_css():
    """Inject custom CSS for aesthetic improvements (Gemini style)."""
    st.markdown("""
        <style>
        /* Hide Default Streamlit UI Noise */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Sleek overall styling - Limit width for readability */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 900px !important;
        }
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-size: 20px;
            font-weight: 500;
        }
        
        /* Title adjustments */
        h1 {
            font-family: 'Inter', system-ui, sans-serif;
            font-weight: 600;
            background: -webkit-linear-gradient(45deg, #4A90E2, #7f8c8d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
        }
        
        /* Subtitle */
        .subtitle {
            font-size: 1.1rem;
            font-weight: 400;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Sidebar Chat Thread Menu Customization */
        [data-testid="stSidebar"] button[kind="tertiary"] {
            justify-content: flex-start;
            padding-left: 0.5rem;
            font-weight: 400;
            opacity: 0.8;
            transition: all 0.2s;
        }
        [data-testid="stSidebar"] button[kind="tertiary"]:hover {
            opacity: 1.0;
            background-color: rgba(128, 128, 128, 0.1);
        }
        
        /* Floating Chat Uploader Adjustments */
        [data-testid="stPopover"] {
            padding-top: 1px;
        }
        /* Completely hide the arrow icon next to the popover button safely */
        [data-testid="stPopover"] summary > svg:last-of-type {
            display: none !important;
        }
        [data-testid="stPopover"] summary {
            list-style: none !important;
        }
        [data-testid="stPopover"] summary::-webkit-details-marker {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)


def chunk_text(text: str, chunk_size: int = 300) -> list:
    """Split text efficiently into token-safe chunks for optimized vector embedding."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def parse_and_add_documents(uploaded_files):
    """Parse CSV, PDF, and JSON chunks cleanly into the vector-store context with chunking optimization."""
    import json
    try:
        import pandas as pd
        from pypdf import PdfReader
    except ImportError:
        st.error("Missing dependencies. Backend needs to have installed: pypdf pandas")
        return

    sys_engine = st.session_state.query_system
    if not sys_engine:
        st.error("System offline")
        return
        
    for file in uploaded_files:
        content = ""
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file)
                # Cap the CSV read to reasonable limits just in case it's millions of rows, but convert all safe rows to string structure
                content = f"CSV Filename: {file.name}\n" + df.head(1000).to_string()
            elif file.name.endswith(".json"):
                data = json.load(file)
                content = json.dumps(data, indent=2)
            else:
                content = file.getvalue().decode('utf-8')
                
            if content.strip():
                metadata = {"session_id": st.session_state.current_session_id}
                
                # OPTIMIZED SAVING: Chunk large data so the context window safely parses it without truncation loss
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file.name}_chunk_{i}"
                    sys_engine.tag.add_document(doc_id=chunk_id, content=chunk, metadata=metadata)
                    
                st.toast(f"Optimized strictly into {len(chunks)} contextual chunks!", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Failed to process {file.name}: {str(e)}")


def initialize_session_state():
    """Initialize Streamlit session state and Chat Session Tracking."""
    if "query_system" not in st.session_state:
        with st.spinner("Initializing AI Query System..."):
            try:
                st.session_state.query_system = AIQuerySystem()
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                st.session_state.query_system = None
                
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "chat_sessions" not in st.session_state:
        load_chat_sessions()
        st.session_state.active_filters = [st.session_state.current_session_id]
        
    st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_session_id]


def display_lineage(lineage):
    """Display lineage trace in an expander using tabs."""
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


def render_sidebar():
    """Render the sidebar carefully engineered like Gemini."""
    with st.sidebar:
        # Top Native Button
        if st.button("New Chat", icon=":material/add:", type="primary", use_container_width=True):
            st.session_state.session_counter += 1
            new_id = f"Session {st.session_state.session_counter}"
            st.session_state.chat_sessions[new_id] = []
            st.session_state.current_session_id = new_id
            save_chat_sessions()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Chat History")
            
        # List sessions with minimal tertiary link style
        # Iterate in REVERSE order so newest is at the top
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

        # Add vertical whitespace to push settings to bottom
        st.markdown("<br>"*5, unsafe_allow_html=True)
        
        # Advanced Technical Settings in a neat expander
        with st.expander("Settings & Resources", icon=":material/settings:", expanded=False):
            st.caption("System Resources")
            if st.session_state.query_system:
                stats = st.session_state.query_system.get_stats()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Cache", stats['cache_stats']['total_entries'])
                with c2:
                    st.metric("Docs", stats['tag_collections']['documents'])
                
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
                    st.rerun()
            else:
                st.warning("System Offline")

def render_auth_screen():
    """Render a premium front-end login gateway (Logic mocked per user request)."""
    st.write("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>Nexus Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d; margin-bottom: 3rem;'>Enterprise Knowledge & Semantic RAG Engine</p>", unsafe_allow_html=True)
    
    # To make the login box wider, we increase the middle column's ratio.
    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        with st.container(border=True):
            tab1, tab2 = st.tabs(["Log In", "Sign Up"])
            
            with tab1:
                st.markdown("### Welcome Back")
                st.caption("Please authenticate to access your Nexus Data Warehouse.")
                st.text_input("Work Email", placeholder="name@company.com", key="login_email")
                st.text_input("Password", type="password", key="login_pass")
                st.write("")
                if st.button("Secure Login", icon=":material/login:", use_container_width=True, type="primary"):
                    # Dummy logic - immediately pass through
                    st.session_state.authenticated = True
                    st.rerun()
                    
            with tab2:
                st.markdown("### Create Account")
                st.caption("Your data remains isolated under Enterprise standards.")
                st.text_input("Full Name", placeholder="Jane Doe")
                st.text_input("Work Email", placeholder="name@company.com", key="signup_email")
                st.text_input("Password", type="password", key="signup_pass")
                st.write("")
                if st.button("Register Account", icon=":material/person_add:", use_container_width=True, type="primary"):
                    st.toast("Registration Successful! Signing you in...", icon=":material/check_circle:")
                    st.session_state.authenticated = True
                    st.rerun()


def render_welcome_screen():
    """Render exactly like the Google Gemini welcome screen with contextual ideas."""
    st.markdown("<h3 style='text-align: center; margin-top: 4rem; font-family: Inter, sans-serif;'>Hello. How can I help you today?</h3>", unsafe_allow_html=True)
    st.write("<br><br>", unsafe_allow_html=True)
    
    st.markdown("#### Quick actions & ideas from your context")
    col1, col2, col3 = st.columns(3)
    
    # Base examples just in case there is no history yet!
    examples = [
        ("Sample: Customer Insights", "How many customers do we have?"),
        ("Sample: Financial Overview", "What is the total generated revenue?"),
        ("Sample: Operations Update", "Show me the 5 most recent orders")
    ]
    
    # Intelligently grab multiple old contextual queries from user history
    recent_searches = []
    seen = set()
    # Iterate dynamically to pull out unique questions
    for s_id, msgs in reversed(st.session_state.chat_sessions.items()):
        for m in reversed(msgs):
            if m["role"] == "user":
                if m["content"] not in seen:
                    # Ignore the base examples from polluting real history logs
                    if m["content"] not in [e[1] for e in examples]:
                        seen.add(m["content"])
                        recent_searches.append((s_id, m["content"]))
                    
    # Map their history smartly into the Quick Action UI
    if len(recent_searches) >= 1:
        examples[0] = (f"From {recent_searches[0][0]}", recent_searches[0][1])
    if len(recent_searches) >= 2:
        examples[1] = (f"From {recent_searches[1][0]}", recent_searches[1][1])
    if len(recent_searches) >= 3:
        examples[2] = (f"From {recent_searches[2][0]}", recent_searches[2][1])
    
    for (name, query), col in zip(examples, [col1, col2, col3]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                
                # Truncate string gracefully for UI appearance, but pass pure query to backend
                display_query = f'"{query[:40]}..."' if len(query) > 40 else f'"{query}"'
                st.caption(display_query)
                
                if st.button("Query", icon=":material/play_arrow:", key=f"ex_{name}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    save_chat_sessions()
                    st.rerun()

def main():
    """Main Streamlit application."""
    inject_custom_css()
    initialize_session_state()
    
    # Gateway Security Intercept (UI Only Mock)
    if not st.session_state.get("authenticated", False):
        render_auth_screen()
        return

    # Authorized Content Below
    st.title("Nexus Intelligence")
    st.markdown('<p class="subtitle">Enterprise Knowledge & Semantic RAG Engine</p>', unsafe_allow_html=True)

    render_sidebar()

    # Render Welcome Screen OR History
    if len(st.session_state.messages) == 0:
        render_welcome_screen()
    else:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    st.write(message["content"])
                    
                    # Native Streamlit Thumbs Feedback
                    fb = st.feedback("thumbs", key=f"feed_{st.session_state.current_session_id}_{i}")
                    if fb is not None and message.get("feedback") != fb:
                        message["feedback"] = fb
                        save_chat_sessions()
                    
                    # Surfacing Pure Text RAG Contexts explicitly so user can read/copy
                    raw_docs = message.get("raw_docs")
                    if raw_docs:
                        with st.expander("📄 Sourced Knowledge Contexts", expanded=True):
                            for idx, doc in enumerate(raw_docs):
                                doc_id = doc.get("id", "Document snippet")
                                # Remove system chunk identifier string if present for cleaner UI
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
    input_col, attach_col = st.columns([1, 15])
    with input_col:
        with st.popover("", icon=":material/attach_file:", use_container_width=True):
            st.markdown("**Knowledge & Context Management**")
            st.caption("Documents uploaded here will ONLY be readable within this specific Chat Thread to prevent confusion.")
            
            uploaded_files = st.file_uploader("Upload local files", accept_multiple_files=True, type=['csv', 'pdf', 'json'], label_visibility="collapsed")
            if uploaded_files and st.button("Ingest Files", icon=":material/upload_file:", use_container_width=True):
                with st.spinner("Processing semantics..."):
                    parse_and_add_documents(uploaded_files)
                    
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_sessions()
        st.rerun()

    # Engine Processing Trigger
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        
        if st.session_state.query_system:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                with st.spinner("Processing your request..."):
                    try:
                        selected_sessions = st.session_state.get("active_filters", [st.session_state.current_session_id])
                        context_filter = {"session_id": {"$in": selected_sessions}}
                        
                        response = st.session_state.query_system.run_pipeline(
                            user_query=user_prompt, 
                            context_filter=context_filter
                        )
                        st.write(response.answer)
                        if response.lineage.cache_hit:
                            st.toast("Answered from Cache", icon=":material/bolt:")
                        display_lineage(response.lineage)
                        
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
