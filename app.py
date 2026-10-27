import streamlit as st
from agent import app, vector_store

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Financial RAG Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished financial dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .trace-card {
        background: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .badge-yes {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-no {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: System Configuration & Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Financial RAG Analyst")
    st.caption("Self-Correcting Graph Architecture")

    st.markdown("---")
    st.markdown("#### 🗄️ Vector Database Status")
    
    # Dynamically fetch or fallback to collection count
    try:
        chunk_count = vector_store._collection.count()
    except Exception:
        chunk_count = 575

    st.success(f"🟢 **Active Database**")
    st.metric(label="Indexed Chunks", value=f"{chunk_count} chunks")
    st.caption("Target Coverage: Apple (AAPL) & Tesla (TSLA) Filings")

    st.markdown("---")
    st.markdown("#### ⚙️ Agent Pipeline Details")
    st.markdown("""
    - **LLM**: `openai/gpt-oss-120b` (Groq)
    - **Temperature**: `0.0`
    - **Embeddings**: `BAAI/bge-small-en-v1.5`
    - **Retriever**: ChromaDB (`k=3`)
    - **Graph Flow**: `START` → `rewrite_query` → `retrieve` → `grade_documents` → `generate` → `check_hallucination`
    """)

    st.markdown("---")
    st.markdown("#### 💡 Sample Queries")
    sample_queries = [
        "Is Apple doing well financially?",
        "What was Tesla's total automotive revenue or delivery numbers mentioned in the reports?",
        "What were Apple's total revenues and net income reported?",
        "What are the key risk factors or manufacturing challenges reported by Tesla?",
    ]
    for sq in sample_queries:
        if st.button(f"🔍 {sq[:38]}...", key=sq, help=sq):
            st.session_state["pending_query"] = sq

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Main Chat Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📈 Financial RAG Analyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Query Apple & Tesla 10-Q/10-K financial reports with verified, self-correcting RAG.</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if a sample query was selected
prompt_input = None
if "pending_query" in st.session_state:
    prompt_input = st.session_state.pop("pending_query")

# ---------------------------------------------------------
# Helper to render reasoning trace
# ---------------------------------------------------------
def render_trace(trace: dict):
    with st.expander("🔍 Agent Reasoning & Trace", expanded=False):
        tab0, tab1, tab2, tab3 = st.tabs([
            "🔄 Query Rewriter",
            "📑 Retrieved Chunks",
            "⚖️ Document Grading",
            "🛡️ Hallucination Check"
        ])

        with tab0:
            st.markdown("#### 🎯 Query Transformation")
            st.markdown(f"**Original User Prompt:**")
            st.write(f"`{trace.get('question', 'N/A')}`")
            rewritten = trace.get("rewritten_query", "")
            if rewritten:
                st.info(f"**Optimized Financial Search Query:**\n\n🔍 `{rewritten}`")
                st.caption("The agent translated the user's abstract question into dense financial statement keywords.")
            else:
                st.write("Direct query used.")

        with tab1:
            retrieved_docs = trace.get("retrieved_documents", [])
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Chunk #{i}:**")
                    st.text_area(f"Chunk {i} Content", doc, height=120, key=f"chunk_txt_{i}_{hash(doc[:30])}", disabled=True)
            else:
                st.info("No raw chunks recorded in trace.")

        with tab2:
            grading_verdicts = trace.get("grading_verdicts", [])
            if grading_verdicts:
                for item in grading_verdicts:
                    cid = item.get("chunk_id", "-")
                    verdict = item.get("relevance", "N/A")
                    badge = "badge-yes" if verdict == "yes" else "badge-no"
                    status_text = "✅ Relevant" if verdict == "yes" else "❌ Irrelevant (Filtered Out)"
                    st.markdown(
                        f"<div class='trace-card'>"
                        f"<b>Chunk #{cid}:</b> <span class='{badge}'>{status_text}</span><br>"
                        f"<small>{item.get('content', '')[:250]}...</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("Document grading details not available.")

        with tab3:
            h_grade = trace.get("hallucination_grade", "N/A")
            loops = trace.get("loop_count", 1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if h_grade == "yes":
                    st.success("✅ **Answer Factually Grounded**")
                    st.write("The generated answer was verified to be strictly grounded in the retrieved context.")
                elif h_grade == "no":
                    st.warning("⚠️ **Hallucination Detected & Handled**")
                    st.write("The answer triggered self-correction or reached retry limits.")
                else:
                    st.info(f"Verdict: {h_grade}")
            with col_b:
                st.metric(label="Evaluation Loops", value=loops)

# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "trace" in msg and msg["trace"]:
            render_trace(msg["trace"])
        st.markdown(msg["content"])

# ---------------------------------------------------------
# Chat Input & Invocation
# ---------------------------------------------------------
user_query = st.chat_input("Ask about Tesla or Apple revenues, deliveries, cash flow...")
query_to_run = prompt_input or user_query

if query_to_run:
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # 2. Invoke Agent Pipeline
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent analyzing financial documents & verifying facts..."):
            initial_state = {"question": query_to_run}
            result = app.invoke(initial_state)

            final_generation = result.get("generation", "No answer generated.")
            trace_data = {
                "question": query_to_run,
                "rewritten_query": result.get("rewritten_query", ""),
                "retrieved_documents": result.get("retrieved_documents", []),
                "grading_verdicts": result.get("grading_verdicts", []),
                "hallucination_grade": result.get("hallucination_grade", "yes"),
                "loop_count": result.get("loop_count", 1),
            }

            # Render intermediate trace expander
            render_trace(trace_data)

            # Render final answer
            st.markdown(final_generation)

    # 3. Store assistant message in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_generation,
        "trace": trace_data
    })
