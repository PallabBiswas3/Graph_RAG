import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import tempfile
import networkx as nx
import numpy as np

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Academic",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

/* Main header */
.main-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.sub-header {
    color: #8b949e;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1.5rem;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

/* Answer box */
.answer-box {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e6edf3;
}

/* Chunk card */
.chunk-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
}
.chunk-card .chunk-meta {
    color: #3fb950;
    font-size: 0.72rem;
    margin-bottom: 0.3rem;
}
.chunk-score {
    float: right;
    background: #1f6feb;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
}

/* Strategy badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin: 2px;
}
.badge-vector  { background: #1f3a5f; color: #58a6ff; }
.badge-ics     { background: #1a3a2a; color: #3fb950; }
.badge-iks     { background: #3a1a3a; color: #d2a8ff; }
.badge-uks     { background: #3a2a1a; color: #ffa657; }

/* Section titles */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}

/* Input styling */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 6px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #388bfd !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    border-radius: 6px;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: white !important;
}

/* Metric delta */
.delta-positive { color: #3fb950; font-family: 'IBM Plex Mono', monospace; }
.delta-negative { color: #f85149; font-family: 'IBM Plex Mono', monospace; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline     = None
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result  = None
if "use_demo" not in st.session_state:
    st.session_state.use_demo     = False


# ── Helper: build pipeline ────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    from pipeline import GraphRAGPipeline
    return GraphRAGPipeline(
        openai_client=None,
        chunk_size=800,
        num_keywords=5,
        top_k=10,
        pass_k=10,
    )


# ── Helper: graph visualisation as HTML ──────────────────────────────
def render_graph_html(dkg, ikg, highlight_uris=None) -> str:
    highlight_uris = highlight_uris or set()
    G = dkg.G

    nodes_data, edges_data = [], []
    color_map = {
        "document": "#1f6feb",
        "chapter":  "#3fb950",
        "section":  "#d2a8ff",
        "chunk":    "#8b949e",
        "metadata": "#ffa657",
    }

    for node_id, data in G.nodes(data=True):
        ntype = data.get("type", "chunk")
        uri   = data.get("uri", "")
        label = data.get("title") or data.get("key") or uri.split(":")[-1] if uri else node_id.split(":")[-1]
        label = label[:22] + "…" if len(label) > 22 else label

        is_hit = uri in highlight_uris
        color  = "#f85149" if is_hit else color_map.get(ntype, "#8b949e")
        size   = {"document": 28, "chapter": 20, "section": 16, "chunk": 12, "metadata": 10}.get(ntype, 12)

        nodes_data.append({
            "id":    node_id,
            "label": label,
            "color": color,
            "size":  size + (6 if is_hit else 0),
            "title": f"{ntype.upper()}: {data.get('title', data.get('text',''))[:120]}",
        })

    for src, dst, edata in G.edges(data=True):
        rel   = edata.get("rel", "")
        color = {"HAS_CHAPTER": "#3fb950", "HAS_SECTION": "#d2a8ff",
                 "HAS_CHUNK": "#30363d", "NEXT_CHUNK": "#21262d",
                 "HAS_METADATA": "#ffa657"}.get(rel, "#30363d")
        edges_data.append({"from": src, "to": dst, "color": color, "title": rel})

    # Add IKG keyword edges
    for kw, uris in ikg._kw_to_uris.items():
        uri_list = list(uris)
        for i in range(len(uri_list)):
            for j in range(i+1, len(uri_list)):
                edges_data.append({
                    "from":    f"chunk:{uri_list[i]}",
                    "to":      f"chunk:{uri_list[j]}",
                    "color":   "rgba(210,168,255,0.25)",
                    "title":   f"shared keyword: {kw}",
                    "dashes":  True,
                })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    legend_html = "".join([
        f'<span style="background:{c};padding:2px 10px;border-radius:10px;font-size:11px;margin:2px;display:inline-block;color:white">{t}</span>'
        for t, c in color_map.items()
    ])

    return f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
<style>
  body   {{ margin:0; background:#0d1117; color:#e6edf3; font-family:'IBM Plex Sans',sans-serif; }}
  #graph {{ width:100%; height:520px; border:1px solid #30363d; border-radius:8px; background:#0d1117; }}
  #legend{{ padding:8px 12px; font-size:11px; border-top:1px solid #21262d; background:#0d1117; }}
  #info  {{ padding:8px 12px; font-size:11px; color:#8b949e; min-height:32px; background:#161b22;
            border:1px solid #30363d; border-radius:4px; margin:6px 0; font-family:monospace; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="info">Click a node to inspect it</div>
<div id="legend">
  <b style="color:#8b949e;font-size:10px;letter-spacing:1px">NODE TYPES ·</b>
  {legend_html}
  <span style="background:rgba(210,168,255,0.25);padding:2px 10px;border-radius:10px;font-size:11px;margin:2px;display:inline-block;color:#d2a8ff">-- keyword link (IKG)</span>
  {'<span style="background:#f85149;padding:2px 10px;border-radius:10px;font-size:11px;margin:2px;display:inline-block;color:white">● retrieved chunk</span>' if highlight_uris else ''}
</div>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var options = {{
  nodes: {{ font:{{color:'#e6edf3',size:11}}, borderWidth:1, borderWidthSelected:2 }},
  edges: {{ arrows:{{to:{{enabled:true,scaleFactor:0.4}}}}, smooth:{{type:'cubicBezier',roundness:0.4}} }},
  physics: {{ solver:'forceAtlas2Based', forceAtlas2Based:{{gravitationalConstant:-60,springLength:80}}, stabilization:{{iterations:120}} }},
  interaction: {{ hover:true, tooltipDelay:100 }},
  layout: {{ improvedLayout:true }},
  background: {{ color:'#0d1117' }}
}};
var network = new vis.Network(document.getElementById('graph'), {{nodes,edges}}, options);
network.on('click', function(p) {{
  if (p.nodes.length > 0) {{
    var n = nodes.get(p.nodes[0]);
    document.getElementById('info').innerHTML = '<b style="color:#58a6ff">' + p.nodes[0] + '</b> · ' + (n.title||'');
  }}
}});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="main-header">GraphRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header"> Academic Research Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Document Source</div>', unsafe_allow_html=True)

    source = st.radio("", [" Upload PDFs", " Use Demo Papers"], label_visibility="collapsed")

    if source == " Use Demo Papers":
        if st.button("⚡ Load Demo Dataset", use_container_width=True):
            with st.spinner("Indexing 3 academic papers..."):
                from demo import SAMPLE_DOCS
                pipeline = get_pipeline()
                pipeline.dkg        = pipeline.dkg.__class__()
                pipeline.ikg        = pipeline.ikg.__class__()
                pipeline.vector_db  = pipeline.vector_db.__class__(collection_name="graphrag_ui")
                pipeline._all_chunks = []
                pipeline.index_documents(SAMPLE_DOCS)
                st.session_state.pipeline     = pipeline
                st.session_state.indexed_docs = [d["doc_title"] for d in SAMPLE_DOCS]
                st.session_state.chat_history = []
            st.success(f"✓ {len(SAMPLE_DOCS)} papers indexed")

    else:
        uploaded = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("⚡ Index Documents", use_container_width=True):
            from utils.pdf_loader import load_pdf_as_doc
            from pipeline import GraphRAGPipeline
            docs = []
            progress = st.progress(0)
            for i, uf in enumerate(uploaded):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    doc = load_pdf_as_doc(tmp_path, doc_id=f"doc_{i}", metadata={})
                    doc["doc_title"] = uf.name.replace(".pdf","").replace("_"," ")
                    docs.append(doc)
                except Exception as e:
                    st.error(f"Failed to load {uf.name}: {e}")
                finally:
                    os.unlink(tmp_path)
                progress.progress((i+1)/len(uploaded))

            if docs:
                pipeline = GraphRAGPipeline(chunk_size=800, num_keywords=5, top_k=10, pass_k=10)
                pipeline.index_documents(docs)
                st.session_state.pipeline     = pipeline
                st.session_state.indexed_docs = [d["doc_title"] for d in docs]
                st.session_state.chat_history = []
                st.success(f"✓ {len(docs)} document(s) indexed")

    # ── Indexed docs list ──────────────────────────────────────────
    if st.session_state.indexed_docs:
        st.markdown('<div class="section-title">Indexed Documents</div>', unsafe_allow_html=True)
        for doc in st.session_state.indexed_docs:
            st.markdown(f"<div style='font-size:0.78rem;color:#3fb950;padding:3px 0'>📑 {doc}</div>",
                        unsafe_allow_html=True)

        pipeline = st.session_state.pipeline
        if pipeline:
            stats = pipeline.dkg.get_stats()
            ikg_s = pipeline.ikg.get_stats()
            st.markdown('<div class="section-title">Index Stats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Chunks",   stats.get("chunk", 0))
                st.metric("Chapters", stats.get("chapter", 0))
            with c2:
                st.metric("Keywords", ikg_s.get("unique_keywords", 0))
                st.metric("Sections", stats.get("section", 0))

    # ── Pipeline params ────────────────────────────────────────────
    st.markdown('<div class="section-title">Parameters</div>', unsafe_allow_html=True)
    top_k      = st.slider("Top-k retrieval", 3, 20, 10)
    num_kw     = st.slider("Keywords per chunk", 1, 10, 5)
    chunk_size = st.select_slider("Chunk size (tokens)", [400, 600, 800, 1000, 1300], value=800)

    if st.session_state.pipeline and st.button("Apply Parameters", use_container_width=True):
        st.session_state.pipeline.top_k        = top_k
        st.session_state.pipeline.pass_k       = top_k
        st.session_state.pipeline.num_keywords = num_kw
        st.session_state.pipeline.chunk_size   = chunk_size
        st.info("Parameters updated for next query.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.indexed_docs:
    # ── Welcome screen ───────────────────────────────────────────────
    st.markdown('<div class="main-header">Document GraphRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Knowledge graph enhanced retrieval augmented generation · Knollmeyer et al. 2025</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="label">Step 1</div>
            <div style="color:#3fb950;font-size:1.1rem;font-weight:600;margin:6px 0">Upload or Demo</div>
            <div style="color:#8b949e;font-size:0.82rem">Add PDF papers or load the built-in demo dataset from the sidebar</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="label">Step 2</div>
            <div style="color:#58a6ff;font-size:1.1rem;font-weight:600;margin:6px 0">Ask Questions</div>
            <div style="color:#8b949e;font-size:0.82rem">Single-hop or multi-hop questions across all your documents</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="label">Step 3</div>
            <div style="color:#d2a8ff;font-size:1.1rem;font-weight:600;margin:6px 0">Explore the Graph</div>
            <div style="color:#8b949e;font-size:0.82rem">Visualise DKG structure and IKG keyword links live</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""<div class="metric-card">
        <div class="label">Pipeline Overview</div>
        <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <span class="badge badge-vector">① Vector Search</span>
            <span style="color:#30363d">→</span>
            <span class="badge badge-uks">② UKS · query keywords</span>
            <span style="color:#30363d">→</span>
            <span class="badge badge-ics">③ ICS · chapter expand</span>
            <span style="color:#30363d">→</span>
            <span class="badge badge-iks">④ IKS · keyword links</span>
            <span style="color:#30363d">→</span>
            <span class="badge" style="background:#1a2a3a;color:#58a6ff">⑤ Deduplicate</span>
            <span style="color:#30363d">→</span>
            <span class="badge" style="background:#1a2a3a;color:#ffa657">⑥ Rerank</span>
            <span style="color:#30363d">→</span>
            <span class="badge" style="background:#1a2a3a;color:#3fb950">⑦ Generate</span>
        </div>
    </div>""", unsafe_allow_html=True)

else:
    # ── Main tabs ────────────────────────────────────────────────────
    tab_qa, tab_graph, tab_eval = st.tabs(["💬  Q&A", "🕸  Knowledge Graph", "📊  Evaluation"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — Q&A
    # ══════════════════════════════════════════════════════════════════
    with tab_qa:
        st.markdown('<div class="main-header" style="font-size:1.3rem">Ask Your Documents</div>',
                    unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""<div style="text-align:right;margin:8px 0">
                    <span style="background:#1f6feb;color:white;padding:8px 14px;border-radius:16px 16px 4px 16px;
                    font-size:0.88rem;display:inline-block;max-width:80%">{msg["content"]}</span></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)
                if "stats" in msg:
                    s = msg["stats"]
                    st.markdown(
                        f'<span class="badge badge-vector">vector {s["vector_hits"]}</span>'
                        f'<span class="badge badge-ics">ICS +{s["ics_hits"]}</span>'
                        f'<span class="badge badge-iks">IKS +{s["iks_hits"]}</span>'
                        f'<span class="badge badge-uks">UKS +{s["uks_hits"]}</span>'
                        f'<span class="badge" style="background:#1a2a3a;color:#58a6ff">combined {s["combined"]}</span>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Query input
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            query = st.text_input("", placeholder="Ask a question about your documents…",
                                  label_visibility="collapsed", key="query_input")
        with col_btn:
            ask = st.button("Ask →", use_container_width=True)

        # Example questions
        st.markdown('<div class="section-title">Example Questions</div>', unsafe_allow_html=True)
        examples = [
            "What is the quadratic complexity limitation of self-attention?",
            "How do Knowledge Graphs improve multi-hop reasoning in RAG?",
            "What training approach does RAG use for its retriever?",
            "What are the main limitations of RAG systems?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                ask   = True

        # Run query
        if ask and query and st.session_state.pipeline:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.spinner("Retrieving across DKG + IKG…"):
                result = st.session_state.pipeline.query(query, evaluate=False)
                st.session_state.last_result = result

            answer = result["answer"]
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": answer,
                "stats":   result["stats"],
            })
            st.rerun()

        # Retrieved chunks panel
        if st.session_state.last_result:
            r = st.session_state.last_result
            st.markdown('<div class="section-title">Retrieved Chunks (after rerank)</div>',
                        unsafe_allow_html=True)
            for i, chunk in enumerate(r["final_chunks"][:5], 1):
                score = chunk.get("rerank_score", 0)
                doc   = chunk.get("doc_title", "")
                chap  = chunk.get("chapter", "")
                text  = chunk.get("text", "")[:200]
                st.markdown(f"""<div class="chunk-card">
                    <div class="chunk-meta">
                        <span class="chunk-score">{score:.1f}</span>
                        [{i}] {doc} · {chap}
                    </div>
                    {text}…
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — KNOWLEDGE GRAPH
    # ══════════════════════════════════════════════════════════════════
    with tab_graph:
        st.markdown('<div class="main-header" style="font-size:1.3rem">Knowledge Graph Explorer</div>',
                    unsafe_allow_html=True)

        pipeline = st.session_state.pipeline
        if pipeline:
            highlight_uris = set()
            if st.session_state.last_result:
                highlight_uris = {c.get("uri","") for c in st.session_state.last_result.get("final_chunks", [])}

            c1, c2, c3, c4 = st.columns(4)
            stats = pipeline.dkg.get_stats()
            ikg_s = pipeline.ikg.get_stats()
            c1.metric("Total Nodes",   stats["total_nodes"])
            c2.metric("Total Edges",   stats["total_edges"])
            c3.metric("Unique Keywords", ikg_s["unique_keywords"])
            c4.metric("Avg KW/Chunk",  f"{ikg_s['avg_keywords_per_chunk']:.1f}")

            if highlight_uris:
                st.info(f"🔴 {len(highlight_uris)} retrieved chunks highlighted in red from your last query")

            html = render_graph_html(pipeline.dkg, pipeline.ikg, highlight_uris)
            st.components.v1.html(html, height=620, scrolling=False)

            # IKG keyword table
            st.markdown('<div class="section-title">Top Keywords (IKG)</div>', unsafe_allow_html=True)
            kw_data = [(kw, len(uris)) for kw, uris in pipeline.ikg._kw_to_uris.items()]
            kw_data.sort(key=lambda x: x[1], reverse=True)
            cols = st.columns(4)
            for i, (kw, count) in enumerate(kw_data[:20]):
                cols[i % 4].markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:4px;'
                    f'padding:4px 8px;margin:2px;font-size:0.75rem;font-family:monospace">'
                    f'<span style="color:#d2a8ff">{kw}</span> '
                    f'<span style="color:#8b949e;float:right">{count} chunks</span></div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — EVALUATION
    # ══════════════════════════════════════════════════════════════════
    with tab_eval:
        st.markdown('<div class="main-header" style="font-size:1.3rem">GraphRAG vs Naive RAG</div>',
                    unsafe_allow_html=True)

        if st.button("▶ Run Full Evaluation (5 questions)", use_container_width=False):
            from demo import EVAL_QUESTIONS
            from evaluation.metrics import RAGEvaluator

            pipeline  = st.session_state.pipeline
            evaluator = RAGEvaluator()
            gr_instances, naive_instances = [], []

            progress = st.progress(0)
            status   = st.empty()

            for i, item in enumerate(EVAL_QUESTIONS):
                status.markdown(f'<div style="color:#8b949e;font-size:0.82rem;font-family:monospace">Running Q{i+1}: {item["question"][:60]}…</div>',
                                 unsafe_allow_html=True)
                gr_result    = pipeline.query(item["question"], reference_answer=item["reference"], evaluate=False)
                naive_chunks = pipeline.vector_db.search(item["question"], k=10)
                naive_answer = pipeline._extractive_answer(item["question"], naive_chunks)

                gr_instances.append({"query": item["question"],
                                     "retrieved_chunks": gr_result["final_chunks"],
                                     "answer": gr_result["answer"],
                                     "reference_answer": item["reference"]})
                naive_instances.append({"query": item["question"],
                                        "retrieved_chunks": naive_chunks,
                                        "answer": naive_answer,
                                        "reference_answer": item["reference"]})
                progress.progress((i+1)/len(EVAL_QUESTIONS))

            status.empty()
            gr_avg    = evaluator.evaluate_batch(gr_instances)
            naive_avg = evaluator.evaluate_batch(naive_instances)

            st.markdown('<div class="section-title">Results Comparison</div>', unsafe_allow_html=True)

            metrics_display = {
                "context_recall":  "Context Recall",
                "k_precision":     "K-Precision",
                "answer_recall":   "Answer Recall",
                "faithfulness":    "Faithfulness",
            }

            cols = st.columns(len(metrics_display))
            for col, (key, label) in zip(cols, metrics_display.items()):
                g = gr_avg.get(key, 0)
                n = naive_avg.get(key, 0)
                delta = g - n
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
                color = "#3fb950" if delta > 0 else ("#f85149" if delta < 0 else "#8b949e")
                col.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div class="label">{label}</div>
                    <div class="value">{g:.3f}</div>
                    <div style="color:{color};font-family:'IBM Plex Mono',monospace;font-size:0.8rem">
                        {arrow} {abs(delta):.3f} vs naive
                    </div>
                    <div style="color:#8b949e;font-size:0.72rem;margin-top:4px">naive: {n:.3f}</div>
                </div>""", unsafe_allow_html=True)

            # Per-question breakdown
            st.markdown('<div class="section-title">Per-Question Breakdown</div>', unsafe_allow_html=True)
            for i, (gr, naive, item) in enumerate(zip(gr_instances, naive_instances, EVAL_QUESTIONS), 1):
                gr_m    = evaluator.evaluate(**{k: gr[k] for k in ["query","retrieved_chunks","answer","reference_answer"]})
                naive_m = evaluator.evaluate(**{k: naive[k] for k in ["query","retrieved_chunks","answer","reference_answer"]})
                delta   = gr_m["answer_recall"] - naive_m["answer_recall"]
                color   = "#3fb950" if delta > 0 else ("#f85149" if delta < 0 else "#8b949e")
                st.markdown(f"""<div class="chunk-card">
                    <div class="chunk-meta">[{item["type"].upper()}] Q{i}</div>
                    <div style="color:#e6edf3;margin-bottom:6px">{item["question"]}</div>
                    <span style="color:#8b949e;font-size:0.75rem">GraphRAG AR: <b style="color:#58a6ff">{gr_m['answer_recall']:.3f}</b></span>
                    &nbsp;·&nbsp;
                    <span style="color:#8b949e;font-size:0.75rem">Naive AR: <b style="color:#8b949e">{naive_m['answer_recall']:.3f}</b></span>
                    &nbsp;·&nbsp;
                    <span style="color:{color};font-size:0.75rem;font-family:monospace">Δ {'▲' if delta>0 else '▼'}{abs(delta):.3f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-card" style="text-align:center;padding:2rem">
                <div style="color:#8b949e;font-size:0.88rem">
                    Click the button above to run the full evaluation comparing<br>
                    GraphRAG vs Naive RAG across all 5 questions.<br><br>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#3fb950">
                    Metrics: Context Recall · K-Precision · Answer Recall · Faithfulness
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)