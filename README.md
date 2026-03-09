# Document GraphRAG — Academic Research Pipeline

> Knowledge Graph Enhanced Retrieval Augmented Generation for Academic Document Q&A  
> Based on [Knollmeyer et al., *Electronics* 2025, 14, 2102](https://doi.org/10.3390/electronics14112102)

---

## What is this?

Naive RAG systems retrieve document chunks purely by vector similarity — losing document structure, missing cross-document connections, and struggling with multi-hop questions. **Document GraphRAG** fixes this by layering two knowledge graphs on top of standard vector search:

- **DKG (Document Knowledge Graph)** — mirrors each paper's hierarchy: Document → Chapter → Section → Chunk
- **IKG (Information Knowledge Graph)** — links chunks across documents via shared keywords

Three retrieval strategies then expand the initial vector results using these graphs before a semantic reranker selects the final context for answer generation.

---

## Project Structure

```
graphrag_academic/
├── app.py                      # Streamlit web UI (Q&A + graph visualisation + evaluation)
├── pipeline.py                 # Full GraphRAG pipeline orchestrator
├── demo.py                     # CLI demo — 5 benchmark questions, GraphRAG vs Naive RAG
│
├── graph/
│   ├── dkg.py                  # Document Knowledge Graph (NetworkX tree)
│   └── ikg.py                  # Information Knowledge Graph (keyword inverted index)
│
├── retrieval/
│   ├── vector_store.py         # NumPy cosine similarity vector store
│   ├── ics.py                  # Informed Chapter Search  — chapter sibling expansion
│   ├── iks.py                  # Informed Keyword Search  — cross-doc keyword expansion
│   ├── uks.py                  # Uninformed Keyword Search — query keyword parallel path
│   └── reranker.py             # Semantic reranker (sentence-transformers cosine)
│
├── evaluation/
│   └── metrics.py              # Recall@k, MRR@k, Context Precision/Recall, Faithfulness
│
├── utils/
│   ├── chunker.py              # Structure-aware chunker (preserves chapter hierarchy)
│   ├── keyword_extractor.py    # YAKE keyword extraction + optional LLM extractor
│   └── pdf_loader.py           # PDF ingestion via PyMuPDF / pdfplumber / pypdf
│
├── data/                       # Place your PDF documents here
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Install core dependencies

```bash
pip install sentence-transformers streamlit numpy networkx
```

### 2. Install a PDF library (to use your own papers)

```bash
pip install pymupdf          # recommended — best quality, heading detection
# OR
pip install pdfplumber        # alternative
# OR
pip install pypdf             # lightweight fallback
```

### 3. Optional — install YAKE for better keyword extraction

```bash
pip install yake
```
Without YAKE, a simple frequency-based fallback is used automatically.

### 4. Optional — OpenAI API key for LLM generation & reranking

```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```
Without a key, the system uses semantic reranking (sentence-transformers) and extractive answer generation — fully functional offline.

---

## Quick Start

### Run the CLI demo (built-in academic papers, no setup needed)

```bash
python demo.py
```

Indexes 3 sample papers (Transformer, RAG, Knowledge Graphs), runs 5 benchmark questions, and prints a GraphRAG vs Naive RAG comparison table.

### Launch the Streamlit UI

```bash
streamlit run app.py
```

Opens a browser interface with:
- PDF upload or one-click demo dataset loading
- Interactive Q&A chat
- Live knowledge graph visualisation
- GraphRAG vs Naive RAG evaluation dashboard

### Run on your own PDF papers

```bash
python demo.py  my_papers/
```

Indexes all PDFs in the folder and starts an interactive Q&A session in the terminal.

---

## Pipeline Overview

```
Documents
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 1 — INDEXING                                 │
│  Chunk → assign URI → build DKG (hierarchy)         │
│                     → build IKG (keywords)          │
│                     → embed → VectorStore           │
└─────────────────────────────────────────────────────┘
    │
    ▼  (query arrives)
┌─────────────────────────────────────────────────────┐
│  PHASE 2 — RETRIEVAL                                │
│  ① Vector Search   → top-k by cosine similarity    │
│  ② UKS             → query keywords → IKG          │
│  ③ ICS             → chapter siblings → DKG        │
│  ④ IKS             → keyword edges   → IKG         │
│  ⑤ Deduplicate     → URI-based                     │
│  ⑥ Rerank          → semantic cosine score         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 3 — GENERATION                               │
│  Top-k chunks → structured prompt → LLM → Answer   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 4 — EVALUATION                               │
│  Recall@k · MRR@k · CP · CR · K-Prec · AR · Faith  │
└─────────────────────────────────────────────────────┘
```

---

## The Three Retrieval Strategies

| Strategy | Abbrev | Graph Used | How It Works | Best For |
|---|---|---|---|---|
| Informed Chapter Search | ICS | DKG | Fetches all chunks from the same chapter as each retrieved chunk | Single-hop, structured docs |
| Informed Keyword Search | IKS | IKG | Fetches all chunks sharing any keyword with each retrieved chunk | Multi-hop, cross-document |
| Uninformed Keyword Search | UKS | IKG | Extracts keywords from the query directly, runs in parallel with vector search | Technical queries, synonym gaps |

All three run on every query. Results are merged, deduplicated by URI, and reranked semantically.

---

## Evaluation Metrics

All implemented in `evaluation/metrics.py`. Values in [0, 1], higher is better.

| Metric | Type | Description |
|---|---|---|
| Recall@k | Retrieval | Proportion of relevant URIs found in top-k |
| MRR@k | Retrieval | Reciprocal rank of first relevant result |
| Context Precision | Retrieval | Fraction of retrieved chunks that are relevant |
| Context Recall | Retrieval | Fraction of reference answer tokens covered by context |
| K-Precision | Generation | Fraction of answer tokens grounded in context (hallucination proxy) |
| Answer Recall | Generation | Fraction of reference answer tokens in the generated answer |
| Faithfulness | Generation | Fraction of answer sentences ≥60% grounded in context |

> **Note:** Recall@k, MRR@k, and Context Precision require labelled ground-truth URI sets. The other four metrics run without labels using lexical matching.

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `800` | Approximate tokens per chunk (paper optimum: 1000) |
| `num_keywords` | `5` | Keywords extracted per chunk for the IKG |
| `top_k` | `10` | Chunks retrieved from vector search |
| `pass_k` | `10` | Chunks passed to generation after reranking |
| `embedding_model` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings + reranking |
| `gen_model` | `gpt-3.5-turbo` | OpenAI model for generation (ignored if no API key) |
| `rerank_model` | `gpt-3.5-turbo` | OpenAI model for reranking (falls back to semantic cosine) |

Parameters can be adjusted via the Streamlit sidebar sliders at runtime, or passed directly to `GraphRAGPipeline(...)`.

---

## Using the Streamlit UI

```bash
streamlit run app.py
```

**Sidebar** — toggle PDF upload vs demo dataset, view index stats, adjust parameters live  
**Q&A Tab** — chat interface with retrieval badges showing how many chunks each strategy contributed  
**Knowledge Graph Tab** — interactive vis.js graph; blue=Documents, green=Chapters, purple=Sections, grey=Chunks; dashed lines=IKG keyword links; retrieved chunks highlight red after each query  
**Evaluation Tab** — one-click GraphRAG vs Naive RAG benchmark across all 5 questions with metric deltas  

---

## Demo Results (offline, sentence-transformers)

| Metric | Naive RAG | GraphRAG | Δ |
|---|---|---|---|
| Context Recall | 0.6769 | 0.6769 | – |
| K-Precision | 0.8844 | 0.8716 | ▼ 0.013 |
| **Answer Recall** | **0.1800** | **0.3077** | **▲ +12.8%** |
| Faithfulness | 0.6000 | 0.6000 | – |

> K-Precision is slightly lower on the 18-chunk demo corpus because graph expansion retrieves broader context. This effect reverses at scale (500+ chunks) where vector search starts missing relevant content. Answer Recall improvement is consistent.

---

## Embedding Priority

The system selects the best available method automatically:

```
1. OpenAI text-embedding-ada-002   (if OPENAI_API_KEY is set)
2. sentence-transformers           (if installed — recommended)
3. Hash-based BoW fallback         (always available, no semantic understanding)
```

The `all-MiniLM-L6-v2` model (~90MB) downloads automatically on first run and caches locally.

---

## Reference

```
Knollmeyer, S.; Caymazer, O.; Grossmann, D.
Document GraphRAG: Knowledge Graph Enhanced Retrieval Augmented Generation
for Document Question Answering Within the Manufacturing Domain.
Electronics 2025, 14, 2102.
https://doi.org/10.3390/electronics14112102
```