# Academic GraphRAG Pipeline

A full implementation of the Document GraphRAG framework (Knollmeyer et al., 2025)
applied to **Academic Research** documents.

## Project Structure

```
graphrag_academic/
├── data/                   # Place your PDF/text documents here
│   └── sample_docs/        # Sample academic papers (auto-generated for demo)
├── graph/
│   ├── dkg.py              # Document Knowledge Graph construction
│   └── ikg.py              # Information Knowledge Graph (keyword layer)
├── retrieval/
│   ├── vector_store.py     # Embedding + Chroma vector DB
│   ├── ics.py              # Informed Chapter Search
│   ├── iks.py              # Informed Keyword Search
│   ├── uks.py              # Uninformed Keyword Search
│   └── reranker.py         # LLM-based reranker
├── evaluation/
│   └── metrics.py          # Recall@k, MRR@k, Context Precision/Recall
├── utils/
│   ├── chunker.py          # Document chunking
│   └── keyword_extractor.py# YAKE-based keyword extraction
├── pipeline.py             # Full GraphRAG pipeline orchestrator
├── demo.py                 # End-to-end demo with sample data
├── requirements.txt        # All dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key (used for embeddings + LLM reranking/generation):
```bash
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

```bash
# Run full demo with built-in sample academic documents
python demo.py

# Or use the pipeline directly on your own docs
python pipeline.py --docs_dir data/ --query "What are the limitations of transformer models?"
```

## Pipeline Overview

1. **Indexing**: Documents → Chunks → DKG (hierarchy) + IKG (keywords) + Vector DB
2. **Retrieval**: Vector search → ICS + IKS + UKS graph expansion → Deduplication → Reranking
3. **Generation**: Top-k reranked chunks → LLM → Answer with citations
4. **Evaluation**: Recall@k, MRR@k, Context Precision, Context Recall

## Parameters

| Parameter     | Default | Description                          |
|---------------|---------|--------------------------------------|
| chunk_size    | 1000    | Tokens per chunk                     |
| keywords      | 5       | Keywords extracted per chunk         |
| top_k         | 10      | Chunks retrieved from vector DB      |
| pass_k        | 10      | Chunks passed to LLM after reranking |
