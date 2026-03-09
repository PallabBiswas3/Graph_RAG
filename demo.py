import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
demo.py  —  End-to-End GraphRAG Demo
--------------------------------------
Runs the full pipeline on 3 sample academic paper summaries covering:
  1. Transformer / Attention mechanisms
  2. Retrieval-Augmented Generation
  3. Knowledge Graphs in NLP

Then evaluates 5 questions:
  - 2 single-hop (SQuAD-style)
  - 2 multi-hop (HotpotQA-style, answers span multiple papers)
  - 1 unanswerable

Compares GraphRAG vs Naive RAG on all evaluation metrics.
"""

import os
import json
from pipeline import GraphRAGPipeline
from evaluation.metrics import RAGEvaluator

# ──────────────────────────────────────────────────────────────────────
# Optional: plug in your OpenAI key for LLM embeddings + generation
# Without it, the demo uses deterministic fallbacks (still runs fully)
# ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("[Demo] OpenAI client initialised.")
else:
    print("[Demo] No OPENAI_API_KEY found — using fallback embeddings & extraction.")

# ──────────────────────────────────────────────────────────────────────
# Sample Academic Documents
# ──────────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    {
        "doc_id": "paper_transformer",
        "doc_title": "Attention Is All You Need",
        "metadata": {"author": "Vaswani et al.", "year": "2017"},
        "text": """
# 1. Introduction

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
The best performing models also connect the encoder and decoder through an attention mechanism.
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely.

The Transformer allows for significantly more parallelization and can reach a new state of the art
in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# 2. Background

The goal of reducing sequential computation forms the foundation of the Extended Neural GPU,
ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block.
In these models, the number of operations required to relate signals from two arbitrary input or output positions
grows in the distance between positions.

Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions
of a single sequence in order to compute a representation of the sequence. Self-attention has been used
successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations.

# 3. Model Architecture

## 3.1 Encoder and Decoder Stacks

The encoder maps an input sequence to a sequence of continuous representations.
The encoder is composed of a stack of N=6 identical layers.
Each layer has two sub-layers: a multi-head self-attention mechanism, and a fully connected feed-forward network.

The decoder is also composed of a stack of N=6 identical layers.
In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer,
which performs multi-head attention over the output of the encoder stack.

## 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output.
The output is computed as a weighted sum of the values, where the weight assigned to each value is computed
by a compatibility function of the query with the corresponding key.

We call our particular attention "Scaled Dot-Product Attention".
The input consists of queries and keys of dimension dk, and values of dimension dv.
We compute the dot products of the query with all keys, divide each by sqrt(dk),
and apply a softmax function to obtain the weights on the values.

Multi-head attention allows the model to jointly attend to information from different representation subspaces
at different positions. With a single attention head, averaging inhibits this.

## 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in the encoder and decoder contains
a fully connected feed-forward network, which is applied to each position separately and identically.
This consists of two linear transformations with a ReLU activation in between.

# 4. Limitations and Future Work

The Transformer relies entirely on attention and does not use any recurrence.
One limitation is that self-attention is quadratic in the sequence length, making it expensive
for very long sequences. This has motivated research into sparse attention patterns.

Another limitation is that the model requires a large amount of training data to perform well.
Transfer learning and pre-training approaches have been developed to address this limitation.
The model also lacks an explicit memory mechanism for storing long-term dependencies.
"""
    },
    {
        "doc_id": "paper_rag",
        "doc_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "metadata": {"author": "Lewis et al.", "year": "2020"},
        "text": """
# 1. Introduction

Large pre-trained language models have been shown to store factual knowledge in their parameters.
However, their ability to access and manipulate knowledge is still limited compared to task-specific architectures.
We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG).

RAG models combine pre-trained parametric and non-parametric memory for language generation.
The parametric memory is a pre-trained language model and the non-parametric memory is a dense vector index
of Wikipedia, accessed with a pre-trained neural retriever.

Key advantages of RAG include the ability to update knowledge without full retraining,
improved factual accuracy, and reduced hallucinations compared to pure parametric models.
RAG has become the dominant paradigm for question answering over large document collections.

# 2. Methods

## 2.1 Retriever

The retriever takes a query and returns the top-k most relevant documents from an external corpus.
We use a bi-encoder retriever based on Dense Passage Retrieval (DPR).
The retriever encodes both the query and documents into a shared vector space using BERT.
Documents are retrieved using maximum inner product search.

## 2.2 Generator

The generator takes the input query concatenated with the retrieved documents and produces an output.
We fine-tune a pre-trained sequence-to-sequence model (BART) as the generator.
The generator must learn to attend to the most relevant parts of the retrieved context.

## 2.3 Training

Both the retriever and generator can be fine-tuned end-to-end.
However, the retriever index is typically kept fixed during training due to computational constraints.
The model is trained to maximize the likelihood of the correct answer.

# 3. Limitations

RAG systems have several key limitations that motivate further research.
First, retrieval quality is a bottleneck: if relevant documents are not retrieved, the generator cannot produce correct answers.
Second, multi-hop reasoning is difficult because each retrieval step is independent.
Third, the system struggles with domain-specific or long-tail knowledge that is sparse in the corpus.
Chunking strategies also significantly affect retrieval quality, as poor chunk boundaries can separate related information.
Context window limitations constrain how many retrieved documents can be provided to the generator.

# 4. Evaluation

We evaluate on three knowledge-intensive NLP benchmarks: Natural Questions, TriviaQA, and WebQuestions.
RAG outperforms pure parametric models on all benchmarks.
The model also achieves competitive results on fact verification tasks.
Open-domain question answering remains a challenging benchmark for RAG systems.
"""
    },
    {
        "doc_id": "paper_kg",
        "doc_title": "Knowledge Graphs: Opportunities and Challenges",
        "metadata": {"author": "Peng et al.", "year": "2023"},
        "text": """
# 1. Introduction

Knowledge Graphs (KGs) represent structured relationships between entities in a graph format.
Each node represents an entity such as a person, place, concept, or event.
Each edge represents a relationship between two entities, such as 'bornIn', 'worksAt', or 'partOf'.
KGs originated from the Semantic Web vision and have evolved into practical tools for knowledge management.

Major examples include Wikidata, Freebase, DBpedia, and domain-specific KGs in medicine and science.
KGs are used in web search, recommendation systems, question answering, and natural language understanding.

# 2. Knowledge Graph Construction

## 2.1 Entity Extraction

Named Entity Recognition (NER) is the foundational task for KG construction.
Modern NER systems use transformer-based models fine-tuned on annotated corpora.
Entities are classified into types such as Person, Organization, Location, and Event.
Disambiguation is required when the same surface form refers to multiple distinct entities.

## 2.2 Relation Extraction

Relation extraction identifies semantic relationships between entity pairs.
Supervised approaches require large annotated datasets with labeled entity-relation triples.
Distant supervision uses existing KGs to automatically label training data.
Open Information Extraction extracts relations in a schema-free manner.

# 3. Knowledge Graphs and Language Models

## 3.1 Integration Patterns

KGs can enhance language models in several ways.
First, KG embeddings can be injected into the model's token representations.
Second, retrieved KG triples can be provided as additional context during inference.
Third, the model can be trained to generate structured KG triples as output.

## 3.2 KG-Enhanced Retrieval

When combined with RAG systems, KGs provide structured retrieval pathways beyond vector similarity.
Entities and their relationships can guide the retrieval of contextually relevant passages.
Multi-hop reasoning becomes possible by traversing KG edges across multiple steps.
This is a key advantage over naive vector-based RAG for complex question answering.

KG-enhanced RAG systems have shown improvements on multi-hop benchmarks like HotpotQA.
The structured nature of KGs complements the semantic flexibility of dense vector retrieval.

# 4. Challenges

Key challenges in KG development include knowledge incompleteness, temporal staleness, and noise.
Constructing high-quality KGs remains labor-intensive, requiring significant domain expertise.
Aligning KGs across languages and domains is an unsolved problem.
Scalability is a concern as KGs grow to billions of triples.
Integrating KGs with modern neural language models requires careful architectural design.
"""
    },
]

# ──────────────────────────────────────────────────────────────────────
# Evaluation Questions
# ──────────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "What is the quadratic complexity limitation of self-attention?",
        "reference": "Self-attention is quadratic in the sequence length, making it expensive for very long sequences.",
        "relevant_doc": "paper_transformer",
        "type": "single-hop",
    },
    {
        "question": "What are the main limitations of RAG systems for question answering?",
        "reference": "RAG limitations include retrieval quality bottleneck, difficulty with multi-hop reasoning, poor handling of domain-specific long-tail knowledge, chunking strategy effects, and context window constraints.",
        "relevant_doc": "paper_rag",
        "type": "single-hop",
    },
    {
        "question": "How do Knowledge Graphs improve multi-hop reasoning in RAG systems, and what was the original model architecture that KG-enhanced RAG aims to complement?",
        "reference": "KGs provide structured retrieval by traversing edges across multiple steps enabling multi-hop reasoning. The Transformer architecture uses attention mechanisms and was the basis for RAG retrievers.",
        "relevant_doc": "multi-doc",
        "type": "multi-hop",
    },
    {
        "question": "What training approach does RAG use for its retriever and how does the encoder design in transformers support this?",
        "reference": "RAG uses a bi-encoder retriever based on Dense Passage Retrieval using BERT. The Transformer encoder uses stacked self-attention and feed-forward layers to produce rich representations.",
        "relevant_doc": "multi-doc",
        "type": "multi-hop",
    },
    {
        "question": "What was the stock price of OpenAI in 2023?",
        "reference": "",
        "relevant_doc": None,
        "type": "unanswerable",
    },
]


# ──────────────────────────────────────────────────────────────────────
# Main Demo
# ──────────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "="*60)
    print("  ACADEMIC GRAPHRAG DEMO")
    print("  Based on: Knollmeyer et al. (2025)")
    print("="*60)

    # ── Build GraphRAG pipeline ─────────────────────────────────────
    print("\n[1/4] Initialising GraphRAG Pipeline...")
    graphrag = GraphRAGPipeline(
        openai_client=openai_client,
        chunk_size=800,
        num_keywords=5,
        top_k=10,
        pass_k=10,
    )

    # ── Index documents ─────────────────────────────────────────────
    print("\n[2/4] Indexing 3 academic papers...")
    graphrag.index_documents(SAMPLE_DOCS)

    # ── Build naive RAG pipeline (no graph, same vector DB) ─────────
    from retrieval.vector_store import VectorStore
    naive_vs = VectorStore(openai_client=openai_client,
                           collection_name="naive_rag_academic")
    # Reuse same chunks by re-indexing
    for doc in SAMPLE_DOCS:
        from utils.chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=800)
        chunks = chunker.chunk_document(
            doc["text"], doc["doc_id"], doc["doc_title"], doc.get("metadata")
        )
        naive_vs.add_chunks(chunks)

    # ── Run evaluation ───────────────────────────────────────────────
    print("\n[3/4] Running evaluation on all questions...\n")
    evaluator = RAGEvaluator()
    graphrag_instances = []
    naive_instances    = []

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        q    = item["question"]
        ref  = item["reference"]
        qtype = item["type"]

        print(f"\n{'─'*55}")
        print(f"Q{i} [{qtype.upper()}]: {q}")

        # --- GraphRAG ---
        gr_result = graphrag.query(q, reference_answer=ref)

        # --- Naive RAG ---
        naive_chunks = naive_vs.search(q, k=10)
        naive_answer = graphrag.generate(q, naive_chunks[:10])

        print(f"\n  GraphRAG Answer : {gr_result['answer'][:200]}...")
        print(f"  Naive Answer    : {naive_answer[:200]}...")
        print(f"\n  Retrieval stats :")
        for k, v in gr_result["stats"].items():
            print(f"    {k:<20} {v}")

        # Collect for batch evaluation
        graphrag_instances.append({
            "query":            q,
            "retrieved_chunks": gr_result["final_chunks"],
            "answer":           gr_result["answer"],
            "reference_answer": ref,
        })
        naive_instances.append({
            "query":            q,
            "retrieved_chunks": naive_chunks,
            "answer":           naive_answer,
            "reference_answer": ref,
        })

    # ── Aggregate results ────────────────────────────────────────────
    print("\n[4/4] Aggregated Evaluation Results")
    graphrag_avg = evaluator.evaluate_batch(graphrag_instances)
    naive_avg    = evaluator.evaluate_batch(naive_instances)

    evaluator.print_results(naive_avg,    "Naive RAG  (average)")
    evaluator.print_results(graphrag_avg, "GraphRAG   (average)")

    # ── Delta summary ────────────────────────────────────────────────
    print("\n📊  IMPROVEMENT (GraphRAG vs Naive RAG)")
    print(f"{'Metric':<25} {'Naive':>8} {'GraphRAG':>10} {'Δ':>8}")
    print("─" * 55)
    for metric in graphrag_avg:
        g = graphrag_avg[metric]
        n = naive_avg.get(metric, 0)
        delta = g - n
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        print(f"  {metric:<23} {n:>8.4f} {g:>10.4f} {arrow}{abs(delta):>6.4f}")

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    output = {
        "graphrag_avg": graphrag_avg,
        "naive_avg":    naive_avg,
        "questions":    EVAL_QUESTIONS,
    }
    with open("output/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅  Results saved to output/results.json")
    print("\n" + "="*60)


if __name__ == "__main__":
    run_demo()


# ──────────────────────────────────────────────────────────────────────
# BONUS: Load your own PDFs
# ──────────────────────────────────────────────────────────────────────

def run_with_pdfs(pdf_folder: str):
    """
    Run GraphRAG on your own PDF academic papers.

    Usage:
        Place PDFs in a folder, then call:
        >>> run_with_pdfs("my_papers/")

    Install a PDF library first:
        pip install pymupdf        (recommended)
        pip install pdfplumber     (alternative)
        pip install pypdf          (lightweight)
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from utils.pdf_loader import load_pdf_folder
    from pipeline import GraphRAGPipeline

    print(f"\n{'='*60}")
    print(f"  GraphRAG on YOUR PDFs: {pdf_folder}")
    print(f"{'='*60}")

    docs = load_pdf_folder(pdf_folder)
    if not docs:
        print("No PDFs found. Add PDF files to the folder and try again.")
        return

    pipeline = GraphRAGPipeline(
        openai_client=openai_client,
        chunk_size=1000,
        num_keywords=5,
        top_k=10,
        pass_k=10,
    )
    pipeline.index_documents(docs)

    print(f"\nReady! Ask questions about your {len(docs)} paper(s).")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = pipeline.query(question, evaluate=False)
        print(f"\nAnswer: {result['answer']}")
        print(f"Retrieval: vector={result['stats']['vector_hits']} "
              f"ics={result['stats']['ics_hits']} "
              f"iks={result['stats']['iks_hits']} "
              f"uks={result['stats']['uks_hits']} "
              f"combined={result['stats']['combined']}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run on PDF folder: python demo.py my_papers/
        run_with_pdfs(sys.argv[1])
    else:
        # Run default demo
        run_demo()