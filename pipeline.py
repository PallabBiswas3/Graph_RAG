from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
pipeline.py  —  GraphRAG Pipeline Orchestrator
-------------------------------------------------
Ties together all components into the full 3-phase pipeline
described in the paper (Figure 3):

    Phase 1 — INDEXING
        Documents → Chunks → DKG + IKG + VectorDB

    Phase 2 — RETRIEVAL
        Query → Vector Search → ICS + IKS + UKS → Dedup → Rerank

    Phase 3 — GENERATION
        Top-k Chunks → Prompt → LLM → Answer

Usage:
    pipeline = GraphRAGPipeline(openai_client=client)
    pipeline.index_documents(docs)
    result = pipeline.query("What are the limitations of transformers?")
    print(result["answer"])
"""

import os
from typing import List, Dict, Optional, Set

from utils.chunker import DocumentChunker, Chunk
from utils.keyword_extractor import YAKEExtractor
from graph.dkg import DocumentKnowledgeGraph
from graph.ikg import InformationKnowledgeGraph
from retrieval.vector_store import VectorStore
from retrieval.ics import informed_chapter_search
from retrieval.iks import informed_keyword_search
from retrieval.uks import uninformed_keyword_search
from retrieval.reranker import LLMReranker
from evaluation.metrics import RAGEvaluator


# ---------------------------------------------------------------------------
# Generation prompt (Appendix A of the paper)
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = (
    "You are an AI assistant working in a Retrieval Augmented Generation pipeline. "
    "Your task is to generate an accurate answer based solely on the provided context."
)

GENERATION_USER = """You will be provided with context chunks and a question.
Answer the question using ONLY information from the context. Cite the source chunk number.

<context>
{context}
</context>

<question>
{question}
</question>

Instructions:
1. Carefully read the context.
2. Answer the question using only the context.
3. Cite which chunk(s) your answer comes from (e.g., [Chunk 2]).
4. If the answer is not in the context, say: "This question cannot be answered with the provided context."
5. Do not use external knowledge.

<answer>"""


class GraphRAGPipeline:
    """
    Full GraphRAG pipeline with all three retrieval strategies.

    Parameters
    ----------
    openai_client   : OpenAI client instance (optional; uses fallbacks if None)
    chunk_size      : tokens per chunk (default 1000)
    num_keywords    : keywords extracted per chunk (default 5)
    top_k           : chunks from vector search (default 10)
    pass_k          : chunks passed to LLM after reranking (default 10)
    gen_model       : LLM model for answer generation
    rerank_model    : LLM model for reranking
    """

    def __init__(
        self,
        openai_client=None,
        chunk_size: int = 1000,
        num_keywords: int = 5,
        top_k: int = 10,
        pass_k: int = 10,
        gen_model: str = "gpt-3.5-turbo",
        rerank_model: str = "gpt-3.5-turbo",
    ):
        self.client       = openai_client
        self.chunk_size   = chunk_size
        self.num_keywords = num_keywords
        self.top_k        = top_k
        self.pass_k       = pass_k
        self.gen_model    = gen_model

        # Components
        self.chunker    = DocumentChunker(chunk_size=chunk_size)
        self.kw_extract = YAKEExtractor(num_keywords=num_keywords)
        self.dkg        = DocumentKnowledgeGraph()
        self.ikg        = InformationKnowledgeGraph()
        self.vector_db  = VectorStore(openai_client=openai_client)
        self.reranker   = LLMReranker(openai_client=openai_client,
                                       model=rerank_model)
        self.evaluator  = RAGEvaluator()

        self._all_chunks: List[Chunk] = []

    # ==================================================================
    # PHASE 1 — INDEXING
    # ==================================================================

    def index_document(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Index a single document. Returns number of chunks created.
        """
        chunks = self.chunker.chunk_document(
            text=text, doc_id=doc_id, doc_title=doc_title, metadata=metadata
        )
        if not chunks:
            print(f"[Pipeline] No chunks created for '{doc_title}'")
            return 0

        # DKG: hierarchical structure
        self.dkg.add_document(doc_id, doc_title, chunks, metadata)

        # IKG: keyword semantic layer
        self.ikg.build_from_chunks(chunks, self.kw_extract, self.num_keywords)

        # Vector DB: dense embeddings
        self.vector_db.add_chunks(chunks)

        self._all_chunks.extend(chunks)
        print(f"[Pipeline] Indexed '{doc_title}': {len(chunks)} chunks.")
        return len(chunks)

    def index_documents(self, docs: List[dict]) -> None:
        """
        Batch index multiple documents.

        Each doc dict: { text, doc_id, doc_title, metadata (optional) }
        """
        total = 0
        for doc in docs:
            n = self.index_document(
                text=doc["text"],
                doc_id=doc["doc_id"],
                doc_title=doc["doc_title"],
                metadata=doc.get("metadata"),
            )
            total += n
        print(f"\n[Pipeline] Indexing complete. Total chunks: {total}")
        print(f"[Pipeline] DKG stats: {self.dkg.get_stats()}")
        print(f"[Pipeline] IKG stats: {self.ikg.get_stats()}")

    # ==================================================================
    # PHASE 2 — RETRIEVAL
    # ==================================================================

    def retrieve(self, query: str) -> Dict:
        """
        Full GraphRAG retrieval for a query.

        Steps:
            1. Vector search → initial_chunks
            2. UKS (parallel, query keywords)
            3. ICS (chapter expansion of initial)
            4. IKS (keyword expansion of initial + UKS)
            5. Combine all → deduplicate → rerank → top pass_k

        Returns dict with all intermediate + final chunk sets.
        """
        # ---- Step 1: Vector search ------------------------------------
        initial_chunks = self.vector_db.search(query, k=self.top_k)
        initial_uris: Set[str] = {c["uri"] for c in initial_chunks}

        # ---- Step 2: UKS (query keywords → IKG) ----------------------
        uks_chunks = uninformed_keyword_search(
            query=query,
            ikg=self.ikg,
            extractor=self.kw_extract,
            existing_uris=initial_uris,
            num_keywords=self.num_keywords,
        )
        uks_uris = {c["uri"] for c in uks_chunks}

        # Merge UKS into pool for ICS/IKS
        pool = initial_chunks + uks_chunks
        pool_uris = initial_uris | uks_uris

        # ---- Step 3: ICS (chapter expansion) --------------------------
        ics_chunks = informed_chapter_search(
            initial_chunks=pool,
            dkg=self.dkg,
        )
        ics_uris = {c["uri"] for c in ics_chunks}

        # ---- Step 4: IKS (keyword expansion) --------------------------
        iks_chunks = informed_keyword_search(
            initial_chunks=pool,
            ikg=self.ikg,
        )
        iks_uris = {c["uri"] for c in iks_chunks}

        # ---- Step 5: Combine + deduplicate ----------------------------
        all_chunks_map: Dict[str, dict] = {}
        for c in (initial_chunks + uks_chunks + ics_chunks + iks_chunks):
            if c and c.get("uri"):
                all_chunks_map[c["uri"]] = c

        combined = list(all_chunks_map.values())

        # ---- Step 6: Rerank → top pass_k ------------------------------
        reranked = self.reranker.rerank(combined, query, top_k=self.pass_k)

        return {
            "query": query,
            "initial_chunks":  initial_chunks,
            "uks_chunks":      uks_chunks,
            "ics_chunks":      ics_chunks,
            "iks_chunks":      iks_chunks,
            "combined_chunks": combined,
            "final_chunks":    reranked,
            "stats": {
                "vector_hits": len(initial_chunks),
                "uks_hits":    len(uks_chunks),
                "ics_hits":    len(ics_chunks),
                "iks_hits":    len(iks_chunks),
                "combined":    len(combined),
                "after_rerank": len(reranked),
            }
        }

    # ==================================================================
    # PHASE 3 — GENERATION
    # ==================================================================

    def generate(self, query: str, chunks: List[dict]) -> str:
        """
        Generate an answer given a query and the final reranked chunks.
        Uses prompt template from Appendix A of the paper.
        """
        if not chunks:
            return "No relevant context found to answer this question."

        # Build context block with numbered chunks
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            doc  = chunk.get("doc_title", "Unknown")
            chap = chunk.get("chapter", "")
            text = chunk.get("text", "")
            context_parts.append(
                f"[Chunk {i}] Source: {doc} | Chapter: {chap}\n{text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = GENERATION_USER.format(context=context, question=query)

        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=self.gen_model,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=600,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Pipeline] Generation error: {e}")

        # Fallback: extractive answer (no LLM)
        return self._extractive_answer(query, chunks)

    # ==================================================================
    # FULL QUERY (retrieval + generation)
    # ==================================================================

    def query(
        self,
        question: str,
        relevant_uris: Optional[Set[str]] = None,
        reference_answer: str = "",
        evaluate: bool = True,
    ) -> dict:
        """
        End-to-end: retrieve → generate → (optionally) evaluate.

        Returns dict with answer, retrieved chunks, and metrics.
        """
        retrieval_result = self.retrieve(question)
        final_chunks = retrieval_result["final_chunks"]

        answer = self.generate(question, final_chunks)

        result = {
            "question": question,
            "answer":   answer,
            **retrieval_result,
        }

        if evaluate:
            metrics = self.evaluator.evaluate(
                query=question,
                retrieved_chunks=final_chunks,
                answer=answer,
                relevant_uris=relevant_uris,
                reference_answer=reference_answer,
                k=self.pass_k,
            )
            result["metrics"] = metrics

        return result

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _extractive_answer(query: str, chunks: List[dict]) -> str:
        """
        Simple extractive fallback: find the sentence in top chunks
        with the most query word overlap.
        """
        import re
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        best_sent, best_score = "", 0

        for chunk in chunks[:3]:
            text = chunk.get("text", "")
            for sent in re.split(r'(?<=[.!?])\s+', text):
                words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
                score = len(words & query_words)
                if score > best_score:
                    best_score = score
                    best_sent = sent

        if best_sent:
            src = chunks[0].get("doc_title", "context")
            return f"{best_sent} [Source: {src}]"
        return "Unable to generate an answer from the provided context."

    def print_query_result(self, result: dict) -> None:
        """Pretty-print a full query result."""
        print("\n" + "="*60)
        print(f"QUESTION: {result['question']}")
        print("="*60)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nRETRIEVAL STATS:")
        for k, v in result.get("stats", {}).items():
            print(f"  {k:<20} {v}")
        if "metrics" in result:
            self.evaluator.print_results(result["metrics"], "Evaluation Metrics")