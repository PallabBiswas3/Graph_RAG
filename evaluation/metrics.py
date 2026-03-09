"""
metrics.py  —  RAG Evaluation Metrics
---------------------------------------
Implements all evaluation dimensions described in the paper (Section 6.4):

RETRIEVAL METRICS (Context Relevance):
  - Recall@k          : proportion of relevant chunks retrieved in top-k
  - MRR@k             : Mean Reciprocal Rank of first relevant chunk
  - Context Precision : fraction of retrieved chunks that are relevant (LLM-free approx)
  - Context Recall    : fraction of needed context that was retrieved

GENERATION METRICS:
  - K-Precision       : token overlap between answer and retrieved context
  - Answer Recall     : token overlap between answer and ground-truth
  - Faithfulness      : fraction of answer claims supported by context

All metrics return values in [0, 1] — higher is better.
"""

from __future__ import annotations
import re
from typing import List, Set, Optional, Dict


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple word tokenizer (lowercase, alphanumeric only)."""
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())


def _token_set(text: str) -> Set[str]:
    return set(_tokenize(text))


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_uris: List[str],
    relevant_uris: Set[str],
    k: int = 10,
) -> float:
    """
    Recall@k: proportion of relevant URIs found within the top-k retrieved.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_uris : ranked list of retrieved chunk URIs
        relevant_uris  : set of ground-truth relevant URIs
        k              : cutoff rank

    Returns float in [0, 1].
    """
    if not relevant_uris:
        return 0.0
    top_k = set(retrieved_uris[:k])
    hits = top_k & relevant_uris
    return len(hits) / len(relevant_uris)


def mrr_at_k(
    retrieved_uris: List[str],
    relevant_uris: Set[str],
    k: int = 10,
) -> float:
    """
    Mean Reciprocal Rank@k: reciprocal of the rank of the FIRST relevant result.

    MRR@k = 1/rank_first_relevant   (0 if none found in top-k)

    Returns float in [0, 1].
    """
    for rank, uri in enumerate(retrieved_uris[:k], start=1):
        if uri in relevant_uris:
            return 1.0 / rank
    return 0.0


def context_precision(
    retrieved_chunks: List[dict],
    relevant_uris: Set[str],
    k: int = 10,
) -> float:
    """
    Context Precision@k: fraction of top-k retrieved chunks that ARE relevant.

    CP@k = |retrieved[:k] ∩ relevant| / k

    Returns float in [0, 1].
    """
    if not retrieved_chunks or not relevant_uris:
        return 0.0
    top_k = retrieved_chunks[:k]
    hits = sum(1 for c in top_k if c.get("uri") in relevant_uris)
    return hits / len(top_k)


def context_recall(
    retrieved_chunks: List[dict],
    reference_answer: str,
    k: int = 10,
) -> float:
    """
    Context Recall (lexical approximation):
    Fraction of reference answer tokens covered by the retrieved context.

    CR = |tokens(reference) ∩ tokens(context)| / |tokens(reference)|

    Returns float in [0, 1].
    """
    if not reference_answer.strip():
        return 0.0

    context_text = " ".join(c.get("text", "") for c in retrieved_chunks[:k])
    ref_tokens  = _token_set(reference_answer)
    ctx_tokens  = _token_set(context_text)

    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ctx_tokens) / len(ref_tokens)


# ---------------------------------------------------------------------------
# Generation Metrics
# ---------------------------------------------------------------------------

def k_precision(answer: str, retrieved_chunks: List[dict]) -> float:
    """
    K-Precision: proportion of tokens in the generated answer that
    appear in the retrieved context.

    KP = |tokens(answer) ∩ tokens(context)| / |tokens(answer)|

    Higher = answer is well-grounded in retrieved content.
    Returns float in [0, 1].
    """
    if not answer.strip():
        return 0.0
    context = " ".join(c.get("text", "") for c in retrieved_chunks)
    ans_tokens = _token_set(answer)
    ctx_tokens = _token_set(context)
    if not ans_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def answer_recall(answer: str, reference: str) -> float:
    """
    Answer Recall: proportion of reference answer tokens present in the
    generated answer.

    AR = |tokens(reference) ∩ tokens(answer)| / |tokens(reference)|

    Measures correctness without penalizing extra (non-contradictory) info.
    Returns float in [0, 1].
    """
    if not reference.strip():
        return 0.0
    ref_tokens = _token_set(reference)
    ans_tokens = _token_set(answer)
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ans_tokens) / len(ref_tokens)


def faithfulness_score(answer: str, retrieved_chunks: List[dict]) -> float:
    """
    Faithfulness (lexical approximation):
    Fraction of answer sentences where MOST tokens appear in the context.

    A sentence is "faithful" if ≥60% of its content tokens are in the context.
    Returns float in [0, 1].
    """
    if not answer.strip() or not retrieved_chunks:
        return 0.0

    context_tokens = _token_set(
        " ".join(c.get("text", "") for c in retrieved_chunks)
    )

    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    faithful = 0

    for sent in sentences:
        sent_tokens = _token_set(sent)
        content_tokens = {t for t in sent_tokens if len(t) > 3}
        if not content_tokens:
            faithful += 1   # short / functional sentences are counted as faithful
            continue
        overlap = len(content_tokens & context_tokens) / len(content_tokens)
        if overlap >= 0.6:
            faithful += 1

    return faithful / max(len(sentences), 1)


# ---------------------------------------------------------------------------
# Aggregate Evaluation Runner
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates a complete RAG run (retrieval + generation) for one query.

    Usage:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(
            query="What is transformer attention?",
            retrieved_chunks=[...],
            answer="Transformer attention is...",
            relevant_uris={"urn:graphrag:doc1:chunk:0003"},
            reference_answer="Attention mechanisms allow...",
            k=10,
        )
        print(results)
    """

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[dict],
        answer: str,
        relevant_uris: Optional[Set[str]] = None,
        reference_answer: str = "",
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single QA instance.

        Returns dict with keys:
            recall_at_k, mrr_at_k, context_precision, context_recall,
            k_precision, answer_recall, faithfulness
        """
        retrieved_uris = [c.get("uri", "") for c in retrieved_chunks]
        relevant_uris  = relevant_uris or set()

        return {
            # --- Retrieval ---
            "recall_at_k":        recall_at_k(retrieved_uris, relevant_uris, k),
            "mrr_at_k":           mrr_at_k(retrieved_uris, relevant_uris, k),
            "context_precision":  context_precision(retrieved_chunks, relevant_uris, k),
            "context_recall":     context_recall(retrieved_chunks, reference_answer, k),
            # --- Generation ---
            "k_precision":        k_precision(answer, retrieved_chunks),
            "answer_recall":      answer_recall(answer, reference_answer),
            "faithfulness":       faithfulness_score(answer, retrieved_chunks),
        }

    def evaluate_batch(
        self,
        instances: List[dict],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Average metrics over a list of evaluation instances.

        Each instance dict must contain:
            query, retrieved_chunks, answer,
            relevant_uris (optional), reference_answer (optional)
        """
        if not instances:
            return {}

        totals: Dict[str, float] = {}
        for inst in instances:
            scores = self.evaluate(
                query=inst.get("query", ""),
                retrieved_chunks=inst.get("retrieved_chunks", []),
                answer=inst.get("answer", ""),
                relevant_uris=inst.get("relevant_uris"),
                reference_answer=inst.get("reference_answer", ""),
                k=k,
            )
            for metric, val in scores.items():
                totals[metric] = totals.get(metric, 0.0) + val

        n = len(instances)
        return {m: round(v / n, 4) for m, v in totals.items()}

    @staticmethod
    def print_results(metrics: Dict[str, float], title: str = "Results") -> None:
        """Pretty-print a metrics dict."""
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
        print(f"  RETRIEVAL METRICS")
        print(f"  {'Recall@k':<25} {metrics.get('recall_at_k', 0):.4f}")
        print(f"  {'MRR@k':<25} {metrics.get('mrr_at_k', 0):.4f}")
        print(f"  {'Context Precision':<25} {metrics.get('context_precision', 0):.4f}")
        print(f"  {'Context Recall':<25} {metrics.get('context_recall', 0):.4f}")
        print(f"\n  GENERATION METRICS")
        print(f"  {'K-Precision':<25} {metrics.get('k_precision', 0):.4f}")
        print(f"  {'Answer Recall':<25} {metrics.get('answer_recall', 0):.4f}")
        print(f"  {'Faithfulness':<25} {metrics.get('faithfulness', 0):.4f}")
        print(f"{'='*50}\n")
