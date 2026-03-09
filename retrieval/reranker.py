"""
reranker.py — Reranker using sentence-transformers cosine similarity.
Falls back to LLM scoring if OpenAI client provided, else uses semantic similarity.
Mirrors paper Appendix C: scores 1-10, sorted descending, top-k returned.
"""

from __future__ import annotations
import re
from typing import List, Optional
import numpy as np


RERANK_SYSTEM_PROMPT = (
    "You are an expert in processing and filtering search results. "
    "Help the user in sorting the search results by relevance."
)

RERANK_USER_TEMPLATE = """You will be provided with text chunks and a question.
Evaluate each chunk's relevance to the question and assign a score from 1 to 10.
10 = highly relevant. 1 = not relevant at all.

Question: {query}

Chunks to evaluate:
{chunks_text}

For each chunk, respond with EXACTLY this format on separate lines:
CHUNK_ID: <id> | SCORE: <1-10> | REASON: <brief reason>

Evaluate ALL {n_chunks} chunks. Output nothing else."""


# Load sentence-transformers model (reuse if already loaded in vector_store)
try:
    from sentence_transformers import SentenceTransformer, util
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    _ST_MODEL = None


class LLMReranker:
    """
    Reranks chunks by relevance to a query.
    Priority: OpenAI LLM → sentence-transformers cosine → keyword overlap
    """

    def __init__(
        self,
        openai_client=None,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 2,
    ):
        self.client = openai_client
        self.model = model
        self.max_retries = max_retries

    def rerank(self, chunks: List[dict], query: str, top_k: int = 10) -> List[dict]:
        if not chunks:
            return []

        if self.client:
            scored = self._llm_rerank(chunks, query)
        elif ST_AVAILABLE and _ST_MODEL is not None:
            scored = self._semantic_rerank(chunks, query)
        else:
            scored = self._keyword_rerank(chunks, query)

        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Semantic reranking via sentence-transformers (FREE, LOCAL)
    # ------------------------------------------------------------------

    def _semantic_rerank(self, chunks: List[dict], query: str) -> List[dict]:
        """
        Score each chunk by cosine similarity to the query embedding.
        Maps similarity [-1, 1] → score [1, 10].
        """
        texts = [c.get("text", "") for c in chunks]
        query_emb  = _ST_MODEL.encode(query,  convert_to_tensor=True)
        chunk_embs = _ST_MODEL.encode(texts,  convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_emb, chunk_embs)[0].cpu().numpy()

        for i, chunk in enumerate(chunks):
            sim   = float(cosine_scores[i])
            score = round(1 + (sim + 1) / 2 * 9, 2)   # map [-1,1] → [1,10]
            chunk["rerank_score"]  = score
            chunk["rerank_reason"] = f"Semantic similarity: {sim:.3f}"

        return chunks

    # ------------------------------------------------------------------
    # LLM reranking (OpenAI)
    # ------------------------------------------------------------------

    def _llm_rerank(self, chunks: List[dict], query: str) -> List[dict]:
        chunks_text = "\n\n".join(
            f"[CHUNK {i}]\n{c.get('text', '')[:600]}"
            for i, c in enumerate(chunks)
        )
        prompt = RERANK_USER_TEMPLATE.format(
            query=query, chunks_text=chunks_text, n_chunks=len(chunks)
        )
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0, max_tokens=800,
                )
                raw    = resp.choices[0].message.content
                scores = self._parse_scores(raw, len(chunks))
                if len(scores) == len(chunks):
                    for i, chunk in enumerate(chunks):
                        chunk["rerank_score"]  = scores[i]["score"]
                        chunk["rerank_reason"] = scores[i]["reason"]
                    return chunks
            except Exception as e:
                print(f"[Reranker] LLM error (attempt {attempt+1}): {e}")

        return self._semantic_rerank(chunks, query) if ST_AVAILABLE else self._keyword_rerank(chunks, query)

    @staticmethod
    def _parse_scores(raw: str, n_chunks: int) -> List[dict]:
        pattern = re.compile(
            r'CHUNK_ID:\s*(\d+)\s*\|\s*SCORE:\s*(\d+)\s*\|\s*REASON:\s*(.+)',
            re.IGNORECASE,
        )
        results = {}
        for line in raw.split('\n'):
            m = pattern.search(line)
            if m:
                idx   = int(m.group(1))
                score = max(1, min(10, int(m.group(2))))
                results[idx] = {"score": score, "reason": m.group(3).strip()}
        return [results.get(i, {"score": 5, "reason": "not scored"}) for i in range(n_chunks)]

    # ------------------------------------------------------------------
    # Keyword overlap fallback (last resort)
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_rerank(chunks: List[dict], query: str) -> List[dict]:
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        for chunk in chunks:
            text_words = set(re.findall(r'\b\w{3,}\b', chunk.get("text", "").lower()))
            overlap    = len(query_words & text_words)
            score      = min(10, max(1, int(overlap * 10 / max(len(query_words), 1))))
            chunk["rerank_score"]  = score
            chunk["rerank_reason"] = f"Keyword overlap: {overlap}/{len(query_words)}"
        return chunks