"""
vector_store.py — In-memory numpy vector store with cosine similarity.
Uses sentence-transformers (all-MiniLM-L6-v2) for proper semantic embeddings.
Falls back to hash-based vectors if sentence-transformers is not installed.
"""

from __future__ import annotations
import math
from typing import List, Dict, Optional
import numpy as np

# Try sentence-transformers first (proper semantic embeddings)
try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
    print("[VectorStore] sentence-transformers loaded (all-MiniLM-L6-v2).")
except ImportError:
    ST_AVAILABLE = False
    _ST_MODEL = None
    print("[VectorStore] sentence-transformers not found — using hash fallback.")


class VectorStore:
    def __init__(
        self,
        openai_client=None,
        collection_name: str = "graphrag_academic",
        embedding_model: str = "text-embedding-ada-002",
        persist_dir: str = "./output",
    ):
        self.openai_client   = openai_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._uris:     List[str]         = []
        self._vectors:  List[List[float]] = []
        self._metadata: List[dict]        = []
        self._texts:    List[str]         = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks) -> None:
        if not chunks:
            return
        texts, uris, metas = [], [], []
        for c in chunks:
            if hasattr(c, "uri"):
                uri, text = c.uri, c.text
                meta = {
                    "doc_id":      c.doc_id,
                    "doc_title":   c.doc_title,
                    "chapter":     c.chapter,
                    "section":     c.section or "",
                    "chunk_index": c.chunk_index,
                }
            else:
                uri  = c["uri"]
                text = c["text"]
                meta = {k: v for k, v in c.items()
                        if k not in ("uri", "text")
                        and isinstance(v, (str, int, float))}
            if uri in self._uris:
                continue
            texts.append(text)
            uris.append(uri)
            metas.append(meta)

        if not texts:
            return

        embeddings = self._embed_batch(texts)
        self._uris.extend(uris)
        self._vectors.extend(embeddings)
        self._metadata.extend(metas)
        self._texts.extend(texts)
        print(f"[VectorStore:{self.collection_name}] Indexed {len(texts)} chunks "
              f"(total: {len(self._uris)}).")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[dict]:
        if not self._vectors:
            return []

        q_vec  = np.array(self._embed_single(query), dtype=np.float32)
        matrix = np.array(self._vectors, dtype=np.float32)

        # Cosine similarity via normalised dot product
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = m_norm @ q_norm

        top_k       = min(k, len(self._uris))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "uri":   self._uris[i],
                "text":  self._texts[i],
                "score": float(scores[i]),
                **self._metadata[i],
            }
            for i in top_indices
        ]

    def count(self) -> int:
        return len(self._uris)

    # ------------------------------------------------------------------
    # Embedding  (priority: OpenAI → sentence-transformers → hash)
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # 1. OpenAI (if client provided)
        if self.openai_client:
            try:
                resp = self.openai_client.embeddings.create(
                    model=self.embedding_model, input=texts
                )
                return [item.embedding for item in resp.data]
            except Exception as e:
                print(f"[VectorStore] OpenAI embedding failed: {e}. Trying next.")

        # 2. sentence-transformers (free, local, semantic)
        if ST_AVAILABLE and _ST_MODEL is not None:
            vecs = _ST_MODEL.encode(texts, show_progress_bar=False)
            return vecs.tolist()

        # 3. Hash-based fallback (deterministic, no dependencies)
        return [self._simple_embed(t) for t in texts]

    def _embed_single(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    @staticmethod
    def _simple_embed(text: str, dim: int = 512) -> List[float]:
        """Deterministic hash-based BoW embedding — used only as last resort."""
        import hashlib
        vec = [0.0] * dim
        for w in text.lower().split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]