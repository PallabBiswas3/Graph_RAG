"""
ikg.py  —  Information Knowledge Graph (Keyword Layer)
--------------------------------------------------------
Implements the IKG as described in the paper (Section 5.1, Figure 4).

Structure:
    Chunk ──HAS_KEYWORD──► Keyword ◄──HAS_KEYWORD── Chunk

This allows two chunks from *different chapters or documents* to be
connected if they share a keyword, enabling cross-document retrieval.

The IKG is stored in-memory using a simple inverted index:
    keyword  →  [uri1, uri2, ...]
    uri      →  [keyword1, keyword2, ...]
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional
from utils.chunker import Chunk
from utils.keyword_extractor import YAKEExtractor


class InformationKnowledgeGraph:
    """
    Lightweight in-memory IKG using an inverted index.

    Key operations
    --------------
    add_chunk(chunk, keywords)    – index chunk keywords
    get_chunks_by_keyword(kw)     – find all chunks sharing a keyword
    get_keywords_for_chunk(uri)   – fetch stored keywords for a chunk
    get_related_chunks(uri)       – IKS: all chunks sharing any keyword with uri
    get_chunks_by_query_keywords  – UKS: chunks matching query-level keywords
    """

    def __init__(self):
        # keyword (lowercase) → set of chunk URIs
        self._kw_to_uris: Dict[str, Set[str]] = {}
        # uri → list of keywords
        self._uri_to_kws: Dict[str, List[str]] = {}
        # uri → chunk data dict (text, doc, chapter, etc.)
        self._uri_to_data: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: Chunk, keywords: List[str]) -> None:
        """
        Register a chunk and its keywords in the IKG.
        Called during the indexing phase after keyword extraction.
        """
        self._uri_to_kws[chunk.uri] = keywords
        self._uri_to_data[chunk.uri] = {
            "uri": chunk.uri,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
        }
        for kw in keywords:
            kw_norm = kw.lower().strip()
            if kw_norm not in self._kw_to_uris:
                self._kw_to_uris[kw_norm] = set()
            self._kw_to_uris[kw_norm].add(chunk.uri)

    def build_from_chunks(
        self,
        chunks: List[Chunk],
        extractor: Optional[YAKEExtractor] = None,
        num_keywords: int = 5,
    ) -> None:
        """
        Convenience method: extract keywords for each chunk and index all at once.
        """
        if extractor is None:
            extractor = YAKEExtractor(num_keywords=num_keywords)

        for chunk in chunks:
            keywords = extractor.extract(chunk.text, n=num_keywords)
            chunk.keywords = keywords        # also store on the Chunk object
            self.add_chunk(chunk, keywords)

        print(f"[IKG] Indexed {len(chunks)} chunks with "
              f"{len(self._kw_to_uris)} unique keywords.")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_keywords_for_chunk(self, uri: str) -> List[str]:
        """Return the keyword list stored for this chunk URI."""
        return self._uri_to_kws.get(uri, [])

    def get_chunks_by_keyword(self, keyword: str) -> List[dict]:
        """Return all chunk data dicts that are linked to `keyword`."""
        uris = self._kw_to_uris.get(keyword.lower().strip(), set())
        return [self._uri_to_data[u] for u in uris if u in self._uri_to_data]

    def get_related_chunks(self, uri: str) -> List[dict]:
        """
        IKS — Informed Keyword Search:
        Given a source URI, find ALL chunks that share at least one keyword.
        Excludes the source URI itself.
        """
        keywords = self.get_keywords_for_chunk(uri)
        related_uris: Set[str] = set()

        for kw in keywords:
            for u in self._kw_to_uris.get(kw.lower(), set()):
                if u != uri:
                    related_uris.add(u)

        return [self._uri_to_data[u] for u in related_uris
                if u in self._uri_to_data]

    def get_chunks_by_query_keywords(
        self,
        query_keywords: List[str],
        exclude_uris: Optional[Set[str]] = None,
    ) -> List[dict]:
        """
        UKS — Uninformed Keyword Search:
        Given keywords extracted directly from the user query,
        return all chunks linked to ANY of these keywords.
        """
        exclude_uris = exclude_uris or set()
        matched_uris: Set[str] = set()

        for kw in query_keywords:
            kw_norm = kw.lower().strip()
            # Exact match
            for u in self._kw_to_uris.get(kw_norm, set()):
                matched_uris.add(u)
            # Partial / substring match for robustness
            for stored_kw, uris in self._kw_to_uris.items():
                if kw_norm in stored_kw or stored_kw in kw_norm:
                    matched_uris.update(uris)

        matched_uris -= exclude_uris
        return [self._uri_to_data[u] for u in matched_uris
                if u in self._uri_to_data]

    def get_stats(self) -> dict:
        """Summary statistics."""
        return {
            "total_chunks_indexed": len(self._uri_to_kws),
            "unique_keywords": len(self._kw_to_uris),
            "avg_keywords_per_chunk": (
                sum(len(v) for v in self._uri_to_kws.values()) /
                max(len(self._uri_to_kws), 1)
            ),
        }
