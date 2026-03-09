"""
iks.py  —  Informed Keyword Search

Retrieval Strategy 2 (Green path in Figure 5 of the paper).

Algorithm:
    1. Start with top-k chunks from vector search (initial_chunks)
    2. For EACH retrieved chunk, look up its keywords in the IKG
    3. For EACH keyword, fetch ALL other chunks that share that keyword
    4. Union the results → keyword-linked retrieval set

Rationale (from paper Section 5.2):
    "IKS enhances naive RAG retrieval by incorporating domain-specific
     keywords to refine search results. In academic contexts, texts often
     contain specialized terminology which can be effectively utilized
     to improve retrieval accuracy."

This enables inter-document connections: two papers in different chapters
or files can be retrieved together if they share domain keywords.
"""

from __future__ import annotations
from typing import List, Dict, Set
from graph.ikg import InformationKnowledgeGraph


def informed_keyword_search(
    initial_chunks: List[dict],
    ikg: InformationKnowledgeGraph,
) -> List[dict]:
    """
    Expand an initial retrieval set using keyword edges in the IKG.

    Args:
        initial_chunks : list of chunk dicts returned by vector search
        ikg            : the indexed InformationKnowledgeGraph

    Returns:
        List of NEW chunk dicts linked via shared keywords.
        Does NOT include chunks already in initial_chunks.
    """
    initial_uris: Set[str] = {c["uri"] for c in initial_chunks}
    expanded: Dict[str, dict] = {}

    for chunk in initial_chunks:
        uri = chunk.get("uri")
        if not uri:
            continue

        # Get all chunks that share at least one keyword with this chunk
        related = ikg.get_related_chunks(uri)
        for rel_chunk in related:
            if rel_chunk and rel_chunk.get("uri") not in initial_uris:
                expanded[rel_chunk["uri"]] = rel_chunk

    return list(expanded.values())
