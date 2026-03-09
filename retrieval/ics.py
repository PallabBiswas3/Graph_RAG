"""
ics.py  —  Informed Chapter Search
-------------------------------------
Retrieval Strategy 1 (Blue path in Figure 5 of the paper).

Algorithm:
    1. Start with top-k chunks from vector search (initial_chunks)
    2. For EACH retrieved chunk, query the DKG for ALL other chunks
       in the SAME chapter
    3. Union the results → expanded chapter-aware retrieval set

Rationale (from paper Section 5.2):
    "Relevant content is often concentrated within the same chapter,
     particularly in technical documents, where chapters provide a
     structured organization of information."
"""

from __future__ import annotations
from typing import List, Dict, Set
from graph.dkg import DocumentKnowledgeGraph


def informed_chapter_search(
    initial_chunks: List[dict],
    dkg: DocumentKnowledgeGraph,
) -> List[dict]:
    """
    Expand an initial retrieval set using the DKG's chapter structure.

    Args:
        initial_chunks : list of chunk dicts returned by vector search
        dkg            : the indexed DocumentKnowledgeGraph

    Returns:
        List of NEW chunk dicts (not already in initial_chunks).
        Caller is responsible for combining with initial_chunks and
        deduplicating by URI.
    """
    initial_uris: Set[str] = {c["uri"] for c in initial_chunks}
    expanded: Dict[str, dict] = {}

    for chunk in initial_chunks:
        uri = chunk.get("uri")
        if not uri:
            continue

        # Fetch all siblings in the same chapter via DKG
        siblings = dkg.get_chapter_chunks(uri)
        for sibling in siblings:
            if sibling and sibling.get("uri") not in initial_uris:
                expanded[sibling["uri"]] = sibling

    return list(expanded.values())
