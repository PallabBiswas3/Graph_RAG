"""
uks.py  —  Uninformed Keyword Search

Retrieval Strategy 3 (Orange path in Figure 5 of the paper).

Algorithm:
    1. Extract keywords DIRECTLY from the user query (no prior retrieval needed)
    2. Look up ALL chunks in the IKG that are linked to ANY of these keywords
    3. Add these chunks to the initial retrieval pool
    4. This expanded pool then feeds into ICS and IKS steps

Key difference from IKS:
    - IKS starts from *retrieved chunk keywords* (informed by vector search)
    - UKS starts from *query keywords* (parallel / independent pathway)
    - UKS can surface relevant chunks that vector search missed entirely

Rationale (from paper Section 5.2):
    "UKS introduces an additional retrieval pathway that operates in
     parallel with vector-based retrieval, expanding the initial set
     of retrieved chunks."
"""

from __future__ import annotations
from typing import List, Set
from graph.ikg import InformationKnowledgeGraph
from utils.keyword_extractor import YAKEExtractor


def uninformed_keyword_search(
    query: str,
    ikg: InformationKnowledgeGraph,
    extractor: YAKEExtractor,
    existing_uris: Set[str] = None,
    num_keywords: int = 5,
) -> List[dict]:
    """
    Retrieve chunks by extracting keywords from the user query directly.

    Args:
        query         : the raw user query string
        ikg           : the indexed InformationKnowledgeGraph
        extractor     : keyword extractor to parse the query
        existing_uris : URIs already in the retrieval pool (to avoid duplicates)
        num_keywords  : how many keywords to extract from the query

    Returns:
        List of NEW chunk dicts not already in existing_uris.
    """
    existing_uris = existing_uris or set()

    # Step 1: Extract keywords from the query itself
    query_keywords = extractor.extract(query, n=num_keywords)

    if not query_keywords:
        return []

    # Step 2: Retrieve all chunks matching these keywords from the IKG
    matched_chunks = ikg.get_chunks_by_query_keywords(
        query_keywords=query_keywords,
        exclude_uris=existing_uris,
    )

    return matched_chunks
