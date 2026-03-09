"""
dkg.py  —  Document Knowledge Graph
-------------------------------------
Builds a tree-structured graph that mirrors each document's hierarchy:

    Document
      └── Chapter
            └── Section
                  └── Chunk  (URI links to vector DB)

Uses NetworkX as a lightweight in-process graph store.
In production this would be replaced by an RDF/GraphDB instance (as in the paper).

Node types:
    - document  : top-level node, carries metadata
    - chapter   : direct child of document
    - section   : child of chapter
    - chunk     : leaf node, holds the URI + text

Edge types:
    - HAS_CHAPTER   : document  → chapter
    - HAS_SECTION   : chapter   → section
    - HAS_CHUNK     : section/chapter → chunk
    - NEXT_CHUNK    : chunk → chunk (sequential ordering within a chapter)
"""

from __future__ import annotations
import networkx as nx
from typing import List, Dict, Optional, Tuple
from utils.chunker import Chunk


class DocumentKnowledgeGraph:
    """
    In-memory DKG built on top of a directed NetworkX graph.

    Key operations
    --------------
    add_document(chunks)      – index all chunks for one document
    get_chapter_chunks(uri)   – ICS: fetch all siblings in the same chapter
    get_chunk_by_uri(uri)     – direct lookup
    get_all_chunks()          – full corpus
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        # Fast lookup tables
        self._uri_to_node: Dict[str, str]          = {}  # uri -> node_id
        self._chapter_to_chunks: Dict[str, List[str]] = {}  # chapter_key -> [uri]
        self._doc_metadata: Dict[str, dict]        = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        doc_title: str,
        chunks: List[Chunk],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a full document's chunk list to the DKG.
        Creates document, chapter, section, and chunk nodes automatically.
        """
        meta = metadata or {}
        doc_node = f"doc:{doc_id}"

        # Document node
        self.G.add_node(doc_node, type="document", doc_id=doc_id,
                        title=doc_title, **meta)
        self._doc_metadata[doc_id] = {"title": doc_title, **meta}

        # Optional metadata child nodes (author, year)
        for key, val in meta.items():
            meta_node = f"meta:{doc_id}:{key}"
            self.G.add_node(meta_node, type="metadata", key=key, value=val)
            self.G.add_edge(doc_node, meta_node, rel="HAS_METADATA")

        chapter_nodes: Dict[str, str] = {}   # chapter_text -> node_id
        section_nodes: Dict[str, str] = {}   # (chapter, section) -> node_id
        prev_chunk_uri: Optional[str] = None

        for chunk in chunks:
            # ---- Chapter node ----------------------------------------
            chap_key = f"{doc_id}::{chunk.chapter}"
            if chap_key not in chapter_nodes:
                chap_node = f"chapter:{chap_key}"
                self.G.add_node(chap_node, type="chapter",
                                doc_id=doc_id, title=chunk.chapter)
                self.G.add_edge(doc_node, chap_node, rel="HAS_CHAPTER")
                chapter_nodes[chap_key] = chap_node
                self._chapter_to_chunks[chap_node] = []
            else:
                chap_node = chapter_nodes[chap_key]

            # ---- Section node (optional) --------------------------------
            if chunk.section:
                sec_key = f"{doc_id}::{chunk.chapter}::{chunk.section}"
                if sec_key not in section_nodes:
                    sec_node = f"section:{sec_key}"
                    self.G.add_node(sec_node, type="section",
                                    doc_id=doc_id, title=chunk.section)
                    self.G.add_edge(chap_node, sec_node, rel="HAS_SECTION")
                    section_nodes[sec_key] = sec_node
                parent_node = section_nodes[sec_key]
            else:
                parent_node = chap_node

            # ---- Chunk node --------------------------------------------
            chunk_node = f"chunk:{chunk.uri}"
            self.G.add_node(
                chunk_node,
                type="chunk",
                uri=chunk.uri,
                text=chunk.text,
                doc_id=doc_id,
                doc_title=doc_title,
                chapter=chunk.chapter,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
            )
            self.G.add_edge(parent_node, chunk_node, rel="HAS_CHUNK")
            self._uri_to_node[chunk.uri] = chunk_node
            self._chapter_to_chunks[chap_node].append(chunk.uri)

            # Sequential ordering
            if prev_chunk_uri:
                prev_node = f"chunk:{prev_chunk_uri}"
                self.G.add_edge(prev_node, chunk_node, rel="NEXT_CHUNK")
            prev_chunk_uri = chunk.uri

        print(f"[DKG] Indexed '{doc_title}': {len(chunks)} chunks, "
              f"{len(chapter_nodes)} chapters, {len(section_nodes)} sections.")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_chapter_chunks(self, uri: str) -> List[dict]:
        """
        ICS — Informed Chapter Search:
        Given a chunk URI, return ALL chunk data objects in the same chapter.
        """
        chunk_node = self._uri_to_node.get(uri)
        if not chunk_node:
            return []

        # Walk up: chunk → section/chapter
        chapter_node = self._find_chapter_ancestor(chunk_node)
        if not chapter_node:
            return []

        sibling_uris = self._chapter_to_chunks.get(chapter_node, [])
        return [self._node_data(f"chunk:{u}") for u in sibling_uris]

    def get_chunk_by_uri(self, uri: str) -> Optional[dict]:
        """Direct lookup of a single chunk by URI."""
        node = self._uri_to_node.get(uri)
        return self._node_data(node) if node else None

    def get_all_chunks(self) -> List[dict]:
        """Return data dicts for every chunk node."""
        return [
            dict(self.G.nodes[n])
            for n in self.G.nodes
            if self.G.nodes[n].get("type") == "chunk"
        ]

    def get_stats(self) -> dict:
        """Summary statistics about the current graph."""
        type_counts: dict = {}
        for n in self.G.nodes:
            t = self.G.nodes[n].get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            **type_counts,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_chapter_ancestor(self, chunk_node: str) -> Optional[str]:
        """Walk predecessors until we find a chapter node."""
        for ancestor in nx.ancestors(self.G, chunk_node):
            if self.G.nodes[ancestor].get("type") == "chapter":
                return ancestor
        return None

    def _node_data(self, node_id: str) -> Optional[dict]:
        if node_id and node_id in self.G:
            return dict(self.G.nodes[node_id])
        return None
