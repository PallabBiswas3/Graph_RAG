"""
chunker.py
----------
Splits documents into semantically coherent chunks while preserving
the hierarchical structure (Document > Chapter > Section > Chunk).

Mirrors the paper's approach: chunk at paragraph/sentence boundaries,
assign each chunk a URI, retain chapter/section membership.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """A single text chunk with full structural metadata."""
    uri: str                          # Unique Resource Identifier
    text: str                         # Raw chunk text
    doc_id: str                       # Parent document ID
    doc_title: str                    # Parent document title
    chapter: str                      # Chapter heading
    section: str                      # Section heading (sub-chapter)
    chunk_index: int                  # Sequential index within document
    token_count: int = 0
    keywords: List[str] = field(default_factory=list)

    def __repr__(self):
        return (f"Chunk(uri={self.uri[:30]}..., doc='{self.doc_title}', "
                f"chapter='{self.chapter}', tokens={self.token_count})")


class DocumentChunker:
    """
    Parses a plain-text (or markdown) document into structured Chunk objects.

    Heading detection supports both:
      - Markdown style:   ## 2. Introduction
      - Numbered style:   2.1 Background
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        """
        Args:
            chunk_size: Approximate max tokens per chunk (1 token ≈ 4 chars).
            overlap:    Character overlap between consecutive chunks (default 0,
                        matching the paper's no-overlap setting).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 1 token ≈ 4 characters (rough heuristic)
        self._char_limit = chunk_size * 4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Main entry point.  Returns a list of Chunk objects for one document.
        """
        sections = self._split_into_sections(text)
        chunks: List[Chunk] = []
        chunk_index = 0

        for chapter, section, body in sections:
            # Further split the body into size-limited pieces
            sub_texts = self._split_body(body)
            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                uri = self._make_uri(doc_id, chunk_index)
                chunks.append(Chunk(
                    uri=uri,
                    text=sub_text.strip(),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chapter=chapter,
                    section=section,
                    chunk_index=chunk_index,
                    token_count=len(sub_text) // 4,
                ))
                chunk_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str):
        """
        Detect headings and split text into (chapter, section, body) tuples.
        """
        # Patterns: markdown headings OR numbered headings
        heading_re = re.compile(
            r'^(#{1,4}\s+.+|(?:\d+\.)+\d*\s+[A-Z].+)$',
            re.MULTILINE,
        )

        results = []
        current_chapter = "Introduction"
        current_section = ""
        current_body_lines: List[str] = []
        depth_map: dict = {}  # heading text -> depth

        lines = text.split('\n')
        for line in lines:
            m = heading_re.match(line.strip())
            if m:
                # Save accumulated body
                if current_body_lines:
                    results.append((
                        current_chapter,
                        current_section,
                        '\n'.join(current_body_lines),
                    ))
                    current_body_lines = []

                heading_text = re.sub(r'^#+\s*', '', line.strip())
                depth = self._heading_depth(line)
                depth_map[heading_text] = depth

                if depth <= 1:
                    current_chapter = heading_text
                    current_section = ""
                else:
                    current_section = heading_text
            else:
                current_body_lines.append(line)

        # Flush last section
        if current_body_lines:
            results.append((current_chapter, current_section,
                            '\n'.join(current_body_lines)))

        return results if results else [("Document", "", text)]

    def _heading_depth(self, line: str) -> int:
        """Return depth: 1 for top-level, 2 for sub, etc."""
        md_match = re.match(r'^(#+)', line.strip())
        if md_match:
            return len(md_match.group(1))
        num_match = re.match(r'^(\d+\.)+', line.strip())
        if num_match:
            return num_match.group(0).count('.')
        return 1

    def _split_body(self, body: str) -> List[str]:
        """
        Split body text into chunks no larger than self._char_limit.
        Splits preferentially at paragraph boundaries (blank lines),
        then at sentence boundaries.
        """
        if len(body) <= self._char_limit:
            return [body]

        # Try paragraph splits first
        paragraphs = re.split(r'\n\s*\n', body)
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self._char_limit:
                current = (current + "\n\n" + para).lstrip()
            else:
                if current:
                    chunks.append(current)
                # Para itself might be too long → split by sentence
                if len(para) > self._char_limit:
                    chunks.extend(self._split_by_sentence(para))
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks if chunks else [body]

    def _split_by_sentence(self, text: str) -> List[str]:
        """Fallback: split oversized paragraph by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self._char_limit:
                current = (current + " " + sent).lstrip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _make_uri(doc_id: str, index: int) -> str:
        """
        Produces a deterministic URI like:
            urn:graphrag:doc_id:chunk:0042
        """
        return f"urn:graphrag:{doc_id}:chunk:{index:04d}"
