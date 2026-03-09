"""
keyword_extractor.py

Wraps YAKE for fast, unsupervised keyword extraction (primary method).
Also provides an LLM-based extractor for domain-specific keywords
(alternative, as discussed in the paper Section 5.1 and 7.4).
"""

from typing import List, Optional
import re

# YAKE import with graceful fallback
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("[WARNING] yake not installed. Using simple TF-based fallback.")


class YAKEExtractor:
    """
    Keyword extraction using the YAKE algorithm.
    Extracts unigrams + bigrams, deduplicates at threshold 0.8.
    Mirrors the paper's configuration exactly.
    """

    def __init__(
        self,
        language: str = "en",
        max_ngram: int = 2,
        dedup_threshold: float = 0.8,
        num_keywords: int = 5,
    ):
        self.language = language
        self.max_ngram = max_ngram
        self.dedup_threshold = dedup_threshold
        self.num_keywords = num_keywords

        if YAKE_AVAILABLE:
            self._extractor = yake.KeywordExtractor(
                lan=language,
                n=max_ngram,
                dedupLim=dedup_threshold,
                top=num_keywords * 3,   # Over-extract, then filter
                features=None,
            )
        else:
            self._extractor = None

    def extract(self, text: str, n: Optional[int] = None) -> List[str]:
        """
        Returns up to `n` (or self.num_keywords) keywords from `text`.
        Keywords are lowercased strings (unigrams or bigrams).
        """
        n = n or self.num_keywords

        if YAKE_AVAILABLE and self._extractor:
            try:
                raw = self._extractor.extract_keywords(text)
                # YAKE returns (keyword, score) — lower score = more relevant
                keywords = [kw for kw, _ in raw]
                return self._clean(keywords)[:n]
            except Exception:
                pass

        # Fallback: simple frequency-based extraction
        return self._simple_extract(text, n)

    def _clean(self, keywords: List[str]) -> List[str]:
        """Normalize and filter trivial keywords."""
        stopwords = {
            "the", "a", "an", "in", "of", "to", "and", "or", "is",
            "are", "was", "were", "for", "on", "with", "that", "this",
            "it", "as", "at", "be", "by", "from", "have", "has",
        }
        cleaned = []
        for kw in keywords:
            kw = kw.lower().strip()
            if len(kw) < 3:
                continue
            if kw in stopwords:
                continue
            cleaned.append(kw)
        return cleaned

    def _simple_extract(self, text: str, n: int) -> List[str]:
    """Fallback TF-based extractor when YAKE is unavailable."""
    
        stopwords = {
        "the", "a", "an", "in", "of", "to", "and", "or", "is",
        "are", "was", "were", "for", "on", "with", "that", "this",
        "it", "as", "at", "be", "by", "from", "have", "has",
        "we", "our", "their", "its", "also", "can", "may",
    }

        words = re.findall(r'\b[A-Za-z]{2,}\b', text)

        freq = {}

        for w in words:
            w_lower = w.lower()

        # Rule 1: Keep ALL CAPS words
            if w.isupper():
                freq[w] = freq.get(w, 0) + 2   # boost importance

        # Rule 2: Normal words
            if w_lower not in stopwords and len(w) >= 4:
                freq[w_lower] = freq.get(w_lower, 0) + 1

        sorted_words = sorted(freq, key=lambda x: freq[x], reverse=True)

        return sorted_words[:n]


class LLMKeywordExtractor:
    """
    LLM-based keyword extractor for domain-specific terminology.
    Uses the OpenAI Chat API with the prompt template from Appendix B
    of the paper.

    More computationally expensive than YAKE but produces
    higher-quality, context-aware keywords.
    """

    SYSTEM_PROMPT = (
        "You are an AI assistant specialized in identifying and extracting "
        "key technical terminology and significant concepts from academic texts. "
        "Your task is to analyze provided text passages and identify the most "
        "relevant and distinctive keywords that capture the main concepts."
    )

    USER_TEMPLATE = """Please extract the top {n} keywords from the provided text.
The keywords can be unigrams or bigrams.
The keywords should be unique and significant.
Do not use any special characters or symbols; only letters, numbers, spaces, hyphens (-), and underscores (_).
Common stopwords or generic phrases should not be included.

<context>
{text}
</context>

Instructions:
1. Extract unique and significant unigrams or bigrams.
2. Ensure no special characters other than hyphens and underscores.
3. If fewer than {n} keywords exist, return only what is significant.
4. Do not include explanations outside the JSON array.
5. Ensure output is a valid JSON array with double quotes only.

Return format: ["keyword1", "keyword2", ..., "keywordN"]
"""

    def __init__(self, client, model: str = "gpt-3.5-turbo", num_keywords: int = 5):
        self.client = client
        self.model = model
        self.num_keywords = num_keywords

    def extract(self, text: str, n: Optional[int] = None) -> List[str]:
        """Extract keywords using LLM. Falls back to empty list on error."""
        import json
        n = n or self.num_keywords
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": self.USER_TEMPLATE.format(
                        text=text[:2000], n=n
                    )},
                ],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
            keywords = json.loads(raw)
            return [str(k).lower().strip() for k in keywords if k][:n]
        except Exception as e:
            print(f"[LLMKeywordExtractor] Error: {e}")
            return []
