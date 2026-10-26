from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.config import CHUNKS_DIR, REFERENCE_PENALTY
from src.retrieval.retrieval_utils import annotate_result, normalize_chunk_id

logger = logging.getLogger(__name__)

# Keep clinical negation words ("no", "not", "without"); they change meaning.
_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "at", "to",
    "from", "with", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "which", "what", "how",
    "does", "do", "did", "can", "could", "should", "would", "may", "might",
}

# Letters/digits with an optional decimal part, so "0.5" stays one token.
# Hyphens and other punctuation split tokens ("range-of-motion" -> 3 tokens).
_TOKEN_RE = re.compile(r"[^\W_]+(?:\.\d+)?", flags=re.UNICODE)

try:  # Optional dependency: pip install nltk
    from nltk.stem.snowball import SnowballStemmer

    _STEMMER = SnowballStemmer("english")
except Exception:  # pragma: no cover - optional dependency
    _STEMMER = None


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    if _STEMMER is None or token.isdigit() or any(char.isdigit() for char in token):
        return token
    return _STEMMER.stem(token)


def tokenize(text: str, use_stemming: bool = True) -> List[str]:
    """Tokenize English clinical text for BM25.

    Lowercases and strips diacritics, keeps decimal values such as ``0.5``,
    removes a small stopword list while retaining negation, and optionally
    applies English Snowball stemming.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("–", "-").replace("—", "-")
    tokens = [token for token in _TOKEN_RE.findall(text) if token not in _ENGLISH_STOPWORDS]
    if use_stemming:
        tokens = [_stem(token) for token in tokens]
    return tokens


class BM25Retriever:
    """BM25 retrieval with a conservative, precomputed bibliography penalty."""

    def __init__(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        reference_penalty: float = REFERENCE_PENALTY,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = True,
        title_fields: Sequence[str] = ("section_title", "title"),
        title_repeat: int = 0,
    ):
        """
        reference_penalty: multiplier for bibliography/reference chunks.
        title_repeat: number of times to prepend the title/section to indexed
            text; 0 disables this BM25F approximation.
        """
        if not 0.0 < reference_penalty <= 1.0:
            raise ValueError("reference_penalty must be in the range (0, 1].")
        if title_repeat < 0:
            raise ValueError("title_repeat must be >= 0.")

        self.chunks_dir = Path(chunks_dir)
        self.reference_penalty = reference_penalty
        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        self.title_fields = tuple(title_fields)
        self.title_repeat = title_repeat

        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        self._load_chunks()
        if not self.documents:
            raise RuntimeError("No chunks found to build the BM25 index.")

        self._id_to_index: Dict[str, int] = {
            chunk_id: index for index, chunk_id in enumerate(self.ids)
        }

        tokenized_documents = [
            tokenize(self._index_text(document, metadata), self.use_stemming)
            for document, metadata in zip(self.documents, self.metadatas)
        ]
        self.bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)
        self._penalties, self._roles = self._precompute_annotations()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _index_text(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Return indexed text, optionally boosting title/section terms."""
        if self.title_repeat <= 0:
            return text
        metadata = metadata or {}
        title = ""
        for field in self.title_fields:
            value = metadata.get(field)
            if value:
                title = str(value)
                break
        if not title:
            return text
        return " ".join([title] * self.title_repeat + [text])

    def _precompute_annotations(self) -> Tuple[np.ndarray, List[str]]:
        penalties = np.ones(len(self.ids), dtype=np.float64)
        roles: List[str] = []
        for index in range(len(self.ids)):
            probe = annotate_result({
                "id": self.ids[index],
                "document": self.documents[index],
                "metadata": self.metadatas[index],
            })
            if probe.get("reference_penalty", 1.0) < 1.0:
                penalties[index] = self.reference_penalty
            roles.append(probe.get("chunk_role", ""))
        return penalties, roles

    def _load_chunks(self) -> None:
        json_files = sorted(self.chunks_dir.glob("*.json"))
        seen_ids: Set[str] = set()

        for json_file in json_files:
            if json_file.name == "_quality_report.json":
                continue

            logger.info("Reading chunks: %s", json_file.name)
            with json_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                chunks = data.get("chunks", [])
            elif isinstance(data, list):
                chunks = data
            else:
                logger.warning("Unexpected JSON format, skipped: %s", json_file.name)
                continue

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text")
                if not chunk_id or not text:
                    continue

                chunk_id = normalize_chunk_id(chunk_id)
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                self.ids.append(chunk_id)
                self.documents.append(text)
                self.metadatas.append(chunk.get("metadata") or {})

        logger.info("BM25 total chunks: %d", len(self.documents))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score(
        self,
        query: str,
        apply_reference_penalty: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw and adjusted scores; both are empty for an empty query."""
        query_tokens = tokenize(query, self.use_stemming)
        if not query_tokens:
            empty = np.array([], dtype=np.float64)
            return empty, empty

        raw = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        adjusted = raw * self._penalties if apply_reference_penalty else raw
        return raw, adjusted

    def search(
        self,
        query: str,
        n_results: int = 20,
        apply_reference_penalty: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search BM25 results; chunks with zero scores are not returned.

        Set ``apply_reference_penalty=False`` for a pure BM25 ablation.
        """
        raw, adjusted = self._score(query, apply_reference_penalty)
        if adjusted.size == 0 or n_results <= 0:
            return []

        n = min(n_results, adjusted.size)
        if n < adjusted.size:
            top = np.argpartition(-adjusted, n - 1)[:n]
        else:
            top = np.arange(adjusted.size)
        top = top[np.argsort(-adjusted[top], kind="stable")]

        results: List[Dict[str, Any]] = []
        for index in top:
            if adjusted[index] <= 0:
                break
            result = annotate_result({
                "id": self.ids[index],
                "score": float(adjusted[index]),
                "raw_score": float(raw[index]),
                "document": self.documents[index],
                "metadata": self.metadatas[index],
            })
            result["rank"] = len(results) + 1
            results.append(result)
        return results

    def debug_rank(
        self,
        query: str,
        target_ids: Set[str],
        apply_reference_penalty: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Return raw and adjusted ranks for selected IDs.

        IDs absent from the index are returned with ``found=False``.
        """
        normalized_target_ids = {normalize_chunk_id(chunk_id) for chunk_id in target_ids}
        if not normalized_target_ids:
            return {}

        raw, adjusted = self._score(query, apply_reference_penalty)
        if adjusted.size == 0:
            return {}

        order = np.argsort(-adjusted, kind="stable")
        rank_of = np.empty(adjusted.size, dtype=np.int64)
        rank_of[order] = np.arange(1, adjusted.size + 1)

        found: Dict[str, Dict[str, Any]] = {}
        for chunk_id in normalized_target_ids:
            index = self._id_to_index.get(chunk_id)
            if index is None:
                found[chunk_id] = {"found": False, "rank": None}
                continue
            found[chunk_id] = {
                "found": True,
                "rank": int(rank_of[index]),
                "score": float(adjusted[index]),
                "raw_score": float(raw[index]),
                "chunk_role": self._roles[index],
                "document": self.documents[index],
                "metadata": self.metadatas[index],
            }
        return found
