from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set
import json
import re

from rank_bm25 import BM25Okapi

from src.retrieval.retrieval_utils import annotate_result


CHUNKS_DIR = Path("data/processed/chunks")

# Keep clinical negation words; they can materially change meaning.
_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "at", "to",
    "from", "with", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "which", "what", "how",
    "does", "do", "did", "can", "could", "should", "would", "may", "might",
}


def tokenize(text: str) -> List[str]:
    """Clinical English tokenizer for BM25.

    Hyphenated clinical terms are split into useful components while
    punctuation is ignored. Negation terms such as 'not' and 'no' are kept.
    """
    text = (text or "").lower().replace("–", "-").replace("—", "-")
    tokens = re.findall(r"[a-z0-9]+", text, flags=re.UNICODE)
    return [token for token in tokens if token not in _ENGLISH_STOPWORDS]


class BM25Retriever:
    """BM25 retrieval with a conservative bibliography penalty."""

    def __init__(
        self,
        chunks_dir: Path = CHUNKS_DIR,
        reference_penalty: float = 0.25,
    ):
        if not 0.0 < reference_penalty <= 1.0:
            raise ValueError("reference_penalty 0 ile 1 arasında olmalıdır.")

        self.chunks_dir = chunks_dir
        self.reference_penalty = reference_penalty

        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        self._load_chunks()

        if not self.documents:
            raise RuntimeError("BM25 index oluşturmak için chunk bulunamadı.")

        tokenized_documents = [tokenize(document) for document in self.documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    def _load_chunks(self):
        json_files = sorted(self.chunks_dir.glob("*.json"))
        seen_ids: Set[str] = set()

        for json_file in json_files:
            if json_file.name == "_quality_report.json":
                continue

            print(f"BM25 okunuyor: {json_file.name}")

            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                chunks = data.get("chunks", [])
            elif isinstance(data, list):
                chunks = data
            else:
                print("  UYARI: Beklenmeyen JSON formatı, atlandı.")
                continue

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue

                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text")

                if not chunk_id or not text:
                    continue

                chunk_id = str(chunk_id).strip()
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                self.ids.append(chunk_id)
                self.documents.append(text)
                self.metadatas.append(chunk.get("metadata", {}))

        print(f"BM25 toplam chunk: {len(self.documents)}")

    def _rank_indices(self, query: str, apply_reference_penalty: bool = True):
        query_tokens = tokenize(query)
        if not query_tokens:
            return [], []

        raw_scores = self.bm25.get_scores(query_tokens)

        adjusted_scores = []
        for index, raw_score in enumerate(raw_scores):
            if apply_reference_penalty:
                probe = annotate_result({
                    "id": self.ids[index],
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                })
                penalty = probe["reference_penalty"]
            else:
                penalty = 1.0
            adjusted_scores.append(float(raw_score) * penalty)

        ranked_indices = sorted(
            range(len(adjusted_scores)),
            key=lambda i: adjusted_scores[i],
            reverse=True,
        )
        return ranked_indices, adjusted_scores

    def debug_rank(
        self,
        query: str,
        target_ids: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Return raw + adjusted BM25 ranks for selected IDs."""
        if not target_ids:
            return {}

        ranked_indices, adjusted_scores = self._rank_indices(query)
        if not ranked_indices:
            return {}

        raw_scores = self.bm25.get_scores(tokenize(query))
        found: Dict[str, Dict[str, Any]] = {}

        for rank, index in enumerate(ranked_indices, start=1):
            chunk_id = self.ids[index]
            if chunk_id in target_ids:
                role = annotate_result({
                    "id": chunk_id,
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                })["chunk_role"]

                found[chunk_id] = {
                    "rank": rank,
                    "score": adjusted_scores[index],
                    "raw_score": float(raw_scores[index]),
                    "chunk_role": role,
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                }

                if len(found) == len(target_ids):
                    break

        return found

    def search(
        self,
        query: str,
        n_results: int = 20,
        apply_reference_penalty: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search BM25 results.

        By default the production bibliography penalty is preserved. Set
        apply_reference_penalty=False for a pure BM25 ablation.
        """
        ranked_indices, adjusted_scores = self._rank_indices(
            query,
            apply_reference_penalty=apply_reference_penalty,
        )

        raw_scores = self.bm25.get_scores(tokenize(query))
        results = []
        for rank, index in enumerate(ranked_indices[:n_results], start=1):
            result = {
                "id": self.ids[index],
                "score": adjusted_scores[index],
                "raw_score": float(raw_scores[index]),
                "document": self.documents[index],
                "metadata": self.metadatas[index],
            }
            result = annotate_result(result)
            result["rank"] = rank
            results.append(result)

        return results
