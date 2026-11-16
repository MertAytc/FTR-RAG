"""Document-level retrieval over source metadata.

This channel intentionally retrieves source records rather than chunks. It is
useful for questions about document identity, organization, and publication or
revision year. Its results can be combined with chunk retrieval at context
assembly time without pretending that a document record is a content chunk.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.config import CHUNKS_DIR

logger = logging.getLogger(__name__)

_INDEX_FIELDS = (
    "doc_name",
    "title",
    "organization",
    "year",
    "doc_version",
    "doc_family",
    "domain",
    "condition",
    "population",
    # Reviewed document-level fields propagated from the guideline workbook.
    "guideline_title",
    "guideline_source_authority",
    "guideline_filename_year",
    "guideline_version",
    "guideline_publication_date",
    "guideline_last_updated",
    "guideline_diseases_covered",
    "guideline_conditions_covered",
    "guideline_target_population",
    "guideline_age_group",
    "guideline_clinical_phase",
    "guideline_treatment_context",
    "guideline_injury_context",
)


class MetadataRetriever:
    """BM25 search over one metadata record per source document."""

    def __init__(self, chunks_dir: Path = CHUNKS_DIR):
        self.chunks_dir = Path(chunks_dir)
        records: Dict[str, Dict[str, Any]] = {}
        json_files = sorted(self.chunks_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No chunk JSON files found in {self.chunks_dir}")

        for json_file in json_files:
            if json_file.name == "_quality_report.json":
                continue
            with json_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                chunks = data.get("chunks", [])
            elif isinstance(data, list):
                chunks = data
            else:
                logger.warning("Skipping unexpected chunk JSON format: %s", json_file)
                continue

            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                metadata = chunk.get("metadata") or {}
                doc_name = str(metadata.get("doc_name") or "").strip()
                if not doc_name:
                    continue

                record = records.setdefault(
                    doc_name,
                    {
                        "document_id": doc_name,
                        "doc_name": doc_name,
                        "_values": defaultdict(set),
                        "chunk_count": 0,
                    },
                )
                record["chunk_count"] += 1
                for field in _INDEX_FIELDS:
                    value = metadata.get(field)
                    if isinstance(value, (list, tuple, set)):
                        record["_values"][field].update(
                            str(item).strip() for item in value if str(item).strip()
                        )
                    elif value is not None and str(value).strip():
                        record["_values"][field].add(str(value).strip())

        self.documents: List[Dict[str, Any]] = []
        for doc_name in sorted(records):
            record = records[doc_name]
            values = record.pop("_values")
            metadata: Dict[str, Any] = {"doc_name": doc_name}
            for field, field_values in values.items():
                ordered_values = sorted(field_values)
                if len(ordered_values) == 1:
                    metadata[field] = ordered_values[0]
                elif ordered_values:
                    # Preserve conflicts instead of silently choosing a year.
                    metadata[field] = ordered_values
            record["metadata"] = metadata
            self.documents.append(record)

        if not self.documents:
            raise RuntimeError(f"No document metadata found in {self.chunks_dir}")

        self.bm25 = BM25Okapi([
            tokenize(self._record_text(record)) for record in self.documents
        ])
        logger.info(
            "Metadata retriever indexed %d documents from %s",
            len(self.documents),
            self.chunks_dir,
        )

    @staticmethod
    def _record_text(record: Dict[str, Any]) -> str:
        metadata = record.get("metadata") or {}
        parts: List[str] = []
        for field in _INDEX_FIELDS:
            value = metadata.get(field)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
        return " ".join(parts)

    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Return ranked source metadata records; no chunk text is indexed."""
        if n_results <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        results: List[Dict[str, Any]] = []
        for index in order:
            score = float(scores[index])
            if score <= 0:
                break
            record = self.documents[int(index)]
            results.append({
                "document_id": record["document_id"],
                "doc_name": record["doc_name"],
                "metadata": dict(record["metadata"]),
                "chunk_count": record["chunk_count"],
                "score": score,
                "rank": len(results) + 1,
            })
            if len(results) >= n_results:
                break
        return results
