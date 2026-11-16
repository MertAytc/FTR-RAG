"""Run first-50 raw-text pipeline with baseline + expert metadata.

Chunk IDs and source text come from the latest inline-table corpus. Ranking
metadata is the union of the 25 searchable baseline fields and the 14 expert
fields used in the earlier raw run. The previous run's files are preserved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import chromadb
import ollama

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import run_first50_metadata_generation_ablation as base
from src.retrieval import reranker as reranker_module
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding import embed_texts
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retrieval_utils import _NON_SEARCHABLE_METADATA_FIELDS

BASELINE_CHUNKS_DIR = ROOT / "data/processed/chunks"
LATEST_CHUNKS_DIR = ROOT / "data/processed/chunks_context_tables_v1_workbook_metadata"
FILTERED_CHUNKS_DIR = ROOT / "data/processed/chunks_latest_baseline_plus_expert_metadata_20261031"
OUTPUT_DIR = ROOT / "data/processed/generation_first50_baseline_plus_expert_metadata_20261031"
BENCHMARK_PATH = ROOT / "data/processed/metadata_reranker_embedding_ablation_20261030/benchmark_first50.json"
RAW_DB = ROOT / "data/processed/vector_db_inline_tables_v1_20261030"
RAW_COLLECTION = "ftr_chunks_inline_tables_v1_20261030"
RUN_DIR = OUTPUT_DIR / "first50_raw_text_embedding"

OPERATIONAL_METADATA_FIELDS = {
    "context_before_chunk_ids",
    "context_after_chunk_ids",
    "related_text_chunk_ids",
    "relation_evidence_pages",
}

LOG = logging.getLogger("first50_combined_metadata")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _chunk_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.chunks.json")):
        records.extend(read_json(path).get("chunks", []))
    return records


def _baseline_metadata_keys() -> set[str]:
    keys = {
        str(key)
        for chunk in _chunk_records(BASELINE_CHUNKS_DIR)
        for key in (chunk.get("metadata") or {})
    }
    table_data = read_json(BASELINE_CHUNKS_DIR / "table_chunks.json")
    keys.update(
        str(key)
        for chunk in table_data.get("chunks", [])
        for key in (chunk.get("metadata") or {})
    )
    return keys


def _text_id_digest(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: str(item.get("chunk_id", ""))):
        digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_filtered_corpus() -> dict[str, Any]:
    baseline_keys = _baseline_metadata_keys()
    latest_files = sorted(LATEST_CHUNKS_DIR.glob("*.chunks.json"))
    if not latest_files:
        raise FileNotFoundError(f"No latest chunk files found in {LATEST_CHUNKS_DIR}")

    latest_chunks: list[dict[str, Any]] = []
    latest_keys: set[str] = set()
    for path in latest_files:
        for chunk in read_json(path).get("chunks", []):
            latest_chunks.append(chunk)
            latest_keys.update((chunk.get("metadata") or {}).keys())

    expert_fields = set(base.RERANKER_CONTEXT_FIELDS)
    if len(expert_fields) != 14:
        raise RuntimeError(f"Expected 14 expert metadata fields, found {len(expert_fields)}")

    common_baseline_keys = baseline_keys & latest_keys
    expert_fields_present = expert_fields & latest_keys
    baseline_searchable_fields = common_baseline_keys - _NON_SEARCHABLE_METADATA_FIELDS
    # This union contains 38 configured names because recommendation_strength
    # is shared by the 25 baseline fields and 14 expert fields. Some expert
    # chunk-level fields are absent in the corpus; they remain empty and are
    # never synthesized.
    selected_ranking_keys = baseline_searchable_fields | expert_fields
    selected_corpus_keys = common_baseline_keys | expert_fields
    selected_keys = selected_corpus_keys | OPERATIONAL_METADATA_FIELDS
    reranker_fields = sorted(selected_ranking_keys)
    source_digest = _text_id_digest(latest_chunks)

    if FILTERED_CHUNKS_DIR.exists():
        manifest_path = FILTERED_CHUNKS_DIR / "metadata_profile_manifest.json"
        if not manifest_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing unmanifested corpus: {FILTERED_CHUNKS_DIR}"
            )
        manifest = read_json(manifest_path)
        if manifest.get("source_chunk_text_id_sha256") != source_digest:
            raise RuntimeError("Existing filtered corpus was built from different chunk text/IDs.")
        if set(manifest.get("reranker_metadata_fields", [])) != set(reranker_fields):
            raise RuntimeError("Existing combined-metadata corpus has a different metadata profile.")
        return manifest

    FILTERED_CHUNKS_DIR.mkdir(parents=True)
    filtered_chunks: list[dict[str, Any]] = []
    stripped_metadata_keys: set[str] = set()
    for path in latest_files:
        data = read_json(path)
        for chunk in data.get("chunks", []):
            original_meta = chunk.get("metadata") or {}
            filtered_meta = {key: value for key, value in original_meta.items() if key in selected_keys}
            stripped_metadata_keys.update(set(original_meta) - set(filtered_meta))
            clean_chunk = dict(chunk)
            clean_chunk["metadata"] = filtered_meta
            clean_chunk.pop("commented_text", None)
            filtered_chunks.append(clean_chunk)
        target = FILTERED_CHUNKS_DIR / path.name
        for chunk in data.get("chunks", []):
            original_meta = chunk.get("metadata") or {}
            chunk["metadata"] = {key: value for key, value in original_meta.items() if key in selected_keys}
            chunk.pop("commented_text", None)
        write_json(target, data)

    filtered_digest = _text_id_digest(filtered_chunks)
    if filtered_digest != source_digest:
        raise RuntimeError("Filtered corpus changed chunk text or IDs.")

    manifest = {
        "profile": "latest_chunk_text_with_baseline25_plus_expert14_metadata",
        "created_from": str(LATEST_CHUNKS_DIR.relative_to(ROOT)),
        "baseline_metadata_source": str(BASELINE_CHUNKS_DIR.relative_to(ROOT)),
        "baseline_metadata_keys": sorted(baseline_keys),
        "baseline_metadata_key_count": len(baseline_keys),
        "latest_metadata_key_count": len(latest_keys),
        "baseline_latest_intersection_keys": sorted(common_baseline_keys),
        "baseline_latest_intersection_count": len(common_baseline_keys),
        "expert_metadata_fields": sorted(expert_fields),
        "expert_metadata_fields_present": sorted(expert_fields_present),
        "expert_metadata_field_count": len(expert_fields),
        "baseline_searchable_metadata_fields": sorted(baseline_searchable_fields),
        "baseline_searchable_metadata_field_count": len(baseline_searchable_fields),
        "selected_ranking_metadata_keys": reranker_fields,
        "selected_ranking_metadata_key_count": len(reranker_fields),
        "metadata_field_slots": len(baseline_searchable_fields) + len(expert_fields),
        "metadata_field_overlap": sorted(baseline_searchable_fields & expert_fields),
        "reranker_metadata_fields": reranker_fields,
        "reranker_excluded_non_searchable_fields": sorted(
            common_baseline_keys & _NON_SEARCHABLE_METADATA_FIELDS
        ),
        "operational_metadata_retained_for_context_only": sorted(OPERATIONAL_METADATA_FIELDS),
        "stripped_latest_metadata_keys": sorted(stripped_metadata_keys),
        "chunk_count": len(filtered_chunks),
        "source_chunk_text_id_sha256": source_digest,
        "filtered_chunk_text_id_sha256": filtered_digest,
        "commented_text_removed": True,
        "chunk_text_or_ids_changed": False,
    }
    write_json(FILTERED_CHUNKS_DIR / "metadata_profile_manifest.json", manifest)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    manifest = build_filtered_corpus()
    LOG.info(
        "Combined metadata profile ready: chunks=%d baseline searchable=%d expert=%d unique=%d",
        manifest["chunk_count"], manifest["baseline_searchable_metadata_field_count"],
        manifest["expert_metadata_field_count"], manifest["selected_ranking_metadata_key_count"],
    )

    base.BENCHMARK_PATH = BENCHMARK_PATH
    base.CHUNKS_DIR = FILTERED_CHUNKS_DIR
    base.OUTPUT_BASE = OUTPUT_DIR
    base.RERANKER_CONTEXT_FIELDS = tuple(manifest["reranker_metadata_fields"])
    reranker_module.RERANKER_METADATA_FIELDS = tuple(manifest["reranker_metadata_fields"])

    questions = base._load_questions(BENCHMARK_PATH)
    chunk_map = base._load_chunk_map(FILTERED_CHUNKS_DIR)
    bm25 = BM25Retriever(chunks_dir=FILTERED_CHUNKS_DIR, metadata_mode="all")
    if len(bm25.ids) != len(set(bm25.ids)):
        raise RuntimeError("BM25 chunk IDs are not unique.")
    if set(chunk_map) != set(bm25.ids):
        raise RuntimeError(f"Chunk map/BM25 IDs differ: chunks={len(chunk_map)}, BM25={len(bm25.ids)}")

    client = chromadb.PersistentClient(path=str(RAW_DB))
    collection = client.get_collection(RAW_COLLECTION)
    base._validate_dense_collection(collection, bm25)

    available_models = {
        str(item.get("model") or item.get("name") or "")
        for item in ollama.list().models
    }
    if base.MODEL not in available_models:
        raise RuntimeError(f"Ollama model {base.MODEL!r} is unavailable: {sorted(available_models)}")

    query_texts = [str(question["question"]).strip() for question in questions]
    query_embeddings = embed_texts(query_texts, input_type="query")
    if len(query_embeddings) != len(questions):
        raise RuntimeError("Query embedding count mismatch.")

    bm25_rows = {}
    for index, question in enumerate(questions, 1):
        qid = str(question["id"]).strip()
        bm25_rows[qid] = bm25.search(str(question["question"]).strip(), n_results=base.BM25_TOP_N)
        if index % 10 == 0:
            print(f"BM25 combined-metadata results prepared: {index}/{len(questions)}", flush=True)

    metadata_retriever = MetadataRetriever(chunks_dir=FILTERED_CHUNKS_DIR)
    reranker = CrossEncoderReranker(model_name=base.RERANKER_MODEL, batch_size=4, max_length=512)
    report = base._run_one(
        "raw_text_embedding", collection, questions, query_embeddings, bm25_rows,
        bm25, reranker, chunk_map, metadata_retriever, base.MODEL,
    )
    report["metadata_profile"] = manifest["profile"]
    report["baseline_metadata_manifest"] = str(
        (FILTERED_CHUNKS_DIR / "metadata_profile_manifest.json").relative_to(ROOT)
    )
    report["bm25_metadata_mode"] = (
        "all searchable values from the union of baseline common fields and the 14 expert fields; "
        "BM25 non-searchable technical fields remain excluded by retrieval_utils"
    )
    report["reranker_metadata_fields"] = manifest["reranker_metadata_fields"]
    report_path = RUN_DIR / "run_report.json"
    base._write_json(report_path, report)
    base._write_report_markdown(RUN_DIR / "run_report.md", report)
    print(json.dumps({
        "run_report": str(report_path.relative_to(ROOT)),
        "metadata_manifest": str((FILTERED_CHUNKS_DIR / "metadata_profile_manifest.json").relative_to(ROOT)),
        "question_count": len(questions),
        "retrieval_eligible": report.get("retrieval_metric_eligible_questions"),
        "generation_completed": report.get("generation", {}).get("completed"),
        "ragas_scored": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
