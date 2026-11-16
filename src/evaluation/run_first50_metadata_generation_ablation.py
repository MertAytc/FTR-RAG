"""Run the first 50 questions with raw-text and commentary-embedding indexes.

BM25 indexes all semantic chunk metadata in both runs. Dense search uses either
the existing raw-text or commentary-augmented embedding collection. RRF keeps
the project weights, and the reranker sees only the agreed clinical metadata
fields. Generation outputs are RAGAS-ready; this script does not score RAGAS.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import chromadb
import ollama

from src.evaluation.run_ragas_generation_dataset import EXPECTED_BEHAVIOR, RAGAS_METRICS
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import ENGLISH_SYSTEM_PROMPT, validate_citations
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.config import EMBEDDING_MODEL, PROJECT_ROOT
from src.retrieval.embedding import embed_texts
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.reranker import CrossEncoderReranker, RERANKER_MODEL, _reranker_document
from src.retrieval.retrieval_utils import annotate_result, normalize_chunk_id

LOG = logging.getLogger("first50_metadata_generation_ablation")

BENCHMARK_PATH = PROJECT_ROOT / "data/processed/metadata_reranker_embedding_ablation_20261030/benchmark_first50.json"
CHUNKS_DIR = PROJECT_ROOT / "data/processed/chunks_context_tables_v1_workbook_metadata"
RAW_DB = PROJECT_ROOT / "data/processed/vector_db_inline_tables_v1_20261030"
RAW_COLLECTION = "ftr_chunks_inline_tables_v1_20261030"
COMMENTED_DB = PROJECT_ROOT / "data/processed/vector_db_inline_tables_commentary_embed_v1_20261030"
COMMENTED_COLLECTION = "ftr_chunks_inline_tables_commentary_embed_v1_20261030"
OUTPUT_BASE = PROJECT_ROOT / "data/processed/generation_first50_workbook_metadata_20261030"

MODEL = "qwen3:8b"
DENSE_TOP_N = 40
BM25_TOP_N = 40
RRF_TOP_K = 20
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.85
GENERATION_TOP_K = 5
TABLE_NEIGHBOR_LIMIT = 1
RERANKER_CONTEXT_FIELDS = (
    "guideline_target_population",
    "guideline_age_group",
    "guideline_clinical_phase",
    "guideline_treatment_context",
    "guideline_injury_context",
    "chunk_interventions",
    "chunk_population",
    "chunk_clinical_phase",
    "chunk_recommendation_status",
    "recommendation_strength",
    "evidence_level",
    "chunk_contraindications",
    "chunk_adverse_effects",
    "chunk_clinical_warnings",
)
VARIANTS = ("strict_canonical", "acceptable_full", "acceptable_plus_partial")
STAGES = ("dense", "bm25", "rrf", "reranker")
CUTOFFS = (1, 3, 5, 10)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            LOG.warning("Ignoring malformed JSONL line %d in %s: %s", line_no, path, exc)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    raw = _read_json(path)
    rows = raw.get("questions", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError(f"Expected exactly 50 benchmark questions, found {len(rows) if isinstance(rows, list) else 'invalid'}")
    for index, question in enumerate(rows, 1):
        if not question.get("id") or not question.get("question"):
            raise ValueError(f"Benchmark question row {index} is missing id or question.")
    return rows


def _load_chunk_map(chunks_dir: Path) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")
    for path in files:
        data = _read_json(path)
        for chunk in data.get("chunks", []):
            chunk_id = normalize_chunk_id(chunk.get("chunk_id") or chunk.get("id") or "")
            if chunk_id:
                found.setdefault(chunk_id, chunk)
    return found


def _validate_dense_collection(collection, bm25: BM25Retriever) -> None:
    if collection.count() != len(bm25.ids):
        raise RuntimeError(
            f"Dense/BM25 count mismatch for {collection.name}: dense={collection.count()}, bm25={len(bm25.ids)}"
        )
    stored: Dict[str, str] = {}
    for offset in range(0, collection.count(), 256):
        batch = collection.get(limit=256, offset=offset, include=["documents"])
        stored.update({normalize_chunk_id(cid): text for cid, text in zip(batch.get("ids", []), batch.get("documents", []))})
    expected = dict(zip(bm25.ids, bm25.documents))
    if set(stored) != set(expected):
        raise RuntimeError(f"Dense/BM25 ID mismatch for {collection.name}.")
    mismatches = [cid for cid, text in expected.items() if stored.get(cid) != text]
    if mismatches:
        raise RuntimeError(f"Dense documents differ from source chunks for {collection.name}: {mismatches[:5]}")


class ChunkMapCollection:
    """Fresh source chunks for table-neighbor context, avoiding stale Chroma metadata."""

    def __init__(self, chunks: Dict[str, Dict[str, Any]]):
        self.chunks = chunks

    def get(self, ids: Sequence[str], include: Sequence[str] | None = None) -> Dict[str, Any]:
        found = [normalize_chunk_id(cid) for cid in ids if normalize_chunk_id(cid) in self.chunks]
        documents = [self.chunks[cid].get("text") or self.chunks[cid].get("document") or "" for cid in found]
        metadatas = [self.chunks[cid].get("metadata") or {} for cid in found]
        return {"ids": found, "documents": documents, "metadatas": metadatas}


def _relevance_groups(question: Dict[str, Any]) -> Dict[str, List[Set[str]]]:
    gold = {normalize_chunk_id(cid) for cid in question.get("gold_chunk_ids", []) if normalize_chunk_id(cid)}
    full = [
        {normalize_chunk_id(cid) for cid in group if normalize_chunk_id(cid)}
        for group in question.get("relevance_groups", [])
    ]
    full = [group for group in full if group]
    if not full and gold:
        full = [{cid} for cid in sorted(gold)]
    strict = [group & gold for group in full]
    strict = [group for group in strict if group]
    partial = [set(group) for group in full]
    for entry in question.get("partial_relevance_groups", []):
        if entry.get("decision") != "partial":
            continue
        ids = {normalize_chunk_id(cid) for cid in entry.get("chunk_ids", []) if normalize_chunk_id(cid)}
        indexes = entry.get("support_group_indexes") or entry.get("partial_group_indexes") or []
        for index in indexes:
            if isinstance(index, int) and 0 <= index < len(partial):
                partial[index].update(ids)
    return {"strict_canonical": strict, "acceptable_full": full, "acceptable_plus_partial": partial}


def _metric(ranked_ids: Sequence[str], groups: Sequence[Set[str]], gains: Dict[str, int]) -> Dict[str, Any]:
    if not groups:
        return {"eligible": False}
    first = next((rank for rank, cid in enumerate(ranked_ids, 1) if any(cid in group for group in groups)), None)
    result: Dict[str, Any] = {
        "eligible": True,
        "group_count": len(groups),
        "first_relevant_rank": first,
        "mrr@20": (1.0 / first) if first and first <= 20 else 0.0,
        "ndcg@10": 0.0,
    }
    for k in CUTOFFS:
        selected = set(ranked_ids[:k])
        covered = sum(bool(group & selected) for group in groups)
        result[f"hit@{k}"] = float(covered > 0)
        result[f"recall@{k}"] = covered / len(groups)
        result[f"full_hit@{k}"] = float(covered == len(groups))
        result[f"covered_groups@{k}"] = covered

    dcg = 0.0
    covered_groups: Set[int] = set()
    for rank, chunk_id in enumerate(ranked_ids[:10], 1):
        group_gain = 0
        for group_index, group in enumerate(groups):
            if group_index not in covered_groups and chunk_id in group:
                group_gain += int(gains.get(chunk_id, 1))
                covered_groups.add(group_index)
        if group_gain:
            dcg += group_gain / math.log2(rank + 1)
    ideal_gains = sorted((max((int(gains.get(cid, 1)) for cid in group), default=1) for group in groups), reverse=True)
    ideal = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal_gains[:10]))
    result["ndcg@10"] = dcg / ideal if ideal else 0.0
    return result


def _metrics_for_question(question: Dict[str, Any], ranked_by_stage: Dict[str, Sequence[str]]) -> Dict[str, Any] | None:
    if question.get("evaluation_track") != "retrieval":
        return None
    groups = _relevance_groups(question)
    if not groups["acceptable_full"]:
        return None
    gold = {normalize_chunk_id(cid) for cid in question.get("gold_chunk_ids", []) if normalize_chunk_id(cid)}
    acceptable_ids = set().union(*groups["acceptable_full"])
    partial_ids = set().union(*groups["acceptable_plus_partial"])
    gains = {
        "strict_canonical": {cid: 1 for cid in gold},
        "acceptable_full": {cid: 1 for cid in acceptable_ids},
        "acceptable_plus_partial": {cid: (2 if cid in acceptable_ids else 1) for cid in partial_ids},
    }
    return {
        stage: {
            variant: _metric(ranked_by_stage[stage], groups[variant], gains[variant])
            for variant in VARIANTS
        }
        for stage in STAGES
    }


def _dense_results(collection, query_embedding: Sequence[float], top_n: int, metadata_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    response = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=top_n,
        include=["documents", "distances"],
    )
    ids = response.get("ids", [[]])[0]
    docs = response.get("documents", [[]])[0]
    distances = response.get("distances", [[]])[0]
    output = []
    for rank, cid in enumerate(ids, 1):
        chunk_id = normalize_chunk_id(cid)
        item = {
            "id": chunk_id,
            "rank": rank,
            "score": float(distances[rank - 1]),
            "document": docs[rank - 1],
            "metadata": metadata_by_id.get(chunk_id, {}),
        }
        output.append(annotate_result(item))
    return output


def _rrf(dense: Sequence[Dict[str, Any]], bm25: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    for channel, rows, weight in (("dense", dense, DENSE_WEIGHT), ("bm25", bm25, BM25_WEIGHT)):
        for item in rows:
            cid = normalize_chunk_id(item.get("id", ""))
            if cid not in fused:
                fused[cid] = {
                    "id": cid,
                    "document": item.get("document", ""),
                    "metadata": item.get("metadata") or {},
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }
            fused[cid][f"{channel}_rank"] = item["rank"]
            quality = item.get("reference_penalty", 1.0)
            fused[cid]["rrf_score"] += weight * quality / (RRF_K + item["rank"])
    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
        item["rrf_rank"] = rank
    return ranked


def _rank_summary(items: Sequence[Dict[str, Any]], chunk_map: Dict[str, Dict[str, Any]], limit: int, include_full_text: bool = False) -> List[Dict[str, Any]]:
    output = []
    for rank, item in enumerate(items[:limit], 1):
        cid = normalize_chunk_id(item.get("id") or item.get("chunk_id") or "")
        chunk = chunk_map.get(cid, {})
        metadata = chunk.get("metadata") or item.get("metadata") or {}
        text = str(chunk.get("text") or item.get("document") or "")
        row = {
            "rank": rank,
            "chunk_id": cid,
            "score": item.get("score", item.get("rrf_score")),
            "rrf_rank": item.get("rrf_rank", item.get("rank")),
            "reranker_score": item.get("reranker_score"),
            "doc_name": metadata.get("doc_name"),
            "guideline_title": metadata.get("guideline_title"),
            "section_path": metadata.get("section_path"),
            "page_start": metadata.get("page_start"),
            "metadata": metadata,
            "text": text if include_full_text else None,
            "excerpt": " ".join(text.split())[:500],
        }
        output.append(row)
    return output


def _retrieval_for_run(
    questions: List[Dict[str, Any]],
    query_embeddings: Sequence[Sequence[float]],
    bm25_rows: Dict[str, List[Dict[str, Any]]],
    collection,
    bm25: BM25Retriever,
    reranker: CrossEncoderReranker,
    chunk_map: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], float]:
    metadata_by_id = {cid: chunk.get("metadata") or {} for cid, chunk in chunk_map.items()}
    rows = []
    started_all = time.perf_counter()
    for index, (question, query_embedding) in enumerate(zip(questions, query_embeddings), 1):
        qid = str(question["id"]).strip()
        query = str(question["question"]).strip()
        started = time.perf_counter()
        dense_rows = _dense_results(collection, query_embedding, DENSE_TOP_N, metadata_by_id)
        bm25_results = bm25_rows[qid]
        rrf_full = _rrf(dense_rows, bm25_results)
        rrf_top = rrf_full[:RRF_TOP_K]
        reranker_input = []
        for item in rrf_top:
            full_metadata = item.get("metadata") or {}
            partial = {field: full_metadata.get(field) for field in RERANKER_CONTEXT_FIELDS if full_metadata.get(field) not in (None, "", [], {})}
            reranker_input.append({
                "id": item["id"],
                "document": item["document"],
                "metadata": partial,
                "rrf_rank": item["rrf_rank"],
            })
        rerank_started = time.perf_counter()
        reranked = reranker.rerank(
            query,
            reranker_input,
            top_k=RRF_TOP_K,
            include_metadata_context=True,
        )
        rerank_seconds = time.perf_counter() - rerank_started
        retrieval_seconds = time.perf_counter() - started
        ranked_ids = {
            "dense": [item["id"] for item in dense_rows],
            "bm25": [normalize_chunk_id(item.get("id", "")) for item in bm25_results],
            "rrf": [item["id"] for item in rrf_top],
            "reranker": [normalize_chunk_id(item.get("id", "")) for item in reranked],
        }
        metric_rows = _metrics_for_question(question, ranked_ids)
        rows.append({
            "question_id": qid,
            "evaluation_track": question.get("evaluation_track") or "retrieval",
            "question": query,
            "gold_answer": question.get("gold_answer") or "",
            "gold_passage": question.get("gold_passage") or "",
            "gold_chunk_ids": question.get("gold_chunk_ids") or [],
            "relevance_available": bool(metric_rows),
            "metrics": metric_rows,
            "retrieval_seconds": retrieval_seconds,
            "reranker_seconds": rerank_seconds,
            "dense_top20": _rank_summary(dense_rows, chunk_map, 20),
            "bm25_top20": _rank_summary(bm25_results, chunk_map, 20),
            "rrf_top20": _rank_summary(rrf_top, chunk_map, 20),
            "reranker_top20": _rank_summary(reranked, chunk_map, 20),
            "generation_top5": _rank_summary(reranked, chunk_map, GENERATION_TOP_K, include_full_text=True),
        })
        print(
            f"[retrieval {index:02}/{len(questions)}] {qid} "
            f"RRF top5={','.join(ranked_ids['rrf'][:5])} | "
            f"reranker top5={','.join(ranked_ids['reranker'][:5])} | "
            f"retrieval={retrieval_seconds:.3f}s rerank={rerank_seconds:.3f}s",
            flush=True,
        )
    return rows, time.perf_counter() - started_all


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for stage in STAGES:
        aggregate[stage] = {}
        for variant in VARIANTS:
            eligible = [row for row in rows if row["metrics"] and row["metrics"][stage][variant].get("eligible")]
            summary: Dict[str, Any] = {"eligible_questions": len(eligible)}
            for name in [*(f"hit@{k}" for k in CUTOFFS), *(f"recall@{k}" for k in CUTOFFS), *(f"full_hit@{k}" for k in CUTOFFS), "mrr@20", "ndcg@10"]:
                values = [float(row["metrics"][stage][variant][name]) for row in eligible]
                summary[f"macro_{name}"] = sum(values) / len(values) if values else None
            group_total = sum(row["metrics"][stage][variant]["group_count"] for row in eligible)
            for k in CUTOFFS:
                covered = sum(row["metrics"][stage][variant][f"covered_groups@{k}"] for row in eligible)
                summary[f"micro_recall@{k}"] = covered / group_total if group_total else None
            aggregate[stage][variant] = summary
    return aggregate


def _context_for_question(
    question: Dict[str, Any],
    retrieval_row: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    metadata_retriever: MetadataRetriever,
    chunk_collection: ChunkMapCollection,
    include_commentary: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates = []
    for item in retrieval_row["generation_top5"]:
        cid = item["chunk_id"]
        chunk = chunk_map.get(cid)
        if not chunk:
            continue
        candidates.append({
            "id": cid,
            "document": chunk.get("text") or "",
            "metadata": chunk.get("metadata") or {},
            "commented_text": chunk.get("commented_text") or (chunk.get("metadata") or {}).get("commented_text"),
        })
    metadata_results = []
    if question.get("evaluation_track") == "metadata_attribution":
        metadata_results = metadata_retriever.search(str(question["question"]), n_results=5)
    context = assemble_context(
        candidates,
        max_sources=GENERATION_TOP_K,
        metadata_results=metadata_results,
        max_metadata_sources=5,
        include_table_neighbors=True,
        table_neighbor_limit=TABLE_NEIGHBOR_LIMIT,
        include_commentary=include_commentary,
        collection=chunk_collection,
    )
    return context, metadata_results


def _generate(model: str, question: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    track = question.get("evaluation_track") or "retrieval"
    expected_behavior = EXPECTED_BEHAVIOR.get(track, EXPECTED_BEHAVIOR["retrieval"])
    user_message = (
        f"CLINICAL QUESTION:\n{question['question']}\n\n"
        f"EXPECTED RESPONSE BEHAVIOR:\n{expected_behavior}\n\n"
        f"CONTEXT:\n{context.get('context_text', '')}\n\n"
        "Answer the question using only the context."
    )
    started = time.perf_counter()
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        think=False,
        stream=True,
        options={"temperature": 0.0, "num_predict": 300},
    )
    first_token_at = None
    pieces: List[str] = []
    final: Dict[str, Any] = {}
    for item in stream:
        text = ((item.get("message") or {}).get("content") or "")
        if text and first_token_at is None:
            first_token_at = time.perf_counter()
        if text:
            pieces.append(text)
        if item.get("done"):
            final = item
    ended = time.perf_counter()
    response = "".join(pieces).strip()
    if not response:
        raise RuntimeError("Ollama returned an empty response.")
    return {
        "response": response,
        "ttft_seconds": first_token_at - started if first_token_at is not None else None,
        "generation_response_time_seconds": ended - started,
        "ollama_total_duration_ns": final.get("total_duration"),
        "ollama_load_duration_ns": final.get("load_duration"),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_count": final.get("eval_count"),
        "eval_duration_ns": final.get("eval_duration"),
    }


def _generation_row(
    question: Dict[str, Any],
    context: Dict[str, Any],
    metadata_results: List[Dict[str, Any]],
    generated: Dict[str, Any],
    retrieval_row: Dict[str, Any],
    model: str,
    run_name: str,
    include_commentary: bool,
) -> Dict[str, Any]:
    track = question.get("evaluation_track") or "retrieval"
    row: Dict[str, Any] = {
        "question_id": question["id"],
        "evaluation_track": track,
        "expected_behavior": EXPECTED_BEHAVIOR.get(track, EXPECTED_BEHAVIOR["retrieval"]),
        "user_input": question["question"],
        "retrieved_contexts": context.get("retrieved_contexts", []),
        "response": generated["response"],
        "reference": question.get("gold_answer") or "",
        "reference_passage": question.get("gold_passage") or "",
        "retrieved_chunk_ids": [item["chunk_id"] for item in retrieval_row["generation_top5"]],
        "reranker_top5": retrieval_row["generation_top5"],
        "context_sources": context.get("sources", []),
        "metadata_source_documents": [item.get("doc_name") for item in metadata_results],
        "retrieval_metrics": retrieval_row.get("metrics"),
        "generation_model": model,
        "run_name": run_name,
        "embedding_variant": run_name,
        "context_includes_commented_text": include_commentary,
        "ttft_seconds": generated["ttft_seconds"],
        "generation_response_time_seconds": generated["generation_response_time_seconds"],
        "retrieval_seconds": retrieval_row["retrieval_seconds"],
        "reranker_seconds": retrieval_row["reranker_seconds"],
        "end_to_end_seconds": retrieval_row["retrieval_seconds"] + retrieval_row["reranker_seconds"] + generated["generation_response_time_seconds"],
        "grounded_citation_valid": validate_citations(generated["response"], context.get("source_count", 0)),
    }
    for metric in RAGAS_METRICS:
        row[metric] = None
    return row


def _write_report_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        f"# {report['run_name']}",
        "",
        "RAGAS scores are not computed in this run. `ragas_input_generation.jsonl` contains the required inputs and null metric columns.",
        "",
        f"- Questions generated: {report['generation']['completed']}/{report['generation']['question_count']}",
        f"- Retrieval-metric eligible questions: {report['retrieval_metric_eligible_questions']}",
        f"- Chunk corpus: `{report['chunks_dir']}` ({report['bm25_index_count']:,} unique chunks)",
        f"- Dense collection: `{report['dense_collection']}` ({report['dense_index_count']:,} chunks)",
        f"- BM25: `all` semantic metadata; Dense: `{report['embedding_variant']}`; reranker metadata fields: {len(report['reranker_metadata_fields'])}",
        f"- Retrieval elapsed: {report['retrieval_elapsed_seconds']:.1f}s; generation elapsed: {report['generation']['elapsed_seconds']:.1f}s",
        "",
        "## Retrieval metrics (macro group recall, eligible retrieval-track questions)",
        "",
        "| Stage | Relevance | n | Hit@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR@20 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        for variant in VARIANTS:
            values = report["aggregate_metrics"][stage][variant]
            fmt = lambda key: "n/a" if values.get(key) is None else f"{values[key]:.3f}"
            lines.append(
                f"| {stage.upper()} | {variant} | {values['eligible_questions']} | {fmt('macro_hit@1')} | "
                f"{fmt('macro_recall@3')} | {fmt('macro_recall@5')} | {fmt('macro_recall@10')} | "
                f"{fmt('macro_ndcg@10')} | {fmt('macro_mrr@20')} |"
            )
    lines.extend([
        "",
        "## Generation timing",
        "",
        f"- Mean TTFT: {report['generation'].get('mean_ttft_seconds')}",
        f"- Median TTFT: {report['generation'].get('median_ttft_seconds')}",
        f"- Mean generation response time: {report['generation'].get('mean_response_time_seconds')}",
        f"- Median generation response time: {report['generation'].get('median_response_time_seconds')}",
        "",
        "## Files",
        "",
        f"- RAGAS input: `{report['files']['ragas_input']}`",
        f"- Per-question retrieval: `{report['files']['retrieval_rows']}`",
        f"- Full report: `{report['files']['report_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one(
    run_name: str,
    collection,
    questions: List[Dict[str, Any]],
    query_embeddings: Sequence[Sequence[float]],
    bm25_rows: Dict[str, List[Dict[str, Any]]],
    bm25: BM25Retriever,
    reranker: CrossEncoderReranker,
    chunk_map: Dict[str, Dict[str, Any]],
    metadata_retriever: MetadataRetriever,
    model: str,
) -> Dict[str, Any]:
    include_commentary = run_name == "commented_text_embedding"
    out_dir = OUTPUT_BASE / f"first50_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = out_dir / "retrieval_per_question.jsonl"
    generation_path = out_dir / "ragas_input_generation.jsonl"
    failures_path = out_dir / "generation_failures.jsonl"
    report_path = out_dir / "run_report.json"

    retrieval_rows, retrieval_elapsed = _retrieval_for_run(
        questions, query_embeddings, bm25_rows, collection, bm25, reranker, chunk_map
    )
    _write_jsonl(retrieval_path, retrieval_rows)
    retrieval_by_id = {row["question_id"]: row for row in retrieval_rows}

    existing = {row["question_id"]: row for row in _read_jsonl(generation_path) if row.get("response")}
    failures: List[Dict[str, Any]] = []
    chunk_collection = ChunkMapCollection(chunk_map)
    ttft_values: List[float] = []
    response_values: List[float] = []
    generation_started = time.perf_counter()
    for index, question in enumerate(questions, 1):
        qid = question["id"]
        if qid in existing:
            generated_row = existing[qid]
            if generated_row.get("ttft_seconds") is not None:
                ttft_values.append(float(generated_row["ttft_seconds"]))
            response_values.append(float(generated_row.get("generation_response_time_seconds", 0.0)))
            print(f"[generation {index:02}/{len(questions)}] {run_name} {qid} already complete", flush=True)
            continue

        retrieval_row = retrieval_by_id[qid]
        context, metadata_results = _context_for_question(
            question, retrieval_row, chunk_map, metadata_retriever, chunk_collection, include_commentary
        )
        last_error = None
        for attempt in range(1, 4):
            try:
                generated = _generate(model, question, context)
                generated_row = _generation_row(
                    question, context, metadata_results, generated, retrieval_row,
                    model, run_name, include_commentary,
                )
                with generation_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(generated_row, ensure_ascii=False) + "\n")
                    handle.flush()
                existing[qid] = generated_row
                if generated["ttft_seconds"] is not None:
                    ttft_values.append(float(generated["ttft_seconds"]))
                response_values.append(float(generated["generation_response_time_seconds"]))
                last_error = None
                print(
                    f"[generation {index:02}/{len(questions)}] {run_name} {qid} "
                    f"TTFT={generated['ttft_seconds']:.3f}s response={generated['generation_response_time_seconds']:.3f}s",
                    flush=True,
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if last_error:
            failure = {"question_id": qid, "attempts": 3, "error": last_error}
            failures.append(failure)
            with failures_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"[generation {index:02}/{len(questions)}] {run_name} {qid} FAILED {last_error}", flush=True)

    generation_elapsed = time.perf_counter() - generation_started
    total_metrics = _aggregate_metrics(retrieval_rows)
    eligible_n = sum(1 for row in retrieval_rows if row.get("metrics"))
    report = {
        "run_name": run_name,
        "benchmark": str(BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
        "question_count": len(questions),
        "evaluation_track_counts": {
            track: sum(1 for question in questions if question.get("evaluation_track") == track)
            for track in sorted({str(question.get("evaluation_track") or "retrieval") for question in questions})
        },
        "retrieval_metric_eligible_questions": eligible_n,
        "chunks_dir": str(CHUNKS_DIR.relative_to(PROJECT_ROOT)),
        "bm25_index_count": len(bm25.ids),
        "bm25_metadata_mode": "all semantic metadata (technical IDs/geometry/counters/embedding_text excluded)",
        "dense_collection": collection.name,
        "dense_index_count": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_variant": run_name,
        "embedding_input": (
            "Original chunk.text only"
            if run_name == "raw_text_embedding"
            else "Original chunk.text + two newlines + commented_text for chunks with non-empty commentary; stored documents remain original chunk.text"
        ),
        "reranker_model": reranker.model_name,
        "reranker_metadata_fields": list(RERANKER_CONTEXT_FIELDS),
        "dense_top_n": DENSE_TOP_N,
        "bm25_top_n": BM25_TOP_N,
        "rrf_top_k": RRF_TOP_K,
        "rrf_k": RRF_K,
        "dense_weight": DENSE_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "quality_reference_penalty": True,
        "generation_model": model,
        "generation_context": {
            "reranker_top_k": GENERATION_TOP_K,
            "include_table_neighbors": True,
            "table_neighbor_limit_per_side": TABLE_NEIGHBOR_LIMIT,
            "include_commentary": include_commentary,
            "metadata_source_cards_only_for": "metadata_attribution questions",
        },
        "ragas_scored": False,
        "ragas_metrics_to_run": list(RAGAS_METRICS),
        "aggregate_metrics": total_metrics,
        "retrieval_elapsed_seconds": retrieval_elapsed,
        "generation": {
            "question_count": len(questions),
            "completed": len(existing),
            "failures": failures,
            "elapsed_seconds": generation_elapsed,
            "mean_ttft_seconds": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "median_ttft_seconds": sorted(ttft_values)[len(ttft_values) // 2] if ttft_values else None,
            "mean_response_time_seconds": sum(response_values) / len(response_values) if response_values else None,
            "median_response_time_seconds": sorted(response_values)[len(response_values) // 2] if response_values else None,
        },
        "files": {
            "ragas_input": str(generation_path.relative_to(PROJECT_ROOT)),
            "retrieval_rows": str(retrieval_path.relative_to(PROJECT_ROOT)),
            "failures": str(failures_path.relative_to(PROJECT_ROOT)),
            "report_json": str(report_path.relative_to(PROJECT_ROOT)),
        },
    }
    _write_json(report_path, report)
    _write_report_markdown(out_dir / "run_report.md", report)
    print("\n" + json.dumps({
        "run_name": run_name,
        "generation_completed": len(existing),
        "generation_total": len(questions),
        "mean_ttft_seconds": report["generation"]["mean_ttft_seconds"],
        "mean_generation_response_time_seconds": report["generation"]["mean_response_time_seconds"],
        "report": str(report_path),
        "ragas_input": str(generation_path),
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--run", choices=("both", "raw_text_embedding", "commented_text_embedding"), default="both")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")

    questions = _load_questions(BENCHMARK_PATH)
    chunk_map = _load_chunk_map(CHUNKS_DIR)
    bm25 = BM25Retriever(chunks_dir=CHUNKS_DIR, metadata_mode="all")
    if len(bm25.ids) != len(set(bm25.ids)):
        raise RuntimeError("BM25 chunk IDs are not unique.")
    if set(chunk_map) != set(bm25.ids):
        raise RuntimeError(f"Chunk map/BM25 IDs differ: chunk_map={len(chunk_map)}, bm25={len(bm25.ids)}")

    raw_client = chromadb.PersistentClient(path=str(RAW_DB))
    comment_client = chromadb.PersistentClient(path=str(COMMENTED_DB))
    raw_collection = raw_client.get_collection(RAW_COLLECTION)
    commented_collection = comment_client.get_collection(COMMENTED_COLLECTION)
    _validate_dense_collection(raw_collection, bm25)
    _validate_dense_collection(commented_collection, bm25)

    model_available = {str(item.get("model") or item.get("name") or "") for item in ollama.list().models}
    if args.model not in model_available:
        raise RuntimeError(f"Ollama model {args.model!r} is unavailable. Installed models: {sorted(model_available)}")

    queries = [str(question["question"]).strip() for question in questions]
    query_embeddings = embed_texts(queries, input_type="query")
    if len(query_embeddings) != len(questions):
        raise RuntimeError(f"Query embedding count mismatch: {len(query_embeddings)} != {len(questions)}")

    bm25_rows: Dict[str, List[Dict[str, Any]]] = {}
    for index, question in enumerate(questions, 1):
        qid = str(question["id"]).strip()
        bm25_rows[qid] = bm25.search(str(question["question"]).strip(), n_results=BM25_TOP_N)
        if index % 10 == 0:
            print(f"BM25 all-metadata results prepared for {index}/{len(questions)} questions.", flush=True)

    metadata_retriever = MetadataRetriever(chunks_dir=CHUNKS_DIR)
    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL, batch_size=4, max_length=512)
    selected = {
        "raw_text_embedding": raw_collection,
        "commented_text_embedding": commented_collection,
    }
    runs = list(selected.items()) if args.run == "both" else [(args.run, selected[args.run])]
    final_reports = []
    for run_name, collection in runs:
        final_reports.append(_run_one(
            run_name, collection, questions, query_embeddings, bm25_rows, bm25,
            reranker, chunk_map, metadata_retriever, args.model,
        ))
    print("\nRUNS COMPLETE. RAGAS was not run; generation JSONL files contain null metric fields.", flush=True)


if __name__ == "__main__":
    main()
