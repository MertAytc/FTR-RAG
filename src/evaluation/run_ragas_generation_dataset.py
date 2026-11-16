"""Generate Qwen responses and RAGAS-ready rows for the frozen 100 questions.

Retrieval/reranker outputs are read from the saved source-corrected run. The
script calls only the configured Ollama generation model, appending a JSONL row
after each successful answer so interrupted runs can resume safely.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import ollama

from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import generate_grounded_answer
from src.retrieval.config import CHUNKS_DIR, PROJECT_ROOT
from src.retrieval.metadata_retriever import MetadataRetriever

logger = logging.getLogger("ragas_generation_dataset")

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data/processed/retrieval_100_goldchunks_v3/frozen/"
    / "FTR_RAG_100_Benchmark_Questions_GoldChunks_v3.json"
)
FREEZE_MANIFEST_PATH = BENCHMARK_PATH.parent / "freeze_manifest.json"
RUN_DIR = PROJECT_ROOT / "data/processed/retrieval_100_goldchunks_v3/source_corrected_20261028"
RERANKER_REPORT_PATH = RUN_DIR / "retrieval_100_reranker_report.json"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/ragas_100_generation"
OUTPUT_PATH = OUTPUT_DIR / "ragas_100_generation.jsonl"
FAILURE_PATH = OUTPUT_DIR / "generation_failures.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "generation_manifest.json"

RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
    "context_entity_recall",
    "factual_correctness_f1",
    "semantic_similarity",
    "noise_sensitivity_relevant",
)

EXPECTED_BEHAVIOR = {
    "retrieval": (
        "Answer the clinical question directly from retrieved clinical sources. "
        "Cite important claims with the source numbers. If retrieved evidence is "
        "insufficient, say so and do not guess."
    ),
    "metadata_attribution": (
        "Identify the correct source document. Use catalog year only as the year "
        "recorded in corpus metadata; distinguish it from publication, revision, "
        "and filename dates unless retrieved text explicitly connects them. Cite "
        "metadata and text sources. State when the distinction is not established."
    ),
    "clarification_behavior": (
        "Ask concise clarifying questions before selecting an exercise. Do not "
        "diagnose or recommend a condition-specific exercise based only on generic pain."
    ),
    "excluded_invalid_gold": (
        "Answer only if the retrieved context directly supports the requested "
        "details. Otherwise state that the details are not specified; do not infer them."
    ),
    "pending_gold": (
        "Use only retrieved evidence. If it does not establish a universal value or "
        "recommendation, say that it is not specified; do not invent one."
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark(benchmark_path: Path = BENCHMARK_PATH) -> List[Dict[str, Any]]:
    data = load_json(benchmark_path)
    rows = data if isinstance(data, list) else next(
        data[key] for key in ("questions", "data", "items") if isinstance(data.get(key), list)
    )
    if benchmark_path.resolve() == BENCHMARK_PATH.resolve():
        expected_hash = load_json(FREEZE_MANIFEST_PATH)["benchmark_sha256"].lower()
        import hashlib

        actual_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest().lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen benchmark hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        if len(rows) != 100:
            raise RuntimeError(f"Expected the frozen 100-question benchmark, found {len(rows)} rows")
    return rows


def load_chunk_map(chunks_dir: Path) -> Dict[str, Dict[str, Any]]:
    chunk_map: Dict[str, Dict[str, Any]] = {}
    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No chunk JSON files found in {chunks_dir}")
    for path in files:
        if path.name == "_quality_report.json":
            continue
        data = load_json(path)
        chunks = data.get("chunks", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
            if chunk_id:
                chunk_map.setdefault(chunk_id, chunk)
    return chunk_map


def load_existing_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring malformed existing output line %d: %s", line_number, exc)
            continue
        if row.get("question_id") and row.get("response"):
            rows[row["question_id"]] = row
    return rows


def build_context(
    question: Dict[str, Any],
    reranker_row: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
    metadata_retriever: MetadataRetriever,
    *,
    include_table_neighbors: bool = False,
    include_commentary: bool = False,
) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []
    for result in reranker_row.get("reranker_top20", [])[:5]:
        chunk_id = str(result.get("chunk_id") or "").strip()
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        chunk_ids.append(chunk_id)
        candidates.append({
            "id": chunk_id,
            "document": chunk.get("text") or chunk.get("document") or "",
            "metadata": chunk.get("metadata") or {},
            "commented_text": chunk.get("commented_text") or (chunk.get("metadata") or {}).get("commented_text"),
        })

    metadata_results: List[Dict[str, Any]] = []
    if question.get("evaluation_track") == "metadata_attribution":
        metadata_results = metadata_retriever.search(question["question"], n_results=5)

    context = assemble_context(
        candidates,
        max_sources=5,
        metadata_results=metadata_results,
        max_metadata_sources=5,
        include_table_neighbors=include_table_neighbors,
        include_commentary=include_commentary,
    )
    return context, chunk_ids, metadata_results


def create_row(
    question: Dict[str, Any],
    context: Dict[str, Any],
    chunk_ids: List[str],
    metadata_results: List[Dict[str, Any]],
    answer_result: Any,
    model: str,
) -> Dict[str, Any]:
    track = question.get("evaluation_track") or "retrieval"
    row: Dict[str, Any] = {
        "question_id": question["id"],
        "evaluation_track": track,
        "expected_behavior": EXPECTED_BEHAVIOR.get(track, EXPECTED_BEHAVIOR["retrieval"]),
        "user_input": question["question"],
        "retrieved_contexts": context["retrieved_contexts"],
        # Preserve only the model's text. Post-generation citation diagnostics
        # remain in separate boolean fields and must not enter RAGAS response.
        "response": answer_result.english_answer or answer_result.answer,
        "reference": question.get("gold_answer") or "",
        "reference_passage": question.get("gold_passage") or "",
        "reference_review_note": question.get("gold_chunk_review_note") or "",
        "retrieved_chunk_ids": chunk_ids,
        "metadata_source_documents": [hit["doc_name"] for hit in metadata_results],
        "context_sources": [
            {
                "source_number": source["source_number"],
                "source_kind": source["source_kind"],
                "chunk_id": source.get("chunk_id"),
                "document_name": source.get("doc_name"),
                "organization": source.get("organization"),
                "catalog_year": source.get("year"),
            }
            for source in context["sources"]
        ],
        "generation_model": model,
        "grounded_citation_valid": answer_result.grounded,
        "insufficient_evidence": answer_result.insufficient_evidence,
        "out_of_scope": answer_result.out_of_scope,
    }
    for metric in RAGAS_METRICS:
        row[metric] = None
    return row


def write_manifest(
    model: str,
    completed: int,
    total: int,
    tracks: Dict[str, int],
    output_path: Path,
    reranker_report_path: Path,
    benchmark_path: Path,
    include_table_neighbors: bool,
    include_commentary: bool,
    reuse_identical_context_from: Path | None = None,
) -> None:
    import hashlib

    manifest = {
        "model": model,
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "benchmark_path": str(benchmark_path),
        "retrieval_source": str(reranker_report_path),
        "retrieval_policy": "Saved source-corrected reranker top-5; no retrieval/reranker rerun.",
        "metadata_policy": "Up to five document metadata source cards are included for metadata_attribution questions only.",
        "context_options": {
            "include_table_neighbors": include_table_neighbors,
            "include_commentary": include_commentary,
            "table_neighbor_limit_per_side": 1,
            "reuse_identical_context_from": str(reuse_identical_context_from) if reuse_identical_context_from else None,
        },
        "question_count": total,
        "completed_count": completed,
        "track_counts": tracks,
        "ragas_metrics_to_run": list(RAGAS_METRICS),
        "output_jsonl": str(output_path),
        "generation_parameters": {"temperature": 0.0, "num_predict": 300, "think": False},
    }
    (output_path.parent / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(
    model: str,
    chunks_dir: Path,
    output_path: Path,
    reranker_report_path: Path = RERANKER_REPORT_PATH,
    benchmark_path: Path = BENCHMARK_PATH,
    include_table_neighbors: bool = False,
    include_commentary: bool = False,
    reuse_identical_context_from: Path | None = None,
) -> None:
    benchmark = load_benchmark(benchmark_path)
    reranker_report = load_json(reranker_report_path)
    reranker_rows = {row["question_id"]: row for row in reranker_report["per_question"]}
    chunks = load_chunk_map(chunks_dir)
    metadata_retriever = MetadataRetriever(chunks_dir=chunks_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_path.parent / "generation_failures.jsonl"
    existing = load_existing_rows(output_path)
    reusable_rows = load_existing_rows(reuse_identical_context_from) if reuse_identical_context_from else {}
    tracks: Dict[str, int] = {}
    for question in benchmark:
        track = question.get("evaluation_track") or "retrieval"
        tracks[track] = tracks.get(track, 0) + 1

    # Confirm the requested model is available before processing any question.
    model_names = {
        str(getattr(item, "model", "") or getattr(item, "name", ""))
        for item in ollama.list().models
    }
    if model not in model_names:
        raise RuntimeError(f"Ollama model {model!r} is not installed. Available: {sorted(model_names)}")

    print(
        f"Model: {model}; questions: {len(benchmark)}; already complete: {len(existing)}; "
        f"to generate: {len([q for q in benchmark if q['id'] not in existing])}",
        flush=True,
    )
    write_manifest(model, len(existing), len(benchmark), tracks, output_path,
                   reranker_report_path, benchmark_path, include_table_neighbors, include_commentary,
                   reuse_identical_context_from)

    for index, question in enumerate(benchmark, start=1):
        question_id = question["id"]
        if question_id in existing:
            continue
        if question_id not in reranker_rows:
            raise KeyError(f"No saved reranker result for {question_id}")

        context, chunk_ids, metadata_results = build_context(
            question, reranker_rows[question_id], chunks, metadata_retriever,
            include_table_neighbors=include_table_neighbors,
            include_commentary=include_commentary,
        )
        reusable = reusable_rows.get(question_id)
        if reusable and reusable.get("retrieved_contexts") == context["retrieved_contexts"]:
            reused_row = dict(reusable)
            reused_row["generation_reused_from_identical_context"] = str(reuse_identical_context_from)
            with output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(reused_row, ensure_ascii=False) + "\n")
                output.flush()
            existing[question_id] = reused_row
            write_manifest(model, len(existing), len(benchmark), tracks, output_path,
                           reranker_report_path, benchmark_path, include_table_neighbors, include_commentary,
                           reuse_identical_context_from)
            print(f"Reused identical context for {question_id} ({len(existing)}/{len(benchmark)})", flush=True)
            continue
        expected_behavior = EXPECTED_BEHAVIOR.get(
            question.get("evaluation_track"), EXPECTED_BEHAVIOR["retrieval"]
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                answer = generate_grounded_answer(
                    query=question["question"],
                    context_result=context,
                    model=model,
                    expected_behavior=expected_behavior,
                )
                row = create_row(
                    question, context, chunk_ids, metadata_results, answer, model
                )
                with output_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                existing[question_id] = row
                write_manifest(model, len(existing), len(benchmark), tracks, output_path,
                               reranker_report_path, benchmark_path, include_table_neighbors, include_commentary,
                               reuse_identical_context_from)
                print(f"Generated {question_id} ({len(existing)}/{len(benchmark)})", flush=True)
                last_error = None
                break
            except Exception as exc:  # Keep failures visible and resumable.
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            failure = {
                "question_id": question_id,
                "attempts": 3,
                "error": f"{type(last_error).__name__}: {last_error}",
            }
            with failure_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"Failed {question_id}: {failure['error']}", flush=True)

    write_manifest(model, len(existing), len(benchmark), tracks, output_path,
                   reranker_report_path, benchmark_path, include_table_neighbors, include_commentary,
                   reuse_identical_context_from)
    print(json.dumps({
        "model": model,
        "completed": len(existing),
        "total": len(benchmark),
        "failures_file": str(failure_path),
        "output": str(output_path),
        "manifest": str(output_path.parent / "generation_manifest.json"),
    }, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--reranker-report", type=Path, default=RERANKER_REPORT_PATH)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--include-table-neighbors", action="store_true")
    parser.add_argument("--include-commentary", action="store_true")
    parser.add_argument("--reuse-identical-context-from", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run(args.model, args.chunks_dir, args.output, args.reranker_report, args.benchmark,
        args.include_table_neighbors, args.include_commentary,
        args.reuse_identical_context_from)


if __name__ == "__main__":
    main()
