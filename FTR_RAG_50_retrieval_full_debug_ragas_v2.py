"""FTR-RAG 50-question retrieval comparison: BM25, Dense, RRF, BGE reranker (Nano-Jev excluded).

Run from project root. Uses existing index; never rebuilds/deletes it.
Outputs are checkpointed after each method so partial runs remain available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import traceback
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding import EMBEDDING_MODEL
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import process_query
from src.retrieval.reranker import RERANKER_MODEL, CrossEncoderReranker
from src.retrieval.vector_store import get_collection

ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "ftr_rag_50_soru_gold_matched.json"
DEFAULT_PREFIX = "retrieval_benchmark_50_full_debug_v2_14"

# Retrieval configuration. These settings do not alter the index.
DENSE_TOP_N = 40
BM25_TOP_N = 40
RRF_K = 60
DENSE_WEIGHT = 1.10
BM25_WEIGHT = 0.85
RRF_CANDIDATE_K = 30
RERANK_CANDIDATE_K = 30
RERANK_FINAL_K = 20
RERANK_BATCH_SIZE = 16
RERANK_MAX_LENGTH = 512
METHODS = [
    "BM25-only", "Dense-only", "Dense+BM25 RRF",
    "RRF+BGE-Reranker Top-20",
]
METRICS = ["Hit@1", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]


def norm(value) -> str:
    return str(value).strip() if value is not None else ""


def parse_gold(q: dict) -> list[str]:
    values = q.get("gold_chunk_ids", q.get("gold_chunk_id", q.get("gold_ids", [])))
    if isinstance(values, str):
        values = values.replace("|", ";").replace(",", ";").split(";")
    if not isinstance(values, (list, tuple, set)):
        values = [values] if values else []
    return list(dict.fromkeys(norm(v) for v in values if norm(v)))


def load_questions(path: Path, expected_count: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dosyası bulunamadı: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    qs = data.get("questions", data.get("benchmark", data.get("items", []))) if isinstance(data, dict) else data
    if not isinstance(qs, list) or not qs:
        raise ValueError("Benchmark JSON içinde soru listesi bulunamadı.")
    if expected_count is not None and len(qs) != expected_count:
        raise ValueError(f"{expected_count} soru bekleniyordu; bulunan: {len(qs)}")
    out, seen = [], set()
    for i, raw in enumerate(qs, 1):
        q = dict(raw)
        qid = norm(q.get("id") or q.get("question_id") or q.get("qid") or f"Q{i}")
        question = norm(q.get("question") or q.get("query") or q.get("soru"))
        if not question:
            raise ValueError(f"{qid}: soru metni boş.")
        if qid in seen:
            raise ValueError(f"Tekrarlanan soru ID'si: {qid}")
        seen.add(qid)
        q.update(_id=qid, _question=question, _gold=parse_gold(q))
        out.append(q)
    return out


def get_metrics(results: list[dict], gold_ids: list[str]) -> dict:
    # With no gold IDs, retrieval still runs, but quality metrics are N/A, not zero.
    if not gold_ids:
        return {**{m: None for m in METRICS}, "first_relevant_rank": None,
                "gold_count": 0, "gold_found": 0, "gold_ranks": []}
    ids = [norm(x.get("id")) for x in results if norm(x.get("id"))]
    gold = set(gold_ids)
    ranks = [rank for rank, cid in enumerate(ids, 1) if cid in gold]
    first = min(ranks) if ranks else None

    def recall(k: int) -> float:
        return len(set(ids[:k]) & gold) / len(gold)

    def ndcg(k: int) -> float:
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= k)
        ideal_n = min(len(gold), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
        return dcg / idcg if idcg else 0.0

    return {
        "Hit@1": float(bool(ids and ids[0] in gold)),
        "Recall@1": recall(1), "Recall@3": recall(3),
        "Recall@5": recall(5), "Recall@10": recall(10),
        "MRR": 1.0 / first if first else 0.0,
        "nDCG@5": ndcg(5), "nDCG@10": ndcg(10),
        "first_relevant_rank": first, "gold_count": len(gold),
        "gold_found": len(ranks), "gold_ranks": ranks,
    }


def safe_json(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def jsonable(value):
    """Convert nested values into JSON-safe primitives without dropping debug detail."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    try:
        return value.tolist()
    except Exception:
        return str(value)


def first_value(item: dict, keys: tuple[str, ...]):
    for key in keys:
        if item.get(key) is not None:
            return item.get(key)
    return None


def result_debug_item(result: dict, rank: int, gold_ids: list[str]) -> dict:
    """Keep full candidate payload and normalized fields needed to audit retrieval/RAGAS."""
    item = dict(result) if isinstance(result, dict) else {"raw_result": str(result)}
    chunk_id = norm(first_value(item, ("id", "chunk_id", "document_id")))
    content = first_value(item, ("document", "text", "content", "page_content", "chunk_text"))
    metadata = first_value(item, ("metadata", "metadatas", "meta"))
    score = first_value(item, ("rerank_score", "score", "similarity", "distance", "bm25_score", "rrf_score"))
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "is_gold": chunk_id in set(gold_ids),
        "score": jsonable(score),
        "score_fields": {k: jsonable(item.get(k)) for k in
                          ("rerank_score", "score", "similarity", "distance", "bm25_score", "rrf_score", "rrf_score_dense", "rrf_score_bm25") if k in item},
        "text": norm(content),
        "text_chars": len(norm(content)),
        "metadata": jsonable(metadata),
        "raw_result": jsonable(item),
    }


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(payload), ensure_ascii=False, default=safe_json) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank_of_gold(results: list[dict], gold_ids: list[str]) -> dict:
    ids = [norm(r.get("id") or r.get("chunk_id")) for r in results]
    return {g: (ids.index(g) + 1 if g in ids else None) for g in gold_ids}


def make_row(q: dict, method: str, results: list[dict], elapsed_ms: float,
             query_ms: float, rewritten: str, expanded: str, extra: dict | None = None) -> dict:
    met = get_metrics(results, q["_gold"])
    # Keep compact IDs/scores in CSV; full text/metadata is stored in debug JSONL.
    row = {
        "question_id": q["_id"], "category": q.get("category", ""),
        "domain": q.get("domain", ""), "answerability": q.get("answerability", ""),
        "expected_behavior": q.get("expected_behavior", ""), "method": method,
        "question": q["_question"], "rewritten_query": rewritten,
        "expanded_query": expanded, "query_processing_ms": round(query_ms, 2),
        "method_ms": round(elapsed_ms, 2), "gold_chunk_ids": ";".join(q["_gold"]),
        "result_count": len(results),
        "top20_chunk_ids": ";".join(norm(r.get("id") or r.get("chunk_id")) for r in results[:20] if norm(r.get("id") or r.get("chunk_id"))),
        "top20_scores": ";".join(str(safe_json(first_value(r, ("rerank_score", "score", "similarity", "distance", "bm25_score", "rrf_score")))) for r in results[:20]),
        "gold_rank_by_id": json.dumps(rank_of_gold(results, q["_gold"]), ensure_ascii=False),
        **met,
    }
    if extra:
        row.update(extra)
    return row


def print_row(row: dict) -> None:
    # Each method is one physical output line; ANSI colors make methods easier to distinguish.
    colors = {
        "BM25-only": "\033[36m",                 # cyan
        "Dense-only": "\033[34m",                # blue
        "Dense+BM25 RRF": "\033[35m",            # magenta
        "RRF+BGE-Reranker Top-20": "\033[32m",   # green
    }
    reset = "\033[0m"
    color = colors.get(row["method"], "") if sys.stdout.isatty() else ""
    method_no = {name: i for i, name in enumerate(METHODS, 1)}.get(row["method"], 0)
    metrics = " | ".join(
        f"{name}={'N/A' if row[name] is None else f'{row[name]:.3f}'}"
        for name in ("Hit@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5")
    )
    first_rank = row['first_relevant_rank'] if row['first_relevant_rank'] is not None else 'yok'
    print(
        f"{color}[{method_no}] {row['method']:<25}{reset} | {metrics}"
        f" | gold_sırası={first_rank:<3} | gold={row['gold_found']}/{row['gold_count']}"
        f" | süre={row['method_ms']:.0f} ms",
        flush=True,
    )


def aggregate(rows: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method and r["gold_count"] > 0]
        if not subset:
            continue
        summary[method] = {
            "evaluated_questions": len(subset),
            **{metric: mean(r[metric] for r in subset if r[metric] is not None) for metric in METRICS},
            "avg_method_ms": mean(r["method_ms"] for r in subset),
            "avg_query_processing_ms": mean(r["query_processing_ms"] for r in subset),
        }
    return summary


def save_checkpoint(rows: list[dict], out_csv: Path, out_json: Path, meta: dict) -> None:
    if not rows:
        return
    # Stable field union protects against optional metadata fields.
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {**meta, "completed_method_rows": len(rows), "summary": aggregate(rows), "results": rows}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=safe_json), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="FTR-RAG 50 soru: BM25, Dense, RRF ve BGE reranker (Nano-Jev kapalı)")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--expected-count", type=int, default=50)
    ap.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--limit", type=int, default=None, help="ilk N soruyla deneme; ör. --limit 2")
    ap.add_argument("--debug-text-max-chars", type=int, default=12000, help="debug dosyasına yazılacak chunk metni üst sınırı; 0=kesme yok")
    args = ap.parse_args()

    questions = load_questions(args.benchmark.resolve(), args.expected_count)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("Çalıştırılacak soru yok.")

    prefix = args.output_prefix
    out_csv = ROOT / f"{prefix}_results.csv"
    out_json = ROOT / f"{prefix}_results.json"
    out_debug = ROOT / f"{prefix}_debug.jsonl"
    out_ragas = ROOT / f"{prefix}_ragas_retrieval_dataset.jsonl"
    out_manifest = ROOT / f"{prefix}_run_manifest.json"
    out_log = ROOT / f"{prefix}_debug.log"
    # Refuse to overwrite an existing run. User can set a new --output-prefix.
    outputs = [out_csv, out_json, out_debug, out_ragas, out_manifest, out_log]
    existing = [x.name for x in outputs if x.exists()]
    if existing:
        raise FileExistsError(f"Çıktı dosyaları zaten var: {existing}. --output-prefix ile yeni ad ver.")
    logging.basicConfig(filename=out_log, level=logging.DEBUG, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")

    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(f"Dense koleksiyonu boş: {collection.name}")
    dense_ids = {norm(x) for x in collection.get()["ids"]}

    print("=" * 110)
    print("FTR-RAG | RETRIEVAL BENCHMARK | NanoJev HARİÇ")
    print(f"Sorular: {len(questions)} | Benchmark: {args.benchmark.name}")
    print(f"Dense koleksiyonu: {collection.name} ({collection.count()} chunk)")
    print(f"Embedding: {EMBEDDING_MODEL} | BGE: {RERANKER_MODEL}")
    print(f"Ayarlar: Dense Top-{DENSE_TOP_N}, BM25 Top-{BM25_TOP_N}, RRF Top-{RRF_CANDIDATE_K}, "
          f"BGE Top-{RERANK_CANDIDATE_K}->{RERANK_FINAL_K}")
    print("Yöntemler: BM25, Dense, RRF, RRF+BGE Reranker Top-20 | NanoJev kullanılmıyor | İndeks değiştirilmeyecek.")

    print("\nBM25 indeksi yükleniyor...", flush=True)
    bm25 = BM25Retriever()
    bm25_ids = {norm(x) for x in bm25.ids}
    gold_questions = [q for q in questions if q["_gold"]]
    no_gold_ids = [q["_id"] for q in questions if not q["_gold"]]
    missing_dense = [(q["_id"], g) for q in gold_questions for g in q["_gold"] if g not in dense_ids]
    missing_bm25 = [(q["_id"], g) for q in gold_questions for g in q["_gold"] if g not in bm25_ids]
    if missing_dense or missing_bm25:
        print(f"Gold ID eksikleri Dense={len(missing_dense)}, BM25={len(missing_bm25)}")
        print("Dense örnek:", missing_dense[:10])
        print("BM25 örnek:", missing_bm25[:10])
        raise RuntimeError("Gold chunk ID kontrolü başarısız; geçersiz kıyaslama önlendi.")
    print(f"Gold kontrolü: {len(gold_questions)} gold içeren soru, {len(no_gold_ids)} gold içermeyen soru.")
    if no_gold_ids:
        print("Gold içermeyen sorular (metrikleri N/A): " + ", ".join(no_gold_ids))

    hybrid = HybridRetriever(bm25_retriever=bm25, dense_top_n=DENSE_TOP_N,
                             bm25_top_n=BM25_TOP_N, rrf_k=RRF_K,
                             dense_weight=DENSE_WEIGHT, bm25_weight=BM25_WEIGHT)
    bge = CrossEncoderReranker(batch_size=RERANK_BATCH_SIZE, max_length=RERANK_MAX_LENGTH)

    meta = {
        "benchmark_file": str(args.benchmark), "question_count": len(questions),
        "questions_with_gold": len(gold_questions), "questions_without_gold": no_gold_ids,
        "dense_collection": collection.name, "dense_chunk_count": collection.count(),
        "bm25_chunk_count": len(bm25.ids), "embedding_model": EMBEDDING_MODEL,
        "bge_reranker_model": RERANKER_MODEL,
        "settings": {"dense_top_n": DENSE_TOP_N, "bm25_top_n": BM25_TOP_N,
                     "rrf_k": RRF_K, "dense_weight": DENSE_WEIGHT, "bm25_weight": BM25_WEIGHT,
                     "rrf_candidate_k": RRF_CANDIDATE_K,
                     "rerank_candidate_k": RERANK_CANDIDATE_K, "rerank_final_k": RERANK_FINAL_K},
        "metric_note": "Gold içermeyen sorularda retrieval kalite metrikleri N/A; ortalamalara yalnızca gold içeren sorular dahil.",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "platform": platform.platform(),
        "cwd": str(Path.cwd()), "script_path": str(Path(__file__).resolve()),
        "benchmark_sha256": sha256_file(args.benchmark.resolve()),
        "output_files": {"csv": out_csv.name, "json": out_json.name, "debug_jsonl": out_debug.name,
                         "ragas_retrieval_jsonl": out_ragas.name, "manifest": out_manifest.name, "log": out_log.name},
        "ragas_dataset_schema": "ftr-rag-ragas-input-v2",
        "ragas_note": "contexts/IDs/scores and gold passage are exported per method. answer remains null until a real answer is supplied. gold_passage is not silently treated as ground_truth.",
        "diagnostic_note": "Bu betik retrieval yapar; answer/response üretmez. Faithfulness ve Answer Relevancy için gerçek generated answer; Context Recall için uygun reference answer gerekir.",
    }
    rows = []
    start_all = time.perf_counter()
    out_manifest.write_text(json.dumps(jsonable({**meta, "status": "running", "started_monotonic": start_all}), ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("RUN_START %s", json.dumps(jsonable(meta), ensure_ascii=False))
    debug_run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{prefix}"
    pool_gold_found = 0

    def record(q, method, results, elapsed, query_ms, rewritten, expanded, extra=None):
        row = make_row(q, method, results, elapsed, query_ms, rewritten, expanded, extra)
        rows.append(row)
        candidates = [result_debug_item(r, i, q["_gold"]) for i, r in enumerate(results, 1)]
        for item in candidates:
            if args.debug_text_max_chars and len(item["text"]) > args.debug_text_max_chars:
                item["text_original_chars"] = len(item["text"])
                item["text"] = item["text"][:args.debug_text_max_chars]
                item["text_truncated"] = True
        debug_record = {
            "run_id": debug_run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "retrieval_method_result", "question_id": q["_id"], "method": method,
            "question": q["_question"], "category": q.get("category"), "domain": q.get("domain"),
            "answerability": q.get("answerability"), "expected_behavior": q.get("expected_behavior"),
            "gold_chunk_ids": q["_gold"], "rewritten_query": rewritten, "expanded_query": expanded,
            "query_processing_ms": round(query_ms, 3), "method_ms": round(elapsed, 3),
            "metrics": {k: row.get(k) for k in METRICS + ["first_relevant_rank", "gold_count", "gold_found", "gold_ranks"]},
            "gold_rank_by_id": rank_of_gold(results, q["_gold"]),
            "candidate_count": len(results), "candidates": candidates,
            "extra": jsonable(extra or {}),
        }
        append_jsonl(out_debug, debug_record)
        # RAGAS-compatible retrieval input: context strings and optional reference. No fabricated answer.
        reference_answer = norm(
            q.get("ground_truth") or q.get("reference_answer") or
            q.get("expected_answer") or q.get("gold_answer")
        ) or None
        ragas_record = {
            "schema_version": "ftr-rag-ragas-input-v2",
            "run_id": debug_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question_id": q["_id"],
            "question": q["_question"],
            "category": q.get("category"),
            "domain": q.get("domain"),
            "answerability": q.get("answerability"),
            "expected_behavior": q.get("expected_behavior"),
            "method": method,
            "contexts": [c["text"] for c in candidates if c["text"]],
            "context_chunk_ids": [c["chunk_id"] for c in candidates if c["text"]],
            "context_metadata": [c["metadata"] for c in candidates if c["text"]],
            "context_scores": [c["score"] for c in candidates if c["text"]],
            "gold_chunk_ids": q["_gold"],
            "gold_passage": norm(q.get("gold_passage")) or None,
            "source": q.get("source"),
            "source_url": q.get("source_url"),
            "source_location": q.get("source_location"),
            "ground_truth": reference_answer,
            "answer": None,
            "answer_status": "not_generated",
            "answer_note": (
                "Retrieval-only run: answer intentionally null. Do not calculate "
                "Faithfulness or Answer Relevancy until a real generated answer is added. "
                "gold_passage is supporting evidence, not automatically a reference answer."
            ),
            "retrieval_metrics": {k: row.get(k) for k in METRICS + ["first_relevant_rank", "gold_count", "gold_found", "gold_ranks"]},
        }
        append_jsonl(out_ragas, ragas_record)
        rows_to_save = {**meta, "elapsed_seconds": round(time.perf_counter() - start_all, 2),
                        "rrf_pool_gold_found_questions": pool_gold_found, "status": "running"}
        save_checkpoint(rows, out_csv, out_json, rows_to_save)
        logging.info("METHOD_DONE qid=%s method=%s ms=%.3f gold_rank=%s candidates=%s", q["_id"], method, elapsed, row.get("first_relevant_rank"), len(results))
        print_row(row)
        return row

    for qi, q in enumerate(questions, 1):
        print("\n" + "-" * 145, flush=True)
        print(f"SORU {qi:02d}/{len(questions):02d} | {q['_id']} | {q['_question']}", flush=True)
        print(f"GOLD: {';'.join(q['_gold']) if q['_gold'] else 'YOK'} | KATEGORİ: {q.get('category', '')}", flush=True)
        t = time.perf_counter()
        try:
            processed = process_query(q["_question"])
        except Exception as exc:
            logging.exception("QUERY_PROCESSING_FAILED qid=%s", q["_id"])
            append_jsonl(out_debug, {"run_id": debug_run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "error", "stage": "query_processing", "question_id": q["_id"], "question": q["_question"], "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
            raise
        query_ms = (time.perf_counter() - t) * 1000
        rewritten = norm(processed.get("rewritten_query")) or q["_question"]
        expanded = norm(processed.get("expanded_query")) or q["_question"]
        print(f"Sorgu işleme: {query_ms:.0f} ms | rewritten={rewritten} | expanded={expanded}", flush=True)
        append_jsonl(out_debug, {"run_id": debug_run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": "query_processed", "question_id": q["_id"], "question": q["_question"], "raw_question_record": jsonable(q), "rewritten_query": rewritten, "expanded_query": expanded, "query_processing_ms": round(query_ms, 3), "process_query_raw": jsonable(processed)})

        t = time.perf_counter()
        bm25_results = bm25.search(expanded, n_results=BM25_TOP_N, apply_reference_penalty=False)
        record(q, "BM25-only", bm25_results, (time.perf_counter()-t)*1000, query_ms, rewritten, expanded)

        t = time.perf_counter()
        dense_results = hybrid._dense_retrieve(rewritten, top_n=DENSE_TOP_N)
        record(q, "Dense-only", dense_results, (time.perf_counter()-t)*1000, query_ms, rewritten, expanded)

        t = time.perf_counter()
        rrf_results = hybrid._rrf_fusion(dense_results, bm25_results, apply_quality_penalty=False)[:RRF_CANDIDATE_K]
        rrf_ms = (time.perf_counter()-t)*1000
        rrf_row = record(q, "Dense+BM25 RRF", rrf_results, rrf_ms, query_ms, rewritten, expanded)
        if rrf_row["gold_count"] and rrf_row["first_relevant_rank"] is not None:
            pool_gold_found += 1

        t = time.perf_counter()
        bge_results = bge.rerank(rewritten, rrf_results[:RERANK_CANDIDATE_K], top_k=RERANK_FINAL_K)
        record(q, "RRF+BGE-Reranker Top-20", bge_results, (time.perf_counter()-t)*1000,
               query_ms, rewritten, expanded, {"reranker_model": RERANKER_MODEL})


        # Persist per-question checkpoint and a compact diagnostic.
        save_checkpoint(rows, out_csv, out_json, {**meta, "elapsed_seconds": round(time.perf_counter()-start_all, 2),
                                                  "rrf_pool_gold_found_questions": pool_gold_found})
        if q["_gold"] and rrf_row["first_relevant_rank"] is None:
            print(f"  TANI | SORU={q['_id']} | Gold chunk RRF Top-30 havuzunda yok; reranker bunu geri getiremez.", flush=True)

    total = time.perf_counter() - start_all
    summary = aggregate(rows)
    final_meta = {**meta, "elapsed_seconds": round(total, 2), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "rrf_pool_gold_found_questions": pool_gold_found, "summary": summary, "status": "completed",
                  "output_sizes_bytes": {p.name: p.stat().st_size for p in outputs if p.exists()}}
    save_checkpoint(rows, out_csv, out_json, final_meta)
    out_manifest.write_text(json.dumps(jsonable(final_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("RUN_COMPLETED elapsed_seconds=%.3f", total)
    print("\n" + "=" * 110)
    print("GENEL ORTALAMA SONUÇLARI (gold içeren sorular)")
    print(f"{'Yöntem':<30}" + "".join(f"{m:>10}" for m in ["Hit@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]) + f"{'Ort.ms':>12}")
    for method in METHODS:
        s = summary.get(method)
        if not s:
            continue
        print(f"{method:<30}" + "".join(f"{s[m]:>10.4f}" for m in ["Hit@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]) + f"{s['avg_method_ms']:>10.2f}")
    print(f"\nRRF Top-30 içinde gold bulunan soru: {pool_gold_found}/{len(gold_questions)} gold içeren soru")
    print(f"Toplam çalışma süresi: {total:.1f} sn")
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")
    print(f"Debug JSONL (tam aday metni/metadata/score): {out_debug}")
    print(f"RAGAS retrieval dataset (answer=null): {out_ragas}")
    print(f"Run manifest: {out_manifest}")
    print(f"Debug log: {out_log}")
    print("Not: yöntem süresi ve query processing süresi ayrı raporlanır. RRF süresi yalnızca füzyon süresidir.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDurduruldu. Son checkpoint sonuç dosyalarında tutuldu.")
        sys.exit(130)
    except Exception as exc:
        try:
            logging.exception("FATAL_BENCHMARK_ERROR")
        except Exception:
            pass
        print(f"\nBENCHMARK HATASI: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
