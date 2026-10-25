#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FTR-RAG | RRF+BGE-Reranker Top-20 | RAGAS 8-metric evaluation

Inputs:
  1) ftr_rag_50_soru_gold_matched_reviewed.json
  2) generated_answers_checkpoint.jsonl

Optional:
  --contexts path/to/retrieval_contexts.jsonl
  Context JSONL rows should contain question_id and one of:
    retrieval_contexts: ["context 1", "context 2", ...]
    contexts: ["context 1", "context 2", ...]
    retrieved_chunks: [{"text": "...", "chunk_id": "..."}, ...]

Important:
- Uses only method "RRF+BGE-Reranker Top-20".
- Uses reviewed gold_passage as reference, not as retrieved context.
- Questions marked no_gold_in_corpus_expected have no valid gold passage; answer-only
  relevance/correctness-style scores are still attempted, but reference-dependent scores
  are marked unavailable for those rows.
- Context-based metrics are NOT computed unless real retrieved contexts are supplied.
- RAGAS APIs differ by version. This script pins a known compatible 0.2.x API.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

TARGET_METHOD = "RRF+BGE-Reranker Top-20"
NO_GOLD_STATUSES = {"no_gold_in_corpus_expected", "no_verified_chunk_match_manual_review"}
PARTIAL_STATUSES = {"partial_match_manual_review"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ftr_ragas")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("JSONL satırı atlandı (%s:%d): %s", path, line_no, exc)
    return rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_id(value: Any) -> str:
    return clean_text(value).upper()


def find_gold_reference(gold: Dict[str, Any]) -> str:
    passage = clean_text(gold.get("gold_passage"))
    status = clean_text(gold.get("gold_chunk_match_status"))
    if not passage or passage.lower().startswith("gold passage yok:"):
        return ""
    if status in NO_GOLD_STATUSES:
        return ""
    return passage


def load_contexts(path: Optional[str]) -> Dict[str, List[str]]:
    """Load actual retrieved contexts, if supplied."""
    result: Dict[str, List[str]] = {}
    if not path:
        return result
    for row in load_jsonl(path):
        qid = normalize_id(row.get("question_id") or row.get("id"))
        raw = row.get("retrieval_contexts", row.get("contexts", row.get("retrieved_chunks", [])))
        if isinstance(raw, str):
            raw = [raw]
        contexts = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    txt = clean_text(item.get("text") or item.get("content") or item.get("page_content"))
                else:
                    txt = clean_text(item)
                if txt:
                    contexts.append(txt)
        if qid:
            result[qid] = contexts
    return result


def select_records(answer_path: str, gold_path: str, context_path: Optional[str]):
    gold_data = load_json(gold_path)
    if isinstance(gold_data, dict):
        gold_data = gold_data.get("questions", gold_data.get("data", []))
    if not isinstance(gold_data, list):
        raise ValueError("Gold JSON bir liste veya questions/data içeren nesne olmalı.")

    gold_by_id = {normalize_id(g.get("id") or g.get("question_id")): g for g in gold_data}
    answer_rows = load_jsonl(answer_path)
    target_rows = [
        r for r in answer_rows
        if clean_text(r.get("method")) == TARGET_METHOD
    ]
    if not target_rows:
        methods = sorted({clean_text(r.get("method")) for r in answer_rows})
        raise ValueError(
            f"'{TARGET_METHOD}' yöntemi bulunamadı. Dosyadaki method değerleri: {methods}"
        )

    # Keep the latest row per question_id in case the checkpoint contains duplicate records.
    latest_by_id = {}
    for row in target_rows:
        qid = normalize_id(row.get("question_id") or row.get("id"))
        if qid:
            latest_by_id[qid] = row

    contexts_by_id = load_contexts(context_path)
    merged = []
    missing_gold = []
    for qid, ans in sorted(latest_by_id.items()):
        gold = gold_by_id.get(qid)
        if not gold:
            missing_gold.append(qid)
            continue
        answer = clean_text(ans.get("answer"))
        status = clean_text(ans.get("answer_status")).lower()
        error = clean_text(ans.get("generation_error"))
        ref = find_gold_reference(gold)
        question = clean_text(gold.get("question") or ans.get("question"))
        contexts = contexts_by_id.get(qid, [])
        if not contexts:
            # Deliberately do not use gold_passage as retrieved context.
            contexts = []
        merged.append({
            "question_id": qid,
            "user_input": question,
            "response": answer,
            "reference": ref,
            "retrieved_contexts": contexts,
            "gold_chunk_ids": gold.get("gold_chunk_ids", []),
            "gold_chunk_match_status": clean_text(gold.get("gold_chunk_match_status")),
            "answer_status": status,
            "generation_error": error,
            "expected_behavior": clean_text(ans.get("expected_behavior") or "Answer"),
            "generation_model": clean_text(ans.get("generation_model")),
            "method": clean_text(ans.get("method")),
        })
    if missing_gold:
        log.warning("Gold dosyasında eşleşmeyen ID'ler atlandı: %s", ", ".join(missing_gold))
    return merged


def build_ragas_dataset(rows: List[Dict[str, Any]], use_contexts: bool):
    from datasets import Dataset
    data = {
        "question_id": [],
        "user_input": [],
        "response": [],
        "reference": [],
        "retrieved_contexts": [],
    }
    for r in rows:
        data["question_id"].append(r["question_id"])
        data["user_input"].append(r["user_input"])
        data["response"].append(r["response"])
        data["reference"].append(r["reference"])
        data["retrieved_contexts"].append(r["retrieved_contexts"] if use_contexts else [])
    return Dataset.from_dict(data)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RRF+BGE-Reranker Top-20 answers with 8 RAGAS metrics."
    )
    parser.add_argument("--answers", default="generated_answers_checkpoint.jsonl",
                        help="generated_answers_checkpoint.jsonl yolu")
    parser.add_argument("--gold", default="ftr_rag_50_soru_gold_matched_reviewed.json",
                        help="Gold JSON yolu")
    parser.add_argument("--contexts", default=None,
                        help="Gerçek RRF+reranker retrieval contextlerini içeren JSONL (opsiyonel)")
    parser.add_argument("--out", default="ragas_rrf_reranker_results",
                        help="Çıktı dosyalarının temel adı")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        help="Ollama URL")
    parser.add_argument("--llm-model", default=os.getenv("RAGAS_LLM_MODEL", "qwen3:8b"),
                        help="RAGAS değerlendirici LLM modeli (Ollama)")
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
                        help="Answer relevancy/semantic metrics için Ollama embedding modeli")
    parser.add_argument("--timeout", type=int, default=240,
                        help="LLM istek timeout saniyesi")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="RAGAS concurrency; Ollama için 1-2 önerilir")
    parser.add_argument("--retries", type=int, default=2,
                        help="Hata sonrası yeniden deneme sayısı")
    parser.add_argument("--limit", type=int, default=0,
                        help="Hızlı deneme için soru sayısı; 0=tümü")
    parser.add_argument("--skip-context-metrics", action="store_true",
                        help="Context metriklerini her durumda atla")
    args = parser.parse_args()

    rows = select_records(args.answers, args.gold, args.contexts)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("Değerlendirilecek eşleşmiş kayıt bulunamadı.")

    generated = [r for r in rows if r["response"] and r["answer_status"] not in {"failed", "error"}]
    if not generated:
        raise SystemExit("Hedef yöntemde değerlendirilebilir cevap bulunamadı.")
    rows = generated

    context_available = bool(args.contexts) and any(r["retrieved_contexts"] for r in rows)
    run_context_metrics = context_available and not args.skip_context_metrics
    log.info("Hedef yöntem: %s", TARGET_METHOD)
    log.info("Eşleşen gold/cevap: %d", len(rows))
    log.info("Referans pasajı bulunan soru: %d", sum(bool(r["reference"]) for r in rows))
    log.info("Gerçek retrieval contexti mevcut: %s", "EVET" if context_available else "HAYIR")
    if not context_available:
        log.warning(
            "Context dosyası verilmedi. Faithfulness/context_precision/context_recall/"
            "context_entity_recall/noise_sensitivity hesaplanmayacak; gold passage retrieval context "
            "yerine kullanılmayacak."
        )

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            answer_correctness,
            answer_similarity,
            faithfulness,
            context_precision,
            context_recall,
            context_entity_recall,
        )
        # RAGAS 0.2.15 exposes this metric as a class in its internal module.
        from ragas.metrics._noise_sensitivity import NoiseSensitivity
        noise_sensitivity = NoiseSensitivity()
    except ImportError as exc:
        raise SystemExit(
            "Bağımlılık eksik veya RAGAS API uyumsuz. Kurulum için:\n"
            "python -m pip install -U \"ragas==0.2.15\" \"langchain-ollama>=0.2.0\" "
            "\"datasets>=2.18,<4\" \"pandas>=2.0\" \"openpyxl>=3.1\"\n"
            f"İçe aktarma hatası: {exc}"
        )

    # Ollama is local; ensure model names match 'ollama list'.
    llm = ChatOllama(
        model=args.llm_model,
        base_url=args.ollama_base_url,
        temperature=0,
        timeout=args.timeout,
        num_predict=1024,
    )
    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
    )

    # Eight metric suite. Context-dependent metrics are included only with real contexts.
    metric_list = [answer_relevancy, answer_correctness, answer_similarity]
    if run_context_metrics:
        metric_list += [
            faithfulness,
            context_precision,
            context_recall,
            context_entity_recall,
            noise_sensitivity,
        ]
    else:
        # To avoid misleading results, context-dependent metrics are not run.
        pass

    dataset = build_ragas_dataset(rows, use_contexts=run_context_metrics)

    # Compatibility: older RAGAS versions accept llm/embeddings on evaluate;
    # the pinned 0.2.15 API uses this interface.
    kwargs = {
        "dataset": dataset,
        "metrics": metric_list,
        "llm": llm,
        "embeddings": embeddings,
        "raise_exceptions": False,
    }
    try:
        result = evaluate(**kwargs)
    except TypeError as exc:
        raise SystemExit(
            "RAGAS evaluate API beklenen sürümle uyuşmuyor. Şunu çalıştır:\n"
            "python -m pip install --upgrade --force-reinstall \"ragas==0.2.15\"\n"
            f"Detay: {exc}"
        )

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_pandas()

    # Attach stable IDs and audit metadata by dataset order.
    # RAGAS output may contain input columns; retain them and ensure question_id is present.
    if "question_id" not in df.columns:
        df.insert(0, "question_id", [r["question_id"] for r in rows[:len(df)]])
    df["method"] = TARGET_METHOD
    df["evaluator_llm"] = args.llm_model
    df["reference_available"] = [bool(r["reference"]) for r in rows[:len(df)]]
    df["gold_chunk_match_status"] = [r["gold_chunk_match_status"] for r in rows[:len(df)]]
    df["retrieval_context_available"] = [bool(r["retrieved_contexts"]) for r in rows[:len(df)]]

    csv_path = str(out_base) + ".csv"
    json_path = str(out_base) + ".json"
    summary_path = str(out_base) + "_summary.json"
    xlsx_path = str(out_base) + ".xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    metric_cols = []
    for name in [
        "answer_relevancy", "answer_correctness", "answer_similarity",
        "faithfulness", "context_precision", "context_recall",
        "context_entity_recall", "noise_sensitivity",
    ]:
        if name in df.columns:
            metric_cols.append(name)

    summary = {
        "target_method": TARGET_METHOD,
        "total_evaluated": len(rows),
        "answers_file": str(Path(args.answers).resolve()),
        "gold_file": str(Path(args.gold).resolve()),
        "contexts_file": str(Path(args.contexts).resolve()) if args.contexts else None,
        "evaluator_llm": args.llm_model,
        "embedding_model": args.embedding_model,
        "ragas_version_expected": "0.2.15",
        "context_metrics_computed": run_context_metrics,
        "metric_means": {},
        "metric_non_null_counts": {},
        "notes": [
            "Gold passage reference olarak kullanılır; retrieved context yerine geçirilmez.",
            "No-gold sorularında reference-dependent scores null/NaN olabilir.",
            "Bu script model-based evaluation üretir; klinik doğrulama veya uzman değerlendirmesinin yerine geçmez.",
        ],
    }
    for col in metric_cols:
        numeric = __import__("pandas").to_numeric(df[col], errors="coerce")
        summary["metric_means"][col] = None if numeric.notna().sum() == 0 else float(numeric.mean())
        summary["metric_non_null_counts"][col] = int(numeric.notna().sum())

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        log.warning("Excel yazılamadı (CSV/JSON kaydedildi): %s", exc)
        xlsx_path = None

    print("\n" + "=" * 72)
    print("RAGAS DEĞERLENDİRME TAMAMLANDI")
    print("=" * 72)
    print(f"Yöntem             : {TARGET_METHOD}")
    print(f"Değerlendirilen     : {len(rows)}")
    print(f"Context metrikleri : {'Hesaplandı' if run_context_metrics else 'HESAPLANMADI (retrieval context yok)'}")
    print("\nMetrik ortalamaları:")
    for name, value in summary["metric_means"].items():
        print(f"  {name:25s}: {'N/A' if value is None else f'{value:.4f}'} "
              f"(n={summary['metric_non_null_counts'][name]})")
    print("\nÇıktılar:")
    print(f"  CSV   : {csv_path}")
    print(f"  JSON  : {json_path}")
    print(f"  Özet  : {summary_path}")
    if xlsx_path:
        print(f"  Excel : {xlsx_path}")
    print("\nNot: Sonuçları yorumlarken no-gold ve partial-match kayıtlarını ayrıca incele.")


if __name__ == "__main__":
    main()
