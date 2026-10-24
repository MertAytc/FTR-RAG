"""
FTR-RAG RAGAS preparation and evaluation helper.

Input:
  - retrieval *_ragas_retrieval_dataset.jsonl produced by the retrieval benchmark
  - optional answers JSON/JSONL with question_id, method, answer, and optionally ground_truth

This script never fabricates answers or reference answers. It exports a validated
dataset even when RAGAS cannot yet run because answers/references are missing.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows

def read_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("answers", "results", "items", "questions"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError("JSON should contain an answers/results/items/questions list.")
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported answer file format.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="retrieval RAGAS JSONL")
    ap.add_argument("--answers", type=Path, default=None, help="optional JSON/JSONL with question_id, method, answer, ground_truth")
    ap.add_argument("--output-prefix", default="ftr_rag_ragas_eval")
    ap.add_argument("--run-ragas", action="store_true", help="run RAGAS if package and valid fields are available")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("Input JSONL is empty.")

    answer_map = {}
    if args.answers:
        for a in read_records(args.answers):
            qid = str(a.get("question_id") or a.get("id") or "").strip()
            method = str(a.get("method") or "").strip()
            if not qid:
                continue
            answer_map[(qid, method)] = a
            # methodless answer can be used for all methods only when explicitly provided
            if not method:
                answer_map[(qid, "*")] = a

    merged = []
    for r in rows:
        a = answer_map.get((str(r.get("question_id")), str(r.get("method"))), answer_map.get((str(r.get("question_id")), "*"), {}))
        item = dict(r)
        if a.get("answer") is not None:
            item["answer"] = str(a["answer"])
            item["answer_status"] = "provided"
        if a.get("ground_truth") is not None:
            item["ground_truth"] = str(a["ground_truth"])
        merged.append(item)

    prefix = Path(args.output_prefix)
    out_jsonl = prefix.with_name(prefix.name + "_validated.jsonl")
    out_csv = prefix.with_name(prefix.name + "_readiness.csv")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    readiness = []
    for r in merged:
        answer = r.get("answer")
        gt = r.get("ground_truth")
        contexts = r.get("contexts") or []
        readiness.append({
            "question_id": r.get("question_id"),
            "method": r.get("method"),
            "context_count": len(contexts),
            "gold_chunk_count": len(r.get("gold_chunk_ids") or []),
            "has_ground_truth": bool(gt),
            "has_answer": bool(answer),
            "can_attempt_faithfulness": bool(answer and contexts),
            "can_attempt_answer_relevancy": bool(answer and r.get("question")),
            "can_attempt_context_precision_recall": bool(gt and contexts),
            "answerability": r.get("answerability"),
            "expected_behavior": r.get("expected_behavior"),
        })
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(readiness[0].keys()))
        writer.writeheader()
        writer.writerows(readiness)

    print(f"Okunan kayıt: {len(merged)}")
    print(f"Doğrulanmış JSONL: {out_jsonl.resolve()}")
    print(f"Hazırlık raporu: {out_csv.resolve()}")
    print("Not: Bu adım eksik cevapları veya referansları üretmez; yalnızca hazır olma durumunu denetler.")

    if args.run_ragas:
        ready = [r for r in merged if r.get("answer") and r.get("question") and r.get("contexts")]
        if not ready:
            print("RAGAS çalıştırılmadı: gerçek answer + question + contexts bulunan kayıt yok.")
            return
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        except Exception as e:
            raise RuntimeError("RAGAS/datasets import edilemedi. Kurulu RAGAS sürümüne göre API uyarlaması gerekebilir.") from e

        # RAGAS legacy evaluate API; versions differ, so fail transparently if incompatible.
        metric_rows = []
        for r in ready:
            base = {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
            if r.get("ground_truth"):
                base["ground_truth"] = r["ground_truth"]
            metric_rows.append(base)
        dataset = Dataset.from_list(metric_rows)
        selected = [faithfulness, answer_relevancy]
        if any(x.get("ground_truth") for x in ready):
            selected.extend([context_precision, context_recall])
        result = evaluate(dataset, metrics=selected)
        out_metrics = prefix.with_name(prefix.name + "_ragas_scores.csv")
        result.to_pandas().to_csv(out_metrics, index=False, encoding="utf-8-sig")
        print(f"RAGAS sonuçları: {out_metrics.resolve()}")

if __name__ == "__main__":
    main()
