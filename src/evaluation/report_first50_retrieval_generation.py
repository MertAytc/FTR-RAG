"""Build an auditable first-50 retrieval report from the frozen-v3 saved run.

This script does not call the retriever, reranker, or generation model. It joins
the saved source-corrected rankings to the frozen benchmark and source chunks,
computes strict/acceptable/partial metrics, and prepares exact top-5 generation
contexts for a later Qwen run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/retrieval_100_goldchunks_v3"
FROZEN = BASE / "frozen/FTR_RAG_100_Benchmark_Questions_GoldChunks_v3.json"
RUN = BASE / "source_corrected_20261028"
RETRIEVAL = RUN / "retrieval_100_report.json"
RERANKER = RUN / "retrieval_100_reranker_report.json"
METADATA = RUN / "metadata_retrieval_experiment/metadata_retrieval_report.json"
CHUNKS = ROOT / "data/processed/chunks_source_corrected_20261028"
OUT = ROOT / "data/processed/retrieval_first50_generation_20261029"
EXPECTED_HASH = "7ba106ea22b376658e498210532d1cb14037fbcc750eafd82e207f73ba024c3b"
SEED = 20261029
BOOTSTRAP_REPS = 10_000
CUTOFFS = (1, 3, 5, 10, 20)
STAGES = ("dense", "bm25", "rrf", "reranker")
VARIANTS = ("strict_canonical", "acceptable_full", "acceptable_plus_partial")
OUT_OF_SCOPE_IDS = {f"FTR-{i:03d}" for i in (44, 45, 46, 47, 48)}
STAGE_FIELDS = {
    "dense": ("dense_top20", "dense_top20"),
    "bm25": ("bm25_top20", "bm25_top20"),
    "rrf": ("rrf_top20", "rrf_top20"),
    "reranker": ("reranker_top20", "reranker_top20"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_ids(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def groups_for(question: Dict[str, Any]) -> Dict[str, List[Set[str]]]:
    gold = set(clean_ids(question.get("gold_chunk_ids", [])))
    full = [set(clean_ids(group)) for group in question.get("relevance_groups", [])]
    if not full and gold:
        full = [{cid} for cid in sorted(gold)]
    strict = [group & gold for group in full]
    if any(not group for group in strict):
        raise ValueError(f"Gold IDs do not cover every relevance group: {question['id']}")
    partial = [set(group) for group in full]
    partial_ids: Set[str] = set()
    for item in question.get("partial_relevance_groups", []):
        if item.get("decision") != "partial":
            continue
        ids = set(clean_ids(item.get("chunk_ids", [])))
        partial_ids.update(ids)
        indexes = item.get("support_group_indexes") or item.get("partial_group_indexes") or []
        for index in indexes:
            if isinstance(index, int) and 0 <= index < len(partial):
                partial[index].update(ids)
    if partial_ids and not partial:
        partial = [set(partial_ids)]
    elif partial_ids and not any(partial_ids & group for group in partial):
        partial[0].update(partial_ids)
    return {
        "strict_canonical": strict,
        "acceptable_full": full,
        "acceptable_plus_partial": partial,
    }


def metric_score(ranked: Sequence[str], groups: Sequence[Set[str]], gains: Dict[str, int]) -> Dict[str, Any]:
    relevant = set().union(*groups) if groups else set()
    first = next((rank for rank, cid in enumerate(ranked, 1) if cid in relevant), None)
    result: Dict[str, Any] = {
        "support_groups": len(groups),
        "first_relevant_rank": first,
        "mrr@20": 1.0 / first if first and first <= 20 else 0.0,
    }
    ideal_gains = sorted(gains.values(), reverse=True)
    ideal = sum(value / math.log2(rank + 2) for rank, value in enumerate(ideal_gains[:10]))
    actual = sum(gains.get(cid, 0) / math.log2(rank + 2) for rank, cid in enumerate(ranked[:10]))
    result["ndcg@10"] = actual / ideal if ideal else 0.0
    for k in CUTOFFS:
        selected = set(ranked[:k])
        matched = sum(bool(group & selected) for group in groups)
        result[f"hit@{k}"] = float(bool(relevant & selected))
        result[f"matched_groups@{k}"] = matched
        result[f"recall@{k}"] = matched / len(groups) if groups else 0.0
        result[f"full_hit@{k}"] = float(bool(groups) and matched == len(groups))
    return result


def bootstrap(values: Sequence[float], seed_offset: int = 0) -> List[float]:
    if not values:
        return [0.0, 0.0]
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.integers(0, len(arr), size=(BOOTSTRAP_REPS, len(arr)))
    means = np.sort(arr[indices].mean(axis=1))
    return [float(means[int(0.025 * BOOTSTRAP_REPS)]), float(means[min(BOOTSTRAP_REPS - 1, int(0.975 * BOOTSTRAP_REPS))])]


def load_chunks() -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    files = sorted(CHUNKS.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No source-corrected chunk JSON files found in {CHUNKS}")
    for path in files:
        if path.name == "_quality_report.json":
            continue
        raw = read_json(path)
        items = raw.get("chunks", []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        for chunk in items:
            if not isinstance(chunk, dict):
                continue
            cid = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
            if cid:
                found.setdefault(cid, chunk)
    return found


def ranking_details(items: Sequence[Dict[str, Any]], chunk_map: Dict[str, Dict[str, Any]], full_text: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for fallback_rank, item in enumerate(items, 1):
        cid = str(item.get("chunk_id") or item.get("id") or "").strip()
        chunk = chunk_map.get(cid, {})
        metadata = chunk.get("metadata") or {}
        rows.append({
            "rank": item.get("rank", fallback_rank),
            "chunk_id": cid,
            "score": item.get("score", item.get("rrf_score", item.get("reranker_score"))),
            "rrf_rank": item.get("rrf_rank"),
            "reranker_score": item.get("reranker_score"),
            "doc_name": item.get("doc_name") or metadata.get("doc_name") or metadata.get("source_file"),
            "section_path": item.get("section_path") or metadata.get("section_path") or metadata.get("section_title"),
            "page_number": item.get("page_number", item.get("page", metadata.get("page_number", metadata.get("page")))),
            "text": str(chunk.get("text") or chunk.get("document") or "") if full_text else None,
            "excerpt": item.get("excerpt") or " ".join(str(chunk.get("text") or "").split())[:500],
            "in_source_chunk_map": bool(chunk),
        })
    return rows


def build() -> Dict[str, Any]:
    actual_hash = sha256(FROZEN)
    if actual_hash != EXPECTED_HASH:
        raise ValueError(f"Frozen v3 hash mismatch: expected {EXPECTED_HASH}, got {actual_hash}")
    questions = read_json(FROZEN)
    if len(questions) < 50:
        raise ValueError(f"Expected at least 50 frozen questions, found {len(questions)}")
    questions = questions[:50]
    retrieval_report = read_json(RETRIEVAL)
    reranker_report = read_json(RERANKER)
    metadata_report = read_json(METADATA)
    if retrieval_report.get("completed_count") != retrieval_report.get("question_count"):
        raise ValueError("Saved retrieval report is incomplete")
    if reranker_report.get("completed_count") != reranker_report.get("question_count"):
        raise ValueError("Saved reranker report is incomplete")
    retrieval_by_id = {r["question_id"]: r for r in retrieval_report["per_question"]}
    reranker_by_id = {r["question_id"]: r for r in reranker_report["per_question"]}
    metadata_by_id = {r["question_id"]: r for r in metadata_report.get("per_question", [])}
    chunk_map = load_chunks()

    out_rows: List[Dict[str, Any]] = []
    for q in questions:
        qid = q["id"]
        if qid not in retrieval_by_id or qid not in reranker_by_id:
            raise KeyError(f"Saved retrieval results are missing {qid}")
        ret = retrieval_by_id[qid]
        rer = reranker_by_id[qid]
        is_out_of_scope = qid in OUT_OF_SCOPE_IDS
        groups = groups_for(q) if q.get("evaluation_track") in {"retrieval", "excluded_invalid_gold"} else {v: [] for v in VARIANTS}
        metric_eligible = {
            "strict_canonical": not is_out_of_scope and q.get("evaluation_track") == "retrieval" and bool(q.get("gold_chunk_ids")) and bool(groups["strict_canonical"]),
            "acceptable_full": not is_out_of_scope and q.get("evaluation_track") == "retrieval" and bool(q.get("gold_chunk_ids")) and bool(groups["acceptable_full"]),
            "acceptable_plus_partial": not is_out_of_scope and bool(groups["acceptable_plus_partial"]),
        }
        available = any(metric_eligible.values())
        stages: Dict[str, Any] = {}
        raw_rankings = {
            "dense": ret.get("dense_top20", []),
            "bm25": ret.get("bm25_top20", []),
            "rrf": ret.get("rrf_top20", []),
            "reranker": rer.get("reranker_top20", []),
        }
        details = {stage: ranking_details(items, chunk_map) for stage, items in raw_rankings.items()}
        top5 = ranking_details(rer.get("reranker_top20", [])[:5], chunk_map, full_text=True)
        if available:
            gold = set(clean_ids(q.get("gold_chunk_ids", [])))
            acceptable = set().union(*groups["acceptable_full"]) if groups["acceptable_full"] else set()
            partial_all = set().union(*groups["acceptable_plus_partial"]) if groups["acceptable_plus_partial"] else set()
            for stage, items in raw_rankings.items():
                ranked = [x["chunk_id"] for x in details[stage]]
                stages[stage] = {}
                for variant in VARIANTS:
                    if not metric_eligible[variant]:
                        continue
                    if variant == "strict_canonical":
                        gains = {cid: 1 for cid in gold}
                    elif variant == "acceptable_full":
                        gains = {cid: 1 for cid in acceptable}
                    else:
                        gains = {cid: (2 if cid in acceptable else 1) for cid in partial_all}
                    stages[stage][variant] = metric_score(ranked, groups[variant], gains)
        metadata_row = metadata_by_id.get(qid, {})
        metadata_candidates = metadata_row.get("metadata_top10", []) if q.get("evaluation_track") == "metadata_attribution" else []
        out_rows.append({
            "question_id": qid,
            "evaluation_track": q.get("evaluation_track"),
            "question": q.get("question"),
            "gold_answer": q.get("gold_answer"),
            "gold_chunk_ids": clean_ids(q.get("gold_chunk_ids", [])),
            "relevance_available": available,
            "metric_eligible": metric_eligible,
            "out_of_scope": is_out_of_scope,
            "support_groups": {k: [sorted(g) for g in v] for k, v in groups.items()},
            "metrics": stages,
            "rankings": details,
            "generation_top5": top5,
            "metadata_candidates": metadata_candidates,
            "generation_status": "excluded_out_of_scope" if is_out_of_scope else "pending_generation",
            "generation_response": None,
            "ttft_seconds": None,
            "total_generation_seconds": None,
            "generation_context_chunk_ids": [],
            "retrieved_contexts": [],
            "context_sources": [],
        })

    aggregates: Dict[str, Any] = {}
    for stage_index, stage in enumerate(STAGES):
        aggregates[stage] = {}
        for variant_index, variant in enumerate(VARIANTS):
            eligible_rows = [r for r in out_rows if r["metric_eligible"][variant]]
            records = [r["metrics"][stage][variant] for r in eligible_rows]
            count = len(records)
            metric_names = [f"hit@{k}" for k in CUTOFFS] + [f"recall@{k}" for k in CUTOFFS] + [f"full_hit@{k}" for k in CUTOFFS] + ["mrr@20", "ndcg@10"]
            summary = {name: {
                "mean": sum(float(x[name]) for x in records) / count if count else 0.0,
                "ci95": bootstrap([float(x[name]) for x in records], stage_index * 100 + variant_index * 20 + i),
            } for i, name in enumerate(metric_names)}
            for k in CUTOFFS:
                matched = sum(int(x[f"matched_groups@{k}"]) for x in records)
                group_total = sum(int(x["support_groups"]) for x in records)
                summary[f"micro_recall@{k}"] = {
                    "mean": matched / group_total if group_total else 0.0,
                    "matched_groups": matched,
                    "total_groups": group_total,
                }
            aggregates[stage][variant] = {"n": count, "metrics": summary}

    result = {
        "report_title": "First 50 questions: retrieval and Qwen generation",
        "cohort": {"n": 50, "question_ids": [q["id"] for q in questions]},
        "evaluation_cohort": {"n": 45, "out_of_scope_excluded": sorted(OUT_OF_SCOPE_IDS), "question_ids": [q["id"] for q in questions if q["id"] not in OUT_OF_SCOPE_IDS]},
        "benchmark_sha256": actual_hash,
        "retrieval_source": str(RETRIEVAL.relative_to(ROOT)),
        "reranker_source": str(RERANKER.relative_to(ROOT)),
        "retrieval_was_rerun": False,
        "retrieval_config": retrieval_report.get("config"),
        "reranker_model": reranker_report.get("reranker_model"),
        "generation_model_requested": "qwen3:8b",
        "generation_status": "pending_generation",
        "track_counts": {track: sum(r["evaluation_track"] == track for r in out_rows) for track in sorted({r["evaluation_track"] for r in out_rows})},
        "retrieval_scored_question_ids": [r["question_id"] for r in out_rows if r["metric_eligible"]["strict_canonical"]],
        "metric_eligible_question_ids": {v: [r["question_id"] for r in out_rows if r["metric_eligible"][v]] for v in VARIANTS},
        "aggregates": aggregates,
        "per_question": out_rows,
        "limitations": [
            "Retrieval rankings are the saved source-corrected run; no new retrieval/reranker calls were made.",
            "FTR-044–FTR-048 are excluded as the requested five out-of-scope questions.",
            "Metadata-attribution questions have no chunk gold IDs and are excluded from chunk-retrieval metrics; generation uses document metadata cards.",
            "FTR-020 has no valid canonical gold. It is excluded from strict and acceptable metrics, but its one partial judgment is included only in the acceptable+partial upper-bound analysis.",
            "Acceptable alternatives are the frozen-v3 full relevance groups. Partial is an upper-bound sensitivity analysis.",
            "Generation uses Qwen3:8b and the reranker top-5; TTFT is measured from the Ollama request start to the first non-empty streamed answer token.",
        ],
    }
    return result


def write_outputs(report: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_first50_generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Per-question metric table plus ranked IDs for all four stages.
    csv_path = OUT / "retrieval_first50_per_question.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        metric_names = ([f"hit@{k}" for k in CUTOFFS]
                        + [f"recall@{k}" for k in CUTOFFS]
                        + [f"full_hit@{k}" for k in CUTOFFS]
                        + ["mrr@20", "ndcg@10"])
        columns = ["question_id", "evaluation_track", "question", "relevance_available", "out_of_scope", "gold_chunk_ids", "dense_top20_ids", "bm25_top20_ids", "rrf_top20_ids", "reranker_top20_ids", "generation_top5_ids", "generation_context_chunk_ids", "generation_status", "ttft_seconds", "total_generation_seconds", "generation_response"]
        writer = csv.DictWriter(handle, fieldnames=columns + [f"{s}.{v}.{m}" for s in STAGES for v in VARIANTS for m in metric_names])
        writer.writeheader()
        for row in report["per_question"]:
            flat: Dict[str, Any] = {
                "question_id": row["question_id"], "evaluation_track": row["evaluation_track"], "question": row["question"],
                "relevance_available": row["relevance_available"], "gold_chunk_ids": ";".join(row["gold_chunk_ids"]),
                "generation_top5_ids": ";".join(x["chunk_id"] for x in row["generation_top5"]),
                "out_of_scope": row["out_of_scope"],
                "generation_context_chunk_ids": ";".join(row["generation_context_chunk_ids"]),
                "generation_status": row["generation_status"],
                "ttft_seconds": row["ttft_seconds"],
                "total_generation_seconds": row["total_generation_seconds"],
                "generation_response": row["generation_response"],
            }
            for stage in STAGES:
                flat[f"{stage}_top20_ids"] = ";".join(x["chunk_id"] for x in row["rankings"][stage])
                for variant in VARIANTS:
                    for metric in metric_names:
                        flat[f"{stage}.{variant}.{metric}"] = row["metrics"].get(stage, {}).get(variant, {}).get(metric)
            writer.writerow(flat)

    lines = [
        "# First 50 questions: retrieval and generation report",
        "",
        f"- Cohort: n={report['cohort']['n']} ({report['cohort']['question_ids'][0]}–{report['cohort']['question_ids'][-1]})",
        f"- Frozen benchmark SHA-256: `{report['benchmark_sha256']}`",
        f"- Tracks: {report['track_counts']}",
        f"- Chunk-retrieval strict/acceptable scored questions: n={len(report['retrieval_scored_question_ids'])}",
        f"- Metric cohort sizes: { {k: len(v) for k, v in report['metric_eligible_question_ids'].items()} }",
        f"- Retrieval source: `{report['retrieval_source']}` (saved rankings; no rerun)",
        f"- Reranker: `{report['reranker_model']}`; generation context = reranker top 5",
        f"- Generation status: **{report['generation_status']}** ({sum(r['generation_status']=='completed' for r in report['per_question'])}/45 completed)",
        "",
        "## Aggregate retrieval metrics",
        "",
        "Macro means have question-bootstrap 95% CIs. Recall is support-group recall. Micro recall pools support groups. nDCG@10 is binary for strict/acceptable and graded for acceptable+partial (full/gold=2, partial=1).",
        "",
        "| Stage | Relevance | n | Hit@1/3/5/10/20 | Recall@1/3/5/10/20 | Full-hit@1/3/5/10/20 | MRR@20 | nDCG@10 |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    def fmt(stat: Dict[str, Any]) -> str:
        return f"{stat['mean']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"
    for stage in STAGES:
        for variant in VARIANTS:
            obj = report["aggregates"][stage][variant]
            m = obj["metrics"]
            hit_values = "<br>".join(fmt(m[f"hit@{k}"]) for k in CUTOFFS)
            recall_values = "<br>".join(fmt(m[f"recall@{k}"]) for k in CUTOFFS)
            full_values = "<br>".join(fmt(m[f"full_hit@{k}"]) for k in CUTOFFS)
            lines.append(f"| {stage} | {variant} | {obj['n']} | {hit_values} | {recall_values} | {full_values} | {fmt(m['mrr@20'])} | {fmt(m['ndcg@10'])} |")
    lines.extend(["", "Cutoff order in the three multi-value columns is @1, @3, @5, @10, @20.", "", "## Micro support-group recall", "", "| Stage | Relevance | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 |", "|---|---|---:|---:|---:|---:|---:|"])
    for stage in STAGES:
        for variant in VARIANTS:
            m = report["aggregates"][stage][variant]["metrics"]
            values = " | ".join(f"{m[f'micro_recall@{k}']['mean']:.3f} ({m[f'micro_recall@{k}']['matched_groups']}/{m[f'micro_recall@{k}']['total_groups']})" for k in CUTOFFS)
            lines.append(f"| {stage} | {variant} | {values} |")
    lines.extend(["", "## Per-question retrieval metrics and generation top 5", "", "For each question, the report lists all stage rankings and the full text of the five chunks supplied as the intended generation context. Non-retrieval tracks are shown but have no chunk-ID retrieval score.", ""])
    for row in report["per_question"]:
        lines.extend([f"### {row['question_id']} — {row['evaluation_track']}", "", row["question"], ""])
        if row["relevance_available"]:
            lines.append("| Stage | Variant | Hit@1/3/5/10/20 | Recall@1/3/5/10/20 | Full-hit@1/3/5/10/20 | MRR@20 | nDCG@10 |")
            lines.append("|---|---|---|---|---|---:|---:|")
            for stage in STAGES:
                for variant in VARIANTS:
                    if variant not in row["metrics"].get(stage, {}):
                        continue
                    m = row["metrics"][stage][variant]
                    hit_values = "/".join(f"{m[f'hit@{k}']:.2f}" for k in CUTOFFS)
                    recall_values = "/".join(f"{m[f'recall@{k}']:.2f}" for k in CUTOFFS)
                    full_values = "/".join(f"{m[f'full_hit@{k}']:.2f}" for k in CUTOFFS)
                    lines.append(f"| {stage} | {variant} | {hit_values} | {recall_values} | {full_values} | {m['mrr@20']:.2f} | {m['ndcg@10']:.2f} |")
        else:
            lines.append(f"Chunk retrieval metrics: N/A (track={row['evaluation_track']}; gold IDs unavailable or not scored).")
        for stage in STAGES:
            lines.append(f"- {stage} top-20 IDs: " + ", ".join(x["chunk_id"] for x in row["rankings"][stage]))
        lines.extend(["", "#### Reranker top 5 supplied to generation", ""])
        for chunk in row["generation_top5"]:
            lines.extend([f"**Rank {chunk['rank']} — `{chunk['chunk_id']}` — {chunk.get('doc_name') or 'source name unavailable'} — page {chunk.get('page_number') or 'n/a'}**", "", chunk["text"] or "[Chunk text missing from source-corrected chunk files.]", ""])
        if row["metadata_candidates"]:
            lines.append("Metadata retrieval candidates: " + json.dumps(row["metadata_candidates"], ensure_ascii=False))
            lines.append("")
        if row["generation_status"] == "completed":
            lines.extend(["**Generated answer**", "", row["generation_response"] or "", "", f"TTFT: {row['ttft_seconds']:.3f}s; total generation time: {row['total_generation_seconds']:.3f}s.", ""])
    completed_rows = [r for r in report["per_question"] if r["generation_status"] == "completed"]
    lines.extend(["## Generation execution summary", "", f"Completed answers: {len(completed_rows)}/45.", ""])
    ttfts = sorted(float(r["ttft_seconds"]) for r in completed_rows if r["ttft_seconds"] is not None)
    if ttfts:
        lines.append(f"Mean TTFT: {sum(ttfts)/len(ttfts):.3f}s; median TTFT: {ttfts[len(ttfts)//2]:.3f}s.")
    else:
        lines.append("TTFT and answers will appear here after generation completes.")
    lines.append("")
    (OUT / "retrieval_first50_generation_report.md").write_text("\n".join(lines), encoding="utf-8")

    # A resumable, exact-input file for a later Qwen run; it contains no generated answers.
    with (OUT / "generation_inputs_first50.jsonl").open("w", encoding="utf-8") as handle:
        for row in report["per_question"]:
            if row["out_of_scope"]:
                continue
            handle.write(json.dumps({
                "question_id": row["question_id"],
                "evaluation_track": row["evaluation_track"],
                "user_input": row["question"],
                "reference": row["gold_answer"],
                "generation_model": "qwen3:8b",
                "retrieved_chunk_ids": [x["chunk_id"] for x in row["generation_top5"]],
                "retrieved_contexts": [x["text"] for x in row["generation_top5"]],
                "metadata_candidates": row["metadata_candidates"],
                "generation_status": row["generation_status"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    result = build()
    write_outputs(result)
    print(json.dumps({
        "output_dir": str(OUT),
        "cohort_n": result["cohort"]["n"],
        "retrieval_scored_n": len(result["retrieval_scored_question_ids"]),
        "track_counts": result["track_counts"],
        "generation_status": result["generation_status"],
    }, ensure_ascii=False, indent=2))
