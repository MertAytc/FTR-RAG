"""Create the requested retrieval-only first-50 report from frozen saved runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.report_first50_retrieval_generation import (
    OUT_OF_SCOPE_IDS,
    STAGES,
    VARIANTS,
    build,
)

OUT = ROOT / "data/processed/retrieval_first50_report_20261029"
CUTOFFS = (1, 3, 5, 10, 20)


def metric_str(value: Dict[str, Any]) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]"


def prepare() -> Dict[str, Any]:
    report = build()
    all_rows = report.pop("per_question")
    included = [row for row in all_rows if row["question_id"] not in OUT_OF_SCOPE_IDS]
    excluded = [row for row in all_rows if row["question_id"] in OUT_OF_SCOPE_IDS]
    for row in included:
        row["reranker_top5"] = row.pop("generation_top5")
        for key in (
            "generation_status", "generation_response", "ttft_seconds",
            "total_generation_seconds", "generation_context_chunk_ids",
            "retrieved_contexts", "context_sources",
        ):
            row.pop(key, None)
    report["report_title"] = "First 50 questions: retrieval-only evaluation"
    report["cohort"] = {
        "original_n": 50,
        "included_n": 45,
        "question_ids": [row["question_id"] for row in included],
        "excluded_out_of_scope": sorted(OUT_OF_SCOPE_IDS),
    }
    report["track_counts_in_included_cohort"] = {
        track: sum(row["evaluation_track"] == track for row in included)
        for track in sorted({row["evaluation_track"] for row in included})
    }
    report["per_question"] = included
    report["excluded_out_of_scope_questions"] = [
        {"question_id": row["question_id"], "question": row["question"], "track": row["evaluation_track"]}
        for row in excluded
    ]
    report.pop("generation_status", None)
    report.pop("generation_model_requested", None)
    report["limitations"] = [
        "The five out-of-scope questions FTR-044–FTR-048 are excluded; the scored cohort has 45 questions.",
        "Five metadata-attribution questions have no chunk gold IDs and are listed but excluded from chunk-retrieval metrics.",
        "FTR-020 has no canonical gold ID. It is excluded from strict and acceptable metrics; one partial-only judgment is counted only in the acceptable+partial upper-bound variant, giving n=40 for that variant.",
        "Strict and acceptable main retrieval metrics use n=39 valid retrieval-track questions. Rankings are from the saved source-corrected run; no new retrieval or reranker call was made.",
        "The reranker top-5 is shown as retrieval output only. Generation and TTFT are out of scope for this report.",
    ]
    return report


def write(report: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_first50_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metric_names = (
        [f"hit@{k}" for k in CUTOFFS]
        + [f"recall@{k}" for k in CUTOFFS]
        + [f"full_hit@{k}" for k in CUTOFFS]
        + ["mrr@20", "ndcg@10"]
    )
    columns = [
        "question_id", "evaluation_track", "question", "gold_chunk_ids",
        "strict_eligible", "acceptable_eligible", "partial_eligible",
        "dense_top20_ids", "bm25_top20_ids", "rrf_top20_ids",
        "reranker_top20_ids", "reranker_top5_ids",
    ] + [f"{stage}.{variant}.{metric}" for stage in STAGES for variant in VARIANTS for metric in metric_names]
    with (OUT / "retrieval_first50_per_question.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["per_question"]:
            flat: Dict[str, Any] = {
                "question_id": row["question_id"],
                "evaluation_track": row["evaluation_track"],
                "question": row["question"],
                "gold_chunk_ids": ";".join(row["gold_chunk_ids"]),
                "strict_eligible": row["metric_eligible"]["strict_canonical"],
                "acceptable_eligible": row["metric_eligible"]["acceptable_full"],
                "partial_eligible": row["metric_eligible"]["acceptable_plus_partial"],
                "reranker_top5_ids": ";".join(x["chunk_id"] for x in row["reranker_top5"]),
            }
            for stage in STAGES:
                flat[f"{stage}_top20_ids"] = ";".join(x["chunk_id"] for x in row["rankings"][stage])
                for variant in VARIANTS:
                    for metric in metric_names:
                        flat[f"{stage}.{variant}.{metric}"] = row["metrics"].get(stage, {}).get(variant, {}).get(metric)
            writer.writerow(flat)

    lines = [
        "# First 50 benchmark questions — retrieval evaluation",
        "",
        "## Cohort and scoring rules",
        "",
        "The first 50 frozen-v3 questions were reviewed. FTR-044–FTR-048 (five out-of-scope questions) are excluded, leaving 45 questions.",
        "",
        "- Strict canonical and Acceptable: n=39 valid retrieval-track questions.",
        "- Acceptable + Partial: n=40; FTR-020 contributes only through one partial judgment. Its strict and acceptable scores remain N/A because it has no valid canonical gold.",
        "- Five metadata-attribution questions have no chunk gold IDs. They are shown in the cohort inventory but excluded from chunk-retrieval metrics.",
        "- Rankings come from the saved source-corrected run; no new retrieval or reranker call was made.",
        f"- Frozen benchmark SHA-256: `{report['benchmark_sha256']}`.",
        f"- Reranker: `{report['reranker_model']}`; top-5 is included below as retrieved candidates. Generation answers and TTFT are outside this retrieval-only report.",
        "",
        "## Aggregate metrics",
        "",
        "Cells show macro mean with question-bootstrap 95% CI. Recall is support-group recall; MRR is MRR@20; nDCG uses binary gain for Strict/Acceptable and graded gain for Acceptable+Partial (full/gold=2, partial=1).",
        "",
        "| Stage | Relevance | n | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@20 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        for variant in VARIANTS:
            summary = report["aggregates"][stage][variant]
            metrics = summary["metrics"]
            lines.append(
                f"| {stage} | {variant} | {summary['n']} | "
                f"{metric_str(metrics['hit@1'])} | {metric_str(metrics['recall@3'])} | "
                f"{metric_str(metrics['recall@5'])} | {metric_str(metrics['recall@10'])} | "
                f"{metric_str(metrics['mrr@20'])} | {metric_str(metrics['ndcg@10'])} |"
            )
    lines.extend(["", "## Per-question results and returned chunks", ""])
    for row in report["per_question"]:
        lines.extend([f"### {row['question_id']} — {row['evaluation_track']}", "", row["question"], ""])
        if not row["relevance_available"]:
            lines.append("Chunk retrieval metrics: N/A — no chunk gold for this question/track.")
        else:
            lines.append("| Stage | Relevance | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@20 | nDCG@10 |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for stage in STAGES:
                for variant in VARIANTS:
                    metric = row["metrics"].get(stage, {}).get(variant)
                    if metric is None:
                        lines.append(f"| {stage} | {variant} | N/A | N/A | N/A | N/A | N/A | N/A |")
                    else:
                        lines.append(
                            f"| {stage} | {variant} | {metric['hit@1']:.2f} | {metric['recall@3']:.2f} | "
                            f"{metric['recall@5']:.2f} | {metric['recall@10']:.2f} | "
                            f"{metric['mrr@20']:.2f} | {metric['ndcg@10']:.2f} |"
                        )
        for stage in STAGES:
            values = ", ".join(f"{x['rank']}:{x['chunk_id']}" for x in row["rankings"][stage])
            lines.append(f"- {stage} top-20: {values}")
        lines.extend(["", "#### Reranker top-5 retrieved chunks", ""])
        for item in row["reranker_top5"]:
            lines.extend([
                f"**Rank {item['rank']} — `{item['chunk_id']}` — {item.get('doc_name') or 'source unavailable'} — page {item.get('page_number') or 'n/a'}**",
                "",
                item.get("text") or "[Chunk text not found in the source-corrected chunk files.]",
                "",
            ])
        if row["evaluation_track"] == "metadata_attribution":
            lines.extend(["Metadata retrieval was inspected separately. No chunk gold exists, so chunk Hit/Recall/MRR/nDCG are N/A.", ""])
    lines.extend(["## Excluded questions", "", "FTR-044, FTR-045, FTR-046, FTR-047, FTR-048 were identified as the five out-of-scope questions and excluded from the 45-question cohort.", ""])
    (OUT / "retrieval_first50_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = prepare()
    write(result)
    print(json.dumps({
        "report_dir": str(OUT),
        "included_questions": len(result["per_question"]),
        "tracks": result["track_counts_in_included_cohort"],
        "metric_n": {variant: len(ids) for variant, ids in result["metric_eligible_question_ids"].items()},
        "files": ["retrieval_first50_report.md", "retrieval_first50_report.json", "retrieval_first50_per_question.csv"],
    }, ensure_ascii=False, indent=2))
