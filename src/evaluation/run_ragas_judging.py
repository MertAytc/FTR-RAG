"""Score saved Qwen generations with RAGAS using GPT-6 Luna as judge.

The judge uses the OpenAI Responses API. Embedding-based metrics use the OpenAI
Embeddings API. Inputs are resumable JSONL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from openai import OpenAI
from langchain_core.outputs import Generation, LLMResult
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerRelevancy,
    ContextEntityRecall,
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NoiseSensitivity,
    SemanticSimilarity,
)

LOG = logging.getLogger("ragas_judging")
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
    "context_entity_recall",
    "factual_correctness_f1",
    "semantic_similarity",
    "noise_sensitivity_relevant",
)
INPUT_PRICE_PER_M = 0.10
CACHED_INPUT_PRICE_PER_M = 0.01
OUTPUT_PRICE_PER_M = 0.50
EMBEDDING_PRICE_PER_M = 0.02


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict) or not row.get("question_id"):
                raise ValueError(f"Expected question_id object at {path}:{line_no}")
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


class OpenAIResponsesJudge(BaseRagasLLM):
    """Minimal RAGAS LLM adapter for GPT-6 Luna's Responses endpoint."""

    def __init__(self, model: str, usage_path: Path, max_concurrent: int = 4):
        super().__init__()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self.model = model
        self.usage_path = usage_path
        self.client = OpenAI(max_retries=1, timeout=90.0)
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._usage_lock = threading.Lock()
        self.usage_totals = {
            "calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0,
        }

    def get_temperature(self, n: int) -> float:
        # The Responses request intentionally omits temperature for this model.
        return 0.0

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        prompt_text = prompt.to_string()
        generations = []
        for _ in range(max(1, n)):
            with self._semaphore:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt_text,
                    reasoning={"effort": "low"},
                    store=False,
                )
            text = str(getattr(response, "output_text", "") or "").strip()
            if not text:
                raise RuntimeError("OpenAI Responses API returned empty output_text.")
            generations.append(Generation(text=text))
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            input_details = getattr(usage, "input_tokens_details", None)
            output_details = getattr(usage, "output_tokens_details", None)
            cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
            reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
            cost = (
                max(0, input_tokens - cached_tokens) * INPUT_PRICE_PER_M
                + cached_tokens * CACHED_INPUT_PRICE_PER_M
                + output_tokens * OUTPUT_PRICE_PER_M
            ) / 1_000_000
            usage_row = {
                "response_id": getattr(response, "id", None),
                "model": self.model,
                "endpoint": "responses",
                "reasoning_effort": "low",
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "estimated_cost_usd": cost,
            }
            with self._usage_lock:
                _append_jsonl(self.usage_path, usage_row)
                self.usage_totals["calls"] += 1
                self.usage_totals["input_tokens"] += input_tokens
                self.usage_totals["cached_input_tokens"] += cached_tokens
                self.usage_totals["output_tokens"] += output_tokens
                self.usage_totals["reasoning_tokens"] += reasoning_tokens
                self.usage_totals["estimated_cost_usd"] += cost
        return LLMResult(generations=[generations], llm_output={"model_name": self.model})

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class OpenAIRagasEmbeddings(BaseRagasEmbeddings):
    """Use OpenAI embeddings for RAGAS answer relevancy and semantic similarity."""

    def __init__(self, model: str, usage_path: Path, max_inputs_per_request: int = 128):
        super().__init__()
        self.model = model
        self.usage_path = usage_path
        self.max_inputs_per_request = max_inputs_per_request
        self.client = OpenAI(max_retries=2, timeout=90.0)
        self._usage_lock = threading.Lock()
        self.usage_totals = {"calls": 0, "input_tokens": 0, "estimated_cost_usd": 0.0}
        self.run_config = RunConfig()

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self.max_inputs_per_request):
            batch = [str(text) for text in texts[start:start + self.max_inputs_per_request]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            if len(response.data) != len(batch):
                raise RuntimeError(
                    "OpenAI embedding response count did not match input count."
                )
            for item in response.data:
                vectors[start + int(item.index)] = [float(value) for value in item.embedding]

            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            cost = input_tokens * EMBEDDING_PRICE_PER_M / 1_000_000
            usage_row = {
                "model": self.model,
                "endpoint": "embeddings",
                "input_count": len(batch),
                "input_tokens": input_tokens,
                "estimated_cost_usd": cost,
            }
            with self._usage_lock:
                _append_jsonl(self.usage_path, usage_row)
                self.usage_totals["calls"] += 1
                self.usage_totals["input_tokens"] += input_tokens
                self.usage_totals["estimated_cost_usd"] += cost

        if any(vector is None for vector in vectors):
            raise RuntimeError("OpenAI embedding response left missing vectors.")
        return [vector for vector in vectors if vector is not None]

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _build_metrics(llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings) -> Dict[str, Any]:
    items = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        ContextEntityRecall(llm=llm),
        FactualCorrectness(llm=llm, mode="f1"),
        SemanticSimilarity(embeddings=embeddings),
        NoiseSensitivity(llm=llm, mode="relevant"),
    ]
    for metric, name in zip(items, METRICS):
        metric.name = name
    return dict(zip(METRICS, items))


def _sample(row: Dict[str, Any]) -> SingleTurnSample:
    contexts = row.get("retrieved_contexts") or []
    contexts = [str(item) for item in contexts if item is not None]
    return SingleTurnSample(
        user_input=str(row.get("user_input") or ""),
        retrieved_contexts=contexts,
        response=str(row.get("response") or ""),
        reference=str(row.get("reference") or ""),
    )


def _has_reference(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("reference") or "").strip())


def _metric_applicable(name: str, row: Dict[str, Any]) -> bool:
    if name in {
        "llm_context_precision_with_reference", "context_recall",
        "context_entity_recall", "factual_correctness_f1", "semantic_similarity",
    }:
        return _has_reference(row)
    return True


async def _score_one(metric, sample: SingleTurnSample, timeout_seconds: float) -> float:
    value = await metric.single_turn_ascore(sample, timeout=timeout_seconds)
    return float(value)


def _load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for row in _read_jsonl(path):
        result[row["question_id"]] = row
    return result


def _bootstrap_mean(values: Sequence[float], seed: int, repetitions: int) -> Dict[str, Any]:
    numbers = np.asarray(list(values), dtype=np.float64)
    if not len(numbers):
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numbers), size=(repetitions, len(numbers)))
    samples = numbers[indices].mean(axis=1)
    return {
        "n": int(len(numbers)),
        "mean": float(numbers.mean()),
        "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
    }


def _summarize(rows: List[Dict[str, Any]], seed: int, repetitions: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "by_evaluation_track": {}}
    tracks = sorted({str(row.get("evaluation_track") or "unknown") for row in rows})
    for scope, selected in [("overall", rows)] + [
        (track, [row for row in rows if (row.get("evaluation_track") or "unknown") == track])
        for track in tracks
    ]:
        metrics = {}
        for name in METRICS:
            values = [
                float(row.get(name, row.get("scores", {}).get(name)))
                for row in selected
                if isinstance(row.get(name, row.get("scores", {}).get(name)), (int, float))
            ]
            metrics[name] = _bootstrap_mean(values, seed, repetitions)
        if scope == "overall":
            summary["overall"] = metrics
        else:
            summary["by_evaluation_track"][scope] = metrics
    summary["question_count"] = len(rows)
    summary["metric_scored_counts"] = {
        name: sum(isinstance(row.get(name), (int, float)) for row in rows)
        for name in METRICS
    }
    summary["metric_error_counts"] = {
        name: sum(row.get("metric_status", {}).get(name) == "error" for row in rows)
        for name in METRICS
    }
    return summary


async def _evaluate_input(
    input_path: Path,
    output_path: Path,
    metrics: Dict[str, Any],
    batch_questions: int,
    metric_timeout_seconds: float,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    inputs = _read_jsonl(input_path)
    if limit is not None:
        inputs = inputs[:limit]
    existing = _load_existing(output_path)
    completed: Dict[str, Dict[str, Any]] = dict(existing)
    remaining = [row for row in inputs if not _is_complete(existing.get(row["question_id"]))]
    LOG.info("%s: %d rows; %d complete; %d to score", input_path.name, len(inputs), len(inputs)-len(remaining), len(remaining))

    async def score_row(source: Dict[str, Any]) -> Dict[str, Any]:
        previous = existing.get(source["question_id"], {})
        result = {
            "question_id": source["question_id"],
            "evaluation_track": source.get("evaluation_track", "unknown"),
            "user_input": source.get("user_input"),
            "generation_model": source.get("generation_model"),
            "response": source.get("response"),
            "scores_model": metrics["faithfulness"].llm.model,
            "scores": dict(previous.get("scores") or {}),
            "metric_status": dict(previous.get("metric_status") or {}),
        }
        sample = _sample(source)
        applicable = []
        for name in METRICS:
            old_status = result["metric_status"].get(name)
            if old_status in {"success", "not_applicable_missing_reference"}:
                result[name] = previous.get(name, result["scores"].get(name))
                continue
            if _metric_applicable(name, source):
                applicable.append(name)
            else:
                result["scores"][name] = None
                result[name] = None
                result["metric_status"][name] = "not_applicable_missing_reference"

        async def score_metric(name: str):
            try:
                return name, await _score_one(metrics[name], sample, metric_timeout_seconds), None
            except Exception as exc:  # Keep the failure auditable and resumable.
                return name, None, f"{type(exc).__name__}: {exc}"

        for name, value, error in await asyncio.gather(*(score_metric(name) for name in applicable)):
            result["scores"][name] = value
            result[name] = value
            result["metric_status"][name] = "success" if error is None else "error"
            if error is not None:
                result.setdefault("metric_errors", {})[name] = error
        result["_complete"] = not any(status == "error" for status in result["metric_status"].values())
        return result

    for start in range(0, len(remaining), batch_questions):
        batch = remaining[start:start + batch_questions]
        for task in asyncio.as_completed([score_row(row) for row in batch]):
            scored = await task
            _append_jsonl(output_path, scored)
            completed[scored["question_id"]] = scored
            LOG.info("Scored %s (%d/%d)", scored["question_id"], len(completed), len(inputs))
    input_ids = [row["question_id"] for row in inputs]
    input_id_set = set(input_ids)
    ordered_ids = input_ids + [qid for qid in completed if qid not in input_id_set]
    ordered_rows = [completed[qid] for qid in ordered_ids if qid in completed]
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_rows),
        encoding="utf-8",
    )
    return ordered_rows[:len(inputs)]


def _is_complete(row: Dict[str, Any] | None) -> bool:
    if not row:
        return False
    statuses = row.get("metric_status", {})
    return all(status in {"success", "not_applicable_missing_reference"} for status in statuses.values()) and len(statuses) == len(METRICS)


def _paired_comparison(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], seed: int, repetitions: int) -> Dict[str, Any]:
    by_a = {row["question_id"]: row for row in rows_a}
    by_b = {row["question_id"]: row for row in rows_b}
    out = {}
    for name in METRICS:
        qids = sorted(
            qid for qid in by_a.keys() & by_b.keys()
            if isinstance(by_a[qid].get(name, by_a[qid].get("scores", {}).get(name)), (int, float))
            and isinstance(by_b[qid].get(name, by_b[qid].get("scores", {}).get(name)), (int, float))
        )
        differences = [
            float(by_b[qid].get(name, by_b[qid].get("scores", {}).get(name)))
            - float(by_a[qid].get(name, by_a[qid].get("scores", {}).get(name)))
            for qid in qids
        ]
        out[name] = _bootstrap_mean(differences, seed, repetitions)
        out[name]["comparison"] = "second input minus first input"
    return out


def _sum_usage(path: Path) -> Dict[str, Any]:
    totals = {
        "calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
        "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0,
    }
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            totals["calls"] += 1
            for key in totals:
                if key == "calls":
                    continue
                totals[key] += row.get(key, 0) or 0
    return totals


def _sum_embedding_usage(path: Path) -> Dict[str, Any]:
    totals = {"calls": 0, "input_count": 0, "input_tokens": 0, "estimated_cost_usd": 0.0}
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            totals["calls"] += 1
            for key in totals:
                if key == "calls":
                    continue
                totals[key] += row.get(key, 0) or 0
    return totals


async def main_async(args) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    usage_path = out_dir / "openai_usage.jsonl"
    embedding_usage_path = out_dir / "openai_embedding_usage.jsonl"
    judge = OpenAIResponsesJudge(args.judge_model, usage_path, args.max_concurrent)
    embeddings = OpenAIRagasEmbeddings(args.embedding_model, embedding_usage_path)
    metric_objects = _build_metrics(judge, embeddings)
    all_results = []
    for input_path in args.inputs:
        input_path = input_path.resolve()
        output_path = out_dir / f"{input_path.parent.name}_ragas_scores.jsonl"
        rows = await _evaluate_input(
            input_path, output_path, metric_objects, args.batch_questions,
            args.metric_timeout_seconds, args.limit,
        )
        summary = {
            "input": str(input_path), "output": str(output_path),
            "judge_model": args.judge_model, "judge_endpoint": "OpenAI Responses API",
            "reasoning_effort": "low", "embedding_model": args.embedding_model,
            "metrics": list(METRICS), "summary": _summarize(rows, args.seed, args.bootstrap_repetitions),
        }
        summary_path = out_dir / f"{input_path.parent.name}_ragas_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_results.append((input_path, rows, summary_path))

    report = {
        "judge_model": args.judge_model,
        "judge_endpoint": "OpenAI Responses API",
        "reasoning_effort": "low",
        "embedding_model": args.embedding_model,
        "embedding_endpoint": "OpenAI Embeddings API",
        "metric_definitions": {
            "faithfulness": "Response claims supported by retrieved contexts.",
            "answer_relevancy": "How relevant the answer is to the question.",
            "llm_context_precision_with_reference": "Ranking quality of retrieved contexts against the reference.",
            "context_recall": "Reference information supported by retrieved contexts.",
            "context_entity_recall": "Reference entities present in retrieved contexts.",
            "factual_correctness_f1": "RAGAS factual correctness F1 against the reference.",
            "semantic_similarity": "Embedding similarity of answer and reference.",
            "noise_sensitivity_relevant": "Whether irrelevant context affects the answer to relevant claims.",
        },
        "inputs": [],
        "judge_usage_this_process": dict(judge.usage_totals),
        "judge_usage_total_in_usage_log": _sum_usage(usage_path),
        "usage_log": str(usage_path),
        "embedding_usage_this_process": dict(embeddings.usage_totals),
        "embedding_usage_total_in_usage_log": _sum_embedding_usage(embedding_usage_path),
        "embedding_usage_log": str(embedding_usage_path),
        "confidence_intervals": {"method": "question-level bootstrap mean, 95% percentile interval", "repetitions": args.bootstrap_repetitions, "seed": args.seed},
    }
    for path, rows, summary_path in all_results:
        report["inputs"].append({"input": str(path), "summary": str(summary_path), "metrics": _summarize(rows, args.seed, args.bootstrap_repetitions)})
    if len(all_results) == 2:
        report["paired_second_minus_first"] = _paired_comparison(all_results[0][1], all_results[1][1], args.seed, args.bootstrap_repetitions)
        report["paired_order"] = [str(all_results[0][0]), str(all_results[1][0])]
    report_path = out_dir / "ragas_judging_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RAGAS evaluation report", "",
        f"- Judge: {args.judge_model} via Responses API; reasoning effort low.",
        f"- Embeddings: OpenAI Embeddings API `{args.embedding_model}`.",
        f"- Bootstrap: {args.bootstrap_repetitions:,} resamples; seed {args.seed}.",
        f"- Judge calls recorded: {report['judge_usage_total_in_usage_log']['calls']}; tokens in/out: {report['judge_usage_total_in_usage_log']['input_tokens']:,}/{report['judge_usage_total_in_usage_log']['output_tokens']:,}; estimated cost: ${report['judge_usage_total_in_usage_log']['estimated_cost_usd']:.6f}.",
        f"- Embedding calls recorded: {report['embedding_usage_total_in_usage_log']['calls']}; input tokens: {report['embedding_usage_total_in_usage_log']['input_tokens']:,}; estimated cost: ${report['embedding_usage_total_in_usage_log']['estimated_cost_usd']:.6f}.",
    ]
    for input_info in report["inputs"]:
        lines.extend(["", f"## {Path(input_info['input']).name}", "", "| Metric | n | Mean | 95% CI |", "|---|---:|---:|---:|"])
        for name in METRICS:
            item = input_info["metrics"]["overall"][name]
            mean = "N/A" if item["mean"] is None else f"{item['mean']:.4f}"
            ci = "N/A" if item["mean"] is None else f"[{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
            lines.append(f"| {name} | {item['n']} | {mean} | {ci} |")
        lines.extend(["", "### By evaluation track", ""])
        for track, track_metrics in input_info["metrics"]["by_evaluation_track"].items():
            lines.append(f"**{track}**")
            lines.append("")
            lines.append("| Metric | n | Mean | 95% CI |")
            lines.append("|---|---:|---:|---:|")
            for name in METRICS:
                item = track_metrics[name]
                mean = "N/A" if item["mean"] is None else f"{item['mean']:.4f}"
                ci = "N/A" if item["mean"] is None else f"[{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
                lines.append(f"| {name} | {item['n']} | {mean} | {ci} |")
            lines.append("")
    if "paired_second_minus_first" in report:
        lines.extend(["", "## Paired commentary effect (commented arm minus raw arm)", "", "| Metric | n | Mean difference | 95% CI |", "|---|---:|---:|---:|"])
        for name, item in report["paired_second_minus_first"].items():
            ci = item["ci95"]
            lines.append(f"| {name} | {item['n']} | {item['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |" if item["mean"] is not None else f"| {name} | 0 | N/A | N/A |")
    (out_dir / "ragas_judging_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "markdown": str(out_dir / 'ragas_judging_report.md'), "usage": str(usage_path), "judge_usage": judge.usage_totals}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="One or two generation JSONL files; use raw first and commented second for paired comparison.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--judge-model", default="gpt-6-luna")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--batch-questions", type=int, default=4)
    parser.add_argument("--metric-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows (useful for an API smoke run; resume later without this flag).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if len(args.inputs) not in {1, 2}:
        parser.error("Provide one input or exactly two inputs for a paired comparison.")
    if args.max_concurrent < 1 or args.batch_questions < 1:
        parser.error("Concurrency and batch question count must be positive.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    # RAGAS 0.2.x combined with Python 3.14 can expose an empty C-level
    # current_task lookup in asyncio.timeout despite the running task being
    # present in asyncio's task registry. Restore the stdlib lookup semantics.
    def _registered_current_task(loop=None):
        if loop is None:
            loop = asyncio.get_running_loop()
        return asyncio.tasks._current_tasks.get(loop)
    asyncio.tasks.current_task = _registered_current_task
    asyncio.current_task = _registered_current_task
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
