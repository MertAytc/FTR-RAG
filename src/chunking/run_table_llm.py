#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run table-chunk prompts against OpenAI Responses API, with resumable JSONL output."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit("Missing dependency: install the OpenAI Python SDK with `pip install openai`.") from exc


ALLOWED_TABLE_TYPES = [
    "outcome_measures", "recommendation_summary", "evidence_summary",
    "intervention_protocol", "dosage_or_parameters", "diagnostic_criteria",
    "classification_or_staging", "risk_factors_or_screening", "drug_or_medication",
    "timeline_or_phases", "glossary_or_definitions", "other",
]

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "table_id": {"type": "string"},
        "table_type": {"type": "string", "enum": ALLOWED_TABLE_TYPES},
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_title": {"type": "string"},
                    "source_row_indices": {"type": "array", "items": {"type": "integer"}},
                    "text": {"type": "string"},
                    "embedding_text": {"type": "string"},
                },
                "required": ["chunk_title", "source_row_indices", "text", "embedding_text"],
                "additionalProperties": False,
            },
        },
        "unparsed_rows": {"type": "array", "items": {"type": "integer"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["table_id", "table_type", "chunks", "unparsed_rows", "warnings"],
    "additionalProperties": False,
}

# Standard GPT-6 Luna text-token rates, USD per million tokens (checked against
# the official model page). Custom --model values need explicit price flags.
MODEL_PRICES_PER_MILLION = {
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50},
}


class OutputValidationError(Exception):
    pass


class FatalAPIError(Exception):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("table_id"), str):
            raise ValueError(f"Input line {line_no} must be an object with a string table_id.")
        records.append(item)
    return records


def load_system_prompt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    marker = "Everything below this line up to `END OF PROMPT` is the prompt."
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    end = raw.rfind("END OF PROMPT")
    if end < 0:
        raise ValueError(f"Prompt terminator END OF PROMPT not found in {path}.")
    prompt = raw[:end].strip()
    if not prompt:
        raise ValueError(f"System prompt is empty: {path}")
    return prompt


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in existing results {path} line {line_no}: {exc}") from exc
        if isinstance(obj, dict) and isinstance(obj.get("table_id"), str):
            ids.add(obj["table_id"])
    return ids


def safe_error_message(exc: Exception, api_key: str) -> str:
    # Do not print request headers or the secret. SDK exception messages are
    # surfaced so authentication/model/schema errors remain actionable.
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    return str(message).replace(api_key, "[REDACTED]")


def get_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                "total_tokens": None, "cached_input_tokens": 0, "cache_write_tokens": 0}
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(input_details, "cache_write_tokens", 0) or 0),
    }


def write_usage_event(
    usage_path: Path, *, table_id: str, model: str, response_id: str | None,
    output_attempt: int, transport_attempt: int, counts: dict[str, int | None],
    error: str | None = None,
) -> None:
    append_jsonl(usage_path, {
        "table_id": table_id,
        "model": model,
        "response_id": response_id,
        "output_attempt": output_attempt,
        "transport_attempt": transport_attempt,
        **counts,
        "error": error,
    })


def retry_delay(exc: Exception, attempt: int) -> float:
    return min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise OutputValidationError("Response contained no output_text (possibly refusal or incomplete output).")


def validate_model_object(obj: Any, expected_id: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise OutputValidationError("JSON root must be one object.")
    expected_keys = {"table_id", "table_type", "chunks", "unparsed_rows", "warnings"}
    if set(obj) != expected_keys:
        raise OutputValidationError(f"Expected exactly keys {sorted(expected_keys)}; got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}.")
    if obj.get("table_id") != expected_id:
        raise OutputValidationError(f"table_id mismatch: expected {expected_id!r}, got {obj.get('table_id')!r}.")
    if obj.get("table_type") not in ALLOWED_TABLE_TYPES:
        raise OutputValidationError(f"Invalid table_type: {obj.get('table_type')!r}.")
    if not isinstance(obj.get("chunks"), list) or not isinstance(obj.get("unparsed_rows"), list) or not isinstance(obj.get("warnings"), list):
        raise OutputValidationError("chunks, unparsed_rows, and warnings must be arrays.")
    for index, chunk in enumerate(obj["chunks"]):
        if not isinstance(chunk, dict) or set(chunk) != {"chunk_title", "source_row_indices", "text", "embedding_text"}:
            raise OutputValidationError(f"Chunk {index} does not match the required chunk object schema.")
        if not isinstance(chunk["chunk_title"], str) or not isinstance(chunk["text"], str) or not isinstance(chunk["embedding_text"], str):
            raise OutputValidationError(f"Chunk {index} text fields must be strings.")
        if not isinstance(chunk["source_row_indices"], list) or any(type(i) is not int for i in chunk["source_row_indices"]):
            raise OutputValidationError(f"Chunk {index} source_row_indices must be an integer array.")
    if any(type(i) is not int for i in obj["unparsed_rows"]):
        raise OutputValidationError("unparsed_rows must contain integers only.")
    if any(not isinstance(w, str) for w in obj["warnings"]):
        raise OutputValidationError("warnings must contain strings only.")
    return obj


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = int(getattr(exc, "status_code", 0) or 0)
        return status in {408, 409, 429} or status >= 500
    return False


def call_with_transient_retries(
    client: OpenAI, *, model: str, prompt: str, record: dict[str, Any],
    api_key: str, usage_path: Path, output_attempt: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usage_events: list[dict[str, Any]] = []
    for transport_attempt in range(1, 6):
        try:
            response = client.responses.create(
                model=model,
                instructions=prompt,
                input=json.dumps(record, ensure_ascii=False),
                reasoning={"effort": "low"},
                text={"format": {
                    "type": "json_schema",
                    "name": "table_chunk_result",
                    "strict": True,
                    "schema": RESULT_SCHEMA,
                }},
                store=False,
            )
            counts = get_usage(response)
            event = {
                "table_id": record["table_id"], "model": model,
                "response_id": getattr(response, "id", None),
                "output_attempt": output_attempt,
                "transport_attempt": transport_attempt,
                **counts, "error": None,
            }
            usage_events.append(event)
            try:
                obj = json.loads(response_text(response))
            except (json.JSONDecodeError, OutputValidationError) as exc:
                error = OutputValidationError(f"Could not parse API response as the required JSON object: {exc}")
                error.usage_events = usage_events
                raise error from exc
            try:
                validated = validate_model_object(obj, record["table_id"])
            except OutputValidationError as exc:
                exc.usage_events = usage_events
                raise
            return validated, usage_events
        except OutputValidationError:
            raise
        except Exception as exc:
            message = safe_error_message(exc, api_key)
            event = {
                "table_id": record["table_id"], "model": model,
                "response_id": None, "output_attempt": output_attempt,
                "transport_attempt": transport_attempt,
                "input_tokens": None, "output_tokens": None,
                "reasoning_tokens": None, "total_tokens": None,
                "cached_input_tokens": None, "cache_write_tokens": None,
                "error": message,
            }
            usage_events.append(event)
            if is_transient(exc):
                if transport_attempt < 5:
                    time.sleep(retry_delay(exc, transport_attempt))
                    continue
                fatal = FatalAPIError(f"Transient API failure persisted after 5 attempts: {message}")
                fatal.usage_events = usage_events
                raise fatal from exc
            status = getattr(exc, "status_code", None)
            prefix = f"OpenAI API request failed (HTTP {status})" if status else "OpenAI API request failed"
            fatal = FatalAPIError(f"{prefix}: {message}")
            fatal.usage_events = usage_events
            raise fatal from exc
    raise FatalAPIError("OpenAI API request failed after 5 attempts.")


def process_one(
    client: OpenAI, *, model: str, prompt: str, record: dict[str, Any],
    api_key: str, usage_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    all_usage: list[dict[str, Any]] = []
    for output_attempt in (1, 2):
        try:
            obj, events = call_with_transient_retries(
                client, model=model, prompt=prompt, record=record,
                api_key=api_key, usage_path=usage_path, output_attempt=output_attempt,
            )
            all_usage.extend(events)
            return obj, all_usage, None
        except OutputValidationError as exc:
            # A malformed object or a mismatched ID receives exactly one new model response.
            all_usage.extend(getattr(exc, "usage_events", []))
            error = str(exc)
            if output_attempt == 2:
                return None, all_usage, error
        except FatalAPIError:
            raise
    return None, all_usage, "Output failed validation twice."


def cost_estimate(events: list[dict[str, Any]], model: str, custom_prices: dict[str, float] | None) -> float | None:
    total = 0.0
    for event in events:
        input_tokens = event.get("input_tokens")
        output_tokens = event.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            continue
        event_model = event.get("model") or model
        prices = (custom_prices if event_model == model else None) or MODEL_PRICES_PER_MILLION.get(event_model)
        if prices is None:
            return None
        cached = int(event.get("cached_input_tokens") or 0)
        cache_write = int(event.get("cache_write_tokens") or 0)
        uncached = max(0, int(input_tokens) - cached - cache_write)
        total += (
            uncached * prices["input"]
            + cached * prices["cached_input"]
            + cache_write * prices["cache_write"]
            + int(output_tokens) * prices["output"]
        ) / 1_000_000
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna", help="OpenAI model ID (default: gpt-6-luna)")
    parser.add_argument("--input", default="data/output_v215/table_llm_v3/table_chunk_inputs_10.jsonl")
    parser.add_argument("--prompt", default="src/chunking/table_chunk_prompt_v2 (1).md")
    parser.add_argument("--output-dir", default=None, help="Defaults to the input file's directory")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum 4 concurrent requests (default: 4)")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--cached-input-price-per-million", type=float)
    parser.add_argument("--cache-write-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    args = parser.parse_args()

    if not 1 <= args.concurrency <= 4:
        parser.error("--concurrency must be between 1 and 4.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set; no API request was made.", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2
    if not prompt_path.is_file():
        print(f"ERROR: prompt file not found: {prompt_path}", file=sys.stderr)
        return 2
    try:
        records = read_jsonl(input_path)
        prompt = load_system_prompt(prompt_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "llm_results.jsonl"
    usage_path = output_dir / "usage.jsonl"
    failed_path = output_dir / "llm_failed.jsonl"

    try:
        completed_ids = read_existing_ids(results_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    work = []
    queued_ids = set()
    for record in records:
        tid = record["table_id"]
        if tid in completed_ids or tid in queued_ids:
            continue
        work.append(record)
        queued_ids.add(tid)

    custom_price_values = [
        args.input_price_per_million, args.cached_input_price_per_million,
        args.cache_write_price_per_million, args.output_price_per_million,
    ]
    custom_prices = None
    if any(value is not None for value in custom_price_values):
        if any(value is None for value in custom_price_values):
            parser.error("Supply all four price flags together.")
        custom_prices = {
            "input": args.input_price_per_million,
            "cached_input": args.cached_input_price_per_million,
            "cache_write": args.cache_write_price_per_million,
            "output": args.output_price_per_million,
        }

    print(f"Model: {args.model}; input records: {len(records)}; already complete: {len(completed_ids)}; to process: {len(work)}")
    new_usage_events: list[dict[str, Any]] = []
    new_successes = 0
    new_failures = 0
    client = OpenAI(api_key=api_key, max_retries=0, timeout=120.0)

    fatal_api_error: str | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(
                    process_one, client, model=args.model, prompt=prompt,
                    record=record, api_key=api_key, usage_path=usage_path,
                ): record
                for record in work
            }
            for future in concurrent.futures.as_completed(future_map):
                record = future_map[future]
                if future.cancelled():
                    continue
                try:
                    result, events, error = future.result()
                except FatalAPIError as exc:
                    events = getattr(exc, "usage_events", [])
                    fatal_api_error = fatal_api_error or str(exc)
                    for pending in future_map:
                        if pending is not future:
                            pending.cancel()
                    result, error = None, None
                except concurrent.futures.CancelledError:
                    continue
                for event in events:
                    append_jsonl(usage_path, event)
                    new_usage_events.append(event)
                if fatal_api_error and result is None and error is None:
                    continue
                if result is not None:
                    append_jsonl(results_path, result)
                    new_successes += 1
                    print(f"Completed {record['table_id']}")
                else:
                    append_jsonl(failed_path, {"table_id": record["table_id"], "model": args.model, "error": error})
                    new_failures += 1
                    print(f"Failed output validation for {record['table_id']}; see llm_failed.jsonl", file=sys.stderr)
    finally:
        client.close()

    if usage_path.exists():
        try:
            usage_events_all = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON in {usage_path}: {exc}", file=sys.stderr)
            return 2
    else:
        usage_events_all = []
    totals = {key: 0 for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cached_input_tokens", "cache_write_tokens")}
    for event in usage_events_all:
        for key in totals:
            value = event.get(key)
            if value is not None:
                totals[key] += int(value)
    estimate = cost_estimate(usage_events_all, args.model, custom_prices)
    summary = {
        "model": args.model,
        "new_successes": new_successes,
        "new_failures": new_failures,
        "skipped_existing": len(records) - len(work),
        **totals,
        "estimated_cost_usd": round(estimate, 8) if estimate is not None else None,
        "cost_note": ("Standard rates; excludes any taxes or account-level adjustments." if estimate is not None
                      else "Unknown model price; supply all four price flags to estimate cost."),
        "results": str(results_path),
        "usage": str(usage_path),
        "failed": str(failed_path),
    }
    if fatal_api_error:
        print(f"ERROR: {fatal_api_error}", file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if new_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
