#!/usr/bin/env python3
"""Transcribe table content from the prepared PDF page images with the OpenAI API.

Reads only the image/text pairs listed in manifest.json, writes actual model responses
to table_extractions.jsonl, and checks numeric tokens against each page's extracted text.
The API key is read only from OPENAI_API_KEY and is never printed or written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


DEFAULT_MODEL = "gpt-6-luna"
TRANSCRIPTION_INSTRUCTIONS = """Transcribe only tabular content visibly present in the supplied PDF page image.
Return every visually separate table as one item. In each table, rows must be arrays of
cell strings in reading order, and the first row must contain the visible column headers
when headers are present. Preserve cell wording, numbers, punctuation, blank cells, and
bullet/list text. Do not summarize, infer, repair OCR, complete clipped text, or invent
cells. Keep multi-line lists inside their visible cell using newline characters. If a
region is unreadable, leave only that cell's legible text and describe the location in
unreadable_regions; never guess its contents. If no table is visible, return an empty
tables array. Do not add source metadata; the caller attaches it."""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["title", "rows"],
            },
        },
        "unreadable_regions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tables", "unreadable_regions"],
}

TRANSIENT_STATUS = {408, 409, 429}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def redact_error(error: BaseException) -> str:
    message = str(error)
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    return message[:2000]


def request_page(client: OpenAI, model: str, image_path: Path, source_pdf: str, page_no: int):
    mime = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_url = f"data:{mime};base64,{encoded}"
    user_text = (
        f"Transcribe the visible table(s) from this page image. Source file: {source_pdf}; "
        f"physical PDF page: {page_no}. These labels identify the source only; do not infer content from them."
    )
    return client.responses.create(
        model=model,
        instructions=TRANSCRIPTION_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "page_table_transcription",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        store=False,
    )


def call_with_retry(client: OpenAI, model: str, image_path: Path, source_pdf: str, page_no: int):
    last_error: BaseException | None = None
    for attempt in range(5):
        try:
            return request_page(client, model, image_path, source_pdf, page_no)
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code not in TRANSIENT_STATUS and exc.status_code < 500:
                raise
            last_error = exc
        if attempt < 4:
            delay = min(60.0, 2.0**attempt) + random.uniform(0.0, 0.5)
            print(f"Transient API error on PDF page {page_no}; retry {attempt + 2}/5 in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def numeric_tokens(value: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", value)


def number_audit(result: dict[str, Any], page_text: str) -> dict[str, Any]:
    source_tokens = set(numeric_tokens(page_text))
    checks = []
    for table_index, table in enumerate(result.get("tables", [])):
        fields = [("title", table.get("title", ""))]
        for row_index, row in enumerate(table.get("rows", [])):
            for cell_index, cell in enumerate(row):
                fields.append((f"rows[{row_index}][{cell_index}]", cell))
        for field, text in fields:
            for token in numeric_tokens(text):
                checks.append({
                    "table_index": table_index,
                    "field": field,
                    "number": token,
                    "found_in_page_text": token in source_tokens,
                })
    return {
        "numeric_tokens_checked": len(checks),
        "all_numbers_found_in_page_text": all(item["found_in_page_text"] for item in checks),
        "checks": checks,
        "unmatched_numbers": [item for item in checks if not item["found_in_page_text"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set; no API request was made.", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = root / "table_extractions.jsonl"
    usage_path = root / "usage.jsonl"
    failed_path = root / "failed_pages.jsonl"
    audit_path = root / "number_validation.jsonl"
    existing = read_jsonl(result_path)
    done = {(row.get("source_pdf"), row.get("pdf_page_1based")) for row in existing}

    client = OpenAI(api_key=api_key, max_retries=0, timeout=180.0)
    succeeded = failed = skipped = 0
    for page in manifest["pages"]:
        source_pdf = page["source_pdf"]
        page_no = int(page["pdf_page_1based"])
        key = (source_pdf, page_no)
        if key in done:
            skipped += 1
            continue
        image_path = root / page["image"]
        text_path = root / page["page_text"]
        try:
            response = call_with_retry(client, args.model, image_path, source_pdf, page_no)
            result = json.loads(response.output_text)
            record = {
                "model": args.model,
                "source_pdf": source_pdf,
                "pdf_page_1based": page_no,
                "transcription": result,
            }
            append_jsonl(result_path, record)
            usage = getattr(response, "usage", None)
            usage_record = {
                "model": args.model,
                "source_pdf": source_pdf,
                "pdf_page_1based": page_no,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            append_jsonl(usage_path, usage_record)
            audit = number_audit(result, text_path.read_text(encoding="utf-8"))
            append_jsonl(audit_path, {
                "source_pdf": source_pdf,
                "pdf_page_1based": page_no,
                **audit,
            })
            succeeded += 1
            print(f"Completed {Path(source_pdf).name} page {page_no}; numeric mismatches: {len(audit['unmatched_numbers'])}")
        except Exception as exc:
            append_jsonl(failed_path, {
                "model": args.model,
                "source_pdf": source_pdf,
                "pdf_page_1based": page_no,
                "error": redact_error(exc),
            })
            failed += 1
            print(f"FAILED {Path(source_pdf).name} page {page_no}: {redact_error(exc)}", file=sys.stderr)

    print(json.dumps({
        "model": args.model,
        "pages_total": len(manifest["pages"]),
        "new_successes": succeeded,
        "new_failures": failed,
        "skipped_existing": skipped,
        "results": str(result_path),
        "usage": str(usage_path),
        "number_validation": str(audit_path),
        "failed": str(failed_path),
    }, ensure_ascii=False, indent=2))
    total_successes = len(existing) + succeeded
    manifest["api_model"] = args.model
    manifest["api_status"] = (
        "complete" if total_successes == len(manifest["pages"]) and failed == 0 else "partial"
    )
    manifest["api_successful_pages"] = total_successes
    manifest["api_failed_pages_this_run"] = failed
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
