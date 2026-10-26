#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table chunk pipeline around an LLM step (LLM call itself is NOT done here).

  prepare  : all_tables.jsonl -> table_chunk_inputs.jsonl (+ table_inputs_excluded.jsonl)
             - drops tables already emitted as AHA recommendation chunks
             - drops clear noise (empty, reference lists, author/work-group lists, garbled) WITH a recorded reason
             - merges multi-page continuations into one table
  finalize : LLM output + all_tables.jsonl -> table_chunks.json (old schema + metadata) + table_chunks_review.json
             - validates row coverage, text fidelity, numbers
             - attaches ALL metadata from the sidecar in code (the LLM never touches it)

Usage:
  python table_chunks_pipeline.py prepare  --tables out/all_tables.jsonl --out-dir out/table_llm
  (run your LLM with table_chunk_prompt_v2.md on each line of table_chunk_inputs.jsonl;
   save results as a JSON list or JSONL, one result object per table)
  python table_chunks_pipeline.py finalize --tables out/all_tables.jsonl --llm-output out/table_llm/llm_results.json --out-dir out/table_llm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.3"
PIPELINE_VERSION = "table-chunks-1.2"

REFERENCE_CAPTION = re.compile(
    r"bibliograph|search\s+(?:strateg|terms)|database information|work\s*group|guideline development team|"
    r"acknowledg|conflicts?\s+of\s+interest|disclosure|author\s+(?:contributions|information)|"
    r"external\s+review|panel\s+members", re.I)
SEARCH_DB_CAPTION = re.compile(r"\b(?:EMBASE|PubMed|MEDLINE|CINAHL|PsycINFO|PsychINFO|Cochrane|Scopus|Web of Science)\b", re.I)
SEARCH_HEADER = re.compile(r"search\s+statement|controlled\s+vocabulary|\bset\b.{0,20}\bconcept\b", re.I | re.S)
AUTHOR_CELL = re.compile(r"^[a-z]?[A-Z][\w'’-]+,\s*[A-Z]\b")


# --------------------------------------------------------------------------- helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_llm_results(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
    elif raw.startswith("{") and "\n{" not in raw:
        obj = json.loads(raw)
        data = obj.get("results") or obj.get("tables") or [obj]
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [x for x in data if isinstance(x, dict) and x.get("table_id")]


def N(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", s))


LEVEL_TOKEN = re.compile(r"^(?:[IVX]{1,4}|[A-E]|\d{1,2})[a-z]?$")
SPACED_LETTERS = re.compile(r"(?:\b[A-Za-z]\s){8,}[A-Za-z]\b")
SHORT_CELL_CHARS = 45
MIN_FRAGMENT_WORDS = 12


def nonempty_rows(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in rows if any(c.strip() for c in r)]


def detect_has_header(rows: list[list[str]]) -> bool:
    """Row 0 is NOT a header when (1) the table has a single content row, or
    (2) rows 0 and 1 both start with a short level token (I, II, A, B, 1, 2a ...), e.g. evidence-level tables."""
    if len(nonempty_rows(rows)) <= 1:
        return False
    if len(rows) >= 2 and rows[0] and rows[1]:
        a, b = rows[0][0].strip(), rows[1][0].strip()
        if LEVEL_TOKEN.match(a) and LEVEL_TOKEN.match(b):
            return False
    return True


def data_rows(rows: list[list[str]], has_header: bool = True) -> list[list[str]]:
    if not rows:
        return []
    return rows[1:] if has_header else rows


def garbled_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    return sum(len(w) <= 2 for w in words) / max(1, len(words))


CLIP_STOP = frozenset("a an and as at be by do for if in is it no not of on or so the to up vs we with per via".split())


def has_clipped_cell(rows: list[list[str]]) -> bool:
    """A cell that starts with a lowercase 1-4 letter non-word ('nce for Neck Pain', 'ts if the') was cut
    mid-word by the extractor: the row is a clipped heading piece, not data."""
    for row in rows:
        for c in row:
            m = re.match(r"([a-z]+)\b", c.strip())
            if m and len(m.group(1)) <= 4 and m.group(1) not in CLIP_STOP:
                return True
    return False


def is_fragment(rec: dict[str, Any]) -> bool:
    """Headerless piece that carries no usable content: logos, 'Yes', single labels, or a header-only
    skeleton (every cell short, e.g. 'Intervention | Description of the intervention').
    Uses UNIQUE cells so a chain of repeated logo cells cannot look like content.
    A long single cell (e.g. a whole I-V evidence list collapsed into one cell) is real content."""
    cells = sorted({c.strip() for r in rec.get("rows") or [] for c in r if c.strip()})
    if not cells:
        return True
    if max(len(c) for c in cells) < SHORT_CELL_CHARS:
        return True
    return len(re.findall(r"\w+", " ".join(cells))) < MIN_FRAGMENT_WORDS


def noise_reason(rec: dict[str, Any]) -> str | None:
    """Checks that are safe to run on a SINGLE source record BEFORE continuation merging.
    (Size-based checks must wait until after merging, otherwise a short continuation page is
    dropped alone and breaks its chain.)"""
    rows = rec.get("rows") or []
    if rec.get("emitted_as_recommendation_chunks"):
        return "already_emitted_as_recommendation_chunks"
    flat = " ".join(" ".join(r) for r in rows)
    if SPACED_LETTERS.search(flat) and len(nonempty_rows(rows)) <= 1:
        return "spaced_letter_title"
    has_header = detect_has_header(rows)
    body = data_rows(rows, has_header)
    label = " ".join(filter(None, [rec.get("caption"), " ".join(rec.get("section_path") or [])]))
    if REFERENCE_CAPTION.search(label):
        return "reference_or_admin_table"
    # literature-search appendix tables: judged from the RAW header rows (never the placeholder row)
    head_text = " ".join(" ".join(r) for r in rows[:3])
    if SEARCH_HEADER.search(head_text) or SEARCH_DB_CAPTION.search(rec.get("caption") or ""):
        return "search_strategy_table"
    first_col = [r[0] for r in body if r and r[0].strip()]
    if len(first_col) >= 4 and sum(bool(AUTHOR_CELL.match(c.strip())) for c in first_col) / len(first_col) >= 0.4:
        return "reference_list_like"
    if len(re.findall(r"[A-Za-z]+", flat)) >= 12 and garbled_ratio(flat) > 0.45:
        return "garbled_extraction"
    return None


def size_reason(rec: dict[str, Any]) -> str | None:
    """Checks run on the MERGED logical table."""
    rows = rec.get("rows") or []
    has_header = rec.get("has_header", detect_has_header(rows))
    body = data_rows(rows, has_header)
    if not body or all(not any(c.strip() for c in r) for r in body):
        return "no_data_rows"
    if not has_header and is_fragment(rec):
        return "fragment_too_short"
    if not has_header:
        uniq = {c.strip() for r in rows for c in r if c.strip()}
        if len(uniq) <= 4 and has_clipped_cell(rows):
            return "clipped_header_fragment"
    if not has_header and int(rec.get("page") or 99) <= 2:
        return "front_matter_headerless"        # article cover / author block, not a clinical table
    return None


def exclusion_reason(rec: dict[str, Any]) -> str | None:
    return noise_reason(rec) or size_reason(rec)


def merge_continuations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge chains linked by continues_table_id into one logical table (first table's id is kept).
    Also decides has_header on the MERGED table and, for headerless tables, prepends an empty
    placeholder row 0 so real data always starts at index 1."""
    by_id = {r["table_id"]: r for r in records}
    child_of = {r["continues_table_id"]: r["table_id"] for r in records
                if r.get("continues_table_id") in by_id}
    heads = [r for r in records if r.get("continues_table_id") not in by_id]
    merged = []
    for head in heads:
        chain, cur = [head], head
        while cur["table_id"] in child_of:
            cur = by_id[child_of[cur["table_id"]]]
            chain.append(cur)
        rows = [list(r) for r in head["rows"]]
        row_pages = [head["page"]] * len(rows)
        for part in chain[1:]:
            part_rows = [list(r) for r in part["rows"]]
            if part_rows and head["rows"] and N(" | ".join(part_rows[0])) == N(" | ".join(head["rows"][0])):
                part_rows = part_rows[1:]            # drop repeated header
            rows += part_rows
            row_pages += [part["page"]] * len(part_rows)
        has_header = detect_has_header(rows)
        text = head.get("text") or ""
        if len(chain) > 1:
            text = " ".join((p.get("text") or "") for p in chain)
        if not has_header:
            width = max((len(r) for r in rows), default=1)
            rows = [[""] * width] + rows
            row_pages = [head["page"]] + row_pages
        merged.append({
            **head,
            "rows": rows, "row_pages": row_pages, "has_header": has_header, "text": text,
            "source_table_ids": [p["table_id"] for p in chain],
            "pages": sorted({p["page"] for p in chain}),
            "bboxes": [{"page": p["page"], "bbox": p["bbox"]} for p in chain],
            "footnote_candidates": [f for p in chain for f in (p.get("footnote_candidates") or [])],
            "merge_status": "merged_continuation" if len(chain) > 1 else "single_page",
        })
    return merged


def build_logical_tables(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """1) per-record noise filter (COR/LOE already emitted, references, garbled, spaced titles)
    2) merge continuation chains  3) size filter on the merged table (empty / fragment)."""
    excluded, pool = [], []
    for rec in records:
        reason = noise_reason(rec)
        (excluded if reason else pool).append({**rec, "_reason": reason} if reason else rec)
    kept = []
    for m in merge_continuations(pool):
        reason = size_reason(m)
        (excluded if reason else kept).append({**m, "_reason": reason} if reason else m)
    return kept, excluded


# --------------------------------------------------------------------------- prepare
def prepare(tables_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(tables_path)
    merged, excluded = build_logical_tables(records)
    with (out_dir / "table_chunk_inputs.jsonl").open("w", encoding="utf-8") as fh:
        for m in merged:
            fh.write(json.dumps({
                "table_id": m["table_id"], "doc_family": m.get("doc_family"),
                "organization": m.get("organization"), "year": m.get("year"),
                "section_path": m.get("section_path"), "caption": m.get("caption"),
                "context_before": m.get("context_before"),
                "footnote_candidates": m.get("footnote_candidates"),
                "has_header": m["has_header"],
                "header_row": m["rows"][0] if m["has_header"] else [], "rows": m["rows"],
                "pages": m.get("pages") or [m["page"]],
            }, ensure_ascii=False) + "\n")
    with (out_dir / "table_inputs_excluded.jsonl").open("w", encoding="utf-8") as fh:
        for e in excluded:
            fh.write(json.dumps({"table_id": e["table_id"], "doc_name": e["doc_name"], "page": e["page"],
                                 "reason": e["_reason"], "caption": e.get("caption"),
                                 "preview": (e.get("text") or "")[:200]}, ensure_ascii=False) + "\n")
    print(json.dumps({
        "source_records": len(records), "excluded": len(excluded),
        "excluded_by_reason": dict(Counter(e["_reason"] for e in excluded)),
        "llm_inputs_after_merge": len(merged),
        "headerless_tables": sum(not m["has_header"] for m in merged),
        "merged_multi_page_tables": sum(m["merge_status"] == "merged_continuation" for m in merged),
    }, indent=2))


# --------------------------------------------------------------------------- finalize
def validate_result(res: dict[str, Any], src: dict[str, Any]) -> tuple[list[str], dict[int, list[str]]]:
    """Return (table-level problems, chunk-index -> chunk problems)."""
    rows = src["rows"]
    n_data = len(rows) - 1
    problems: list[str] = []
    per_chunk: dict[int, list[str]] = {}
    covered = Counter()
    allowed_numbers = numbers(" ".join([
        src.get("caption") or "", " ".join(src.get("section_path") or []),
        " ".join(src.get("footnote_candidates") or []), " ".join(" ".join(r) for r in rows[:1]),
    ]))
    unparsed = [i for i in res.get("unparsed_rows", []) if isinstance(i, int)]
    for ci, ch in enumerate(res.get("chunks", [])):
        errs = []
        idx = [i for i in ch.get("source_row_indices", []) if isinstance(i, int)]
        if not idx:
            errs.append("no_source_row_indices")
        bad = [i for i in idx if i < 1 or i > n_data]
        if bad:
            errs.append(f"row_index_out_of_range:{bad[:5]}")
        covered.update(idx)
        text_n = N(ch.get("text", ""))
        for i in idx:
            if 1 <= i <= n_data and N(" | ".join(rows[i])) not in text_n:
                errs.append(f"text_missing_source_row:{i}")
                break
        src_nums = numbers(" ".join(" ".join(rows[i]) for i in idx if 1 <= i <= n_data)) | allowed_numbers
        body = "\n".join(ch.get("embedding_text", "").split("\n")[1:])
        extra = sorted(numbers(body) - src_nums)
        if extra:
            errs.append(f"embedding_numbers_not_in_source:{extra[:6]}")
        emb = ch.get("embedding_text", "")
        if not emb.strip():
            errs.append("empty_embedding_text")
        elif re.search(r"(?m)^Row:", emb):
            errs.append("embedding_text_not_sentences(Row: format)")
        if len(ch.get("text", "")) > 2500:
            errs.append("text_over_2500_chars")
        if errs:
            per_chunk[ci] = errs
    dup = [i for i, n in covered.items() if n > 1]
    missing = [i for i in range(1, n_data + 1)
               if i not in covered and i not in unparsed
               and any(c.strip() for c in rows[i])]
    if dup:
        problems.append(f"rows_in_multiple_chunks:{dup[:8]}")
    if missing:
        problems.append(f"rows_not_covered:{missing[:12]}")
    return problems, per_chunk


def finalize(tables_path: Path, llm_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(tables_path)
    kept, _ = build_logical_tables(records)
    merged = {m["table_id"]: m for m in kept}
    results = read_llm_results(llm_path)
    chunks, review, stats = [], [], Counter()
    seen_ids: set[str] = set()
    for res in results:
        src = merged.get(res["table_id"])
        if src is None:
            stats["llm_result_without_source"] += 1
            review.append({"table_id": res["table_id"], "problems": ["unknown_table_id"]})
            continue
        t_problems, c_problems = validate_result(res, src)
        if t_problems:
            review.append({"table_id": res["table_id"], "doc_name": src["doc_name"], "problems": t_problems})
        rows, row_pages = src["rows"], src.get("row_pages") or [src["page"]] * len(src["rows"])
        for ci, ch in enumerate(res.get("chunks", [])):
            idx = [i for i in ch.get("source_row_indices", []) if isinstance(i, int) and 1 <= i < len(rows)]
            pages = sorted({row_pages[i] for i in idx}) or [src["page"]]
            used_bboxes = [b for b in src.get("bboxes", []) if b["page"] in pages] or [{"page": src["page"], "bbox": src.get("bbox")}]
            errs = c_problems.get(ci, []) + (["table_level:" + p for p in t_problems] if t_problems else [])
            chunk_rows = ([rows[0]] if src.get("has_header", True) else []) + [rows[i] for i in idx]
            cid = hashlib.sha1(f"{src['table_id']}|{idx}|{N(ch.get('text',''))[:120]}".encode()).hexdigest()[:24]
            base_cid = cid; k = 1
            while cid in seen_ids:                      # same content twice: disambiguate
                cid = hashlib.sha1(f"{base_cid}|{k}".encode()).hexdigest()[:24]; k += 1
            seen_ids.add(cid)
            ok = not errs
            stats["chunks_ok" if ok else "chunks_needs_review"] += 1
            chunks.append({
                "chunk_id": cid,
                "text": ch.get("text", ""),
                "embedding_text": ch.get("embedding_text", ""),
                "metadata": {
                    "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                    "doc_name": src["doc_name"], "doc_family": src.get("doc_family"),
                    "doc_family_source": src.get("doc_family_source"),
                    "organization": src.get("organization"), "year": src.get("year"),
                    "page_start": min(pages), "page_end": max(pages),
                    "section_id": src.get("section_id"), "section_path": src.get("section_path", []),
                    "heading_level": src.get("heading_level"),
                    "chunk_type": "table", "table_type": res.get("table_type"),
                    "chunk_title": ch.get("chunk_title"),
                    "table_id": src["table_id"], "source_table_ids": src.get("source_table_ids", [src["table_id"]]),
                    "table_caption": src.get("caption"), "context_before": src.get("context_before"),
                    "footnote_candidates": src.get("footnote_candidates", []),
                    "table_header_candidate": (" | ".join(rows[0]) if rows and src.get("has_header", True) else None),
                    "has_header": src.get("has_header", True),
                    "source_row_indices": idx, "table_rows": chunk_rows,
                    "row_count": len(idx), "column_count": max((len(r) for r in chunk_rows), default=0),
                    "bbox": used_bboxes[0]["bbox"], "bboxes": used_bboxes,
                    "source_spans": [{"page": b["page"], "bbox": b["bbox"], "source_table_id": tid}
                                     for b in used_bboxes for tid in src.get("source_table_ids", [src["table_id"]])][:len(used_bboxes)],
                    "extraction_method": f"v2.15 sidecar ({src.get('cell_extraction')}) + LLM chunking",
                    "merge_status": src.get("merge_status"),
                    "token_count": max(1, len(ch.get("text", "")) // 4),
                    "token_count_method": "estimated_chars_div_4_NOT_MODEL_TOKEN_COUNT",
                    "validation_status": "auto_validated" if ok else "needs_review",
                    "validation_problems": errs,
                    "llm_warnings": res.get("warnings", []),
                    "embedding_eligible": ok,
                    "summary_status": "llm_generated",
                },
            })
    artifact = {
        "schema_version": SCHEMA_VERSION, "artifact_name": "table_chunks_json",
        "description": "LLM-chunked tables; all metadata attached from the extraction sidecar in code.",
        "summary": {"llm_results": len(results), "chunks": len(chunks), **dict(stats),
                    "tables_with_table_level_problems": len(review)},
        "chunks": chunks,
    }
    (out_dir / "table_chunks.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "table_chunks_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_inputs = sorted(set(merged) - {r["table_id"] for r in results})
    print(json.dumps({**artifact["summary"], "tables_without_llm_result": len(missing_inputs)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--tables", required=True); p.add_argument("--out-dir", required=True)
    f = sub.add_parser("finalize"); f.add_argument("--tables", required=True)
    f.add_argument("--llm-output", required=True); f.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(Path(a.tables), Path(a.out_dir))
    else:
        finalize(Path(a.tables), Path(a.llm_output), Path(a.out_dir))


if __name__ == "__main__":
    main()
