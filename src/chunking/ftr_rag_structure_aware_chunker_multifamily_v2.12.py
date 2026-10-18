#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FTR-RAG multi-family structure-aware PDF/JSON chunker.

Profiles: JOSPT/APTA, AHA/ESC, WHO, NICE, VA/DoD, ADA, and a conservative
GENERIC fallback. The family profiles change cues, not the underlying schema.

Dependencies:
    pip install pymupdf
Optional tokenizer:
    pip install tiktoken
    --tokenizer-encoding cl100k_base   # only if appropriate for the selected model

Important:
- Without a selected embedding tokenizer, token_count is an estimate, not a
  verified model-token count.
- PDF layout extraction is imperfect; review quality reports and sample chunks.
- Tables are extracted separately and placed into reading order by bbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("ftr_rag_structure_aware")
SCHEMA_VERSION = "2.1"
DEFAULT_CHAR_TARGET = 3000          # measurement baseline only
DEFAULT_CHAR_OVERLAP = 300
PAGE_EDGE_FRACTION = 0.14

FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "JOSPT_APTA": (
        "journal of orthopaedic & sports physical therapy",
        "journal of orthopaedic and sports physical therapy",
        "orthopaedic section of the american physical therapy association",
        "academy of oncologic physical therapy",
        "american physical therapy association",
        "apta clinical practice guideline",
        "clinical practice guidelines linked to the international classification",
    ),
    "AHA_ESC": (
        "american heart association", "american college of cardiology",
        "european society of cardiology", "esc guidelines",
        "aha/acc", "aha/acc/ada/asn",
    ),
    "WHO": (
        "world health organization", "package of interventions for rehabilitation",
        "who guideline", "who guidelines",
    ),
    "NICE": (
        "national institute for health and care excellence", "nice guideline",
        "nice guideline", "www.nice.org.uk",
    ),
    "VA_DOD": (
        "department of veterans affairs", "department of defense",
        "va/dod", "va/dod clinical practice guideline",
    ),
    "ADA": (
        "american diabetes association", "standards of care in diabetes",
        "diabetes care", "american diabetes association professional practice committee",
    ),
}

ORG_BY_FAMILY = {
    "JOSPT_APTA": "JOSPT/APTA",
    "AHA_ESC": "AHA/ACC/ESC",
    "WHO": "World Health Organization",
    "NICE": "NICE",
    "VA_DOD": "VA/DoD",
    "ADA": "American Diabetes Association",
    "GENERIC": None,
}

REFERENCE_HEADINGS = {
    "references", "reference", "bibliography", "literature cited",
    "references and notes", "works cited", "reference list",
    "references/bibliography",
}
APPENDIX_HEADINGS = {"appendix", "appendices", "supplementary appendix", "supplemental material"}

# Evidence grades are retained exactly as found in the text; scheme labels are
# only assigned when a recognizable explicit pattern is present.
GRADE_PATTERNS = [
    (re.compile(r"\b(?:COR\s*[:=]?\s*(?:Class\s+[I-V]+|[1-3])\s*[,;/]?\s*)?LOE\s*[:=]?\s*[A-E]\b", re.I), "COR/LOE"),
    (re.compile(r"\bCOR\s*[:=]?\s*(?:Class\s+[I-V]+|[1-3])\b", re.I), "COR"),
    (re.compile(r"\b(?:Strong|Conditional|Weak)\s+(?:recommendation|recommendations?)\b", re.I), "GRADE"),
    (re.compile(r"\b(?:High|Moderate|Low|Very low)\s+(?:certainty|quality)\s+of\s+evidence\b", re.I), "GRADE"),
    (re.compile(r"\bGRADE\s+(?:of\s+)?evidence\s*[:=]?\s*(?:High|Moderate|Low|Very low)\b", re.I), "GRADE"),
    (re.compile(r"\b(?:Evidence Level|Level of Evidence)\s*[:=]?\s*(?:[I-V]+|[1-5])\b", re.I), "LEVEL_OF_EVIDENCE"),
    (re.compile(r"\b(?:Grade|Strength of recommendation)\s*[:=]?\s*[A-F]\b", re.I), "LETTER_GRADE"),
]
# Deliberately explicit recommendation language. Broad cues such as "should"
# are avoided because they classify ordinary explanatory prose as recommendations.
RECOMMENDATION_CUES = (
    "we recommend", "we strongly recommend", "we conditionally recommend",
    "we suggest", "we conditionally suggest", "the panel recommends",
    "the panel suggests", "clinicians should", "is recommended",
    "are recommended", "recommendation:", "recommendations:",
    "strong recommendation", "conditional recommendation",
)
NUMBERED_HEADING = re.compile(
    r"^\s*((?:chapter|section)\s+)?(\d+(?:\.\d+){0,5})(?:[.)])?\s+(.{2,})$",
    re.I,
)
ROMAN_HEADING = re.compile(r"^\s*([IVX]{1,6})[.)]\s+(.{2,})$")
APPENDIX_HEADING = re.compile(r"^\s*(appendix|appendices|supplement(?:ary)?)\s*([A-Z0-9]*)\b", re.I)
BULLET_START = re.compile(r"^\s*(?:[-•▪◦*]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")
PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_pdf_span_text(spans: list[dict[str, Any]]) -> str:
    """Join positioned PDF spans without blindly concatenating adjacent words."""
    if not spans:
        return ""
    ordered = sorted(spans, key=lambda sp: (float((sp.get("bbox") or [0, 0])[0]), float((sp.get("bbox") or [0, 0, 0, 0])[1])))
    pieces: list[str] = []
    previous = None
    for sp in ordered:
        value = str(sp.get("text", ""))
        if not value:
            continue
        if previous is not None and pieces:
            prev_box = previous.get("bbox") or [0, 0, 0, 0]
            box = sp.get("bbox") or [0, 0, 0, 0]
            gap = float(box[0]) - float(prev_box[2]) if len(box) >= 4 and len(prev_box) >= 4 else 0.0
            prev_text = str(previous.get("text", ""))
            # A visible horizontal gap means a word boundary; avoid adding a
            # space around punctuation and preserve already-present whitespace.
            if gap > 0.8 and prev_text and value and not prev_text[-1].isspace() and not value[0].isspace():
                if value[0] not in ",.;:!?%)]}”’" and prev_text[-1] not in "([{“‘$":
                    pieces.append(" ")
        pieces.append(value)
        previous = sp
    return clean_text("".join(pieces))


def text_quality_flags(text: str) -> list[str]:
    flags = []
    if re.search(r"[A-Za-z]{18,}", text):
        flags.append("long_unspaced_token")
    if re.search(r"[A-Za-z]-\s+[a-z]{2,}", text):
        flags.append("possible_hyphenation_artifact")
    if re.search(r"(?:[A-Za-z][^\w\s]){5,}", text):
        flags.append("dense_punctuation_or_extraction_noise")
    if re.search(r"\b(?:cid:\d+|glyph|�)\b", text, re.I):
        flags.append("possible_glyph_extraction_error")
    return flags


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", clean_text(text).lower()).strip()


def _norm_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value).name.lower())


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load manifest.csv keyed by normalized filename; manifest is authoritative."""
    if path is None:
        return {}
    import csv
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dosya_adı", "doc_family", "organization", "year"}
        headers = set(reader.fieldnames or [])
        # Accept ASCII header spelling as well, without silently guessing columns.
        if not required.issubset(headers):
            aliases = {"dosya_adi": "dosya_adı", "filename": "dosya_adı", "file_name": "dosya_adı"}
            normalized_rows = []
            for raw in reader:
                converted = dict(raw)
                for old, new_name in aliases.items():
                    if old in converted and new_name not in converted:
                        converted[new_name] = converted[old]
                normalized_rows.append(converted)
            if not normalized_rows or not required.issubset(set(normalized_rows[0])):
                raise ValueError(
                    "manifest.csv şu sütunları içermeli: dosya_adı, doc_family, organization, year"
                )
            iterable = normalized_rows
        else:
            iterable = reader
        for raw in iterable:
            filename = str(raw.get("dosya_adı", "")).strip()
            family = str(raw.get("doc_family", "")).strip().upper()
            if not filename:
                continue
            if family not in {"JOSPT_APTA", "AHA_ESC", "WHO", "NICE", "VA_DOD", "ADA", "GENERIC"}:
                raise ValueError(f"Manifest'te geçersiz doc_family: {family!r} ({filename})")
            year_value = str(raw.get("year", "")).strip()
            rows[_norm_filename(filename)] = {
                "doc_family": family,
                "organization": str(raw.get("organization", "")).strip() or None,
                "year": int(year_value) if year_value.isdigit() else None,
            }
    return rows


def _manifest_record_for_name(name: str, manifest: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = _norm_filename(name)
    record = manifest.get(key)
    if record is not None:
        return record
    return next((value for manifest_key, value in manifest.items() if manifest_key.endswith(key)), None)


def infer_family(filename: str, first_page_text: str = "") -> str:
    """Conservative inference. Affiliation text is not used; ties return GENERIC."""
    stem = re.sub(r"[^a-z0-9]+", " ", Path(filename).stem.lower()).strip()
    # Only distinctive filename evidence is considered. The body/affiliation is
    # deliberately excluded because author affiliations caused false positives.
    distinctive = {
        "JOSPT_APTA": ("jospt", "apta clinical practice guideline", "physical therapist management"),
        "AHA_ESC": ("richards et al 2026", "aha acc", "esc guidelines", "european society of cardiology"),
        "WHO": ("who guideline", "who neurological", "musculosketal who", "world health organization"),
        "NICE": ("osteoarthritis in over 16s", "nice guideline", "national institute for health and care excellence"),
        "VA_DOD": ("vadod", "va dod", "va-dod"),
        "ADA": ("standards of care 2026", "standards of care in diabetes", "american diabetes association"),
    }
    matches = []
    for family, markers in distinctive.items():
        if any(re.sub(r"[^a-z0-9]+", " ", marker.lower()).strip() in stem for marker in markers):
            matches.append(family)
    return matches[0] if len(matches) == 1 else "GENERIC"


def infer_year(filename: str, text: str = "") -> int | None:
    # Extract only an explicit four-digit year in plausible publication range.
    candidates = re.findall(r"\b(19[89]\d|20[0-3]\d)\b", Path(filename).stem)
    if not candidates:
        candidates = re.findall(r"\b(19[89]\d|20[0-3]\d)\b", text[:5000])
    return int(candidates[0]) if candidates else None


def span_bold(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & 16) or "bold" in font or "black" in font or "demi" in font


def _extract_aha_recommendation_table(blocks: list[dict[str, Any]], page_num: int) -> list[dict[str, Any]]:
    """Recover AHA/ACC/ESC COR-LOE recommendation rows from positioned text blocks.

    This avoids PyMuPDF's costly generic table finder on dense two-column journal
    PDFs. It only emits a table when the explicit COR/LOE/Recommendations header
    and numbered rows are visible in the extracted page text.
    """
    headers = []
    for block in blocks:
        text = clean_text(block.get("text", ""))
        low = norm(text)
        compact = re.sub(r"\s+", "", low)
        if low in {"cor loe recommendations", "cor loe recommendation"} or compact in {"corloerecommendations", "corloerecommendation"}:
            headers.append(block)
    results = []
    for header_block in headers:
        hb = header_block.get("bbox") or [0, 0, 0, 0]
        y_header = float(hb[1])
        candidates = []
        for block in blocks:
            bbox = block.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = map(float, bbox)
            if y0 <= y_header + 7 or y0 > y_header + 340 or x0 < 105 or x0 > 305:
                continue
            text = clean_text(block.get("text", ""))
            m = re.match(r"^\s*(\d{1,2})[.)]?\s+(.{8,})$", text, re.S)
            if m:
                candidates.append((y0, int(m.group(1)), block, m.group(2).strip()))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            continue
        rows = [["COR", "LOE", "Recommendations"]]
        for idx, (y0, row_num, start_block, first_text) in enumerate(candidates):
            next_y = candidates[idx + 1][0] if idx + 1 < len(candidates) else y_header + 345
            rec_bits = [first_text]
            for block in blocks:
                bbox = block.get("bbox") or [0, 0, 0, 0]
                x0, by0, x1, by1 = map(float, bbox)
                txt = clean_text(block.get("text", ""))
                if block is start_block or not txt:
                    continue
                if x0 >= 110 and x1 <= 305 and y0 - 2 <= by0 < next_y:
                    if not re.match(r"^\s*\d{1,2}[.)]?\s+", txt):
                        rec_bits.append(txt)
            cor, loe = None, None
            for block in blocks:
                bbox = block.get("bbox") or [0, 0, 0, 0]
                x0, by0, x1, by1 = map(float, bbox)
                if x0 > 115 or not (y0 - 3 <= by0 < next_y):
                    continue
                txt = re.sub(r"\s+", " ", clean_text(block.get("text", ""))).strip()
                m = re.match(r"^(1|2a|2b|3)(?:\s*:\s*No\s+Benefit)?\s*(A|B-R|B-NR|B|C-LD|C-EO|C|D)$", txt, re.I)
                if m:
                    cor, loe = m.group(1), m.group(2).upper()
                    break
            rec_text = clean_text(" ".join(rec_bits))
            if rec_text:
                rows.append([cor or "", loe or "", f"{row_num}. {rec_text}"])
        if len(rows) > 1:
            all_boxes = [header_block.get("bbox") or [0, 0, 0, 0]]
            for _, _, block, _ in candidates:
                all_boxes.append(block.get("bbox") or [0, 0, 0, 0])
            bbox = [
                min(float(b[0]) for b in all_boxes), min(float(b[1]) for b in all_boxes),
                max(float(b[2]) for b in all_boxes), max(float(b[3]) for b in all_boxes),
            ]
            results.append({
                "table_index": len(results), "bbox": bbox, "rows": rows,
                "page": page_num, "source_kind": "table", "synthetic_from_positioned_text": True,
            })
    return results


def _page_extract(page: Any, page_num: int, extract_tables: bool = True, family: str = "GENERIC") -> dict[str, Any]:
    raw = page.get_text("dict", sort=True)
    blocks = []
    for raw_block in raw.get("blocks", []):
        if "lines" not in raw_block:
            continue
        spans, lines = [], []
        for line in raw_block.get("lines", []):
            line_spans = []
            for sp in line.get("spans", []):
                txt = sp.get("text", "")
                if not txt.strip():
                    continue
                span = {
                    "text": txt,
                    "size": round(float(sp.get("size", 0)), 2),
                    "bold": span_bold(sp),
                    "bbox": [round(float(v), 2) for v in sp.get("bbox", [])],
                    "page": page_num,
                    "font": str(sp.get("font", "")),
                }
                spans.append(span)
                line_spans.append(span)
            line = join_pdf_span_text(line_spans)
            if line:
                lines.append(line)
        text = clean_text("\n".join(lines))
        if text:
            blocks.append({
                "text": text,
                "bbox": [round(float(v), 2) for v in raw_block.get("bbox", [])],
                "spans": spans,
                "page": page_num,
                "block_no": raw_block.get("number"),
                "source_kind": "text",
            })

    tables = []
    if extract_tables:
        page_text = " ".join(block["text"] for block in blocks).lower()
        if family == "AHA_ESC":
            # Positioned-text reconstruction is substantially faster than find_tables()
            # on this journal layout and preserves explicit COR/LOE cells.
            tables.extend(_extract_aha_recommendation_table(blocks, page_num))
        else:
            table_cues = (
                "table ", "tablo ", "recommendations for", "evidence statement",
                "stage level study", "certainty of evidence", "strength of recommendation",
                "summary of findings", "evidence profile",
            )
            has_cor_loe = bool(re.search(r"\bcor\b", page_text) and re.search(r"\bloe\b", page_text))
            table_candidate = any(cue in page_text for cue in table_cues) or has_cor_loe
            if table_candidate:
                try:
                    finder = page.find_tables()
                    for index, table in enumerate(finder.tables):
                        rows = table.extract()
                        if rows:
                            tables.append({
                                "table_index": index,
                                "bbox": [round(float(v), 2) for v in table.bbox],
                                "rows": rows,
                                "page": page_num,
                                "source_kind": "table",
                            })
                except Exception as exc:
                    LOGGER.debug("Table extraction failed on page %d: %s", page_num, exc)

    return {
        "page": page_num,
        "width": round(float(page.rect.width), 2),
        "height": round(float(page.rect.height), 2),
        "text": "\n".join(b["text"] for b in blocks),
        "blocks": blocks,
        "tables": tables,
    }


def extract_pdf_path(path: Path, extract_tables: bool = True, family_override: str | None = None) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF desteği için `pip install pymupdf` çalıştır.") from exc
    family_hint = family_override or infer_family(path.name, "")
    with fitz.open(path) as pdf:
        pages = [_page_extract(page, i + 1, extract_tables, family_hint) for i, page in enumerate(pdf)]
    first_text = pages[0]["text"] if pages else ""
    family = family_override or infer_family(path.name, first_text)
    return {
        "schema_version": SCHEMA_VERSION, "document": path.name,
        "doc_version": None, "doc_family": family,
        "organization": ORG_BY_FAMILY.get(family),
        "year": infer_year(path.name, first_text),
        "extraction_method": "pymupdf_textdict_and_find_tables",
        "page_count": len(pages), "pages": pages,
    }


def extract_pdf_bytes(name: str, payload: bytes, extract_tables: bool = True, family_override: str | None = None) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF desteği için `pip install pymupdf` çalıştır.") from exc
    family_hint = family_override or infer_family(name, "")
    with fitz.open(stream=payload, filetype="pdf") as pdf:
        pages = [_page_extract(page, i + 1, extract_tables, family_hint) for i, page in enumerate(pdf)]
    first_text = pages[0]["text"] if pages else ""
    family = family_override or infer_family(name, first_text)
    return {
        "schema_version": SCHEMA_VERSION, "document": name,
        "doc_version": None, "doc_family": family,
        "organization": ORG_BY_FAMILY.get(family),
        "year": infer_year(name, first_text),
        "extraction_method": "pymupdf_textdict_and_find_tables",
        "page_count": len(pages), "pages": pages,
    }


def iter_documents(input_path: Path, extract_tables: bool = True, manifest: dict[str, dict[str, Any]] | None = None):
    """Yield one document at a time to avoid holding the entire PDF corpus in RAM."""
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if path.suffix.lower() == ".pdf":
                record = _manifest_record_for_name(path.name, manifest or {})
                yield path.name, extract_pdf_path(path, extract_tables, record.get("doc_family") if record else None)
            elif path.suffix.lower() == ".json":
                yield path.name, json.loads(path.read_text(encoding="utf-8"))
        return
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        record = _manifest_record_for_name(input_path.name, manifest or {})
        yield input_path.name, extract_pdf_path(input_path, extract_tables, record.get("doc_family") if record else None)
        return
    if suffix == ".json":
        yield input_path.name, json.loads(input_path.read_text(encoding="utf-8"))
        return
    if suffix == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            for name in sorted(n for n in zf.namelist() if not n.endswith("/")):
                ext = Path(name).suffix.lower()
                if ext == ".pdf":
                    basename = Path(name).name
                    record = _manifest_record_for_name(basename, manifest or {})
                    yield basename, extract_pdf_bytes(basename, zf.read(name), extract_tables, record.get("doc_family") if record else None)
                elif ext == ".json":
                    yield Path(name).name, json.loads(zf.read(name).decode("utf-8"))
        return
    raise ValueError("Girdi PDF, JSON, ZIP veya klasör olmalı.")


def load_documents(input_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Compatibility helper; use iter_documents for corpus-sized inputs."""
    return list(iter_documents(input_path))


def get_font_features(block: dict[str, Any]) -> tuple[float, bool, bool]:
    spans = block.get("spans", [])
    sizes = [float(s.get("size", 0)) for s in spans if s.get("size")]
    size = statistics.median(sizes) if sizes else 0.0
    bold_ratio = sum(bool(s.get("bold")) for s in spans) / len(spans) if spans else 0
    alpha = re.sub(r"[^A-Za-z]", "", clean_text(block.get("text", "")))
    uppercase = bool(alpha) and alpha.isupper()
    return size, bold_ratio >= 0.55, uppercase


def page_median_font(page: dict[str, Any]) -> float:
    sizes = [float(s.get("size", 0)) for b in page.get("blocks", [])
             for s in b.get("spans", []) if s.get("size")]
    return statistics.median(sizes) if sizes else 0.0


TOC_DOTTED = re.compile(r"\.{2,}|…{2,}|·{3,}")
TOC_TRAILING_PAGE = re.compile(r"(?:\s|\.{2,}|…+)\d{1,4}\s*$")
NICE_RECOMMENDATION = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+(.+)$")
SINGLE_GRADE = re.compile(r"^\s*[A-F]\s*$", re.I)
COR_LOE_ONLY = re.compile(r"^\s*(?:COR\s*)?(?:1|2a|2b|3|B-R|B-NR|C-LD|C-EO|A|B|C|D)\s*$", re.I)


def is_toc_line(text: str) -> bool:
    """Recognize TOC entries, including lines inside a multi-line PDF block."""
    stripped = clean_text(text)
    if not stripped:
        return False
    lines = [re.sub(r"\s+", " ", line).strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        lines = [re.sub(r"\s+", " ", stripped).strip()]
    return any(
        TOC_DOTTED.search(line)
        or (TOC_TRAILING_PAGE.search(line) and len(line.split()) >= 4)
        for line in lines
    )


def heading_candidate(block: dict[str, Any], family: str, median_size: float) -> tuple[str, int] | None:
    text = clean_text(block.get("text", ""))
    if not text or len(text) > 190 or is_toc_line(text):
        return None
    stripped = text.strip()
    flat = re.sub(r"\s+", " ", stripped)
    low = norm(flat)
    if SINGLE_GRADE.fullmatch(flat) or COR_LOE_ONLY.fullmatch(flat):
        return None
    if low in {
        "clinical statements and guidelines", "clinical statements and guidelines clinical statements and guidelines", "stage level study evidence statement",
        "stage level study evidence statements", "cor loe recommendations",
        "cor loe recommendation", "stage level study evidence statement recommendation",
    }:
        return None
    # Table-cell-like combinations must not become section headings.
    if re.fullmatch(r"(?:[1-3]|2a|2b)\s+(?:A|B-R|B-NR|C-LD|C-EO|C|D)(?:\s+.*)?", flat, re.I):
        return None
    # NICE three-level numbered recommendations are content, never headings.
    if family == "NICE" and NICE_RECOMMENDATION.match(stripped):
        return None
    if low in {norm(x) for x in REFERENCE_HEADINGS | APPENDIX_HEADINGS}:
        return stripped, 1
    appendix_match = APPENDIX_HEADING.match(stripped)
    numbered = NUMBERED_HEADING.match(stripped)
    size, bold, uppercase = get_font_features(block)
    sentence_like = stripped.endswith((".", ";", ",", "?", "!")) and len(stripped.split()) > 6
    word_count = len(stripped.split())
    if family == "AHA_ESC" and numbered:
        remainder = numbered.group(3).strip().lower()
        clinical_starts = ("in persons", "for persons", "in patients", "for patients", "patients with", "a symptom", "the optimal", "the use of", "among patients", "in individuals", "for individuals")
        if remainder.startswith(clinical_starts) or word_count >= 13:
            return None
    # "Appendix K contains..." is body prose, not an appendix heading.
    if appendix_match and word_count <= 12 and not sentence_like:
        return stripped, 1

    # Explicit 1.1 / 1.1.1 numbering can be either heading or content.
    # NICE three-level forms were excluded above; other families retain
    # numbered section headings when the line is not TOC-like.
    if numbered and not sentence_like and word_count <= 24:
        number = numbered.group(2)
        if family == "NICE" and number.count(".") >= 2:
            return None
        return stripped, number.count(".") + 1
    roman = ROMAN_HEADING.match(stripped)
    if roman and word_count <= 18 and not sentence_like:
        return stripped, 1

    if family == "JOSPT_APTA":
        if word_count <= 14 and not sentence_like and (bold or uppercase):
            return stripped, 1
    elif family == "AHA_ESC":
        # Repeated running titles are removed before this function. Table cell
        # blocks are excluded by their bbox in parse_document.
        if word_count <= 18 and not sentence_like and (bold or uppercase or (median_size and size >= median_size * 1.08)):
            return stripped, 1
    elif family == "WHO":
        if word_count <= 16 and not sentence_like and (bold or uppercase or re.match(r"^(module|annex|part)\b", low)):
            return stripped, 1
    elif family == "NICE":
        if word_count <= 18 and not sentence_like and (bold or uppercase or re.match(r"^(recommendations?|evidence|rationale|context|overview)\b", low)):
            return stripped, 1
    elif family == "VA_DOD":
        if word_count <= 18 and not sentence_like and (bold or uppercase or re.match(r"^(qualifying statements|introduction|recommendations|background)\b", low)):
            return stripped, 1
    elif family == "ADA":
        if word_count <= 18 and not sentence_like and (bold or uppercase or re.match(r"^(recommendations?|standards of care|table of contents)\b", low)):
            return stripped, 1
    else:
        if word_count <= 12 and not sentence_like and bold and (uppercase or (median_size and size >= median_size)):
            return stripped, 1
    return None


def detect_grade(text: str) -> tuple[str | None, str | None]:
    for pattern, system in GRADE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip(), system
    # A–F alone is not enough evidence to infer a grade system. The raw value
    # is kept only if explicitly labeled in the nearby text.
    return None, None


def stable_section_id(doc_name: str, section_path: list[str]) -> str:
    key = Path(doc_name).stem + "::" + " > ".join(section_path)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def table_to_text(rows: list[list[Any]]) -> str:
    output = []
    for row in rows:
        if row:
            output.append(" | ".join(clean_text(cell) for cell in row))
    return "\n".join(line for line in output if line.strip()).strip()


def _is_edge_block(block: dict[str, Any], page: dict[str, Any]) -> bool:
    bbox, height = block.get("bbox") or [], page.get("height")
    if not bbox or not height:
        return False
    return float(bbox[1]) < float(height) * PAGE_EDGE_FRACTION or float(bbox[3]) > float(height) * (1 - PAGE_EDGE_FRACTION)


def repeated_edge_lines(pages: list[dict[str, Any]]) -> set[str]:
    """Detect repeated one- or multi-block running headers/footers across pages."""
    page_signatures: Counter[str] = Counter()
    total_pages = max(1, len(pages))
    for page in pages:
        edge_blocks = [clean_text(b.get("text", "")) for b in page.get("blocks", [])
                       if _is_edge_block(b, page) and clean_text(b.get("text", ""))]
        # Each block plus adjacent pairs/triples catch multi-line running titles.
        signatures = set(norm(t) for t in edge_blocks if len(t) <= 180)
        for width in (2, 3):
            for i in range(max(0, len(edge_blocks) - width + 1)):
                joined = " ".join(edge_blocks[i:i + width])
                if len(joined) <= 220:
                    signatures.add(norm(joined))
        page_signatures.update(signatures)
    threshold = max(3, int(total_pages * 0.12))
    return {signature for signature, count in page_signatures.items() if count >= threshold}


def is_noise(block: dict[str, Any], page: dict[str, Any], repeated_lines: set[str]) -> bool:
    text = clean_text(block.get("text", ""))
    if not text or PAGE_NUMBER.fullmatch(text):
        return True
    if _is_edge_block(block, page):
        key = norm(text)
        if key in repeated_lines:
            return True
        # Also remove a single block that is a component of a repeated joined
        # multi-line header/footer; avoid removing long edge content.
        if len(text) < 140 and any(key and (key in sig or sig in key) for sig in repeated_lines if len(sig) > len(key)):
            return True
    return False


def bbox_inside(inner: list[float] | None, outer: list[float] | None, tolerance: float = 2.0) -> bool:
    if not inner or not outer or len(inner) < 4 or len(outer) < 4:
        return False
    return (float(inner[0]) >= float(outer[0]) - tolerance
            and float(inner[1]) >= float(outer[1]) - tolerance
            and float(inner[2]) <= float(outer[2]) + tolerance
            and float(inner[3]) <= float(outer[3]) + tolerance)


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Merge/split helper: overlap is used only when one source unit must split."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # Split one overlong source unit at paragraphs, then sentences, then hard cap.
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                # Hard split only as final fallback; overlap only in this case.
                start = 0
                while start < len(sentence):
                    end = min(len(sentence), start + max_chars)
                    units.append(sentence[start:end].strip())
                    if end >= len(sentence):
                        break
                    start = max(start + 1, end - max(0, overlap_chars))
    output, current = [], ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                output.append(current)
            current = unit
    if current:
        output.append(current)
    return output


def get_token_counter(encoding: str | None) -> tuple[Callable[[str], int], str]:
    if encoding:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("Tokenizer için `pip install tiktoken` gerekli.") from exc
        enc = tiktoken.get_encoding(encoding)
        return lambda s: len(enc.encode(s)), f"tiktoken:{encoding}"
    return lambda s: (max(1, round(len(s) / 4)) if s else 0), "estimated_chars_div_4_NOT_MODEL_TOKEN_COUNT"


def _toc_pages(pages: list[dict[str, Any]]) -> set[int]:
    """Identify TOC pages from line-level dotted leaders/page numbers and title cues."""
    result: set[int] = set()
    for page in pages:
        pno = int(page.get("page", 0) or 0)
        texts = [clean_text(b.get("text", "")) for b in page.get("blocks", [])]
        lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
        toc_lines = sum(is_toc_line(line) for line in lines)
        has_toc_title = any(norm(line) in {"contents", "table of contents", "contents list", "table of contents continued"} for line in lines)
        # A contents title plus one entry is sufficient; otherwise require several
        # independent TOC-like lines so an ordinary page number is not enough.
        if toc_lines >= 3 or (has_toc_title and toc_lines >= 1):
            result.add(pno)
    return result


def _table_cor_loe(row: list[Any], header: list[Any]) -> tuple[str | None, str | None, str | None]:
    cells = [clean_text(c) for c in row]
    headers = [norm(clean_text(c)) for c in header]
    cor_idx = next((i for i, h in enumerate(headers) if h in {"cor", "class of recommendation", "recommendation class"}), None)
    loe_idx = next((i for i, h in enumerate(headers) if h in {"loe", "level of evidence"}), None)
    # Some extractors return combined COR/LOE header cells. Those are handled
    # conservatively by the positioned-text reconstruction rather than guessing.
    # Fallback only for recognizable explicit values in a table that has COR/LOE labels.
    cor = cells[cor_idx] if cor_idx is not None and cor_idx < len(cells) else None
    loe = cells[loe_idx] if loe_idx is not None and loe_idx < len(cells) else None
    if cor:
        cor = re.sub(r"\s+", " ", cor).strip()
    if loe:
        loe = re.sub(r"\s+", " ", loe).strip()
    if cor and not re.fullmatch(r"(?:Class\s*)?(?:1|2a|2b|3|I|IIa|IIb|III)", cor, re.I):
        cor = None
    if loe and not re.fullmatch(r"(?:A|B-R|B-NR|B|C-LD|C-EO|C|D|Limited Data|Expert Opinion)", loe, re.I):
        loe = None
    rec_idx = next((i for i, h in enumerate(headers) if any(w in h for w in ("recommendation", "statement", "clinical implication"))), None)
    rec_text = cells[rec_idx] if rec_idx is not None and rec_idx < len(cells) else " | ".join(c for c in cells if c)
    if not rec_text or norm(rec_text) == norm(" | ".join(header)):
        return None, None, None
    return cor, loe, rec_text


def _is_recommendation_start(text: str, family: str) -> bool:
    """Conservative detector for explicitly signalled recommendation units.

    Numbered NICE/WHO/VA-DoD recommendation labels are preserved. For other
    families (notably ACR/ADA), only explicit recommendation wording is used;
    ordinary uses of should/may are not sufficient on their own.
    """
    cleaned = clean_text(text)
    low = cleaned.lower()
    if family == "NICE" and NICE_RECOMMENDATION.match(cleaned):
        return True
    if family in {"WHO", "VA_DOD"} and (
        re.search(r"\brecommendations?\s+\d+[a-z]?\b", low)
        or re.match(r"^(?:strong|conditional|weak)\s+recommendation\b", low)
    ):
        return True
    return any(cue in low for cue in RECOMMENDATION_CUES)


def parse_document(document: dict[str, Any], max_chars: int, overlap_chars: int,
                   tokenizer_encoding: str | None, exclude_references: bool) -> dict[str, Any]:
    """Parse structure, classify recommendation units, merge short prose, emit stable metadata."""
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise ValueError("JSON şeması 'pages' listesi içermeli.")
    doc_name = str(document.get("document") or "unknown_document")
    first_text = pages[0].get("text", "") if pages else ""
    family = document.get("doc_family") or infer_family(doc_name, first_text)
    organization = document.get("organization") or ORG_BY_FAMILY.get(family)
    year = document.get("year") or infer_year(doc_name, first_text)
    count_tokens, token_method = get_token_counter(tokenizer_encoding)
    repeated = repeated_edge_lines(pages)
    toc_pages = _toc_pages(pages)

    events: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    in_references = False
    in_appendix = False
    warnings: list[str] = []
    filename_lower = doc_name.lower()
    first_page_lower = clean_text(first_text).lower()
    if "anterior cruciate ligament" in filename_lower and "cardiopulmonary conditions" in first_page_lower[:1500]:
        warnings.append(
            "SOURCE_TITLE_MISMATCH: filename suggests ACL rehabilitation, but PDF title identifies WHO Module 4 Cardiopulmonary Conditions; this file is not valid evidence for an ACL-specific validation."
        )
    for page in pages:
        pno = int(page.get("page", 0) or 0)
        if pno in toc_pages:
            continue
        blocks = page.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            legacy_text = clean_text(page.get("text", ""))
            blocks = [{"text": legacy_text, "bbox": None, "spans": [], "page": pno}] if legacy_text else []
        median_size = page_median_font(page)
        tables = page.get("tables", []) or []
        table_boxes = [t.get("bbox") for t in tables if t.get("bbox")]
        merged = []
        for block in blocks:
            if is_noise(block, page, repeated):
                continue
            # Text inside a detected table is not a heading candidate and is not
            # separately emitted; the structured table extraction owns it.
            bbox = block.get("bbox") or [0, 0, 0, 0]
            if any(bbox_inside(bbox, tb) for tb in table_boxes):
                continue
            merged.append((float(bbox[1]), 0, "text", block))
        for table in tables:
            bbox = table.get("bbox") or [0, 0, 0, 0]
            merged.append((float(bbox[1]), 1, "table", table))
        merged.sort(key=lambda x: (x[0], x[1]))
        page_events: list[dict[str, Any]] = []

        for _, _, kind, obj in merged:
            if kind == "table":
                rows = obj.get("rows", [])
                table_text = table_to_text(rows)
                if not table_text:
                    continue
                caption = None
                # Use a preceding short table title as a section label only when
                # the wording explicitly signals a recommendation table.
                if page_events:
                    previous = page_events[-1]
                    ptxt = previous.get("text", "")
                    if previous.get("kind") in {"paragraph", "heading"} and re.match(
                        r"^\s*(?:recommendations?\s+for|table\s+\d+|table\s+[A-Z]\d*)\b", ptxt, re.I
                    ):
                        caption = ptxt
                        page_events.pop()
                        if re.match(r"^\s*recommendations?\s+for\b", caption, re.I):
                            stack.append({"title": caption, "level": (stack[-1]["level"] + 1 if stack else 1)})
                path = [entry["title"] for entry in stack]
                header = rows[0] if rows else []
                header_norm = " ".join(norm(clean_text(c)) for c in header)
                is_cor_loe_table = (
                    family == "AHA_ESC"
                    and ("cor" in header_norm or "class of recommendation" in header_norm)
                    and ("loe" in header_norm or "level of evidence" in header_norm)
                    and any("recommendation" in norm(clean_text(c)) or "statement" in norm(clean_text(c)) for c in header)
                )
                if is_cor_loe_table:
                    for row in rows[1:]:
                        cor, loe, rec_text = _table_cor_loe(row, header)
                        if not rec_text:
                            continue
                        # Do not create a recommendation from a repeated header or COR/LOE labels.
                        if norm(rec_text) in {norm(" ".join(clean_text(c) for c in header)), "recommendations"}:
                            continue
                        grade_parts = []
                        if cor:
                            grade_parts.append(f"COR {cor}")
                        if loe:
                            grade_parts.append(f"LOE {loe}")
                        events.append({
                            "kind": "recommendation", "text": rec_text,
                            "page_start": pno, "page_end": pno, "section_path": path,
                            "section_id": stable_section_id(doc_name, path),
                            "heading_level": stack[-1]["level"] if stack else 0,
                            "bbox": obj.get("bbox"), "source_spans": [],
                            "evidence_grade": ", ".join(grade_parts) if grade_parts else None,
                            "grade_system": "COR/LOE", "cor": cor, "loe": loe,
                            "recommendation_id": None, "embedding_eligible": True,
                            "table_caption": caption,
                        })
                else:
                    events.append({
                        "kind": "table", "text": table_text, "rows": rows,
                        "caption": caption, "page_start": pno, "page_end": pno,
                        "bbox": obj.get("bbox"), "section_path": path,
                        "section_id": stable_section_id(doc_name, path),
                        "heading_level": stack[-1]["level"] if stack else 0,
                        "evidence_grade": None, "grade_system": None,
                        "embedding_eligible": True,
                    })
                continue

            text = clean_text(obj.get("text", ""))
            if not text or is_toc_line(text):
                continue
            # JOSPT extraction sometimes places the letter grade and recommendation
            # in one multi-line PDF block; split the grade from recommendation text.
            jospt_inline_grade = re.match(
                r"^\s*([A-F])\s*\n+\s*((?:Clinicians?|We)\s+(?:should|may|should not|may not|recommend)\b.*)$",
                text, re.I | re.S
            ) if family == "JOSPT_APTA" else None
            if jospt_inline_grade:
                grade_value, recommendation_text = jospt_inline_grade.group(1).upper(), jospt_inline_grade.group(2).strip()
                path = [entry["title"] for entry in stack]
                page_events.append({
                    "kind": "recommendation", "text": recommendation_text,
                    "page_start": pno, "page_end": pno, "section_path": path,
                    "section_id": stable_section_id(doc_name, path),
                    "heading_level": stack[-1]["level"] if stack else 0,
                    "bbox": obj.get("bbox"), "source_spans": obj.get("spans", []),
                    "evidence_grade": grade_value, "grade_system": "JOSPT_LETTER",
                    "embedding_eligible": True,
                })
                continue

            # A standalone letter grade is an event, not a heading.
            if family == "JOSPT_APTA" and SINGLE_GRADE.fullmatch(text):
                path = [entry["title"] for entry in stack]
                page_events.append({
                    "kind": "grade_marker", "text": text.upper(), "page_start": pno, "page_end": pno,
                    "section_path": path, "section_id": stable_section_id(doc_name, path),
                    "heading_level": stack[-1]["level"] if stack else 0,
                    "bbox": obj.get("bbox"), "source_spans": obj.get("spans", []),
                    "evidence_grade": text.upper(), "grade_system": "JOSPT_LETTER",
                    "embedding_eligible": True,
                })
                continue

            # NICE numbered recommendations are content, never headings.
            nice_rec = NICE_RECOMMENDATION.match(text) if family == "NICE" else None
            if nice_rec:
                rid = nice_rec.group(1)
                body = text
                path = [entry["title"] for entry in stack]
                page_events.append({
                    "kind": "recommendation", "text": body, "page_start": pno, "page_end": pno,
                    "section_path": path, "section_id": stable_section_id(doc_name, path),
                    "heading_level": stack[-1]["level"] if stack else 0,
                    "bbox": obj.get("bbox"), "source_spans": obj.get("spans", []),
                    "evidence_grade": None, "grade_system": None, "recommendation_id": rid,
                    "embedding_eligible": True,
                })
                continue

            # AHA recommendation-table captions can share a PDF text block with
            # explanatory prose; only the caption belongs in section_path.
            aha_caption_tail = None
            if family == "AHA_ESC" and re.match(r"^\s*recommendations?\s+for\b", text, re.I) and "Referenced studies" in text:
                text, aha_caption_tail = re.split(r"(?=Referenced studies)", text, maxsplit=1, flags=re.I)
                obj = dict(obj)
                obj["text"] = text.strip()
            heading = heading_candidate(obj, family, median_size)
            if heading:
                htext, level = heading
                hnorm = norm(htext)
                is_refs = hnorm in {norm(x) for x in REFERENCE_HEADINGS}
                is_appendix = bool(APPENDIX_HEADING.match(htext)) or hnorm in {norm(x) for x in APPENDIX_HEADINGS}
                # Do not let any TOC-like line or a sentence mentioning an appendix
                # toggle document mode; only short, standalone appendix headings qualify.
                if is_appendix and (is_toc_line(htext) or len(htext.split()) > 12 or htext.rstrip().endswith((".", ";", ","))):
                    is_appendix = False
                if is_refs:
                    in_references, in_appendix = True, False
                    kind_name = "references"
                    while stack and stack[-1]["level"] >= 1:
                        stack.pop()
                    stack.append({"title": htext, "level": 1})
                elif is_appendix:
                    in_appendix, in_references = True, False
                    kind_name = "appendix"
                    while stack and stack[-1]["level"] >= 1:
                        stack.pop()
                    stack.append({"title": htext, "level": 1})
                else:
                    kind_name = "heading"
                    # Appendix mode ends only at a clearly numbered main section
                    # or references, not on arbitrary bold subheadings within appendix.
                    num_match = NUMBERED_HEADING.match(htext)
                    if in_appendix and num_match and num_match.group(2).count(".") == 0:
                        in_appendix = False
                    if level <= 1 and not in_appendix:
                        in_references = False
                    while stack and stack[-1]["level"] >= level:
                        stack.pop()
                    stack.append({"title": htext, "level": level})
                path = [entry["title"] for entry in stack]
                page_events.append({
                    "kind": kind_name, "text": htext, "page_start": pno, "page_end": pno,
                    "section_path": path, "section_id": stable_section_id(doc_name, path),
                    "heading_level": level, "bbox": obj.get("bbox"),
                    "source_spans": obj.get("spans", []),
                })
                if aha_caption_tail:
                    tail_text = clean_text(aha_caption_tail)
                    page_events.append({
                        "kind": "paragraph", "text": tail_text, "page_start": pno, "page_end": pno,
                        "section_path": path, "section_id": stable_section_id(doc_name, path),
                        "heading_level": level, "bbox": obj.get("bbox"),
                        "source_spans": obj.get("spans", []), "embedding_eligible": True,
                    })
                continue

            # Explicit recommendation language is recognized across families.
            # Keep this conservative: the sentence must look directive and use
            # a clinical recommendation verb, rather than merely mention a topic.
            if family in {"JOSPT_APTA", "GENERIC", "WHO", "VA_DOD", "AHA_ESC", "ADA"} and _is_recommendation_start(text, family):
                path = [entry["title"] for entry in stack]
                rid_match = re.match(r"^\s*(?:recommendation\s+)?(\d+(?:\.\d+)*(?:[A-Z])?)\b", text, re.I)
                page_events.append({
                    "kind": "recommendation", "text": text, "page_start": pno, "page_end": pno,
                    "section_path": path, "section_id": stable_section_id(doc_name, path),
                    "heading_level": stack[-1]["level"] if stack else 0,
                    "bbox": obj.get("bbox"), "source_spans": obj.get("spans", []),
                    "evidence_grade": None, "grade_system": None,
                    "recommendation_id": rid_match.group(1) if rid_match else None,
                    "embedding_eligible": True,
                })
                continue

            # Family-specific numbered labels remain available for WHO/VA-DoD.
            low = text.lower()
            if _is_recommendation_start(text, family):
                block_kind = "recommendation"
                rid_match = re.search(r"\brecommendations?\s+(\d+[A-Z]?)\b", text, re.I) if family in {"WHO", "VA_DOD"} else None
                rid = rid_match.group(1) if rid_match else None
            else:
                block_kind = "references" if in_references else ("appendix" if in_appendix else ("list_item" if BULLET_START.match(text) else "paragraph"))
                rid = None
            grade, grade_system = detect_grade(text)
            path = [entry["title"] for entry in stack]
            page_events.append({
                "kind": block_kind, "text": text, "page_start": pno, "page_end": pno,
                "section_path": path, "section_id": stable_section_id(doc_name, path),
                "heading_level": stack[-1]["level"] if stack else 0,
                "bbox": obj.get("bbox"), "source_spans": obj.get("spans", []),
                "evidence_grade": grade, "grade_system": grade_system,
                "recommendation_id": rid,
                "embedding_eligible": not (block_kind == "references" and exclude_references),
            })
        # NICE sometimes places the three-level recommendation identifier on its
        # own line/block, with the recommendation text immediately following.
        if family == "NICE":
            joined_page_events: list[dict[str, Any]] = []
            k = 0
            while k < len(page_events):
                current = page_events[k]
                bare_id = re.fullmatch(r"\s*(\d+\.\d+\.\d+)\s*", current.get("text", ""))
                if bare_id and current.get("kind") in {"paragraph", "list_item"} and k + 1 < len(page_events):
                    following = page_events[k + 1]
                    if following.get("kind") in {"paragraph", "list_item"} and following.get("section_id") == current.get("section_id"):
                        following = dict(following)
                        following["text"] = f"{bare_id.group(1)} {following.get('text', '')}".strip()
                        following["kind"] = "recommendation"
                        following["recommendation_id"] = bare_id.group(1)
                        following["page_start"] = min(current.get("page_start", pno), following.get("page_start", pno))
                        following["source_spans"] = current.get("source_spans", []) + following.get("source_spans", [])
                        joined_page_events.append(following)
                        k += 2
                        continue
                joined_page_events.append(current)
                k += 1
            page_events = joined_page_events
        events.extend(page_events)

    # JOSPT/APTA: join a standalone A-F marker to the following recommendation
    # paragraph(s), stopping at a new grade marker or heading/section boundary.
    rec_events: list[dict[str, Any]] = []
    i = 0
    while i < len(events):
        event = events[i]
        if family == "JOSPT_APTA" and event.get("kind") == "grade_marker":
            j = i + 1
            if j < len(events):
                nxt = events[j]
                starts = re.match(r"^\s*(?:clinicians?\s+(?:should|may|should not|may not)|patients?\s+should|we\s+recommend)\b",
                                  nxt.get("text", ""), re.I)
                same_section = nxt.get("section_id") == event.get("section_id")
                if starts and same_section and nxt.get("kind") in {"paragraph", "list_item"}:
                    group = [nxt]
                    j += 1
                    while j < len(events):
                        candidate = events[j]
                        if candidate.get("kind") in {"heading", "grade_marker", "table", "references", "appendix"}:
                            break
                        if candidate.get("section_id") != event.get("section_id"):
                            break
                        # Only attach explanatory text until the next grade/heading.
                        group.append(candidate)
                        j += 1
                    combined = dict(nxt)
                    combined["text"] = "\n\n".join(x["text"] for x in group)
                    combined["page_start"] = event.get("page_start", nxt["page_start"])
                    combined["page_end"] = max(x.get("page_end", nxt["page_end"]) for x in group)
                    combined["kind"] = "recommendation"
                    combined["evidence_grade"] = event["evidence_grade"]
                    combined["grade_system"] = "JOSPT_LETTER"
                    combined["source_spans"] = event.get("source_spans", []) + [sp for x in group for sp in x.get("source_spans", [])]
                    rec_events.append(combined)
                    i = j
                    continue
            warnings.append(f"JOSPT grade marker not attached to a recommendation at page {event.get('page_start')}")
            i += 1
            continue
        # Other family recommendation paragraphs are classified only when the
        # text has an explicit, family-supported marker.
        if family in {"WHO", "VA_DOD"} and event.get("kind") == "paragraph" and _is_recommendation_start(event.get("text", ""), family):
            event["kind"] = "recommendation"
            rid_match = re.match(r"^\s*recommendation\s+(\d+[A-Z]?)\b", event["text"], re.I)
            event["recommendation_id"] = rid_match.group(1) if rid_match else event.get("recommendation_id")
        rec_events.append(event)
        i += 1

    # Repair PDF block fragmentation inside a single recommendation.
    # Some PDFs split one recommendation across positioned blocks, and the
    # first block may be classified as recommendation while its continuation
    # is classified as ordinary prose. Join only high-confidence continuations:
    # same section, adjacent pages, no new recommendation ID, and either the
    # previous text is syntactically incomplete or the next block starts as a
    # lowercase continuation. Tables/headings/references are never absorbed.
    repaired_events: list[dict[str, Any]] = []
    for event in rec_events:
        if repaired_events:
            prev = repaired_events[-1]
            prev_text = clean_text(prev.get("text", ""))
            next_text = clean_text(event.get("text", ""))
            prev_kind = prev.get("kind")
            next_kind = event.get("kind")
            same_section = prev.get("section_id") == event.get("section_id")
            adjacent_pages = event.get("page_start", 0) <= prev.get("page_end", 0) + 1
            no_conflicting_id = not (
                prev.get("recommendation_id") and event.get("recommendation_id")
                and prev.get("recommendation_id") != event.get("recommendation_id")
            )
            incomplete_tail = bool(prev_text) and not re.search(r"[.!?;:]\s*[\"'’”)]*$", prev_text)
            lowercase_continuation = bool(next_text) and next_text[0].islower()
            continuation_kind = next_kind in {"paragraph", "list_item"}
            if (prev_kind == "recommendation" and continuation_kind and same_section
                    and adjacent_pages and no_conflicting_id and next_text
                    and (incomplete_tail or lowercase_continuation)
                    and len(prev_text) + len(next_text) + 2 <= max_chars):
                prev["text"] = prev_text + " " + next_text
                prev["page_end"] = max(prev.get("page_end", 0), event.get("page_end", 0))
                prev["source_spans"] = prev.get("source_spans", []) + event.get("source_spans", [])
                continue
        repaired_events.append(event)
    rec_events = repaired_events

    # Attach NICE bullet sub-items to the immediately preceding recommendation
    # while section/page context remains adjacent.
    if family == "NICE":
        attached: list[dict[str, Any]] = []
        for event in rec_events:
            if event.get("kind") == "list_item" and attached:
                prev = attached[-1]
                if prev.get("kind") == "recommendation" and prev.get("section_id") == event.get("section_id") and event.get("page_start", 0) <= prev.get("page_end", 0) + 1:
                    prev["text"] += "\n" + event["text"]
                    prev["page_end"] = max(prev.get("page_end", 0), event.get("page_end", 0))
                    continue
            attached.append(event)
        rec_events = attached

    # Merge adjacent prose/list blocks in the same section up to max_chars.
    # Tables, recommendations, references and appendix chunks are never merged
    # with other chunk types. Overlap is used only for forced splitting.
    merged_events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in rec_events:
        kind = event.get("kind", "paragraph")
        mergeable = kind in {"paragraph", "list_item", "appendix"}
        if not mergeable:
            if pending:
                merged_events.append(pending)
                pending = None
            merged_events.append(event)
            continue
        pending_kind = pending.get("kind") if pending else None
        same_merge_class = bool(pending) and (
            (pending_kind in {"paragraph", "list_item"} and kind in {"paragraph", "list_item"})
            or (pending_kind == "appendix" and kind == "appendix")
        )
        if pending and same_merge_class and pending.get("section_id") == event.get("section_id") and pending.get("page_end", 0) <= event.get("page_start", 0) + 1:
            combined_text = pending["text"] + "\n\n" + event["text"]
            if len(combined_text) <= max_chars:
                pending["text"] = combined_text
                pending["page_end"] = max(pending.get("page_end", 0), event.get("page_end", 0))
                pending["source_spans"] = pending.get("source_spans", []) + event.get("source_spans", [])
                if pending.get("kind") != event.get("kind"):
                    pending["kind"] = "paragraph"
                continue
            merged_events.append(pending)
            pending = dict(event)
        else:
            if pending:
                merged_events.append(pending)
            pending = dict(event)
    if pending:
        merged_events.append(pending)

    candidates: list[dict[str, Any]] = []
    for event in merged_events:
        kind = event.get("kind", "paragraph")
        if kind == "heading" or kind == "grade_marker":
            continue
        text = event.get("text", "")
        if not text:
            continue
        if kind == "table":
            rows = event.get("rows", [])
            if len(text) <= max_chars or len(rows) <= 1:
                parts = [(text, rows)]
            else:
                header = rows[0]
                parts, group = [], []
                for row in rows[1:]:
                    trial = group + [row]
                    if group and len(table_to_text([header] + trial)) > max_chars:
                        parts.append((table_to_text([header] + group), [header] + group))
                        group = [row]
                    else:
                        group = trial
                if group:
                    parts.append((table_to_text([header] + group), [header] + group))
            for part_text, part_rows in parts:
                candidates.append({**event, "text": part_text, "rows": part_rows, "kind": "table"})
        else:
            parts = split_text(text, max_chars, overlap_chars)
            for part in parts:
                candidates.append({**event, "text": part, "kind": kind})

    chunks = []
    for idx, item in enumerate(candidates, 1):
        chunk_id = f"{Path(doc_name).stem}_chunk_{idx:05d}"
        # Avoid embedding purely bibliographic/reference chunks by default. Cover/title
        # fragments are flagged rather than silently dropped, because first-page text
        # can contain clinically useful summary recommendations.
        if item.get("kind") == "references" and exclude_references:
            item["embedding_eligible"] = False
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "doc_name": doc_name,
            "doc_version": document.get("doc_version"),
            "doc_family": family,
            "organization": organization,
            "year": year,
            "section_id": item.get("section_id") or stable_section_id(doc_name, item.get("section_path", [])),
            "section_path": item.get("section_path", []),
            "heading_level": item.get("heading_level", 0),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "chunk_type": item.get("kind", "paragraph"),
            "evidence_grade": item.get("evidence_grade"),
            "grade_system": item.get("grade_system"),
            "recommendation_id": item.get("recommendation_id"),
            "cor": item.get("cor"),
            "loe": item.get("loe"),
            "token_count": count_tokens(item["text"]),
            "token_count_method": token_method,
            "embedding_eligible": item.get("embedding_eligible", True),
            "table_caption": item.get("caption") or item.get("table_caption"),
            "table_rows": item.get("rows") if item.get("kind") == "table" else None,
            "bbox": item.get("bbox"),
            "source_spans": item.get("source_spans", []),
        }
        chunks.append({"chunk_id": chunk_id, "text": item["text"], "metadata": metadata})
    for idx, chunk in enumerate(chunks):
        chunk["metadata"]["prev_chunk_id"] = chunks[idx - 1]["chunk_id"] if idx else None
        chunk["metadata"]["next_chunk_id"] = chunks[idx + 1]["chunk_id"] if idx + 1 < len(chunks) else None

    token_counts = [c["metadata"]["token_count"] for c in chunks]
    # Compare chunk text against parsed body events, not raw page text: raw text
    # includes intentionally skipped TOC pages, running furniture, and headings
    # that are stored in section_path rather than repeated in chunk text.
    source_text = "\n".join(
        clean_text(event.get("text", "")) for event in events
        if event.get("kind") not in {"heading", "grade_marker"}
    )
    chunk_text = "\n".join(c["text"] for c in chunks)
    source_words, chunk_words = set(norm(source_text).split()), set(norm(chunk_text).split())
    missing_words = sorted(source_words - chunk_words)
    types = Counter(c["metadata"]["chunk_type"] for c in chunks)
    recommendation_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "recommendation"]
    grade_filled = sum(bool(c["metadata"].get("evidence_grade")) for c in recommendation_chunks)
    small = [c for c in chunks if c["metadata"]["token_count"] < 30]
    non_atomic = [c for c in chunks if c["metadata"]["chunk_type"] not in {"recommendation", "table", "references"}]
    small_non_atomic = [c for c in non_atomic if c["metadata"]["token_count"] < 30]
    appendix_chunks = sum(c["metadata"]["chunk_type"] == "appendix" for c in chunks)
    # Quality diagnostics: "orphan" is explicitly defined as a chunk with no
    # section_path. This is a structural warning, not proof that its text is bad.
    orphan_chunks = [c for c in chunks if not c["metadata"].get("section_path")]
    no_source_span_chunks = [c for c in chunks if not c["metadata"].get("source_spans")]
    no_page_chunks = [c for c in chunks if c["metadata"].get("page_start") is None]
    recommendation_without_id = [c for c in recommendation_chunks if not c["metadata"].get("recommendation_id")]
    normalized_texts = [norm(c.get("text", "")) for c in chunks]
    exact_text_counts = Counter(t for t in normalized_texts if t)
    duplicate_extra_count = sum(n - 1 for n in exact_text_counts.values() if n > 1)
    char_lengths = [len(c.get("text", "")) for c in chunks]
    broken_text_chunks = []
    noisy_heading_chunks = []
    likely_missed_recommendation_chunks = []
    for c in chunks:
        ctext = str(c.get("text", ""))
        cm = c.get("metadata", {})
        flags = text_quality_flags(ctext)
        if flags:
            broken_text_chunks.append({
                "chunk_id": c.get("chunk_id"), "page_start": cm.get("page_start"),
                "chunk_type": cm.get("chunk_type"), "flags": flags,
                "text_preview": re.sub(r"\s+", " ", ctext)[:220],
            })
        path = cm.get("section_path") or []
        if any(text_quality_flags(str(h)) or len(str(h).split()) > 18 or re.search(r"\b(?:JOURNAL OF|PAGE \d+|AUTHOR CONTRIBUTIONS|EXTERNAL REVIEW)\b", str(h), re.I) for h in path):
            noisy_heading_chunks.append({
                "chunk_id": c.get("chunk_id"), "page_start": cm.get("page_start"),
                "section_path": path, "text_preview": re.sub(r"\s+", " ", ctext)[:160],
            })
        if cm.get("chunk_type") in {"paragraph", "list_item"} and re.search(r"\b(?:clinicians?|physical therapists?|therapists?|patients?)\s+(?:should|must|may|need to|are advised to)\b|\bwe recommend\b", ctext, re.I):
            likely_missed_recommendation_chunks.append({
                "chunk_id": c.get("chunk_id"), "page_start": cm.get("page_start"),
                "text_preview": re.sub(r"\s+", " ", ctext)[:220],
            })
    token_bins = {
        "under_10": [c for c in chunks if c["metadata"]["token_count"] < 10],
        "under_20": [c for c in chunks if c["metadata"]["token_count"] < 20],
        "under_30": [c for c in chunks if c["metadata"]["token_count"] < 30],
        "under_50": [c for c in chunks if c["metadata"]["token_count"] < 50],
    }
    non_atomic_bins = {
        "under_10": [c for c in non_atomic if c["metadata"]["token_count"] < 10],
        "under_20": [c for c in non_atomic if c["metadata"]["token_count"] < 20],
        "under_30": [c for c in non_atomic if c["metadata"]["token_count"] < 30],
        "under_50": [c for c in non_atomic if c["metadata"]["token_count"] < 50],
    }
    def ratio(count: int, total: int) -> float:
        return round(count / total, 4) if total else 0.0
    def sample_chunks(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        return [{
            "chunk_id": c.get("chunk_id"),
            "chunk_type": c.get("metadata", {}).get("chunk_type"),
            "token_count": c.get("metadata", {}).get("token_count"),
            "char_count": len(c.get("text", "")),
            "page_start": c.get("metadata", {}).get("page_start"),
            "section_path": c.get("metadata", {}).get("section_path", []),
            "text_preview": re.sub(r"\s+", " ", c.get("text", ""))[:180],
        } for c in items[:limit]]
    stats = {
        "chunk_count": len(chunks),
        "chunk_type_distribution": dict(types),
        "short_under_30_count": len(small),
        "short_under_30_ratio": round(len(small) / len(chunks), 4) if chunks else 0,
        "short_under_30_excluding_atomic_count": len(small_non_atomic),
        "short_under_30_excluding_atomic_ratio": round(len(small_non_atomic) / len(non_atomic), 4) if non_atomic else 0,
        "small_chunk_thresholds_all_chunks": {k: {"count": len(v), "ratio": ratio(len(v), len(chunks))} for k, v in token_bins.items()},
        "small_chunk_thresholds_excluding_atomic": {k: {"count": len(v), "ratio": ratio(len(v), len(non_atomic))} for k, v in non_atomic_bins.items()},
        "median_chars": round(statistics.median(char_lengths), 2) if char_lengths else 0,
        "average_chars": round(statistics.mean(char_lengths), 2) if char_lengths else 0,
        "min_chars": min(char_lengths) if char_lengths else 0,
        "max_chars": max(char_lengths) if char_lengths else 0,
        "median_tokens": round(statistics.median(token_counts), 2) if token_counts else 0,
        "median_tokens_excluding_atomic": round(statistics.median([c["metadata"]["token_count"] for c in non_atomic]), 2) if non_atomic else 0,
        "average_tokens": round(statistics.mean(token_counts), 2) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "chunks_over_char_target": sum(len(c["text"]) > max_chars for c in chunks),
        "recommendation_chunk_count": len(recommendation_chunks),
        "recommendation_evidence_grade_filled_ratio": round(grade_filled / len(recommendation_chunks), 4) if recommendation_chunks else None,
        "reference_chunks_excluded_from_embedding": sum(c["metadata"]["chunk_type"] == "references" and not c["metadata"]["embedding_eligible"] for c in chunks),
        "table_chunk_count": types.get("table", 0),
        "appendix_chunk_count": appendix_chunks,
        "section_path_top5": Counter(" > ".join(c["metadata"]["section_path"]) or "(no heading)" for c in chunks).most_common(5),
        "chunks_missing_section_id": sum(not c["metadata"]["section_id"] for c in chunks),
        "orphan_chunk_definition": "chunk with empty section_path; structural diagnostic, not necessarily an error",
        "orphan_chunk_count": len(orphan_chunks),
        "orphan_chunk_ratio": ratio(len(orphan_chunks), len(chunks)),
        "orphan_chunk_samples": sample_chunks(orphan_chunks),
        "chunks_without_source_spans_count": len(no_source_span_chunks),
        "chunks_without_page_metadata_count": len(no_page_chunks),
        "recommendation_chunks_without_recommendation_id_count": len(recommendation_without_id),
        "duplicate_normalized_text_extra_chunks_count": duplicate_extra_count,
        "duplicate_normalized_text_groups_sample": [{"count": n, "text_preview": t[:180]} for t, n in exact_text_counts.most_common() if n > 1][:10],
        "source_words_not_found_in_chunks_count": len(missing_words),
        "source_words_not_found_in_chunks_sample": missing_words[:100],
        "text_coverage_check": "diagnostic_unique_word_set_not_proof_of_zero_loss",
        "broken_text_chunk_count": len(broken_text_chunks),
        "broken_text_chunk_samples": broken_text_chunks[:100],
        "noisy_heading_path_chunk_count": len(noisy_heading_chunks),
        "noisy_heading_path_samples": noisy_heading_chunks[:100],
        "possible_missed_recommendation_count": len(likely_missed_recommendation_chunks),
        "possible_missed_recommendation_samples": likely_missed_recommendation_chunks[:100],
        "recommendation_detection_note": "Possible missed recommendations are heuristic candidates; review samples before treating them as false negatives.",
        "warnings": warnings,
        "toc_pages_skipped": sorted(toc_pages),
        "repeated_header_footer_signatures_removed": len(repeated),
        "family_inference_warning": "GENERIC profile: no unambiguous family evidence" if family == "GENERIC" else None,
        "recommendation_detection_warning": "No recommendation chunks detected" if not recommendation_chunks else None,
        "quality_flags": [
            flag for condition, flag in [
                (bool(orphan_chunks), f"{len(orphan_chunks)} chunk(s) have empty section_path"),
                (bool(small_non_atomic) and ratio(len(small_non_atomic), len(non_atomic)) >= 0.25, "At least 25% of non-atomic chunks are under 30 estimated tokens"),
                (bool(recommendation_without_id), f"{len(recommendation_without_id)} recommendation chunk(s) have no recommendation_id"),
                (duplicate_extra_count > 0, f"{duplicate_extra_count} extra chunk(s) duplicate normalized text"),
                (bool(no_source_span_chunks), f"{len(no_source_span_chunks)} chunk(s) have no source_spans"),
                (bool(broken_text_chunks), f"{len(broken_text_chunks)} chunk(s) show possible text extraction artifacts"),
                (bool(noisy_heading_chunks), f"{len(noisy_heading_chunks)} chunk(s) have potentially noisy heading paths"),
                (bool(likely_missed_recommendation_chunks), f"{len(likely_missed_recommendation_chunks)} prose chunk(s) may contain missed recommendations; manual review required"),
            ] if condition
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "chunker_version": "structure-aware-multifamily-2.12",
        "document": doc_name, "doc_family": family, "organization": organization, "year": year,
        "token_count_method": token_method, "statistics": stats, "chunks": chunks,
    }


def _aggregate_quality(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-document quality counters without averaging percentages blindly."""
    keys = [
        "chunk_count", "short_under_30_count", "short_under_30_excluding_atomic_count",
        "chunks_over_char_target", "recommendation_chunk_count", "table_chunk_count",
        "chunks_missing_section_id", "orphan_chunk_count", "chunks_without_source_spans_count",
        "chunks_without_page_metadata_count", "recommendation_chunks_without_recommendation_id_count",
        "duplicate_normalized_text_extra_chunks_count", "broken_text_chunk_count",
        "noisy_heading_path_chunk_count", "possible_missed_recommendation_count",
    ]
    totals = {key: sum(int(doc.get(key, 0) or 0) for doc in summaries) for key in keys}
    totals["small_chunk_thresholds_all_chunks"] = {}
    totals["small_chunk_thresholds_excluding_atomic"] = {}
    for stat_key in ("small_chunk_thresholds_all_chunks", "small_chunk_thresholds_excluding_atomic"):
        for threshold in ("under_10", "under_20", "under_30", "under_50"):
            count = sum(int(doc.get(stat_key, {}).get(threshold, {}).get("count", 0) or 0) for doc in summaries)
            denominator = totals["chunk_count"] if stat_key == "small_chunk_thresholds_all_chunks" else sum(
                int(doc.get("chunk_count", 0) or 0) - int(doc.get("recommendation_chunk_count", 0) or 0)
                - int(doc.get("table_chunk_count", 0) or 0)
                - int(doc.get("chunk_type_distribution", {}).get("references", 0) or 0)
                for doc in summaries
            )
            totals[stat_key][threshold] = {"count": count, "ratio": round(count / denominator, 4) if denominator > 0 else 0.0}
    totals["orphan_chunk_ratio"] = round(totals["orphan_chunk_count"] / totals["chunk_count"], 4) if totals["chunk_count"] else 0.0
    return totals


def run(input_path: Path, output_dir: Path, max_chars: int, overlap_chars: int,
        tokenizer_encoding: str | None, exclude_references: bool,
        extract_tables: bool = True, manifest_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        candidates = [input_path / "manifest.csv"] if input_path.is_dir() else []
        candidates.append(input_path.parent / "manifest.csv")
        # A single PDF may sit one level below a corpus folder that owns manifest.csv.
        candidates.append(input_path.parent.parent / "manifest.csv")
        manifest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    manifest = load_manifest(manifest_path)
    summaries, failures = [], []
    progress_path = output_dir / "_progress.jsonl"
    report_path = output_dir / "_quality_report.json"
    if progress_path.exists():
        progress_path.unlink()
    total_documents = 0
    for name, document in iter_documents(input_path, extract_tables, manifest):
        total_documents += 1
        started = __import__("time").time()
        try:
            # Manifest is authoritative and overrides all inferred/source metadata.
            manifest_record = _manifest_record_for_name(name, manifest)
            if manifest_record:
                document["doc_family"] = manifest_record["doc_family"]
                document["organization"] = manifest_record["organization"]
                document["year"] = manifest_record["year"]
                document["manifest_applied"] = True
            else:
                document["manifest_applied"] = False

            result = parse_document(document, max_chars, overlap_chars, tokenizer_encoding, exclude_references)
            output = output_dir / f"{Path(name).stem}.chunks.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {
                "document": name, "doc_family": result["doc_family"], "organization": result["organization"],
                "year": result["year"], "manifest_applied": bool(manifest_record),
                **result["statistics"], "token_count_method": result["token_count_method"],
                "elapsed_seconds": round(__import__("time").time() - started, 2),
                "output": str(output),
            }
            summaries.append(summary)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"status": "success", **summary}, ensure_ascii=False) + "\n")
            LOGGER.info("[%d] %s | family=%s | chunks=%s | %.1fs",
                        total_documents, name, result["doc_family"], result["statistics"]["chunk_count"],
                        summary["elapsed_seconds"])
        except Exception as exc:
            LOGGER.exception("Document failed: %s", name)
            failure = {"document": name, "error": repr(exc)}
            failures.append(failure)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"status": "failed", **failure}, ensure_ascii=False) + "\n")
        # Persist a partial report after every document so interrupted corpus runs
        # still retain their completed work and progress.
        partial = {
            "schema_version": SCHEMA_VERSION, "chunker_version": "structure-aware-multifamily-2.12",
            "input": str(input_path), "manifest": str(manifest_path) if manifest_path else None,
            "total_documents_seen": total_documents, "successful_documents": len(summaries),
            "failed_documents": failures, "documents": summaries,
            "family_distribution": dict(Counter(x["doc_family"] for x in summaries)),
            "corpus_quality_totals": _aggregate_quality(summaries),
        }
        report_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    family_counts = Counter(s["doc_family"] for s in summaries)
    report = {
        "schema_version": SCHEMA_VERSION,
        "chunker_version": "structure-aware-multifamily-2.12",
        "input": str(input_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "total_documents": total_documents,
        "successful_documents": len(summaries),
        "failed_documents": failures,
        "family_distribution": dict(family_counts),
        "documents": summaries,
        # Convenient top-level totals requested for quick inspection.
        "total_chunks": _aggregate_quality(summaries)["chunk_count"],
        "total_orphan_chunks": _aggregate_quality(summaries)["orphan_chunk_count"],
        "orphan_chunk_ratio": _aggregate_quality(summaries)["orphan_chunk_ratio"],
        "corpus_quality_totals": _aggregate_quality(summaries),
        "limitations": [
            "Without --tokenizer-encoding, token_count is an estimate and not the embedding model's exact token count.",
            "PDF table extraction depends on layout; extracted rows require visual spot-checking.",
            "Recommendation extraction is family-specific but remains heuristic; report unclassified/low-confidence cases.",
            "Word coverage is diagnostic and cannot prove zero text loss; it also does not compare source order.",
            "GENERIC means family inference was ambiguous or unsupported; add an explicit manifest row to resolve it.",
            "Random recommendation audit against page images is not automated in this script; the generated report provides page_start/page_end for manual audit.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="FTR-RAG multi-family structure-aware chunker")
    parser.add_argument("--input", required=True, help="PDF / JSON / ZIP / directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None, help="Optional manifest.csv: dosya_adı, doc_family, organization, year")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_CHAR_TARGET,
                        help="Character target; not a model-token limit")
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_CHAR_OVERLAP)
    parser.add_argument("--tokenizer-encoding", default=None,
                        help="Optional tiktoken encoding; use only after model tokenizer is confirmed")
    parser.add_argument("--include-references-in-embedding", action="store_true",
                        help="By default references are retained but embedding_eligible=false")
    parser.add_argument("--skip-table-extraction", action="store_true",
                        help="Faster diagnostic run; table quality cannot be evaluated")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    report = run(
        Path(args.input), Path(args.output_dir), args.max_chars, args.overlap_chars,
        args.tokenizer_encoding, not args.include_references_in_embedding,
        not args.skip_table_extraction, Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps({
        "total_documents": report["total_documents"],
        "successful_documents": report["successful_documents"],
        "failed_documents": len(report["failed_documents"]),
        "total_chunks": report["total_chunks"],
        "total_orphan_chunks": report["total_orphan_chunks"],
        "orphan_chunk_ratio": report["orphan_chunk_ratio"],
        "family_distribution": report["family_distribution"],
        "quality_report": str(Path(args.output_dir) / "_quality_report.json"),
        "progress_log": str(Path(args.output_dir) / "_progress.jsonl"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
