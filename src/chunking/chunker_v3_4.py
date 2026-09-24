from pathlib import Path
from collections import Counter
import json
import re
from statistics import mean, median

"""
FTR-RAG CHUNKER v3.4

FINAL CLEANUP VERSION BEFORE RETRIEVAL
--------------------------------------
Primary strategy:
    Structure-Aware Hierarchical Chunking
    + controlled paragraph/sentence boundaries
    + fixed-size safety cap
    + small overlap
    + contextual metadata
    + table-aware handling
    + ICF functioning-domain-aware splitting inside "Overview of
      interventions" tables (v3.3)

v3.2 fixed quality problems found in v3.1. v3.3 adds exactly one new thing on
top of that: splitting the WHO "Overview of the interventions" tables by ICF
functioning domain (see FUNCTIONING_DOMAIN_HEADERS below), because those
tables were confirmed (FTR-005/008/010) to merge several unrelated domains
into a single chunk, diluting the exact terms a narrow query needs for both
BM25 and dense retrieval. Empirically, 142 chunks across the two sample
documents contain at least one of these domain headers, and 83 of those
genuinely mix 2+ distinct domains in one chunk; after this fix they split
into 325 single-domain chunks. This does not touch the detailed
per-intervention resource tables' internal "Target: ..." sub-rows (Format B)
-- only their outer domain heading, when present on its own line/line-run --
that finer split is a separate follow-up.

Main fixes:
1. Repeated-layout detection only considers page-edge lines. Common clinical
   body terms are never removed merely because they repeat across pages.
2. Numbered-heading detection is more conservative against author lists,
   citations, years, article metadata and strings such as "DCD CPG 297".
3. Table extraction is validated. Paragraph-like false-positive tables are
   retained as text-like chunks instead of being labelled as tables.
4. Very small text chunks are reported as orphan candidates rather than being
   silently discarded.
5. A conservative fallback repairs pages whose extracted text is clearly
   word-per-line layout noise. This is not a replacement for better PDF
   extraction; it only handles an obvious extraction failure mode.
6. QA reports include the new diagnostics so the chunker can be frozen based
   on evidence rather than a single chunk count.
7. (v3.3) "Overview of interventions" tables are split by ICF functioning
   domain instead of being kept as one page-wide block, gated to the
   "Content of the Package of interventions..." subsection so narrative text
   that merely mentions a domain by name (glossary text, References titles)
   is never affected.

Out of scope:
- true embedding-based semantic chunking
- classic sliding-window chunking
- parent-child retrieval
- LLM-generated chunk boundaries/summaries
- reranker
- knowledge graph
- conflict resolution / source prioritization
- splitting the detailed per-intervention resource tables' "Target:" rows
"""

def find_project_root():
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for candidate in candidates:
        if (candidate / "data" / "processed" / "extracted_text").exists():
            return candidate
    # Fallback for a freshly created project before the directories exist.
    if here.parent.name == "chunking" and here.parent.parent.name == "src":
        return here.parents[2]
    return here.parent


BASE_DIR = find_project_root()
INPUT_DIR = BASE_DIR / "data" / "processed" / "extracted_text"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "chunks"

TARGET_CHARS = 3000
OVERLAP_CHARS = 300
MIN_TEXT_CHARS = 300
MIN_TABLE_CHARS = 60
LONG_CHUNK_FACTOR = 1.35
REPEATED_LAYOUT_MIN_COUNT = 3
EDGE_LINES = 3
ORPHAN_TOKEN_THRESHOLD = 150
WORD_PER_LINE_RATIO = 0.65

COMMON_HEADINGS = {
    "abstract", "introduction", "background", "methods", "methodology",
    "results", "discussion", "conclusion", "recommendations", "references",
    "appendix", "scope", "purpose", "objectives", "assessment",
    "outcome measures", "safety", "limitations", "patient population",
    "target population", "risk factors", "overview", "process", "definitions",
    "implementation", "clinical implications", "strengths and limitations",
    "future research", "literature searches", "best-evidence synthesis",
    "peer review and public commentary", "exclusions", "feasibility",
    "role of patient preferences",
    "potential benefits, risks, harms, and costs", "benefit-harm assessment",
    "rationale", "recommendation", "recommendations",
}

LOCAL_HEADINGS = {
    "what this evidence adds", "key points", "highlights",
    "clinical practice guideline", "clinical practice guidelines",
}

BAD_MARKERS = (
    "doi:", "http://", "https://", "www.", "copyright", "©", "isbn", "issn",
    "downloaded from", "all rights reserved",
)

REFERENCE_LIKE_PATTERNS = (
    re.compile(r"^\d+\s+(?:were|was|are|is|the|a|an|in|of|to|for|from|with|on|by|and|or|both|two|one|several|these|those|this|that)\b", re.I),
    re.compile(r"^\d+\s+[A-Z][^.!?]{0,25},\s*$"),
)

AUTHOR_LIKE_PATTERN = re.compile(
    r"(?:\bet\s+al\.?\b|\b[A-Z][a-z]+,\s+[A-Z]\.?|\b[A-Z][a-z]+\s+[A-Z][a-z]+,\s+[A-Z])"
)

# ---------------------------------------------------------------------------
# FTR-005/008/010 chunking-dilution fix
#
# WHO "Package of interventions for rehabilitation" documents contain
# "Overview of the interventions" tables (and the longer per-intervention
# resource tables that follow them) that list several ICF functioning
# domains one after another -- e.g. "Motor functions and mobility",
# "Exercise and fitness", "Activities of daily living" -- each with its own
# Assessment/Intervention rows. The PDF layout wraps each domain's heading
# across 1-4 short lines (column width, not sentence boundaries), so the
# existing classify_heading() never recognizes them: they are not numbered,
# not ALL CAPS, and not in COMMON_HEADINGS/LOCAL_HEADINGS. As a result a
# whole page of unrelated domains collapses into one chunk, diluting the
# exact clinical terms (e.g. "gait", "balance", "muscle") a narrow query
# needs, both for BM25 term frequency/length normalization and for the
# dense embedding.
#
# This canonical list was built empirically from the two sample documents
# (9789240071100, 9789240071131) -- not guessed -- by scanning every chunk
# whose text contains two or more "Assessment of ..." rows and extracting
# the short line-runs that immediately precede them, then confirming the
# remaining domains (education/vocation, seeing, hearing, speech/language)
# by direct inspection. It is intentionally scoped to Format-A style
# "Overview of interventions" tables; the more complex per-intervention
# resource tables (session time / material resources / occupations, with
# "Target: ..." sub-rows) are a related but structurally different problem
# and are only split here incidentally, when a domain heading happens to
# appear on its own unwrapped line -- their "Target:" sub-rows are not
# handled by this pass and would need a separate follow-up.
FUNCTIONING_DOMAIN_HEADERS = {
    "motor functions and mobility",
    "motor function and mobility",
    "cognitive functions",
    "mental/cognitive functions",
    "mental cognitive functions",
    "mental health",
    "pain management",
    "exercise and fitness",
    "activities of daily living",
    "activity of daily living",
    "education and vocation",
    "seeing functions",
    "hearing functions",
    "speech, language and communication",
    "speech language and communication",
    "ingestion and dysphagia management",
    "dysphagia management",
    "bowel and bladder management",
    "bladder and bowel management",
    "sexual functions and intimate relationships",
    "skin care",
    "skin integrity functions",
    "cardiovascular functions",
    "cardiovascular and immunological functions",
    "cardiovascular, haematological, immunological and respiratory functions",
    "nutrition",
    "malnutrition",
    "community and social life",
    "interpersonal interactions and relationships",
    "problems with behaviour",
    "problems with behaviours",
    "respiration function",
    "respiration functions",
    "respiratory functions",
    "pressure ulcers",
    "pneumonia",
    "heterotopic ossification",
    "joint impairments",
    "self-management",
    "selfmanagement",
    "sleep functions",
    "voice functions",
}

# Longest phrase (by word count) is 8 words ("cardiovascular, haematological,
# immunological and respiratory functions"); cap the window a little above
# that so the scan stays cheap without needing to special-case the longest
# entry.
_FUNCTIONING_DOMAIN_MAX_WORDS = max(len(p.split()) for p in FUNCTIONING_DOMAIN_HEADERS)
FUNCTIONING_DOMAIN_MAX_LINES = 4
_SOFT_HYPHEN_CHARS = ("\u00ad", "\xad")


def _normalize_domain_join(lines) -> str:
    """Join candidate header lines the way they'd read on one line."""
    text = " ".join(lines)
    for ch in _SOFT_HYPHEN_CHARS:
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _is_domain_line_candidate(line: str) -> bool:
    """A domain-heading fragment: short, no sentence punctuation, no digits.

    PDF column-wrapped fragments are typically well under 30 characters, but
    a domain header can also appear unwrapped on one (wider) line -- the
    longest known entry, "Cardiovascular, haematological, immunological and
    respiratory functions", is 71 characters, plus soft hyphens the PDF
    extractor sometimes inserts mid-word (e.g. "Cardio\xadvascular"). Strip
    those before measuring so a real, if long, unwrapped header still
    qualifies as a single-line candidate.
    """
    line = line.strip()
    for ch in _SOFT_HYPHEN_CHARS:
        line = line.replace(ch, "")
    if not line or len(line) > 80:
        return False
    if re.search(r"[.:;]$", line):
        return False
    if re.search(r"\d", line):
        return False
    if is_bad_layout_line(line):
        return False
    return True


def find_functioning_domain_runs(lines):
    """Scan a page's lines for known ICF functioning-domain headers.

    Returns {start_index: (end_index_exclusive, display_text)}. Headers can
    be wrapped across up to FUNCTIONING_DOMAIN_MAX_LINES short lines (PDF
    column wrapping), or appear unwrapped on a single line.
    """
    runs = {}
    n = len(lines)
    i = 0
    while i < n:
        if not _is_domain_line_candidate(lines[i]):
            i += 1
            continue
        best = None
        window = []
        for w in range(1, min(FUNCTIONING_DOMAIN_MAX_LINES, n - i) + 1):
            candidate_line = lines[i + w - 1]
            if w > 1 and not _is_domain_line_candidate(candidate_line):
                break
            window.append(candidate_line)
            joined = _normalize_domain_join(window)
            if joined in FUNCTIONING_DOMAIN_HEADERS:
                best = w
        if best:
            display = _normalize_domain_join(lines[i:i + best])
            display = display[:1].upper() + display[1:]
            runs[i] = (i + best, display)
            i += best
        else:
            i += 1
    return runs


def norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[ 	]+", " ", text)
    return text.strip()
def clean_line(text: str) -> str:
    text = norm(text)
    return text.replace("ﬁ", "fi").replace("ﬂ", "fl")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    lines = [clean_line(line) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", clean_line(text)).lower()


def is_bad_layout_line(line: str) -> bool:
    low = normalize_for_compare(line)
    if not low:
        return True
    return any(marker in low for marker in BAD_MARKERS)


def looks_like_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", clean_line(line)))


def looks_like_doi_or_url(line: str) -> bool:
    low = clean_line(line).lower()
    return bool(
        re.search(r"10\.\d{4,9}/\S+", low)
        or re.match(r"^(https?://|www\.)", low)
        or "doi.org/" in low
    )


def spaced_caps(text: str) -> str:
    text = clean_line(text)
    if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", text):
        return re.sub(r"\s+", "", text)
    return text


def is_all_caps_heading(text: str) -> bool:
    text = spaced_caps(text)
    if not text or len(text) > 140:
        return False
    if text.lower() in LOCAL_HEADINGS:
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 5 and letters.upper() == letters and len(text.split()) <= 16


def is_likely_author_or_citation(text: str) -> bool:
    low = text.lower()
    if "et al" in low:
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        # Years are common in references/article metadata, but rare in a true
        # short structural heading candidate.
        return True
    if text.count(",") >= 2:
        return True
    if AUTHOR_LIKE_PATTERN.search(text):
        return True
    if re.search(r"\b(?:doi|pmid|issn|isbn)\b", low):
        return True
    return False


def numbered_heading_parts(text: str):
    """Return (number, title) only for conservative true-heading candidates."""
    text = clean_line(text)
    match = re.match(r"^(\d+(?:\.\d+){0,3})\.?\s+(.+)$", text)
    if not match:
        return None

    number = match.group(1)
    title = clean_line(match.group(2))

    if any(p.match(text) for p in REFERENCE_LIKE_PATTERNS):
        return None
    if looks_like_doi_or_url(text) or is_bad_layout_line(text):
        return None
    if is_likely_author_or_citation(title):
        return None
    if title.endswith((",", ";", ":")):
        return None
    if len(title) < 3 or len(title) > 180:
        return None

    words = title.split()
    first_alpha = next((c for c in title if c.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return None

    # A bare number followed by a short phrase is very often article metadata
    # or a sentence beginning with a numeric value, not a structural heading.
    if number.count(".") == 0 and len(words) <= 2 and title.lower() not in COMMON_HEADINGS:
        return None

    # Block obvious metadata strings such as "DCD CPG 297".
    if re.search(r"\b(?:CPG|DOI|PMID|ISSN|ISBN)\b", title, re.I):
        return None
    if re.fullmatch(r"[A-Za-z][A-Za-z ]+\s+\d{2,4}", title):
        return None

    if len(words) == 1 and title.lower() not in COMMON_HEADINGS:
        return None

    return number, title


def heading_level(number: str) -> int:
    return number.count(".") + 1


def classify_heading(line: str):
    line = clean_line(line)
    if not line or len(line) > 180 or is_bad_layout_line(line):
        return None
    if looks_like_page_number(line) or looks_like_doi_or_url(line):
        return None

    numbered = numbered_heading_parts(line)
    if numbered:
        number, title = numbered
        return {
            "text": f"{number} {title}",
            "number": number,
            "title": title,
            "level": heading_level(number),
            "kind": "numbered",
            "persistent": True,
        }

    compact = spaced_caps(line)
    low = compact.lower().rstrip(":")

    if low in LOCAL_HEADINGS:
        return {"text": compact, "number": None, "title": compact, "level": 0, "kind": "local", "persistent": False}
    if low in COMMON_HEADINGS:
        return {"text": compact, "number": None, "title": compact, "level": 0, "kind": "common", "persistent": False}
    if is_all_caps_heading(compact):
        return {"text": compact, "number": None, "title": compact, "level": 0, "kind": "caps", "persistent": False}
    return None


def detect_toc_page(raw_text: str) -> bool:
    low = normalize_for_compare(raw_text)
    if not low:
        return False
    if "table of contents" in low:
        return True
    lines = [clean_line(x) for x in raw_text.splitlines() if clean_line(x)]
    if len(lines) < 8:
        return False
    dotted = sum(1 for line in lines if re.search(r"\.{3,}\s*\d{1,3}\s*$", line))
    numbered = sum(1 for line in lines if re.match(r"^\d+(?:\.\d+)*\s+", line))
    return dotted >= 3 or (dotted >= 1 and numbered >= 6)


def detect_repeated_layout_lines(pages):
    """Detect only likely running headers/footers from page-edge positions."""
    counts = Counter()
    total_pages = len(pages)
    if total_pages < 2:
        return set()

    for page in pages:
        raw_lines = [clean_line(x) for x in (page.get("text", "") or "").splitlines()]
        raw_lines = [x for x in raw_lines if x]
        if not raw_lines:
            continue
        edge_candidates = raw_lines[:EDGE_LINES] + raw_lines[-EDGE_LINES:]
        seen = set()
        for line in edge_candidates:
            if not line or len(line) > 180:
                continue
            low = normalize_for_compare(line)
            if low in seen:
                continue
            seen.add(low)
            counts[low] += 1

    threshold = max(REPEATED_LAYOUT_MIN_COUNT, round(total_pages * 0.35))
    repeated = {line for line, count in counts.items() if count >= threshold}

    # Never delete very short/common body words even if they occur at edges.
    # Layout candidates should look like a phrase, not a single clinical term.
    safe = set()
    for line in repeated:
        if len(line) < 12:
            continue
        if len(line.split()) < 2:
            continue
        if re.fullmatch(r"[\d\W]+", line):
            continue
        safe.add(line)
    return safe


def repair_word_per_line_text(raw_text: str):
    """Conservative fallback for obvious word-per-line PDF extraction noise."""
    lines = [clean_line(x) for x in (raw_text or "").splitlines() if clean_line(x)]
    if len(lines) < 12:
        return raw_text, False

    short_ratio = sum(1 for x in lines if len(x.split()) <= 2 and len(x) <= 18) / len(lines)
    med_len = median(len(x) for x in lines)
    if short_ratio < WORD_PER_LINE_RATIO or med_len > 18:
        return raw_text, False

    # Join words while preserving a paragraph break after sentence endings.
    joined = " ".join(lines)
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    joined = re.sub(r"([.!?])\s+(?=[A-Z])", r"\1\n", joined)
    return joined, True


def filter_page_lines(raw_text: str, repeated_layout_lines):
    lines = []
    for raw in raw_text.splitlines():
        line = clean_line(raw)
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        low = normalize_for_compare(line)
        if low in repeated_layout_lines:
            continue
        if is_bad_layout_line(line) or looks_like_page_number(line):
            continue
        if low.startswith("downloaded from"):
            continue
        lines.append(line)

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def split_sentences(text: str):
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def split_long_text(text: str):
    sentences = split_sentences(text)
    if not sentences:
        return [text]

    result = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= TARGET_CHARS:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= TARGET_CHARS:
                current = candidate
            else:
                if current:
                    result.append(current.strip())
                current = sentence
        else:
            if current:
                result.append(current.strip())
                current = ""
            for i in range(0, len(sentence), TARGET_CHARS):
                part = sentence[i:i + TARGET_CHARS].strip()
                if part:
                    result.append(part)

    if current:
        result.append(current.strip())
    return result


def build_chunks(blocks):
    """Chunk structural blocks while preserving boundaries and small overlap."""
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        block = clean_text(block)
        if not block:
            continue

        if len(block) > TARGET_CHARS:
            flush()
            parts = split_long_text(block)
            if not parts:
                continue
            for part in parts[:-1]:
                chunks.append(part)
            current = parts[-1]
            continue

        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= TARGET_CHARS:
            current = candidate
        else:
            previous = current
            flush()
            overlap = previous[-OVERLAP_CHARS:].strip() if previous else ""
            current = (overlap + "\n\n" + block).strip() if overlap else block

    flush()
    return chunks


def table_to_text(table):
    rows = table.get("rows", []) if isinstance(table, dict) else []
    if not rows:
        return ""
    lines = []
    for row in rows:
        if not row:
            continue
        cells = [clean_line("" if cell is None else str(cell)) for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def validate_table(table_text: str, table):
    """Return (is_real_table, reason) using conservative structural checks."""
    rows = table.get("rows", []) if isinstance(table, dict) else []
    nonempty_rows = [r for r in rows if r and any(str(c or "").strip() for c in r)]
    if not nonempty_rows:
        return False, "empty_table"

    col_counts = [len(r) for r in nonempty_rows]
    max_cols = max(col_counts)
    avg_cell_len = sum(len(str(c or "")) for r in nonempty_rows for c in r) / max(1, sum(len(r) for r in nonempty_rows))
    long_row_count = sum(1 for r in nonempty_rows if len(" ".join(str(c or "") for c in r)) > 500)

    # A one-column or two-column object containing long prose is often a
    # false-positive table produced by PDF layout detection.
    if max_cols <= 2 and (long_row_count >= 1 or avg_cell_len > 180):
        return False, "paragraph_like_table"
    if len(nonempty_rows) == 1 and max_cols <= 2 and len(table_text) > 500:
        return False, "single_long_row"
    return True, "validated"



# ---------------------------------------------------------------------------
# v3.4: semantic completeness + retrieval context enrichment
# ---------------------------------------------------------------------------

_CONTEXT_DEPENDENT_STARTS = (
    "this ", "these ", "those ", "such ", "therefore ", "however ",
    "additionally ", "furthermore ", "moreover ", "thus ",
    "in this population", "in these patients", "for these patients",
    "the latter", "the former",
)

def assess_semantic_completeness(chunk_text):
    """Conservative diagnostic; does not change chunk boundaries."""
    t = re.sub(r"\s+", " ", (chunk_text or "").strip())
    issues = []
    if not t:
        return {"score": 0.0, "status": "incomplete", "issues": ["empty_text"]}
    lower = t.lower()
    if t[0].islower() or t[0] in ",;:)]":
        issues.append("starts_mid_sentence")
    if t[-1] in ",;:":
        issues.append("ends_mid_sentence")
    if re.search(
        r"\b(and|or|but|because|although|while|which|that|including|such as|"
        r"with|for|to|of|in|on|by|as)\s*$", lower
    ):
        issues.append("trailing_connector")
    if any(lower.startswith(p) for p in _CONTEXT_DEPENDENT_STARTS):
        issues.append("context_dependent_opening")
    if t.count("(") != t.count(")") or t.count("[") != t.count("]"):
        issues.append("unbalanced_delimiters")
    score = max(0.0, round(1.0 - 0.22 * len(issues), 2))
    status = "complete" if score >= 0.78 else ("warning" if score >= 0.55 else "incomplete")
    return {"score": score, "status": status, "issues": issues}


def build_retrieval_text(
    document_title, section, subsection, local_heading,
    functioning_domain, source_type, chunk_text
):
    """Retrieval-only representation; original source text remains unchanged."""
    fields = [
        ("Document", document_title),
        ("Section", section),
        ("Subsection", subsection),
        ("Local heading", local_heading),
        ("Functioning domain", functioning_domain),
        ("Source type", source_type),
    ]
    lines = [
        f"{label}: {value.strip()}"
        for label, value in fields
        if isinstance(value, str) and value.strip()
    ]
    body = (chunk_text or "").strip()
    if body:
        lines.append(f"Content: {body}")
    return "\n".join(lines)


def make_metadata(document_name, page_number, source_type, section, subsection, extra=None):
    metadata = {
        "document_id": Path(document_name).stem,
        "title": document_name,
        "organization": None,
        "year": None,
        "domain": "physiotherapy",
        "condition": None,
        "population": None,
        "section": section,
        "subsection": subsection,
        "page": page_number,
        "source_type": source_type,
    }
    if extra:
        metadata.update(extra)
    return metadata


def process_document(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    pages = document.get("pages")
    if not isinstance(pages, list):
        raise ValueError(
            f"{json_path.name}: raw extraction JSON bekleniyordu ('pages' alanı yok). "
            "Bu dosya zaten chunk çıktısı olabilir."
        )

    document_name = document.get("document", json_path.name)
    repeated_layout = detect_repeated_layout_lines(pages)

    all_chunks = []
    current_section = None
    current_section_number = None
    current_subsection = None
    current_subsection_number = None
    toc_pages = 0
    repaired_pages = []
    suspicious_headings = []
    skipped_tables = []
    reclassified_tables = []
    orphan_candidates = []

    for page in pages:
        page_number = page.get("page")
        raw_text = page.get("text", "") or ""
        repaired_text, was_repaired = repair_word_per_line_text(raw_text)
        if was_repaired:
            repaired_pages.append(page_number)
        raw_text_for_analysis = repaired_text

        is_toc = detect_toc_page(raw_text_for_analysis)
        if is_toc:
            toc_pages += 1

        lines = filter_page_lines(raw_text_for_analysis, repeated_layout)
        blocks = []
        current_block = []
        local_heading = None
        current_functioning_domain = None

        # Only split on ICF functioning-domain headers inside the actual
        # "Overview of interventions" content, gated by subsection so we
        # never touch prose that merely mentions a domain by name (e.g. a
        # "Description of the assessment" glossary entry, or a References
        # list title) -- see FUNCTIONING_DOMAIN_HEADERS above for why.
        domain_runs = find_functioning_domain_runs(lines)

        def flush_block():
            nonlocal current_block
            if current_block:
                text = "\n".join(current_block).strip()
                if text:
                    blocks.append(("text", text, local_heading, None, current_functioning_domain))
                current_block = []

        idx = 0
        n_lines = len(lines)
        while idx < n_lines:
            line = lines[idx]

            if idx in domain_runs:
                end_idx, display_text = domain_runs[idx]
                in_package_overview = bool(
                    current_subsection
                    and "package of interventions" in current_subsection.lower()
                )
                if in_package_overview:
                    flush_block()
                    current_functioning_domain = display_text
                    current_block.append(display_text)
                    idx = end_idx
                    continue
                # Gate not met: fall through and process this single line
                # exactly as before (no split), leaving the run intact.

            heading = classify_heading(line)
            if heading:
                flush_block()
                if heading["persistent"] and not is_toc:
                    level = heading["level"]
                    if level == 1:
                        current_section = heading["title"]
                        current_section_number = heading["number"]
                        current_subsection = None
                        current_subsection_number = None
                    else:
                        current_subsection = heading["title"]
                        current_subsection_number = heading["number"]
                local_heading = heading["text"]
                current_functioning_domain = None
                idx += 1
                continue

            if is_bad_layout_line(line) or looks_like_page_number(line):
                idx += 1
                continue
            current_block.append(line)
            idx += 1

        flush_block()

        for table in page.get("tables", []) or []:
            table_text = table_to_text(table)
            if not table_text:
                continue
            if len(table_text) < MIN_TABLE_CHARS:
                skipped_tables.append({
                    "page": page_number,
                    "table_index": table.get("table_index"),
                    "reason": "below_min_table_chars",
                    "characters": len(table_text),
                })
                continue

            is_real, reason = validate_table(table_text, table)
            if is_real:
                blocks.append(("table", table_text, local_heading, table.get("table_index"), None))
            else:
                blocks.append(("text", table_text, local_heading, None, None))
                reclassified_tables.append({
                    "page": page_number,
                    "table_index": table.get("table_index"),
                    "reason": reason,
                })

        for block in blocks:
            block_type, block_text, block_local_heading, table_index, block_functioning_domain = block

            if block_type == "text":
                pieces = build_chunks([block_text])
                for piece in pieces:
                    piece = piece.strip()
                    if not piece:
                        continue
                    semantic = assess_semantic_completeness(piece)
                    metadata = make_metadata(
                        document_name, page_number, "text",
                        current_section, current_subsection,
                        {"estimated_tokens": estimate_tokens(piece)},
                    )
                    if block_local_heading:
                        metadata["local_heading"] = block_local_heading
                    if block_functioning_domain:
                        metadata["functioning_domain"] = block_functioning_domain
                    metadata["small_chunk"] = len(piece) < MIN_TEXT_CHARS
                    metadata["semantic_completeness"] = semantic["status"]
                    metadata["semantic_completeness_score"] = semantic["score"]
                    metadata["semantic_completeness_issues"] = semantic["issues"]
                    retrieval_text = build_retrieval_text(
                        document_name, current_section, current_subsection,
                        block_local_heading, block_functioning_domain,
                        "text", piece
                    )
                    all_chunks.append({
                        "text": piece,
                        "retrieval_text": retrieval_text,
                        "metadata": metadata,
                    })

            elif block_type == "table":
                metadata = make_metadata(
                    document_name,
                    page_number,
                    "table",
                    current_section,
                    current_subsection,
                    {
                        "table_index": table_index,
                        "estimated_tokens": estimate_tokens(block_text),
                    },
                )
                if block_local_heading:
                    metadata["local_heading"] = block_local_heading
                semantic = assess_semantic_completeness(block_text)
                metadata["small_chunk"] = len(block_text.strip()) < MIN_TABLE_CHARS
                metadata["semantic_completeness"] = semantic["status"]
                metadata["semantic_completeness_score"] = semantic["score"]
                metadata["semantic_completeness_issues"] = semantic["issues"]
                retrieval_text = build_retrieval_text(
                    document_name, current_section, current_subsection,
                    block_local_heading, metadata.get("functioning_domain"),
                    "table", block_text
                )
                all_chunks.append({
                    "text": block_text,
                    "retrieval_text": retrieval_text,
                    "metadata": metadata,
                })

        for line in lines:
            heading = classify_heading(line)
            if heading and heading["persistent"]:
                if is_likely_author_or_citation(heading["title"]):
                    suspicious_headings.append({
                        "page": page_number,
                        "heading": heading["text"],
                        "reason": "author_or_citation_like_heading",
                    })
                elif len(heading["title"].split()) <= 1 and heading["title"].lower() not in COMMON_HEADINGS:
                    suspicious_headings.append({
                        "page": page_number,
                        "heading": heading["text"],
                        "reason": "very_short_numbered_heading",
                    })

    for index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = f"{Path(document_name).stem}_chunk_{index:04d}"
        if chunk["metadata"].get("source_type") == "text":
            meta = chunk["metadata"]
            tokens = meta.get("estimated_tokens", 0)
            semantic_status = meta.get("semantic_completeness", "complete")
            if tokens < ORPHAN_TOKEN_THRESHOLD or semantic_status != "complete":
                reasons = []
                if tokens < ORPHAN_TOKEN_THRESHOLD:
                    reasons.append("very_small_chunk")
                if semantic_status != "complete":
                    reasons.append("semantic_" + semantic_status)
                orphan_candidates.append({
                    "chunk_id": chunk["chunk_id"],
                    "page": meta.get("page"),
                    "estimated_tokens": tokens,
                    "local_heading": meta.get("local_heading"),
                    "reason": ";".join(reasons),
                    "semantic_completeness": semantic_status,
                    "semantic_completeness_issues": meta.get(
                        "semantic_completeness_issues", []
                    ),
                })

    return {
        "chunker_version": "3.4",
        "strategy": "structure-aware hierarchical + paragraph/sentence boundaries + fixed-size cap + overlap + context metadata + table-aware",
        "document": document_name,
        "chunk_count": len(all_chunks),
        "toc_pages": toc_pages,
        "repeated_layout_lines": sorted(repeated_layout),
        "repaired_word_per_line_pages": repaired_pages,
        "reclassified_tables": reclassified_tables,
        "skipped_tiny_tables": skipped_tables,
        "suspicious_headings": suspicious_headings,
        "orphan_candidates": orphan_candidates,
        "chunks": all_chunks,
    }


def generate_quality_report(result):
    chunks = result.get("chunks", [])
    text_chunks = [c for c in chunks if c["metadata"].get("source_type") == "text"]
    table_chunks = [c for c in chunks if c["metadata"].get("source_type") == "table"]

    sizes = [len(c["text"]) for c in text_chunks]
    token_sizes = [c["metadata"].get("estimated_tokens", 0) for c in text_chunks]

    section_count = sum(1 for c in chunks if c["metadata"].get("section"))
    subsection_count = sum(1 for c in chunks if c["metadata"].get("subsection"))
    local_heading_count = sum(1 for c in chunks if c["metadata"].get("local_heading"))

    return {
        "chunker_version": "3.4",
        "document": result.get("document"),
        "chunk_count": len(chunks),
        "text_chunk_count": len(text_chunks),
        "table_chunk_count": len(table_chunks),
        "average_characters": round(mean(sizes), 1) if sizes else 0,
        "average_estimated_tokens": round(mean(token_sizes), 1) if token_sizes else 0,
        "median_estimated_tokens": round(median(token_sizes), 1) if token_sizes else 0,
        "min_characters": min(sizes) if sizes else 0,
        "max_characters": max(sizes) if sizes else 0,
        "section_coverage_percent": round(section_count / len(chunks) * 100, 1) if chunks else 0,
        "subsection_coverage_percent": round(subsection_count / len(chunks) * 100, 1) if chunks else 0,
        "local_heading_chunks": local_heading_count,
        "toc_pages": result.get("toc_pages", 0),
        "repeated_layout_candidates": len(result.get("repeated_layout_lines", [])),
        "repaired_word_per_line_pages": len(result.get("repaired_word_per_line_pages", [])),
        "reclassified_tables": len(result.get("reclassified_tables", [])),
        "suspicious_headings": len(result.get("suspicious_headings", [])),
        "skipped_tiny_tables": len(result.get("skipped_tiny_tables", [])),
        "orphan_candidates": len(result.get("orphan_candidates", [])),
        "short_chunks": sum(1 for c in text_chunks if len(c["text"]) < MIN_TEXT_CHARS),
        "long_chunks": sum(1 for c in text_chunks if len(c["text"]) > TARGET_CHARS * LONG_CHUNK_FACTOR),
    }


def process_all_documents():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(
        p for p in INPUT_DIR.glob("*.json")
        if not p.name.startswith("_")
    )

    if not json_files:
        print("[ERROR] data/processed/extracted_text klasöründe JSON bulunamadı.")
        return

    print("=" * 70)
    print("FTR-RAG CHUNKER v3.4")
    print("=" * 70)
    print(f"JSON sayısı: {len(json_files)}")
    print("Strategy: structure-aware + controlled boundaries + fixed cap + overlap + metadata + table-aware")
    print()

    total_chunks = 0
    reports = []
    failed = []

    for json_path in json_files:
        try:
            result = process_document(json_path)
        except ValueError as error:
            failed.append({"file": json_path.name, "reason": str(error)})
            print(f"[SKIP] {json_path.name}: {error}")
            continue
        except Exception as error:
            failed.append({"file": json_path.name, "reason": repr(error)})
            print(f"[ERROR] {json_path.name}: {error}")
            continue

        output_file = OUTPUT_DIR / json_path.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        report = generate_quality_report(result)
        reports.append(report)
        total_chunks += report["chunk_count"]

        print(
            f"{json_path.name}: "
            f"{report['chunk_count']} chunk | "
            f"text {report['text_chunk_count']} | "
            f"table {report['table_chunk_count']} | "
            f"section {report['section_coverage_percent']}% | "
            f"subsection {report['subsection_coverage_percent']}% | "
            f"reclassified tables {report['reclassified_tables']} | "
            f"orphan {report['orphan_candidates']} | "
            f"suspicious {report['suspicious_headings']} | "
            f"TOC {report['toc_pages']}"
        )

    quality_report = {
        "chunker_version": "3.4",
        "strategy": {
            "primary": "structure-aware hierarchical chunking",
            "boundaries": "headings + paragraphs + sentences for long blocks",
            "target_chars": TARGET_CHARS,
            "approx_target_tokens": "500-800",
            "overlap_chars": OVERLAP_CHARS,
            "tables": "validated separate retrievable table chunks; paragraph-like false positives reclassified as text",
            "heading_policy": "conservative; validated numbered headings are persistent, non-numbered headings are page-local",
            "repeated_layout_policy": "only repeated page-edge lines can be removed",
            "provenance": "document_id + page + source_type + section/subsection when confidently detected",
            "word_per_line_repair": "conservative fallback only when extraction clearly shows word-per-line noise",
        },
        "total_documents": len(reports),
        "total_chunks": total_chunks,
        "failed_files": failed,
        "documents": reports,
    }

    report_file = OUTPUT_DIR / "_quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("CHUNKING v3.3 TAMAMLANDI")
    print(f"Toplam chunk: {total_chunks}")
    print(f"Kalite raporu: {report_file}")
    if failed:
        print(f"Atlanan/hatalı dosya: {len(failed)}")
    print("=" * 70)


if __name__ == "__main__":
    process_all_documents()
