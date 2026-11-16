from __future__ import annotations

import re
from typing import Any, Dict

from src.retrieval.config import REFERENCE_PENALTY


# Compact, source-provided metadata fields shared by the BM25 and dense
# metadata ablations. Long free-text summaries are intentionally excluded.
RETRIEVAL_METADATA_FIELDS = (
    ("doc_name", "Document name"),
    ("title", "Chunk title"),
    ("organization", "Organization"),
    ("year", "Year"),
    ("doc_version", "Document version"),
    ("section_path", "Section"),
    ("table_caption", "Table caption"),
    ("guideline_title", "Guideline title"),
    ("guideline_source_authority", "Publisher"),
    ("guideline_filename_year", "Year"),
    ("guideline_version", "Version"),
    ("guideline_publication_date", "Publication date"),
    ("guideline_last_updated", "Last updated"),
    ("guideline_diseases_covered", "Diseases covered"),
    ("guideline_conditions_covered", "Conditions covered"),
    ("guideline_target_population", "Target population"),
    ("guideline_age_group", "Age group"),
    ("guideline_clinical_phase", "Clinical phase"),
    ("guideline_treatment_context", "Treatment context"),
    ("guideline_injury_context", "Injury context"),
    ("chunk_diseases", "Chunk diseases"),
    ("chunk_conditions", "Chunk conditions"),
    ("chunk_interventions", "Chunk interventions"),
    ("chunk_population", "Chunk population"),
    ("chunk_clinical_phase", "Chunk clinical phase"),
    ("chunk_recommendation_status", "Recommendation status"),
    ("recommendation_strength", "Recommendation strength"),
    ("evidence_level", "Evidence level"),
    ("chunk_contraindications", "Contraindications"),
    ("chunk_adverse_effects", "Adverse effects"),
    ("chunk_clinical_warnings", "Clinical warnings"),
)

_EMPTY_METADATA_VALUES = {"unknown", "unspecified", "not applicable", "n/a", "none"}

_NON_SEARCHABLE_METADATA_FIELDS = {
    "bbox",
    "chunk_id",
    "context_after_chunk_ids",
    "context_before_chunk_ids",
    "embedding_eligible",
    "embedding_text",
    "legacy_chunk_id",
    "legacy_chunk_id_source",
    "parent_section_id",
    "related_text_chunk_ids",
    "relation_evidence_pages",
    "retrieval_terms",
    "schema_version",
    "section_id",
    "source_id",
    "source_spans",
    "table_chunk_group_id",
    "table_id",
    "table_origin_id",
    "token_count",
    "token_count_method",
    "duplicate_count",
}


def _metadata_values(value: Any, max_chars: int) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        candidates = value
    elif value is None:
        candidates = ()
    else:
        candidates = (value,)

    values = []
    for candidate in candidates:
        text = " ".join(str(candidate).split()).strip()
        if not text or text.casefold() in _EMPTY_METADATA_VALUES:
            continue
        values.append(text[:max_chars])
    return values


def retrieval_metadata_values(
    metadata: Dict[str, Any] | None,
    max_value_chars: int = 240,
) -> list[str]:
    """Return compact, non-empty structured guideline/chunk metadata values."""
    metadata = metadata or {}
    values = []
    seen = set()
    for field, _label in RETRIEVAL_METADATA_FIELDS:
        for value in _metadata_values(metadata.get(field), max_value_chars):
            normalized = value.casefold()
            if normalized not in seen:
                seen.add(normalized)
                values.append(value)
    return values


def retrieval_metadata_text(
    metadata: Dict[str, Any] | None,
    max_value_chars: int = 240,
) -> str:
    """Render compact structured metadata as labeled dense passage context."""
    metadata = metadata or {}
    parts = []
    seen = set()
    for field, label in RETRIEVAL_METADATA_FIELDS:
        values = _metadata_values(metadata.get(field), max_value_chars)
        if not values:
            continue
        value = "; ".join(values)
        normalized = (label.casefold(), value.casefold())
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(f"{label}: {value}")
    return " | ".join(parts)


def all_searchable_metadata_values(metadata: Dict[str, Any] | None) -> list[str]:
    """Flatten all semantic metadata values for the BM25 all-fields ablation.

    Structural identifiers, coordinates, token counters, and the duplicated
    embedding input are excluded because they are not useful lexical evidence.
    Free-text fields such as guideline_summary and commented_text are included.
    """
    metadata = metadata or {}
    values: list[str] = []
    seen = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)
        elif isinstance(value, (str, int, float, bool)):
            text = " ".join(str(value).split()).strip()
            if not text or text.casefold() in _EMPTY_METADATA_VALUES:
                return
            normalized = text.casefold()
            if normalized not in seen:
                seen.add(normalized)
                values.append(text)

    for field, value in metadata.items():
        if str(field).casefold() not in _NON_SEARCHABLE_METADATA_FIELDS:
            visit(value)
    return values


# Only strong bibliography signals are used here. Clinical chunks can legitimately
# mention trials, years, authors, or journals, so those signals alone must not
# cause a chunk to be treated as a reference list.
_REFERENCE_PATTERNS = (
    re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.I),
    re.compile(r"\bhttps?://doi\.org/", re.I),
    re.compile(r"\bpmid\s*[:.]?\s*\d+", re.I),
    re.compile(r"^\s*references\s*$", re.I | re.M),
)


def normalize_chunk_id(value: Any) -> str:
    """Return a canonical string form for chunk IDs."""
    return str(value).strip()


def infer_chunk_role(text: str, metadata: Dict[str, Any] | None = None) -> str:
    """Classify bibliography/reference chunks conservatively.

    Clinical evidence text may contain years, trials, authors, or journal names.
    Those are therefore not sufficient by themselves. A chunk is marked as a
    reference only when metadata/heading or multiple strong bibliography
    signals support that interpretation.
    """
    text = text or ""
    metadata = metadata or {}

    section = str(metadata.get("section") or "").strip().lower()
    subsection = str(metadata.get("subsection") or "").strip().lower()
    if section == "references" or subsection == "references":
        return "reference"

    # A chunk that explicitly consists of a References heading is a reference
    # block. Do not match the word anywhere in normal prose.
    if re.search(r"^\s*references\s*$", text, flags=re.I | re.M):
        return "reference"

    year_count = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    doi_count = len(re.findall(r"\bdoi\s*:\s*10\.\d{4,9}/", text, flags=re.I))
    doi_url_count = len(re.findall(r"\bhttps?://doi\.org/", text, flags=re.I))
    pmid_count = len(re.findall(r"\bpmid\s*[:.]?\s*\d+", text, flags=re.I))
    etal_count = len(re.findall(r"\bet al\.", text, flags=re.I))
    author_like = len(
        re.findall(
            r"\b[A-Z][A-Za-zÀ-ÿ'’-]+,\s*[A-Z](?:[A-Za-zÀ-ÿ'’-]+)?"
            r"(?:\s+[A-Z](?:[A-Za-zÀ-ÿ'’-]+)?)?\.",
            text,
        )
    )

    strong_links = doi_count + doi_url_count + pmid_count

    # One DOI/PMID plus bibliographic context is strong evidence.
    if strong_links >= 1 and (year_count >= 1 or author_like >= 1):
        return "reference"

    # A dense bibliography block usually has many author/year markers and
    # repeated "et al." references. This threshold is deliberately high.
    if year_count >= 5 and etal_count >= 2 and author_like >= 3:
        return "reference"

    return "clinical"


def annotate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata") or {}
    role = infer_chunk_role(result.get("document", ""), metadata)
    result["chunk_role"] = role
    result["reference_penalty"] = REFERENCE_PENALTY if role == "reference" else 1.0
    return result
