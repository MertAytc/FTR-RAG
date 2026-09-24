from __future__ import annotations

import re
from typing import Any, Dict


# Only strong bibliography signals are used here. Clinical chunks can legitimately
# mention trials, years, authors, or journals, so those signals alone must not
# cause a chunk to be treated as a reference list.
_REFERENCE_PATTERNS = (
    re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.I),
    re.compile(r"\bhttps?://doi\.org/", re.I),
    re.compile(r"\bpmid\s*[:.]?\s*\d+", re.I),
    re.compile(r"^\s*references\s*$", re.I | re.M),
)


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
    result["reference_penalty"] = 0.25 if role == "reference" else 1.0
    return result
