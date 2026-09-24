from typing import Dict, List, Any


REFERENCE_LABELS = {
    "reference",
    "references",
    "bibliography",
    "bibliographic references",
    "references cited",
    "literature cited",
}


def _clean_text(text: str) -> str:
    """Normalize whitespace without changing source wording."""
    return " ".join(str(text).split()).strip()


def _normalize_label(value: Any) -> str:
    return _clean_text(value).lower().rstrip(":")


def is_reference_only(chunk: Dict[str, Any]) -> bool:
    """
    Conservative deterministic filter for reference/bibliography chunks.

    Priority:
    1. Explicit metadata section/subsection labels.
    2. Explicit source_type.
    3. Strong text-level bibliographic signal.

    This function intentionally does not make semantic judgments.
    """

    metadata = chunk.get("metadata") or {}

    source_type = _normalize_label(metadata.get("source_type"))
    section = _normalize_label(metadata.get("section"))
    subsection = _normalize_label(metadata.get("subsection"))

    if source_type in REFERENCE_LABELS:
        return True

    if section in REFERENCE_LABELS:
        return True

    if subsection in REFERENCE_LABELS:
        return True

    text = _clean_text(
        chunk.get("document")
        or chunk.get("text")
        or ""
    )

    if not text:
        return True

    lower_text = text.lower()

    # Only use text detection when the beginning of the chunk explicitly
    # announces a reference section and bibliographic signals are present.
    starts_with_reference_label = any(
        lower_text.startswith(label)
        for label in REFERENCE_LABELS
    )

    citation_signals = (
        "doi",
        "pmid",
        "et al.",
        "journal",
        "vol.",
        "issue",
    )

    citation_signal_count = sum(
        signal in lower_text
        for signal in citation_signals
    )

    if starts_with_reference_label and citation_signal_count >= 1:
        return True

    return False


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunk IDs and exact duplicate normalized text.
    First occurrence is preserved.
    """
    seen_ids = set()
    seen_text = set()
    unique = []

    for chunk in chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id")

        text = _clean_text(
            chunk.get("document")
            or chunk.get("text")
            or ""
        )

        if chunk_id and chunk_id in seen_ids:
            continue

        text_key = text.lower()

        if text_key and text_key in seen_text:
            continue

        if chunk_id:
            seen_ids.add(chunk_id)

        if text_key:
            seen_text.add(text_key)

        unique.append(chunk)

    return unique


def format_source(
    chunk: Dict[str, Any],
    source_number: int,
) -> Dict[str, Any]:
    metadata = chunk.get("metadata") or {}

    return {
        "source_number": source_number,
        "chunk_id": chunk.get("id") or chunk.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title"),
        "organization": metadata.get("organization"),
        "year": metadata.get("year"),
        "domain": metadata.get("domain"),
        "condition": metadata.get("condition"),
        "population": metadata.get("population"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "page": metadata.get("page"),
        "source_type": metadata.get("source_type"),
        "text": _clean_text(
            chunk.get("document")
            or chunk.get("text")
            or ""
        ),
    }


def _display(value: Any) -> str:
    if value is None or value == "":
        return "belirtilmemiş"
    return str(value)


def assemble_context(
    chunks: List[Dict[str, Any]],
    max_sources: int = 3,
    exclude_reference_only: bool = True,
) -> Dict[str, Any]:
    """
    Convert MMR results into compact, structured LLM context.

    MMR has already performed retrieval and diversification. This layer
    intentionally keeps only the first `max_sources` MMR results so the
    generation model receives a small, high-signal context.

    Reference-only chunks are removed before truncation.
    """
    original_count = len(chunks)

    if exclude_reference_only:
        filtered = [
            chunk
            for chunk in chunks
            if not is_reference_only(chunk)
        ]
    else:
        filtered = list(chunks)

    after_reference_filter = len(filtered)

    filtered = deduplicate_chunks(filtered)
    after_dedup = len(filtered)

    # Keep the highest-ranked MMR results only.
    filtered = filtered[:max_sources]

    sources = [
        format_source(chunk, index)
        for index, chunk in enumerate(filtered, start=1)
    ]

    context_blocks = []

    for source in sources:
        block = (
            f"[SOURCE {source['source_number']}]\n"
            f"chunk_id: {_display(source['chunk_id'])}\n"
            f"document_id: {_display(source['document_id'])}\n"
            f"title: {_display(source['title'])}\n"
            f"organization: {_display(source['organization'])}\n"
            f"year: {_display(source['year'])}\n"
            f"domain: {_display(source['domain'])}\n"
            f"condition: {_display(source['condition'])}\n"
            f"population: {_display(source['population'])}\n"
            f"section: {_display(source['section'])}\n"
            f"subsection: {_display(source['subsection'])}\n"
            f"page: {_display(source['page'])}\n"
            f"source_type: {_display(source['source_type'])}\n"
            f"content:\n{source['text']}\n"
            f"[/SOURCE {source['source_number']}]"
        )
        context_blocks.append(block)

    return {
        "sources": sources,
        "context_text": "\n\n".join(context_blocks),
        "source_count": len(sources),
        "input_count": original_count,
        "reference_filtered_count": original_count - after_reference_filter,
        "duplicate_filtered_count": after_reference_filter - after_dedup,
        "max_sources": max_sources,
    }

