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
    include_commentary: bool = False,
) -> Dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    doc_name = metadata.get("doc_name")

    source = {
        "source_number": source_number,
        "source_kind": "table_context" if chunk.get("_table_context_relation") else "chunk",
        "context_relation": chunk.get("_table_context_relation"),
        "linked_from_table_chunk_id": chunk.get("_linked_from_table_chunk_id"),
        "chunk_id": chunk.get("id") or chunk.get("chunk_id"),
        "document_id": metadata.get("document_id") or doc_name,
        "doc_name": doc_name,
        "title": metadata.get("title") or doc_name,
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
    if include_commentary:
        source["commented_text"] = _clean_text(
            chunk.get("commented_text") or metadata.get("commented_text") or ""
        )
    return source


def _parse_linked_ids(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value or "").split(";")
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _linked_table_context(
    chunks: List[Dict[str, Any]],
    per_side_limit: int,
    collection=None,
) -> List[Dict[str, Any]]:
    """Fetch explicitly linked prose neighbors for selected table chunks."""
    if per_side_limit <= 0:
        return []
    linked: List[tuple[str, str, str]] = []
    requested_ids: List[str] = []
    for table_chunk in chunks:
        metadata = table_chunk.get("metadata") or {}
        if str(metadata.get("chunk_type", "")).lower() != "table":
            continue
        table_chunk_id = str(table_chunk.get("id") or table_chunk.get("chunk_id") or "")
        for field, relation in (
            ("context_before_chunk_ids", "text immediately before related table"),
            ("context_after_chunk_ids", "text immediately after related table"),
        ):
            for chunk_id in _parse_linked_ids(metadata.get(field))[:per_side_limit]:
                linked.append((chunk_id, relation, table_chunk_id))
                requested_ids.append(chunk_id)
    if not requested_ids:
        return []
    if collection is None:
        from src.retrieval.vector_store import open_collection

        collection = open_collection()
    fetched = collection.get(
        ids=list(dict.fromkeys(requested_ids)),
        include=["documents", "metadatas"],
    )
    documents = fetched.get("documents") or []
    metadatas = fetched.get("metadatas") or []
    by_id = {
        str(chunk_id): {
            "id": str(chunk_id),
            "document": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) and metadatas[index] else {},
        }
        for index, chunk_id in enumerate(fetched.get("ids") or [])
    }
    output = []
    seen = set()
    for chunk_id, relation, table_chunk_id in linked:
        if chunk_id in seen or chunk_id not in by_id:
            continue
        neighbor = dict(by_id[chunk_id])
        if str((neighbor.get("metadata") or {}).get("chunk_type", "")).lower() in {"table", "references"}:
            continue
        neighbor["_table_context_relation"] = relation
        neighbor["_linked_from_table_chunk_id"] = table_chunk_id
        output.append(neighbor)
        seen.add(chunk_id)
    return output


def format_metadata_source(
    result: Dict[str, Any],
    source_number: int,
) -> Dict[str, Any]:
    """Format a document-level metadata hit as a citable context source."""
    metadata = result.get("metadata") or {}
    doc_name = result.get("doc_name") or metadata.get("doc_name")
    return {
        "source_number": source_number,
        "source_kind": "document_metadata",
        "chunk_id": None,
        "document_id": result.get("document_id") or doc_name,
        "doc_name": doc_name,
        "title": metadata.get("guideline_title") or metadata.get("title") or doc_name,
        "organization": metadata.get("guideline_source_authority") or metadata.get("organization"),
        "year": metadata.get("guideline_filename_year") or metadata.get("year"),
        "guideline_version": metadata.get("guideline_version"),
        "publication_date": metadata.get("guideline_publication_date"),
        "last_updated": metadata.get("guideline_last_updated"),
        "diseases_covered": metadata.get("guideline_diseases_covered"),
        "conditions_covered": metadata.get("guideline_conditions_covered"),
        "target_population": metadata.get("guideline_target_population"),
        "age_group": metadata.get("guideline_age_group"),
        "clinical_phase": metadata.get("guideline_clinical_phase"),
        "treatment_context": metadata.get("guideline_treatment_context"),
        "injury_context": metadata.get("guideline_injury_context"),
        "domain": metadata.get("domain"),
        "condition": metadata.get("condition"),
        "population": metadata.get("population"),
        "section": None,
        "subsection": None,
        "page": None,
        "source_type": "document_metadata",
        "text": "",
        "chunk_count": result.get("chunk_count"),
    }


def _display(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if value is None or value == "":
        return "not specified"
    return str(value)


def assemble_context(
    chunks: List[Dict[str, Any]],
    max_sources: int = 3,
    exclude_reference_only: bool = True,
    metadata_results: List[Dict[str, Any]] | None = None,
    max_metadata_sources: int = 5,
    include_table_neighbors: bool = False,
    table_neighbor_limit: int = 1,
    include_commentary: bool = False,
    collection=None,
) -> Dict[str, Any]:
    """
    Convert ranked retrieval results into compact, structured LLM context.

    Chunks arrive in ranked order from retrieval/reranking. This layer keeps
    the first `max_sources` eligible chunks so generation receives a compact
    context while retaining that ranking.

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

    # Keep the highest-ranked eligible results only; table neighbors are
    # appended afterward and never affect retrieval or reranker ordering.
    filtered = filtered[:max_sources]
    selected_chunk_ids = {
        str(chunk.get("id") or chunk.get("chunk_id") or "") for chunk in filtered
    }
    table_neighbors = (
        _linked_table_context(filtered, table_neighbor_limit, collection=collection)
        if include_table_neighbors
        else []
    )
    table_neighbors = [
        neighbor for neighbor in table_neighbors
        if str(neighbor.get("id") or "") not in selected_chunk_ids
    ]

    metadata_results = metadata_results or []
    if max_metadata_sources < 0:
        raise ValueError("max_metadata_sources must be >= 0.")
    metadata_sources = [
        format_metadata_source(result, index)
        for index, result in enumerate(metadata_results[:max_metadata_sources], start=1)
        if result.get("doc_name") or (result.get("metadata") or {}).get("doc_name")
    ]
    chunk_sources = [
        format_source(chunk, len(metadata_sources) + index, include_commentary=include_commentary)
        for index, chunk in enumerate(filtered, start=1)
    ]
    neighbor_sources = [
        format_source(neighbor, len(metadata_sources) + len(chunk_sources) + index,
                      include_commentary=include_commentary)
        for index, neighbor in enumerate(table_neighbors, start=1)
    ]
    sources = metadata_sources + chunk_sources + neighbor_sources

    context_blocks = []

    for source in sources:
        if source["source_kind"] == "document_metadata":
            block = (
                f"[SOURCE {source['source_number']}]\n"
                "source_kind: document_metadata\n"
                f"document_name: {_display(source['doc_name'])}\n"
                f"organization: {_display(source['organization'])}\n"
                f"catalog_year: {_display(source['year'])}\n"
                f"guideline_version: {_display(source.get('guideline_version'))}\n"
                f"publication_date: {_display(source.get('publication_date'))}\n"
                f"last_updated: {_display(source.get('last_updated'))}\n"
                f"diseases_covered: {_display(source.get('diseases_covered'))}\n"
                f"conditions_covered: {_display(source.get('conditions_covered'))}\n"
                f"target_population: {_display(source.get('target_population'))}\n"
                f"age_group: {_display(source.get('age_group'))}\n"
                f"clinical_phase: {_display(source.get('clinical_phase'))}\n"
                f"treatment_context: {_display(source.get('treatment_context'))}\n"
                f"injury_context: {_display(source.get('injury_context'))}\n"
                "year_note: This is the year recorded in corpus metadata. "
                "Do not treat it as a publication, revision, or file date unless another source states that explicitly.\n"
                f"[/SOURCE {source['source_number']}]"
            )
        else:
            relation_block = (
                f"relation_to_table: {_display(source['context_relation'])}\n"
                f"linked_from_table_chunk_id: {_display(source['linked_from_table_chunk_id'])}\n"
                if source.get("source_kind") == "table_context"
                else ""
            )
            commentary_block = (
                f"commented_table_note (model-generated aid; raw content remains authoritative):\n"
                f"{source['commented_text']}\n"
                if source.get("commented_text")
                else ""
            )
            block = (
                f"[SOURCE {source['source_number']}]\n"
                f"source_kind: {source['source_kind']}\n"
                f"chunk_id: {_display(source['chunk_id'])}\n"
                f"document_id: {_display(source['document_id'])}\n"
                f"document_name: {_display(source['doc_name'])}\n"
                f"title: {_display(source['title'])}\n"
                f"organization: {_display(source['organization'])}\n"
                f"catalog_year: {_display(source['year'])}\n"
                f"domain: {_display(source['domain'])}\n"
                f"condition: {_display(source['condition'])}\n"
                f"population: {_display(source['population'])}\n"
                f"section: {_display(source['section'])}\n"
                f"subsection: {_display(source['subsection'])}\n"
                f"page: {_display(source['page'])}\n"
                f"source_type: {_display(source['source_type'])}\n"
                f"{relation_block}"
                f"content:\n{source['text']}\n"
                f"{commentary_block}"
                f"[/SOURCE {source['source_number']}]"
            )
        context_blocks.append(block)

    return {
        "sources": sources,
        "context_text": "\n\n".join(context_blocks),
        "retrieved_contexts": context_blocks,
        "source_count": len(sources),
        "chunk_source_count": len(chunk_sources),
        "table_neighbor_count": len(neighbor_sources),
        "metadata_source_count": len(metadata_sources),
        "input_count": original_count,
        "reference_filtered_count": original_count - after_reference_filter,
        "duplicate_filtered_count": after_reference_filter - after_dedup,
        "max_sources": max_sources,
        "max_metadata_sources": max_metadata_sources,
        "include_table_neighbors": include_table_neighbors,
        "table_neighbor_limit": table_neighbor_limit,
        "include_commentary": include_commentary,
    }

