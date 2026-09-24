from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context


def main():
    query = input("FTR sorusunu gir: ").strip()

    print("\nRetrieval + Context Assembly v2 başlatılıyor...")

    retriever = MMRRetriever(
        candidate_k=20,
        final_k=6,
        lambda_mult=0.75,
        rrf_weight=0.75,
        semantic_weight=0.25,
    )

    retrieval_result = retriever.search(query)
    mmr_results = retrieval_result["results"]

    context_result = assemble_context(
        mmr_results,
        max_sources=6,
        exclude_reference_only=True,
    )

    print("\n" + "=" * 80)
    print("CONTEXT ASSEMBLY v2")
    print("=" * 80)
    print(f"MMR kaynak sayısı          : {context_result['input_count']}")
    print(f"Reference filtrelenen      : {context_result['reference_filtered_count']}")
    print(f"Duplicate filtrelenen      : {context_result['duplicate_filtered_count']}")
    print(f"Final context kaynak sayısı: {context_result['source_count']}")

    print("\n" + "-" * 80)
    print("STRUCTURED SOURCES")
    print("-" * 80)

    for source in context_result["sources"]:
        print(
            f"\nSOURCE {source['source_number']} | "
            f"{source['document_id']} | "
            f"page={source['page']} | "
            f"chunk={source['chunk_id']}"
        )
        print(f"title       : {source['title']}")
        print(f"organization: {source['organization']}")
        print(f"year        : {source['year']}")
        print(f"condition   : {source['condition']}")
        print(f"population  : {source['population']}")
        print(f"section     : {source['section']}")
        print(f"subsection  : {source['subsection']}")
        print(f"type        : {source['source_type']}")
        print(f"content     : {source['text'][:700]}")

    print("\n" + "=" * 80)
    print("LLM'E GÖNDERİLECEK CONTEXT")
    print("=" * 80)
    print(context_result["context_text"])


if __name__ == "__main__":
    main()
