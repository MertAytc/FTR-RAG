from src.generation.context_assembly import assemble_context
from src.retrieval.reranker import CrossEncoderReranker


def main():
    query = input("FTR question: ").strip()
    reranker = CrossEncoderReranker()
    pipeline = reranker.rerank_hybrid(query, rrf_top_k=20, top_k=5)
    context_result = assemble_context(
        pipeline["reranked_results"],
        max_sources=5,
        exclude_reference_only=True,
    )

    print("\nRRF Top-20 -> Cross-Encoder Top-5 -> Context Assembly")
    print(f"RRF candidates          : {len(pipeline['rrf_candidates'])}")
    print(f"Reranker results        : {len(pipeline['reranked_results'])}")
    print(f"Reference chunks removed: {context_result['reference_filtered_count']}")
    print(f"Duplicates removed     : {context_result['duplicate_filtered_count']}")
    print(f"Final context sources   : {context_result['source_count']}")

    for source in context_result["sources"]:
        print(
            f"\n[SOURCE {source['source_number']}] "
            f"{source['document_id']} | page={source['page']} | "
            f"chunk={source['chunk_id']}"
        )
        print(f"Title: {source['title']}")
        print(f"Content: {source['text'][:700]}")

    print("\nMODEL CONTEXT")
    print(context_result["context_text"])


if __name__ == "__main__":
    main()
