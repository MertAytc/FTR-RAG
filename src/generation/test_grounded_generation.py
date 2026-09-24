from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import generate_grounded_answer


def run_query(query: str):
    print("\n" + "=" * 80)
    print(f"FULL PIPELINE TEST: {query}")
    print("=" * 80)

    retriever = MMRRetriever(
        candidate_k=20,
        final_k=3,
        lambda_mult=0.75,
        rrf_weight=0.85,
        semantic_weight=0.15,
    )

    retrieval_result = retriever.search(query)
    processed_query = retrieval_result["processed_query"]

    print("\n" + "-" * 80)
    print("QUERY PROCESSING")
    print("-" * 80)
    print(f"Original Turkish : {processed_query['original_query']}")
    print(f"English rewrite  : {processed_query['rewritten_query']}")
    print(f"BM25 query       : {processed_query['expanded_query']}")

    print("\n" + "-" * 80)
    print("MMR RETRIEVAL RESULTS")
    print("-" * 80)
    for index, item in enumerate(retrieval_result["results"], start=1):
        metadata = item.get("metadata") or {}
        print(
            f"{index}. {item.get('id') or item.get('chunk_id')} | "
            f"MMR={item.get('mmr_score', item.get('score', 0)):.4f} | "
            f"RRF={item.get('rrf_score', 0):.6f} | "
            f"page={metadata.get('page', 'belirtilmemiş')}"
        )

    context_result = assemble_context(
        retrieval_result["results"],
        max_sources=3,
        exclude_reference_only=True,
    )

    result = generate_grounded_answer(
        query,
        context_result,
        english_query=processed_query["rewritten_query"],
    )

    print("\n" + "-" * 80)
    print("GROUNDED LLM GENERATION")
    print("-" * 80)
    print(f"Context kaynak sayısı : {context_result['source_count']}")
    print(f"Grounded              : {result.grounded}")
    print(f"Yetersiz kanıt        : {result.insufficient_evidence}")
    print(f"Scope dışı            : {result.out_of_scope}")

    print("\nENGLISH GROUNDED ANSWER")
    print(result.english_answer)

    print("\nTÜRKÇE FINAL ANSWER")
    print(result.answer)

    print("\nKAYNAKLAR")
    for source in result.sources:
        print(
            f"[SOURCE {source['source_number']}] "
            f"{source['document_id']} | page={source['page']} | "
            f"chunk={source['chunk_id']}"
        )


def main():
    query = input("Türkçe FTR sorusu: ").strip()
    run_query(query)


if __name__ == "__main__":
    main()
