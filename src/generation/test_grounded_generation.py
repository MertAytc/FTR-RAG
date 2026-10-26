from src.generation.grounded_generation import generate_grounded_answer
from src.generation.context_assembly import assemble_context
from src.retrieval.reranker import CrossEncoderReranker


def run_query(query: str):
    print("\n" + "=" * 80)
    print(f"RRF Top-20 -> Cross-Encoder Top-5 -> Grounded Generation: {query}")
    print("=" * 80)

    reranker = CrossEncoderReranker()
    pipeline = reranker.rerank_hybrid(query, rrf_top_k=20, top_k=5)
    retrieval = pipeline["retrieval"]
    processed_query = retrieval["processed_query"]
    reranked_results = pipeline["reranked_results"]
    context_result = assemble_context(
        reranked_results,
        max_sources=5,
        exclude_reference_only=True,
    )

    print("\nQUERY PROCESSING")
    print(f"Original English : {query}")
    print(f"Search query      : {processed_query['rewritten_query']}")
    print(f"BM25 query       : {processed_query['expanded_query']}")
    print(f"RRF candidates   : {len(pipeline['rrf_candidates'])}")

    print("\nCROSS-ENCODER RERANKER RESULTS")
    for rank, item in enumerate(reranked_results, start=1):
        metadata = item.get("metadata") or {}
        print(
            f"{rank}. {item.get('id') or item.get('chunk_id')} | "
            f"reranker={item.get('reranker_score', 0):.4f} | "
            f"RRF={item.get('rrf_score', 0):.6f} | "
            f"page={metadata.get('page', 'unknown')}"
        )

    result = generate_grounded_answer(
        query,
        context_result,
        english_query=processed_query["rewritten_query"],
    )

    print("\nGROUNDED GENERATION")
    print(f"Context sources     : {context_result['source_count']}")
    print(f"Grounded            : {result.grounded}")
    print(f"Insufficient evidence: {result.insufficient_evidence}")
    print(f"Out of scope        : {result.out_of_scope}")
    print("\nENGLISH ANSWER")
    print(result.english_answer)
    print("\nFINAL ANSWER")
    print(result.answer)

    print("\nSOURCES")
    for source in result.sources:
        print(
            f"[SOURCE {source['source_number']}] "
            f"{source['document_id']} | page={source['page']} | "
            f"chunk={source['chunk_id']}"
        )


def main():
    run_query(input("English FTR question: ").strip())


if __name__ == "__main__":
    main()
