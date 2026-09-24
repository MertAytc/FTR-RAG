from src.retrieval.mmr_retriever import MMRRetriever


def main():
    query = input("FTR sorusunu gir: ").strip()

    retriever = MMRRetriever(
        candidate_k=20,
        final_k=6,
        lambda_mult=0.75,
        rrf_weight=0.75,
        semantic_weight=0.25,
    )

    result = retriever.search(query)

    print("\n" + "=" * 75)
    print("HYBRID + RRF-AĞIRLIKLI EMBEDDING MMR — FINAL TOP 6")
    print("=" * 75)

    for rank, item in enumerate(result["results"], start=1):
        metadata = item["metadata"]

        print(f"\n[{rank}]")
        print(f"ID                 : {item['id']}")
        print(f"MMR Score          : {item['mmr_score']:.6f}")
        print(f"Final Relevance    : {item['relevance']:.6f}")
        print(f"Normalized RRF     : {item['normalized_rrf']:.6f}")
        print(f"Semantic Similarity: {item['semantic_similarity']:.6f}")
        print(f"Redundancy         : {item['redundancy']:.6f}")
        print(f"RRF Score          : {item['rrf_score']:.6f}")
        print(f"Dense Rank         : {item['dense_rank']}")
        print(f"BM25 Rank          : {item['bm25_rank']}")
        print(f"Document           : {metadata.get('document_id')}")
        print(f"Page               : {metadata.get('page')}")
        print(f"Section            : {metadata.get('section')}")
        print(f"Type               : {metadata.get('source_type')}")

        print("\nTEXT:")
        print(item["document"].replace("\n", " ")[:1000])
        print("\n" + "-" * 75)


if __name__ == "__main__":
    main()
