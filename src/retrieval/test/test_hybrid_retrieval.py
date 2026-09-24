from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    query = input("FTR sorusunu gir: ").strip()

    print("\nHybrid Retrieval başlatılıyor...")

    retriever = HybridRetriever(
        dense_top_n=20,
        bm25_top_n=20,
        rrf_k=60,
    )

    result = retriever.search(
        query,
        n_results=8,
    )

    processed = result["processed_query"]

    print("\nOrijinal sorgu:")
    print(processed["original_query"])

    print("\nEşleşen FTR terimleri:")
    for term in processed["matched_terms"]:
        print(f"- {term}")

    print("\nGenişletilmiş BM25 sorgusu:")
    print(processed["expanded_query"])

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL — RRF TOP 8")
    print("=" * 70)

    for rank, item in enumerate(result["results"], start=1):
        metadata = item["metadata"]

        print(f"\n[{rank}]")
        print(f"ID         : {item['id']}")
        print(f"RRF Score  : {item['rrf_score']:.6f}")
        print(f"Dense Rank : {item['dense_rank']}")
        print(f"BM25 Rank  : {item['bm25_rank']}")
        print(f"Dense Score: {item['dense_score']}")
        print(f"BM25 Score : {item['bm25_score']}")
        print(f"Document   : {metadata.get('document_id')}")
        print(f"Page       : {metadata.get('page')}")
        print(f"Section    : {metadata.get('section')}")
        print(f"Type       : {metadata.get('source_type')}")

        text = item["document"].replace("\n", " ")
        print("\nTEXT:")
        print(text[:800])
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
