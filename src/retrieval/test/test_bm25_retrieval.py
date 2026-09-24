from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.query_processor import process_query


def main():
    query = input("FTR sorusunu gir: ").strip()

    print("\nQuery Processor çalışıyor...")

    # Türkçe sorguyu klinik terminoloji ile genişlet
    processed = process_query(query)

    print("\nOrijinal sorgu:")
    print(processed["original_query"])

    print("\nEşleşen FTR terimleri:")
    for term in processed["matched_terms"]:
        print(f"- {term}")

    print("\nEklenen İngilizce terimler:")
    for term in processed["added_terms"]:
        print(f"- {term}")

    print("\nGenişletilmiş BM25 sorgusu:")
    print(processed["expanded_query"])

    print("\nBM25 index oluşturuluyor...")

    retriever = BM25Retriever()

    print("\nBM25 aranıyor...")

    results = retriever.search(
        processed["expanded_query"],
        n_results=5
    )

    print("\n" + "=" * 70)
    print("BM25 + QUERY PROCESSOR SONUÇLARI")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n[{rank}]")
        print(f"ID       : {result['id']}")
        print(f"Score    : {result['score']:.4f}")
        print(f"Document : {metadata.get('document_id')}")
        print(f"Section  : {metadata.get('section')}")
        print(f"Subsect. : {metadata.get('subsection')}")
        print(f"Page     : {metadata.get('page')}")
        print(f"Type     : {metadata.get('source_type')}")

        print("\nTEXT:")

        text = result["document"].replace("\n", " ")

        print(text[:1000])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()