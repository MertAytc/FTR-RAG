from src.retrieval.embedding import embed_texts
from src.retrieval.vector_store import search


def main():

    query = input("\nFTR sorusunu gir: ").strip()

    if not query:
        print("Soru boş bırakılamaz.")
        return

    print("\nQuery embedding oluşturuluyor...")

    query_embedding = embed_texts([query])[0]

    print("ChromaDB aranıyor...\n")

    results = search(
        query_embedding=query_embedding,
        n_results=5,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("=" * 70)
    print("DENSE RETRIEVAL SONUÇLARI")
    print("=" * 70)

    for i in range(len(ids)):

        metadata = metadatas[i]

        print(f"\n[{i + 1}]")
        print(f"ID       : {ids[i]}")
        print(f"Distance : {distances[i]}")
        print(f"Document : {metadata.get('document_id')}")
        print(f"Section  : {metadata.get('section')}")
        print(f"Subsect. : {metadata.get('subsection')}")
        print(f"Page     : {metadata.get('page')}")
        print(f"Type     : {metadata.get('source_type')}")

        print("\nTEXT:")
        print(documents[i][:1000])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()