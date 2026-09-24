from src.retrieval.mmr_retriever import MMRRetriever


TEST_QUERIES = [
    # 1 — comment/paragraph style
    (
        "Hastanın egzersiz sırasında ağrısı artıyor ve seans sonrasında "
        "uzun süren ağrı ile şişlik oluşuyor. Buna rağmen egzersizleri "
        "ilerletmek güvenli midir, hangi kriterlere göre yük azaltılmalı "
        "veya ilerleme yapılmalıdır?"
    ),

    # 2 — plan generation
    (
        "DCD tanısı olan bir çocuk için fizik tedavi planı hazırlarken "
        "değerlendirme, hedef belirleme, müdahale seçimi ve düzenli "
        "yeniden değerlendirme açısından hangi noktalar dikkate alınmalıdır?"
    ),

    # 3 — dosage
    (
        "Ev egzersiz programında egzersizlerin sıklığı, seans süresi ve "
        "uygulama dozunu belirlerken hastanın fonksiyonel hedefleri ve "
        "toleransı nasıl dikkate alınmalıdır?"
    ),

    # 4 — safety
    (
        "Fizik tedavi sırasında güvenlik açısından hangi durumlarda "
        "egzersiz durdurulmalı, yük azaltılmalı veya hasta yeniden "
        "değerlendirilmelidir?"
    ),

    # 5 — outcome/reassessment
    (
        "Tedavi başladıktan sonra hastanın ilerleme gösterip göstermediği "
        "nasıl değerlendirilir ve ölçülen sonuçlara göre fizik tedavi "
        "müdahaleleri ne zaman değiştirilmelidir?"
    ),

    # 6 — exercise progression
    (
        "Kuvvetlendirme ve progresif egzersiz programında yükün "
        "artırılmasına karar verirken ağrı, şişlik, yorgunluk, fonksiyon "
        "ve egzersiz toleransı gibi bulgular nasıl yorumlanmalıdır?"
    ),
]


def print_result(index, query, result):
    print("\n" + "=" * 90)
    print(f"TEST {index}")
    print("=" * 90)
    print(f"SORGU:\n{query}\n")

    for rank, item in enumerate(result["results"], start=1):
        metadata = item["metadata"]
        text = item["document"].replace("\n", " ")

        print(
            f"[{rank}] {item['id']} | "
            f"MMR={item['mmr_score']:.4f} | "
            f"Rel={item['relevance']:.4f} | "
            f"RRF={item['rrf_score']:.4f} | "
            f"Dense={item['dense_rank']} | "
            f"BM25={item['bm25_rank']} | "
            f"Page={metadata.get('page')}"
        )
        print(f"     {text[:450]}")


def main():
    retriever = MMRRetriever(
        candidate_k=20,
        final_k=6,
        lambda_mult=0.75,
        rrf_weight=0.75,
        semantic_weight=0.25,
    )

    for index, query in enumerate(TEST_QUERIES, start=1):
        result = retriever.search(query)
        print_result(index, query, result)


if __name__ == "__main__":
    main()
