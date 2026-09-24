from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import generate_grounded_answer


TEST_QUERIES = [
    (
        "Egzersiz progresyonu",
        "Fizik tedavide egzersizlerin ilerletilmesi hangi kriterlere göre yapılmalıdır?",
    ),
    (
        "DCD fizik tedavi planı",
        "DCD tanısı olan çocuklarda fizik tedavi hedefleri ve müdahaleler nasıl yönetilmelidir?",
    ),
    (
        "Egzersiz dozajı",
        "Fizik tedavide egzersiz yoğunluğu, sıklığı, süresi veya tekrarları hakkında kaynaklarda hangi bilgiler bulunmaktadır?",
    ),
    (
        "Güvenlik ve yeniden değerlendirme",
        "Egzersiz programı ilerletilirken hangi belirtiler izlenmelidir?",
    ),
    (
        "Yetersiz kanıt / abstention",
        "Her hasta egzersiz progresyonunda tam olarak kaç tekrar yapmalıdır?",
    ),
]


def main():
    hybrid = HybridRetriever(
        dense_top_n=20,
        bm25_top_n=20,
        rrf_k=60,
    )

    mmr = MMRRetriever(
        hybrid_retriever=hybrid,
        candidate_k=20,
        final_k=3,
        lambda_mult=0.75,
        rrf_weight=0.85,
        semantic_weight=0.15,
    )

    summary = []

    for number, (name, query) in enumerate(TEST_QUERIES, start=1):
        retrieval = mmr.search(query)

        context = assemble_context(
            retrieval["results"],
            max_sources=3,
            exclude_reference_only=True,
        )

        result = generate_grounded_answer(query, context)

        print("\n" + "=" * 80)
        print(f"TEST {number}: {name}")
        print(f"SORU: {query}")
        print("-" * 80)

        print("\nRETRIEVAL")
        for i, source in enumerate(context["sources"], start=1):
            print(
                f"{i}. {source.get('chunk_id')} | "
                f"page={source.get('page')} | "
                f"title={source.get('title')}"
            )

        print(f"\nContext kaynak sayısı : {len(context['sources'])}")
        print(f"Reference filtrelenen : {context.get('reference_filtered', 0)}")
        print(f"Duplicate filtrelenen : {context.get('duplicate_filtered', 0)}")

        print("\nÜRETİLEN TÜRKÇE CEVAP")
        print(result.answer)

        print("\nKONTROLLER")
        print(f"Grounded          : {result.grounded}")
        print(f"Yetersiz kanıt    : {result.insufficient_evidence}")
        print(f"Scope dışı        : {result.out_of_scope}")

        summary.append(result)

    print("\n" + "=" * 80)
    print("BENCHMARK ÖZETİ")
    print("=" * 80)
    print(f"Toplam test       : {len(summary)}")
    print(f"Grounded          : {sum(r.grounded for r in summary)}")
    print(f"Yetersiz kanıt    : {sum(r.insufficient_evidence for r in summary)}")
    print(f"Scope dışı        : {sum(r.out_of_scope for r in summary)}")


if __name__ == "__main__":
    main()
