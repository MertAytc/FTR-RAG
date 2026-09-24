from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ollama

from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import (
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    validate_citations,
)


CASES = [
    (
        "A_EN_QUERY_EN_ANSWER",
        "What criteria should be used to progress exercises in physical therapy?",
        "English",
    ),
    (
        "B_EN_QUERY_TR_ANSWER",
        "What criteria should be used to progress exercises in physical therapy?",
        "Turkish",
    ),
    (
        "C_TR_QUERY_TR_ANSWER",
        "Fizik tedavide egzersizlerin ilerletilmesi hangi kriterlere göre yapılmalıdır?",
        "Turkish",
    ),
    (
        "D_TR_QUERY_EN_ANSWER",
        "Fizik tedavide egzersizlerin ilerletilmesi hangi kriterlere göre yapılmalıdır?",
        "English",
    ),
]


def generate_with_same_context(query, context_text, language):
    if language == "English":
        instruction = (
            "Answer in English. Use only the CONTEXT. "
            "Use at most 4 short bullet points. "
            "Put [SOURCE X] after each important clinical claim. "
            "If the requested information is not explicitly in the CONTEXT, "
            "say that it is not specified and do not guess."
        )
    else:
        instruction = (
            "Yanıtı Türkçe ver. Yalnızca CONTEXT'i kullan. "
            "En fazla 4 kısa madde kullan. "
            "Her önemli klinik iddianın sonunda [SOURCE X] kullan. "
            "İstenen bilgi CONTEXT'te açıkça yoksa belirtilmediğini söyle ve tahmin etme."
        )

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"KULLANICI SORUSU:\n{query}\n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"{instruction}"
                ),
            },
        ],
        options={
            "temperature": 0.0,
            "num_predict": 300,
        },
    )

    answer = response["message"]["content"].strip()
    return answer


def print_chunks(chunks):
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        print(
            f"{i}. {metadata.get('document_id', '?')} | "
            f"{chunk.get('id', '?')} | "
            f"page={metadata.get('page', '?')} | "
            f"MMR={chunk.get('mmr_score', 0):.4f} | "
            f"RRF={chunk.get('rrf_score', 0):.6f}"
        )


def run():
    print("=" * 80)
    print("FTR-RAG 4-LU TÜRKÇE / İNGİLİZCE KONTROLLÜ DİL TESTİ v2")
    print("=" * 80)
    print("Aynı retrieved context üzerinde cevap dilini değiştirerek generation etkisini ölçer.")
    print()

    mmr = MMRRetriever()

    for name, query, answer_language in CASES:
        retrieval = mmr.search(query)
        chunks = retrieval["results"][:3]

        # IMPORTANT:
        # grounded_generation.py expects the assembled context_result,
        # not a raw list of chunks. We assemble exactly the same way
        # as the production pipeline.
        context_result = assemble_context(
            chunks,
            max_sources=3,
            exclude_reference_only=True,
        )

        print("\n" + "=" * 80)
        print(name)
        print("QUERY:", query)
        print("ANSWER LANGUAGE:", answer_language)
        print("=" * 80)

        print("\nRETRIEVED CHUNKS")
        print_chunks(chunks)

        print("\nCONTEXT SOURCES USED BY GENERATION")
        for source in context_result["sources"]:
            print(
                f"{source['source_number']}. "
                f"{source['chunk_id']} | page={source['page']}"
            )

        print("\nGENERATION")
        answer = generate_with_same_context(
            query,
            context_result["context_text"],
            answer_language,
        )

        grounded = validate_citations(
            answer,
            context_result["source_count"],
        )

        print("Grounded citation syntax:", grounded)
        print("\nANSWER:")
        print(answer)

    print("\n" + "=" * 80)
    print("YORUMLAMA")
    print("=" * 80)
    print("A vs B: Aynı İngilizce sorgu + aynı retrieved context; yalnızca cevap dili değişir.")
    print("C vs D: Aynı Türkçe sorgu + aynı retrieved context; yalnızca cevap dili değişir.")
    print("A/C: İngilizce ve Türkçe sorguların retrieval sonuçlarını karşılaştırır.")
    print("A/B veya C/D'de chunklar aynı olup cevap kalitesi değişiyorsa generation dili etkisi güçlenir.")
    print("Bu test RAGAS değildir; kontrollü teşhis testidir.")


if __name__ == "__main__":
    run()
