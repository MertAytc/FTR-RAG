
from pathlib import Path
import sys
import re
import ollama

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import GENERATION_MODEL, SYSTEM_PROMPT, validate_citations

QUESTIONS = [
    "DCD tanısı olan çocuklarda fizik tedavi planı nasıl oluşturulmalıdır ve hangi tür müdahaleler önerilebilir?",
    "DCD olan çocuklarda fizik tedavi sırasında hedef belirleme, ilerleme ve yeniden değerlendirme nasıl yapılmalıdır?",
    "Fizik tedavide egzersizlerin ilerletilmesi hangi kriterlere göre yapılmalıdır? Ağrı, şişlik, uzun süren kas ağrısı ve fonksiyon nasıl dikkate alınmalıdır?",
    "Postoperatif dönemde progresif direnç egzersizleri hangi koşullarda uygulanabilir ve ilerleme sırasında hangi riskler izlenmelidir?",
    "Fizik tedavi programında egzersizlerin sıklığı, set ve tekrar sayısı veya süre gibi doz bilgileri kaynaklarda belirtilmişse nasıl uygulanmalıdır? Kaynakta belirtilmeyen değerleri tahmin etme.",
    "Hastanın egzersiz sırasında ağrısı veya şişliği artarsa egzersiz programında nasıl bir değişiklik yapılmalıdır ve hangi durumlarda yeniden değerlendirme gerekir?",
    "Evde uygulanabilecek fiziksel egzersiz programlarının planlanmasında hangi unsurlar dikkate alınmalıdır?",
    "Fizik tedavide egzersiz seçimi ve yoğunluğu hastanın toleransı, fonksiyonu ve hedefleri doğrultusunda nasıl belirlenmelidir?",
    "DCD olan çocuklarda fizik tedavi müdahaleleri sırasında çocuğun ve ailenin hedefleri ve tercihleri nasıl dikkate alınmalıdır?",
    "Fizik tedavi programının etkili olup olmadığı nasıl izlenmeli, hangi durumlarda müdahale değiştirilmeli veya yeniden değerlendirme yapılmalıdır?",
]

USER_INSTRUCTION = """
Yanıtı yalnızca CONTEXT'e dayanarak Türkçe ver.
En fazla 5 kısa madde kullan.
Egzersiz veya tedavi/müdahale önerilerini yalnızca kaynakta destekleniyorsa belirt.
Kaynakta bulunan set, tekrar, süre, sıklık, yoğunluk veya ilerleme kriterlerini aynen koru.
Kaynakta bulunmayan sayısal değerleri kesinlikle tahmin etme veya uydurma.
İlaç, ilaç dozu veya ilaç tedavisi önerme.
Tanı koyma ve tıbbi tedavi planı uydurma.
Her klinik iddianın sonunda [SOURCE X] kullan.
Birden fazla kaynak gerekiyorsa [SOURCE 1, SOURCE 3] biçimini kullan.
Sorunun istediği spesifik bilgi CONTEXT'te yoksa bunu açıkça belirt ve eksik bilgiyi başka bilgilerle doldurma.
"""

def ask_model(query, context_text):
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KULLANICI SORUSU:\n{query}\n\nCONTEXT:\n{context_text}\n\n{USER_INSTRUCTION}"},
        ],
        options={"temperature": 0.0, "num_predict": 350},
    )
    return response["message"]["content"].strip()

def count_citations(answer):
    return len(re.findall(r"\[SOURCE\s+[0-9]+(?:\s*,\s*[0-9]+)*\]", answer, flags=re.I))

def main():
    print("=" * 88)
    print("FTR-RAG 10 SORULUK EGZERSİZ / TEDAVİ-MÜDAHALE TÜRKÇE TESTİ")
    print("=" * 88)
    print(f"Generation model: {GENERATION_MODEL}")
    print("Bu test otomatik RAGAS skoru üretmez; Türkçe kaliteyi manuel değerlendirmek için çıktı toplar.")
    print()

    retriever = MMRRetriever()
    grounded_total = 0
    citation_total = 0
    medication_signal_total = 0

    for i, query in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 88)
        print(f"TEST {i}/10")
        print(f"SORU: {query}")
        print("=" * 88)

        retrieval = retriever.search(query)
        chunks = retrieval["results"][:3]

        print("\nRETRIEVED CHUNKS")
        for rank, chunk in enumerate(chunks, 1):
            print(
                f"{rank}. {chunk.get('document_id')} | {chunk.get('chunk_id')} | "
                f"page={chunk.get('page')} | MMR={chunk.get('mmr_score', 0):.4f} | "
                f"RRF={chunk.get('rrf_score', 0):.6f}"
            )

        context_result = assemble_context(chunks, max_sources=3, exclude_reference_only=True)
        answer = ask_model(query, context_result["context_text"])
        citation_valid = validate_citations(answer, context_result["source_count"])
        medication_signal = any(
            x in answer.lower()
            for x in ["mg", "milligram", "tablet", "kapsül", "ilaç dozu", "ilaç kullan"]
        )

        grounded_total += int(citation_valid)
        citation_total += count_citations(answer)
        medication_signal_total += int(medication_signal)

        print("\nGENERATION")
        print(f"Context source count: {context_result['source_count']}")
        print(f"Grounded citation syntax: {citation_valid}")
        print(f"Potential medication signal: {medication_signal}")
        print("\nANSWER:")
        print(answer)

    print("\n" + "=" * 88)
    print("ÖZET")
    print("=" * 88)
    print(f"Toplam test                     : 10")
    print(f"Geçerli citation syntax         : {grounded_total}/10")
    print(f"Toplam [SOURCE] citation        : {citation_total}")
    print(f"Potansiyel ilaç/doz sinyali     : {medication_signal_total}")
    print("\nMANUEL TÜRKÇE KALİTE PUANI")
    print("Her cevap için 5 başlık: 0-2 puan.")
    print("1) Türkçe akıcılık/anlaşılabilirlik")
    print("2) Klinik doğruluk/terminoloji")
    print("3) Kaynaklara sadakat")
    print("4) Egzersiz/müdahale bilgisinin soruya uygunluğu")
    print("5) Uydurma veya gereksiz bilgi vermeme")
    print("Her soru 10 puan; 10 soru toplam 100 puan.")
    print("Bu skor RAGAS değildir; RAGAS daha sonra ayrı değerlendirilmelidir.")

if __name__ == "__main__":
    main()
