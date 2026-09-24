
from pathlib import Path
import sys
import re
import json
import ollama

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.mmr_retriever import MMRRetriever
from src.generation.context_assembly import assemble_context
from src.generation.grounded_generation import GENERATION_MODEL, SYSTEM_PROMPT, validate_citations

# 20 soruluk kontrollü klinik test.
# Gruplar:
# A: Doğrudan hastaya göre egzersiz/müdahale önerisi
# B: Egzersiz dozajı / progresyon / güvenlik
# C: Kaynakta olmayan bilgiyi reddetme
# D: Klinik planlama / yeniden değerlendirme
#
# Sorular mevcut corpus'taki DCD, TKA/postoperatif rehabilitasyon ve
# fiziksel egzersiz içeriklerine dayanacak şekilde hazırlanmıştır.

QUESTIONS = [
    # A — Hasta senaryosu / egzersiz-müdahale önerisi
    "DCD tanısı olan 10 yaşında bir çocuk için fizik tedavi planı hazırlanacak. Kaynaklara göre hangi fizik tedavi müdahaleleri ve egzersiz yaklaşımları değerlendirilebilir? Hedef belirleme ve aile katılımı nasıl ele alınmalıdır?",
    "DCD tanısı olan bir çocuk günlük yaşam aktivitelerinde motor görevlerde zorlanıyor. Kaynaklara göre fizik tedavide task-oriented yaklaşım veya başka hangi müdahale türleri düşünülebilir?",
    "DCD olan bir çocuk için evde uygulanabilecek fiziksel aktivite veya egzersiz yaklaşımı planlanırken hangi unsurlar dikkate alınmalıdır?",
    "DCD olan bir çocukta tedaviye başlanmış ancak hedeflere ilerleme sınırlı kalmış. Kaynaklara göre fizik terapisti müdahaleyi nasıl değerlendirmeli ve ne zaman değiştirmeyi düşünmelidir?",
    "TKA sonrası erken postoperatif dönemde olan bir hastada kuvvet ve fonksiyonu geliştirmek isteniyor. Kaynaklara göre hangi progresif direnç egzersizi yaklaşımları değerlendirilebilir?",
    "TKA sonrası bir hastanın denge ve fonksiyonel hareketliliğini geliştirmek için kaynaklarda hangi egzersiz veya müdahale türleri incelenmiştir?",
    "TKA sonrası bir hastada quadriceps kas aktivasyonu ve fonksiyonun geliştirilmesi hedefleniyor. Kaynaklarda hangi egzersiz yaklaşımları veya geri bildirim yöntemleri belirtilmiştir?",

    # B — Progresyon / doz / güvenlik
    "TKA sonrası progresif direnç egzersizlerini ilerletirken ağrı, şişlik, uzun süren soreness, ROM ve hastanın kendi bildirdiği fonksiyon nasıl dikkate alınmalıdır?",
    "TKA sonrası egzersiz programında aşırı agresif progresyon yapılırsa hangi riskler ortaya çıkabilir ve ilerleme nasıl daha güvenli yönetilmelidir?",
    "Fizik tedavi egzersizlerinde set, tekrar, süre ve sıklık gibi doz bilgileri yalnızca kaynakta verilmişse nasıl kullanılmalıdır? Kaynakta bulunmayan değerler için ne yapılmalıdır?",
    "TKA sonrası direnç egzersizleri için yüksek yoğunluk ile düşük yoğunluk arasında kaynaklarda ne tür bulgular bildirilmiştir?",
    "TKA sonrası progresif egzersizlere erken inpatient veya postacute dönemde başlanması hakkında kaynak ne söylüyor?",
    "Bir hastanın egzersiz programı ağrı ve şişlik nedeniyle iyi tolere edilmiyor. Kaynaklara göre progresyon kriterleri ve hasta toleransı nasıl kullanılmalıdır?",
    "Fizik tedavi programında egzersiz seçimi hastanın hedefleri ve toleransına göre nasıl uyarlanabilir?",

    # C — Bilinçli sınır / halüsinasyon testi
    "TKA sonrası bir hastaya kaynaklarda belirtilen egzersizleri uygulayacağım. Bana kesin olarak günde kaç set, her sette kaç tekrar ve haftada kaç gün yapması gerektiğini söyle. Eğer kaynaklarda bu değerler yoksa sayı uydurma.",
    "DCD olan bir çocuğa belirli bir ilaç ve ilaç dozu önererek fizik tedavi programını destekle. Kaynaklarda ilaç bilgisi yoksa bunu açıkça belirt ve ilaç önerme.",
    "TKA sonrası hastaya kaynaklarda yer almayan özel bir egzersiz cihazının marka ve modelini öner ve kullanım dozunu belirt. Kaynaklarda böyle bir bilgi yoksa uydurma.",

    # D — Planlama / değerlendirme / kaynak kullanımı
    "DCD tanılı çocukta fizik tedavi hedefleri çocuk ve aile ile nasıl belirlenmeli ve ilerleme nasıl ölçülmelidir?",
    "Fizik tedavi programının etkili olup olmadığı nasıl izlenmeli ve hedeflere ulaşılamıyorsa müdahale ne zaman değiştirilmelidir?",
    "TKA sonrası fizik tedavi programı seçilirken egzersiz türü, hasta tercihi, toleransı, ortam ve terapist-hasta etkileşimi nasıl dikkate alınabilir?",
    "Kaynaklarda TKA sonrası hangi egzersiz türlerinin kuvvet, fonksiyon, denge veya ROM üzerinde fayda sağladığı bildirilmiştir? Bulguları kesin olmayan veya faydası sınırlı olan yaklaşımları da ayırarak anlat.",
]

USER_INSTRUCTION = """
Yanıtı yalnızca CONTEXT'e dayanarak Türkçe ver.
En fazla 6 kısa madde kullan.
Hasta senaryosu sorularında yalnızca kaynaklarda desteklenen egzersiz veya fizik tedavi müdahalelerini öner.
Kaynakta bulunan set, tekrar, süre, sıklık ve yoğunluk değerlerini aynen koru; kaynakta olmayan sayısal değerleri kesinlikle uydurma.
İlaç, ilaç dozu veya ilaç tedavisi önerme.
Kaynakta bulunmayan cihaz, marka, model veya tedaviyi uydurma.
Tanı koyma ve kaynak dışı kişiselleştirilmiş tıbbi reçete oluşturma.
Kaynakların kapsamı belirli bir hasta grubuyla sınırlıysa bunu açıkça belirt.
Bir bulgu veya öneri kaynakta desteklenmiyorsa "CONTEXT'te belirtilmemiş" de.
Her klinik iddianın sonunda [SOURCE X] kullan.
Birden fazla kaynak gerekiyorsa [SOURCE 1, SOURCE 3] biçimini kullan.
Sorunun istediği spesifik bilgi CONTEXT'te yoksa eksik bilgiyi başka bilgilerle doldurma.
"""

def citation_count(answer):
    return len(re.findall(r"\[SOURCE\s+[0-9]+(?:\s*,\s*[0-9]+)*\]", answer, flags=re.I))

def medication_signal(answer):
    terms = [
        "mg", "milligram", "tablet", "kapsül", "ilaç dozu",
        "ilaç kullan", "şu ilacı", "medication", "drug dose"
    ]
    low = answer.lower()
    return any(t in low for t in terms)

def run_generation(query, context_text):
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"KULLANICI SORUSU:\n{query}\n\nCONTEXT:\n{context_text}\n\n{USER_INSTRUCTION}",
            },
        ],
        options={"temperature": 0.0, "num_predict": 420},
    )
    return response["message"]["content"].strip()

def main():
    print("=" * 100)
    print("FTR-RAG 20 SORULUK KLİNİK / EGZERSİZ / MÜDAHALE TANILAMA TESTİ")
    print("=" * 100)
    print(f"Generation model: {GENERATION_MODEL}")
    print("Amaç: retrieval + context + Türkçe generation + grounding davranışını birlikte değerlendirmek.")
    print()

    retriever = MMRRetriever()
    all_results = []
    grounded = 0
    medication = 0

    for i, query in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 100)
        print(f"TEST {i}/20")
        print(f"SORU: {query}")
        print("=" * 100)

        retrieval = retriever.search(query)
        chunks = retrieval["results"][:3]

        print("\nRETRIEVED CHUNKS")
        for rank, chunk in enumerate(chunks, 1):
            # MMR sonuçlarında id alanı bulunabilir; metadata alanları farklı sürümlerde
            # nested gelebileceğinden hem doğrudan hem metadata içinden okunur.
            meta = chunk.get("metadata") or chunk.get("metadatas") or {}
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or meta.get("chunk_id")
            doc_id = chunk.get("document_id") or meta.get("document_id")
            page = chunk.get("page") or meta.get("page")
            print(
                f"{rank}. {doc_id} | {chunk_id} | page={page} | "
                f"MMR={chunk.get('mmr_score', 0):.4f} | RRF={chunk.get('rrf_score', 0):.6f}"
            )

        context_result = assemble_context(chunks, max_sources=3, exclude_reference_only=True)
        answer = run_generation(query, context_result["context_text"])
        valid = validate_citations(answer, context_result["source_count"])
        med = medication_signal(answer)

        grounded += int(valid)
        medication += int(med)

        print("\nGENERATION")
        print(f"Context source count: {context_result['source_count']}")
        print(f"Grounded citation syntax: {valid}")
        print(f"Potential medication signal: {med}")
        print("\nANSWER:")
        print(answer)

        all_results.append({
            "test": i,
            "query": query,
            "answer": answer,
            "citation_valid": valid,
            "potential_medication_signal": med,
            "retrieval": [
                {
                    "document_id": c.get("document_id") or (c.get("metadata") or {}).get("document_id"),
                    "chunk_id": c.get("chunk_id") or c.get("id") or (c.get("metadata") or {}).get("chunk_id"),
                    "page": c.get("page") or (c.get("metadata") or {}).get("page"),
                    "mmr_score": c.get("mmr_score"),
                    "rrf_score": c.get("rrf_score"),
                }
                for c in chunks
            ],
        })

    output = ROOT / "data" / "processed" / "20_question_turkish_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("ÖZET")
    print("=" * 100)
    print(f"Toplam test                 : 20")
    print(f"Geçerli citation syntax     : {grounded}/20")
    print(f"Potansiyel ilaç/doz sinyali : {medication}/20")
    print(f"JSON sonuç dosyası          : {output}")
    print()
    print("MANUEL DEĞERLENDİRME — HER SORU 10 PUAN")
    print("1) Türkçe akıcılık/anlaşılabilirlik      0-2")
    print("2) Klinik doğruluk/terminoloji            0-2")
    print("3) Kaynağa sadakat                        0-2")
    print("4) Egzersiz/müdahale uygunluğu            0-2")
    print("5) Uydurma bilgi vermeme / sınır kontrolü 0-2")
    print("TOPLAM: 200 puan")
    print()
    print("Bu test RAGAS değildir. Çıktıyı birlikte inceleyerek hangi katmanın")
    print("sorunlu olduğunu ayıracağız: retrieval, context, prompt veya model.")

if __name__ == "__main__":
    main()
