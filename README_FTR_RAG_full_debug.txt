FTR-RAG FULL DEBUG + RAGAS HAZIRLIK BETIGI

Dosya: FTR_RAG_50_retrieval_full_debug_ragas.py

NE YAPAR?
- Mevcut Dense/Chroma ve BM25 indekslerini okur; indeks oluşturmaz, silmez veya yeniden kurmaz.
- BM25-only, Dense-only, Dense+BM25 RRF ve RRF+BGE reranker sonuçlarını karşılaştırır.
- Her yöntem için aday chunk ID'leri, sıraları, ham retrieval kayıtları, erişilebilen skor alanları, metin ve metadata'yı debug JSONL'ye yazar.
- Her gold chunk'ın bulunup bulunmadığını ve rank'ini kaydeder.
- Query processing çıktısını, rewritten/expanded sorguları ve süreleri kaydeder.
- Her yöntemden sonra sonuçları checkpoint eder; çalışma yarıda kesilirse tamamlanan sonuçlar korunur.
- RAGAS için question/contexts/ground_truth alanları olan JSONL üretir. Answer alanı özellikle null bırakılır; script yanıt üretmediği için uydurma cevap yazmaz.
- Çalışma manifesti, hash, Python/platform bilgisi ve log dosyası üretir.

NASIL ÇALIŞTIRILIR?
1. Dosyayı FTR-RAG proje kök dizinine kopyala (src/ klasörü ve benchmark JSON ile aynı dizine).
2. Proje sanal ortamını etkinleştir.
3. Önce iki soruluk hızlı deneme:
   python FTR_RAG_50_retrieval_full_debug_ragas.py --limit 2 --expected-count 50 --output-prefix debug_smoke_2q
   Not: --expected-count benchmark dosyasının toplam soru sayısını kontrol eder; --limit çalıştırılacak soru sayısını sınırlar.
4. Başarılıysa tam test:
   python FTR_RAG_50_retrieval_full_debug_ragas.py --output-prefix retrieval_full_debug_v2_14_run01

ÜRETİLEN DOSYALAR (prefix örneği retrieval_full_debug_v2_14_run01)
- *_results.csv: yöntem/soru başına metrikler ve kompakt aday listeleri.
- *_results.json: checkpoint'li sonuçlar, özet metrikler, konfigürasyon.
- *_debug.jsonl: ayrıntılı olay akışı; her retrieval yönteminin aday metinleri, metadata, skorları, gold eşleşmeleri ve ham kayıtları.
- *_ragas_retrieval_dataset.jsonl: RAGAS'a aktarılabilecek retrieval context kayıtları.
- *_run_manifest.json: konfigürasyon, benchmark SHA-256, ortam ve çalışma durumu.
- *_debug.log: log ve hata izleri.

RAGAS İÇİN ÖNEMLİ SINIRLAMA
Bu betik retrieval katmanını ölçer; LLM cevabı üretmez. Dolayısıyla Faithfulness, Answer Relevancy ve answer-grounded metrikleri bu dosyayla tek başına hesaplanamaz. Bunlar için gerçek üretim hattından alınan `answer` alanı eklenmeli; `ground_truth` alanı da benchmark'ta gerçek referans yanıt mevcutsa doldurulmalıdır. Bu betik gold chunk ID'lerini ve retrieval metriklerini ayrıca kaydeder; gold chunk ID, doğal dilde ground_truth yanıtıyla aynı şey değildir.

RAGAS'a geçmeden önce:
- *_ragas_retrieval_dataset.jsonl içindeki `contexts` alanlarının boş olmadığını kontrol et.
- `ground_truth` değerlerinin benchmark'taki doğrulanmış referans yanıtlar olduğunu teyit et; yoksa null kalması doğrudur.
- `answer` alanını gerçek LLM çıktılarıyla eşleştir; boş/null cevaplarla answer-based metrik çalıştırma.
- Bu çıktı RAGAS sürümüne göre Dataset/HF Dataset formatına dönüştürülmelidir; JSONL'nin kendisi her sürümde doğrudan evaluate() girdisi değildir.

DEBUG NOTLARI
- Ayrıntılı debug JSONL büyük olabilir; chunk metinleri varsayılan olarak her sonuç için 12.000 karakterle sınırlandırılır. Kesilmesin istenirse `--debug-text-max-chars 0` kullanılabilir.
- `--output-prefix` yeni ve kullanılmamış bir ad olmalı; betik var olan çıktıları ezmez.
- Betik burada yalnızca statik derleme kontrolünden geçirildi; senin Windows ortamındaki Ollama/Chroma/BM25 modelleriyle uçtan uca çalıştırma doğrulanmadı.
