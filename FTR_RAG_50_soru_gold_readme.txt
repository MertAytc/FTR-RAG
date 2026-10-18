FTR-RAG 50 soru gold chunk eşleştirmesi

Girdi: ftr_rag_50_soru_benchmark(1).json
Chunk kaynağı: output_v2_14.zip içindeki v2.14 chunk JSON dosyaları

Sonuç:
- Soru sayısı: 50
- Chunk JSON dokümanı: 52
- Chunk toplamı: 12.880
- En az bir gold_chunk_id adayı atanan soru: 45
- Korpus dışında/kanıt yok olarak tasarlanan ve gold_chunk_ids boş bırakılan soru: 5 (FTR-044–FTR-048)
- Atanan tüm chunk_id değerleri output_v2_14.zip içindeki chunk_id alanlarında bulundu.

ÖNEMLİ SINIRLAMA:
Bu dosya otomatik metin eşleştirmesiyle üretilmiş aday gold eşleştirmelerini içerir. Benchmark'ın resmi gold standardı kabul edilmeden önce FTR öğrencileri/uzman tarafından gold_passage, kaynak sayfası ve seçilen chunk metni yan yana doğrulanmalıdır. Özellikle çok belgeli sorular ve OCR/tablo metinleri elle kontrol edilmelidir.

Chroma indeksinin dosyası/veritabanı ve benchmark çalıştırma scripti yüklenmediği için mevcut Chroma kayıt sayısı doğrulanmadı ve benchmark çalıştırılmadı. 12.880 sayısı, ZIP'teki chunk JSON çıktılarının toplamıdır; Chroma index count değildir.
