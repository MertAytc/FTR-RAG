# İlk 50 Soru Retrieval Karşılaştırma Raporu

**Tarih:** 31 Ekim 2026  
**Karşılaştırma:** Ham `text` embedding'i ve `text + commented_text` embedding'i  
**Kohort:** Retrieval değerlendirmesine uygun 39 soru

## Özet

İki Dense embedding varyantı aynı 50 soruda, aynı corpus ve aynı retrieval ayarlarıyla değerlendirildi. BM25 her iki run'da da tüm semantik metadata alanlarını kullandı. Reranker, konuşulan seçili klinik metadata alanlarıyla çalıştı. Dense tarafında değişen tek girdi, özgün chunk metnine `commented_text` eklenmesiydi.

Commentary eklemek Dense ve RRF sıralamalarında bazı soruların ilk beşini değiştirdi; ancak makro retrieval metriklerinde anlamlı bir artış üretmedi. Final reranker metrikleri iki run'da da aynı çıktı. Bu karşılaştırmanın sonuçları, bu benchmark ve mevcut relevance etiketleriyle sınırlıdır.

## Kurulum ve değerlendirme kapsamı

- Her run'da ilk 50 benchmark sorusu işlendi. Retrieval metrikleri yalnızca `evaluation_track = retrieval` olan **39 soru** üzerinden hesaplandı.
- Diğer 11 soru retrieval metriklerine katılmadı: 5 metadata-attribution, 5 pending-gold, 1 excluded-invalid-gold.
- Corpus: `chunks_context_tables_v1_workbook_metadata`; Dense koleksiyon başına 13,895 chunk. Chunk kimlikleri, metin ve metadata aynı kaldı.
- **BM25:** tüm semantik metadata (`metadata_mode=all`). İki run'da BM25 sıralaması değişmedi.
- **Dense:** `qwen3-embedding:0.6b`; ilk run'da özgün `text`, ikinci run'da özgün `text` ile varsa `commented_text` birlikte embed edildi.
- **RRF:** BM25 ve Dense adayları birleştirildi; mevcut RRF ağırlıkları ve kalite/atıf cezası korundu.
- **Reranker:** `BAAI/bge-reranker-v2-m3`; seçili klinik metadata bağlamı her iki run'da aynı şekilde kullanıldı.
- Generation ve RAGAS puanlaması bu raporun kapsamı dışındadır. RAGAS çalıştırılmadı.

## Metrik tanımları

- **Strict canonical:** Yalnızca orijinal gold chunk ID'leri ilgili kabul edildi.
- **Acceptable full:** Orijinal gold ID'leri ve havuz yargısında tam destek veren alternatifler ilgili kabul edildi.
- **Acceptable + partial:** Full destekli alternatiflere ek olarak kısmi destekli chunk'lar da ilgili kabul edildi; bu üst sınır yorumudur.
- **Hit@1:** İlk sıradaki chunk'ın en az bir relevance grubunu karşılaması.
- **Recall@k:** Soru başına karşılanan relevance grubu oranlarının ortalaması (makro). `MRR@20` ve `nDCG@10` raporlandı.
- Çoklu relevance grupları ayrı sayıldı. Kaynak rapor JSON'larında micro recall ve full-hit metrikleri de korunmuştur.

## Strict canonical

| Aşama | Embedding | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@20 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Raw text | 0.231 | 0.397 | 0.526 | 0.615 | 0.360 | 0.409 |
| BM25 | Commented text | 0.231 | 0.397 | 0.526 | 0.615 | 0.360 | 0.409 |
| Dense | Raw text | 0.231 | 0.436 | 0.551 | 0.692 | 0.369 | 0.435 |
| Dense | Commented text | 0.231 | 0.436 | 0.551 | 0.692 | 0.369 | 0.435 |
| RRF | Raw text | 0.308 | 0.551 | 0.577 | 0.744 | 0.469 | 0.519 |
| RRF | Commented text | 0.308 | 0.551 | 0.577 | 0.744 | 0.469 | 0.519 |
| RRF + reranker | Raw text | 0.410 | 0.551 | 0.641 | 0.744 | 0.520 | 0.565 |
| RRF + reranker | Commented text | 0.410 | 0.551 | 0.641 | 0.744 | 0.520 | 0.565 |

## Acceptable full

| Aşama | Embedding | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@20 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Raw text | 0.385 | 0.538 | 0.679 | 0.744 | 0.513 | 0.552 |
| BM25 | Commented text | 0.385 | 0.538 | 0.679 | 0.744 | 0.513 | 0.552 |
| Dense | Raw text | 0.410 | 0.564 | 0.731 | 0.846 | 0.547 | 0.603 |
| Dense | Commented text | 0.410 | 0.564 | 0.731 | 0.846 | 0.546 | 0.603 |
| RRF | Raw text | 0.487 | 0.756 | 0.782 | 0.872 | 0.646 | 0.686 |
| RRF | Commented text | 0.462 | 0.756 | 0.782 | 0.872 | 0.633 | 0.676 |
| RRF + reranker | Raw text | 0.667 | 0.769 | 0.795 | 0.872 | 0.741 | 0.758 |
| RRF + reranker | Commented text | 0.667 | 0.769 | 0.795 | 0.872 | 0.741 | 0.758 |

## Acceptable + partial (üst sınır)

| Aşama | Embedding | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@20 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Raw text | 0.667 | 0.821 | 0.846 | 0.885 | 0.772 | 0.615 |
| BM25 | Commented text | 0.667 | 0.821 | 0.846 | 0.885 | 0.772 | 0.615 |
| Dense | Raw text | 0.538 | 0.769 | 0.910 | 0.936 | 0.686 | 0.589 |
| Dense | Commented text | 0.538 | 0.769 | 0.910 | 0.936 | 0.686 | 0.589 |
| RRF | Raw text | 0.718 | 0.897 | 0.910 | 0.962 | 0.835 | 0.708 |
| RRF | Commented text | 0.692 | 0.897 | 0.910 | 0.962 | 0.822 | 0.699 |
| RRF + reranker | Raw text | 0.923 | 0.949 | 0.974 | 0.987 | 0.962 | 0.809 |
| RRF + reranker | Commented text | 0.923 | 0.949 | 0.974 | 0.987 | 0.962 | 0.809 |

## Sıralama değişiklikleri

İki run'ın ilk beş chunk ID'si soru bazında karşılaştırıldı:

| Aşama | İlk 5'i değişen soru sayısı |
|---|---:|
| BM25 | 0 / 50 |
| Dense | 10 / 50 |
| RRF | 9 / 50 |
| Reranker | 2 / 50 |
| Generation'a verilen ilk 5 | 2 / 50 |

BM25'in değişmemesi beklenir; BM25 girdisi iki run'da aynı metadata ve chunk metinlerinden oluşur. Dense ilk beşinde değişiklik görülmesine karşın makro Dense metrikleri neredeyse tamamen aynı kaldı. RRF'de Acceptable full ve A+P için Hit@1, MRR ve nDCG az miktarda geriledi; Recall@3/5/10 aynı kaldı. Reranker sonrasında iki run'ın ölçülen makro metrikleri aynı oldu.

## Sonuç

Bu deney, `commented_text`'i embedding girdisine katmanın bu 39 soruluk retrieval kohortunda ölçülebilir bir genel kazanım sağlamadığını gösteriyor. Bazı Dense/RRF ilk beş sıralamaları değişti, fakat son reranker performansında fark oluşmadı. Dolayısıyla commentary metninin faydası bu deneyle doğrulanmadı; sonraki adımda commentary'nin etkisini ayrı bir retrieval varyantı veya soru bazlı hata incelemesiyle test etmek gerekir.

## Soru bazlı kayıtlar

Her aşamanın top-20 chunk'ları, skorları, kaynak bilgileri ve generation top-5'i aşağıdaki dosyalarda tutuluyor:

- Raw text: `first50_raw_text_embedding/retrieval_per_question.jsonl`
- Commented text: `first50_commented_text_embedding/retrieval_per_question.jsonl`

Run başına tüm metrikler, micro recall, full-hit, çalışma ayarları ve özetler `run_report.json` ile `run_report.md` dosyalarında bulunur.

## Sınırlılıklar

- Retrieval değerlendirmesi 39 soruya dayanır; metadata-attribution, pending-gold ve invalid-gold soruları bu metriklere dahil değildir.
- Acceptable ve partial relevance yargıları gold ID değerlendirmesini genişletir. A+P tablosu üst sınır olarak yorumlanmalı, ana performans sayısı gibi sunulmamalıdır.
- Rapor güven aralıkları içermez; bu run'ın benchmark metrikleri nokta tahminleridir.
- Karşılaştırma yalnızca mevcut corpus, benchmark, metadata ve retrieval ayarları için geçerlidir.
