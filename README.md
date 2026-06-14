# Agentic Document QA System

Çok-modlu (metin + görsel), uzun bağlamlı PDF belgeleri üzerinde soru-cevap yapabilen,
ajan tabanlı (agentic) bir Retrieval-Augmented Generation (RAG) sistemi. Belgeleri
metin/tablo/görsel olarak ayrıştırır, FAISS ile indeksler ve LangGraph tabanlı bir
orkestrasyon döngüsüyle Gemini modeli üzerinden doğrulanmış cevaplar üretir.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Proje Yapısı](#proje-yapısı)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Çalıştırma](#çalıştırma)
- [Çalışma Akışı](#çalışma-akışı)
- [Önbellekleme](#önbellekleme)
- [API Güvenilirliği (Yeniden Deneme Mekanizması)](#api-güvenilirliği-yeniden-deneme-mekanizması)
- [Testler](#testler)
- [Model Değerlendirmesi (LLM-as-a-Judge)](#model-değerlendirmesi-llm-as-a-judge)
- [Örnek Çıktı](#örnek-çıktı)
- [Notlar ve Sınırlamalar](#notlar-ve-sınırlamalar)

---

## Özellikler

- **PDF ön işleme**: Metin, tablo ve görsellerin font/yapı analizine dayalı olarak
  ayrıştırılması (pdfplumber + PyMuPDF).
- **Yapısal chunking**: Belge başlık/bölüm yapısına göre, sayfa ve bölüm bilgisini
  taşıyan, örtüşmeli (overlap) metin parçaları.
- **Çok dilli (TR/EN) embedding ve FAISS tabanlı arama**: Metin ve görseller için ayrı
  vektör indeksleri.
- **Görsel anlama**: Gemini (2.5 flash kullanıldı) ile grafik/tablo/diyagram açıklaması ve analizi.
- **Ajan tabanlı orkestrasyon**: LangGraph ReAct döngüsü ile araç (tool) çağırma,
  yeniden arama ve görsel analiz kararları.
- **Doğrulama katmanı**: Üretilen cevabın, getirilen bağlam (context) ile tutarlılığının
  kontrol edilmesi ve gerekirse düzeltilmesi.
- **Önbellekleme**: PDF başına ve birleşik (merged) seviyede kalıcı önbellek; aynı
  belgeler tekrar işlenmez.

---

## Mimari

Sistem iki ana hattan oluşur:

1. **Ön işleme hattı** (bir kez / belge değiştiğinde çalışır): PDF → metin/tablo/görsel
   chunk'lar → görsel açıklamaları (Gemini) → FAISS indeksleri.
2. **Soru-cevap hattı** (her soru için çalışır): `OrchestratorAgent`, `RetrieverAgent`
   ve `ImageAnalystAgent` araçlarını kullanarak bir taslak cevap üretir;
   `ValidatorAgent` bu cevabı bağlamla karşılaştırıp nihai cevabı döndürür.

Detaylı mimari diyagram ve tasarım kararları için `Mimari Tasarım Dokümanı & Kısa Teknik Not.docx`
dosyasına bakınız.

---

## Proje Yapısı

```
.
├── agents
│   ├── image_analyst_agent.py
│   ├── orchestrator_agent.py
│   ├── retriever_agent.py
│   └── validator_agent.py
├── cache
├── config.py
├── data
├── db
├── eval
│   ├── evaluate_pipeline.py
│   └── evaluator.py
├── llm
│   └── llm.py
├── main.py
├── preprocess
│   └── preprocess.py
├── README.md
├── requirements.txt
├── retrieval
│   └── retrieval.py
└── tests
    ├── test_image_analyst_agent.py
    ├── test_preprocess.py
    └── test_retriever_agent.py

```

---

## Gereksinimler

- Python **3.10+**
- Google Cloud projesi ile **Vertex AI API** erişimi (Gemini modeli için)

---

## Kurulum

1. **Depoyu klonlayın ve dizine girin**

   ```bash
   git clone <repo-url>
   cd <repo-klasoru>
   ```

2. **Conda ortamı oluşturun ve etkinleştirin**

   ```bash
   conda create -n pdf_reader python==3.11
   conda activate pdf_reader
   ```

3. **Bağımlılıkları kurun**

   ```bash
   pip install -r requirements.txt
   ```

NOT: Gemini API kullanabilmeniz için gerekli dosyaların path'lerini .env dosyasına giriniz.
   ```

---

## Yapılandırma

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
GOOGLE_CLOUD_PROJECT=<proje-id>
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/json.json
```

Ayrıca `config.py` içinde aşağıdaki parametreler değiştirilebilir:

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| `GEMINI_MODEL` | Kullanılan Gemini modeli | `gemini-2.5-flash` |
| `CHUNK_SIZE` | Metin chunk boyutu (karakter) | `500` |
| `CHUNK_OVERLAP` | Chunk'lar arası örtüşme (karakter) | `100` |
| `TOP_K` | Retrieval'de getirilecek sonuç sayısı | `5` |
| `IMAGE_THRESHOLD` | Görsel sonuçlar için minimum benzerlik skoru | `0.3` |
| `EMBED_MODEL` | Embedding modeli | `paraphrase-multilingual-mpnet-base-v2` |
| `MIN_IMAGE_SIZE` | Bu boyuttan küçük görseller atlanır (px) | `50` |
| `CACHE_DIR` / `DB_DIR` | Önbellek/indeks dizinleri | `cache` / `db` |

---

## Çalıştırma

PDF'ler `data/` klasöründe ya da tek bir dosya yolu olarak verilebilir.

```bash
# data/ klasöründeki tüm PDF'ler üzerinde soru sor
python main.py --pdf data --question "Sorunuz buraya"

# Tek bir PDF üzerinde çalıştır
python main.py --pdf data/rapor.pdf --question "YOLOv8'in mAP değeri nedir?"
```

### Komut Satırı Argümanları

| Argüman | Açıklama | Varsayılan |
|---|---|---|
| `--pdf` | PDF dosyası veya PDF'lerin bulunduğu klasör | `data` |
| `--question` | Sorulacak soru | (kod içinde tanımlı örnek soru) |

---

## Çalışma Akışı

Program çalıştırıldığında aşağıdaki adımlar sırayla işletilir:

1. **[1/5] Ön işleme**: PDF(ler) ayrıştırılır, metin/tablo/görsel chunk'ları çıkarılır
   (önbellekte varsa atlanır).
2. **[2/5] Görsel açıklama**: Yeni çıkarılan görseller Gemini Vision ile açıklanır.
3. **[3/5] İndeksleme**: Tüm chunk'lar için birleşik FAISS indeksleri oluşturulur
   (önbellekte geçerliyse atlanır).
4. **[4/5] Ajan kurulumu**: `RetrieverAgent`, `ImageAnalystAgent`, `ValidatorAgent` ve
   `OrchestratorAgent` başlatılır.
5. **[5/5] Soru-cevap**: Orchestrator, `search_chunks` ve gerektiğinde `analyze_image`
   araçlarını kullanarak bir taslak cevap üretir; `ValidatorAgent` bu cevabı bağlamla
   doğrulayıp nihai cevabı döndürür.

---

## Önbellekleme

- **`cache/`**: PDF'lerden çıkarılan görseller (`<pdf>_p<sayfa>_img<sıra>.<uzantı>`).
- **`db/per_pdf/<pdf_adı>/`**: Her PDF için chunk'lar ve `mtime` damgası — PDF
  değişmediyse ön işleme ve görsel açıklama adımları tekrar çalıştırılmaz.
- **`db/text.index`, `db/image.index`, `db/merged_stamp.json`**: Birleşik FAISS
  indeksleri — PDF seti değişmediyse doğrudan yüklenir.

PDF'lerden birini güncellediğinizde veya yeni PDF eklediğinizde, ilgili önbellek
otomatik olarak geçersiz sayılır ve yeniden oluşturulur. Tüm önbelleği sıfırlamak için
`cache/` ve `db/` klasörlerini silebilirsiniz.


---

## API Güvenilirliği (Yeniden Deneme Mekanizması)

`GeminiLLM` sınıfı, hem metin üretimi (`__call__`) hem de görsel analiz
(`generate_with_image`) çağrılarında Vertex AI API'sinden geçici bir hata alındığında
(zaman aşımı, kota/rate-limit, geçici sunucu hatası vb.) isteği otomatik olarak yeniden
gönderir:

- Bir hata oluştuğunda istek hemen tekrar denenmez; bekleme süresi her denemede
  **katlanarak (exponential backoff)** artırılır: `wait_time = min(60, 2^retry_count)`
  saniye, üzerine küçük bir rastgele **jitter** (0–1 saniye) eklenir.
- En fazla `max_retries = 6` deneme yapılır; tüm denemeler başarısız olursa akış
  durmaz, ilgili adım için bir hata/placeholder değeri döndürülür.
- Bu mekanizma, API'nin geçici olarak yanıt vermemesi durumunda hem **ön işleme**
  (`describe_image`) hem de **ajan** (validasyon, görsel analiz) adımlarının tek bir
  hatadan dolayı tamamen kesilmesini önler.

---
## Testler

```bash
pip install pytest
pytest -q
```

NOT: İmge analist ajan, retriever ajan ve preprocess kodları için testler yazılmıştır. LLM çağrısı gerektirecek kodlara yazılmamıştır.

---

## Model Değerlendirmesi (LLM-as-a-Judge)

Sistemin ürettiği cevapların kalitesi, **LLM-as-a-Judge** yaklaşımıyla değerlendirilir:
ayrı bir Gemini çağrısı; soru, retrieval'den gelen bağlam (context) ve sistemin nihai
cevabını girdi olarak alır, cevabı önceden tanımlanmış kriterlere göre puanlar.

Değerlendirme kriterleri:

- **Doğruluk / Groundedness**: Cevaptaki iddiaların, getirilen context ile tutarlı
  olup olmadığı (halüsinasyon kontrolü).
- **İlgililik (Relevance)**: Cevabın sorulan soruyla doğrudan ilişkili olup olmadığı.
- **Eksiksizlik (Completeness)**: Context'te bulunan ilgili bilginin cevapta yeterince
  kullanılıp kullanılmadığı.
- **Açıklık / Okunabilirlik**: Cevabın anlaşılır ve doğrudan bir şekilde ifade edilmiş
  olması.

Judge LLM, her kriter için 0-1 arası bir puan ve kısa bir gerekçe üretir. Farklı
sorular ve PDF'ler üzerinde toplanan bu puanlar, sistemin (ve `ValidatorAgent`
düzeltmesinin) genel performansını ölçmek için kullanılır.
---

## Örnek Çıktı

Aşağıda örnek kod çıktıları bulunmaktadır.

> Örnek-1 (Sorudaki bilgi PDF'ten sadece görsel içeriğinden çıkartılabilecek bir bilgidir. Doğru çıkardığı gözlemlenmiştir.)

```
[1/5] Preprocessing YKB_odev.pdf...
  -> 22 text chunks, 0 images found
  -> Cached YKB_odev.pdf
[1/5] Preprocessing yolov11.pdf...
  -> 116 text chunks, 2 images found
[2/5] Describing images for yolov11.pdf...
  -> Describing cache/yolov11_page_3_img_0.png...
  -> Describing cache/yolov11_page_7_img_1.png...

[Warning]: 429 Resource exhausted. Please try again later. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.
Waiting for 2.84 seconds...
  -> Cached yolov11.pdf
[1/5] yolov8.pdf loaded from cache (127 text, 4 images)
[3/5] Building merged FAISS indexes (265 text, 6 images)...
  -> Text index: 265 vectors
  -> Image index: 6 vectors
[4/5] Setting up agents...
  [Orchestrator] Building LangGraph...
  [Orchestrator] Graph compiled
[5/5] Answering: 'What is the k, s and p value of first conv layer in backbone?'

============================================================
  [Orchestrator] Starting pipeline for: 'What is the k, s and p value of first conv layer in backbone?'
  [Orchestrator] Running graph (recursion_limit=10)...
  [Orchestrator] Agent thinking... (2 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'first conv layer backbone k s p value'
  [RetrieverAgent] Found 5 text, 5 image chunks
  [Orchestrator] Agent thinking... (4 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [ImageAnalystAgent] Analyzing image: cache/yolov8_page_4_img_1.png
  [ImageAnalystAgent] Analysis complete (182 chars)
  [Orchestrator] Agent thinking... (6 messages)
  [Orchestrator] Agent decided: final_answer
  [Orchestrator] Routing -> end
  [Orchestrator] Draft answer received (1 chars)
  [Orchestrator] Collected 6734 chars of context for validation
  [ValidatorAgent] Validating answer...
  [ValidatorAgent] Validation complete
  [Orchestrator] Final answer ready
ANSWER
============================================================
The first convolutional layer in the backbone of the YOLOv8 model has the following values:
*   **k (kernel size):** 3
*   **s (stride):** 2
*   **p (padding):** 1

============================================================
```

Örnek-2 (Tablo içerisinde bulunan bir bilgi. Koordinatör ajan sırasıyla retriever, imge analiz, retriever çağırarak doğru cevaba ulaşmıştır.)

```
  -> Found 3 PDF(s): ['YKB_odev.pdf', 'yolov11.pdf', 'yolov8.pdf']
  -> Merged index loaded: 265 text vectors
  -> Merged index loaded: 6 image vectors
[1/5] Merged index valid, skipping preprocessing...
[2/5] Skipped (cached)
[3/5] Skipped (cached)
[4/5] Setting up agents...
  [Orchestrator] Building LangGraph...
  [Orchestrator] Graph compiled
[5/5] Answering: 'What is the map scores of yolov8 versions (YOLOV8n, YOLOV8s, YOLOV8m ...)?'

============================================================
  [Orchestrator] Starting pipeline for: 'What is the map scores of yolov8 versions (YOLOV8n, YOLOV8s, YOLOV8m ...)?'
  [Orchestrator] Running graph (recursion_limit=10)...
  [Orchestrator] Agent thinking... (2 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'YOLOv8 mAP scores'
  [RetrieverAgent] Found 5 text, 5 image chunks
  [Orchestrator] Agent thinking... (4 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'YOLOv8n YOLOv8s YOLOv8m YOLOv8l YOLOv8x mAP scores'
  [RetrieverAgent] Found 5 text, 5 image chunks
  [Orchestrator] Agent thinking... (6 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [ImageAnalystAgent] Analyzing image: cache/yolov8_page_7_img_3.png
  [ImageAnalystAgent] Analysis complete (181 chars)
  [Orchestrator] Agent thinking... (8 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'YOLOv8 performance table mAP'
  [RetrieverAgent] Found 5 text, 5 image chunks
  [Orchestrator] Agent thinking... (10 messages)
  [Orchestrator] Agent decided: final_answer
  [Orchestrator] Routing -> end
  [Orchestrator] Draft answer received (1 chars)
  [Orchestrator] Collected 19444 chars of context for validation
  [ValidatorAgent] Validating answer...
  [ValidatorAgent] Validation complete
  [Orchestrator] Final answer ready
ANSWER
============================================================
The mAP@0.5 scores for the different YOLOv8 versions are:

*   **YOLOv8n:** 47.2%
*   **YOLOv8s:** 58.5%
*   **YOLOv8m:** 66.3%
*   **YOLOv8l:** 69.8%
*   **YOLOv8x:** 71.5%
```

Örnek-3
```
  -> Found 3 PDF(s): ['YKB_odev.pdf', 'yolov11.pdf', 'yolov8.pdf']
  -> Merged index loaded: 265 text vectors
  -> Merged index loaded: 6 image vectors
[1/5] Merged index valid, skipping preprocessing...
[2/5] Skipped (cached)
[3/5] Skipped (cached)
[4/5] Setting up agents...
  [Orchestrator] Building LangGraph...
  [Orchestrator] Graph compiled
[5/5] Answering: 'Ödevin ajan mimarisinin nasıl olması beklenmektedir'

============================================================
  [Orchestrator] Starting pipeline for: 'Ödevin ajan mimarisinin nasıl olması beklenmektedir'
  [Orchestrator] Running graph (recursion_limit=10)...
  [Orchestrator] Agent thinking... (2 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'ajan mimarisi ödev'
  [RetrieverAgent] Found 5 text, 2 image chunks
  [Orchestrator] Agent thinking... (4 messages)
  [Orchestrator] Agent decided: final_answer
  [Orchestrator] Routing -> end
  [Orchestrator] Draft answer received (1 chars)
  [Orchestrator] Collected 3309 chars of context for validation
  [ValidatorAgent] Validating answer...
  [ValidatorAgent] Validation complete
  [Orchestrator] Final answer ready
ANSWER
============================================================
Ödevde ajan mimarisi için bir Mimari Tasarım Dokümanı oluşturulması beklenmektedir. Bu doküman aşağıdaki soruları yanıtlamalıdır:

1.  **Belge ön işleme (Document Pre-processing):** Metin ve görsel içerik nasıl ayrıştırılır? Hangi araçlar/kütüphaneler kullanılır?
2.  **Yapısal navigasyon:** Ajan, uzun bir belgede ilgili bölümü nasıl bulur?

Bu sistem, çok-modlu ve uzun bağlamlı belgeler üzerinde soru-cevap yapabilen bir agentic sistem mimarisi olarak tasarlanmalıdır.
```
> Örnek-4 (PDF'de olmayan bir bilgi)
```
  -> Found 3 PDF(s): ['test_pdf.pdf', 'yolov11.pdf', 'yolov8.pdf']
  -> Merged index loaded: 231 text vectors
  -> Merged index loaded: 6 image vectors
[1/5] Merged index valid, skipping preprocessing...
[2/5] Skipped (cached)
[3/5] Skipped (cached)
[4/5] Setting up agents...
  [Orchestrator] Building LangGraph...
  [Orchestrator] Graph compiled
[5/5] Answering: 'LoRA ile ilgili detaylar nelerdir?'

============================================================
  [Orchestrator] Starting pipeline for: 'LoRA ile ilgili detaylar nelerdir?'
  [Orchestrator] Running graph (recursion_limit=10)...
  [Orchestrator] Agent thinking... (2 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'LoRA detayları'
  [RetrieverAgent] Found 5 text, 4 image chunks
  [Orchestrator] Agent thinking... (4 messages)
  [Orchestrator] Agent decided: tool_call
  [Orchestrator] Routing -> tools
  [RetrieverAgent] Searching: 'LoRA'
  [RetrieverAgent] Found 5 text, 1 image chunks
  [Orchestrator] Agent thinking... (6 messages)
  [Orchestrator] Agent decided: final_answer
  [Orchestrator] Routing -> end
  [Orchestrator] Draft answer received (1 chars)
  [Orchestrator] Collected 8779 chars of context for validation
  [ValidatorAgent] Validating answer...
  [ValidatorAgent] Validation complete
  [Orchestrator] Final answer ready
ANSWER
============================================================
Aradığınız "LoRA" ile ilgili bilgilere belgede rastlanmamıştır.

============================================================
```

---

