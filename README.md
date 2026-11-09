# 📈 BIST Finansal Analist Asistanı (Enterprise RAG Platformu)

Kamuyu Aydınlatma Platformu'nda (KAP) yayımlanan yüzlerce sayfalık faaliyet raporlarını, konsolide bilançoları, kâr/zarar tablolarını ve finansal dipnotları analiz eden, **Retrieval-Augmented Generation (RAG)** mimarisine dayalı kurumsal finansal analitik platformu.

Bu platform; **Akbank (AKBNK), Türk Hava Yolları (THYAO), Tüpraş (TUPRS)** ve BIST 100 şirketlerinin finansal dokümanlarını tarayarak kullanıcının sorduğu sorulara doğrudan resmi raporlardan kanıt göstererek güvenilir, halüsinasyonsuz ve tutarlı yanıtlar üretir.

---

## 🌟 Öne Çıkan Özellikler ve Mimari Yenilikler

- ⚡ **Ultra Hızlı & Temiz Metin Çıkarımı (`pypdfium2`):** Geleneksel OCR kütüphanelerinin tabloları ve dijital PDF mizanpajlarını bozmasının önüne geçilmiş; Google PDFium motoru ile binlerce sayfa sıfır gürültüyle ayrıştırılmıştır.
- 🧠 **Çok Turlu Sohbet Hafızası (Query Rewriting):** Takip sorularında (*"Peki bir önceki yıla göre nasıl değişti?"*) bağlamın kaybolmaması için arka planda soruları bağımsız finansal arama sorgularına dönüştüren dinamik sorgu yeniden yazıcı (Query Condensation) entegre edilmiştir.
- 🎯 **Akıllı Metadata Filtreleme:** Her doküman parçası şirket (`Akbank`, `THY`, `Tüpraş`), mali yıl (`2020-2024`), doküman türü (`Faaliyet Raporu` / `Finansal Tablo`) ve sayfa numarasıyla etiketlenmiştir.
- 📡 **Canlı Yanıt Akışı (Real-time Token Streaming):** Yanıtlar kelime kelime ekrana canlı daktilo efektiyle yazdırılır.
- 📌 **Şeffaf Kaynak & Alıntı Kartları:** Üretilen her yanıtın altında hangi şirketten, hangi mali yıldan, hangi rapordan ve hangi sayfadan alıntı yapıldığını gösteren etkileşimli kaynak kartları sunulur.
- 📥 **Otomatik KAP İndirici (BIST 100 Ölçeklenebilirliği):** Elle rapor indirme zahmetini ortadan kaldıran, BIST 100 hisse kodları üzerinden veya doğrudan KAP bildirim bağlantısıyla yeni raporları sisteme ekleyip anında indeksleyen modül.
- 💾 **Raporu Dışa Aktarma (Export):** Yapılan finansal analizi tek tıkla Markdown/Özet rapor olarak bilgisayara indirebilme.

---

## 🏗️ Sistem Mimarisi

```
[BIST 100 KAP Raporları (PDF)]
            │
            ▼ (pypdfium2 - Dijital Ekstraksiyon & Temizleme)
[Yapılandırılmış Metin & Tablolar]
            │
            ▼ (Metadata Zenginleştirme: Şirket, Yıl, Rapor Tipi, Sayfa No)
[Metin Parçaları (Chunks: 1800 char, 150 overlap)]
            │
            ▼ (Google Gemini Embeddings: models/gemini-embedding-001 - 768-dim)
[FAISS Vektör Veritabanı (vectorstore/db_faiss)]
            │
            ├─── Kullanıcı Sorusu + Sohbet Geçmişi
            ▼ (Query Rewriter: Contextualize Question)
[Bağımsız Arama Sorgusu]
            │
            ▼ (MMR & Metadata Filtreli Arama)
[En Alakalı Kaynak Parçaları (Context)]
            │
            ▼ (Prompt Mühendisliği: Kıdemli Finansal Analist Rolü)
[LLM: Google Gemini 2.5 Flash / Pro]
            │
            ▼ (Canlı Streaming)
[Finansal Analiz Yanıtı + Şeffaf Kaynak Kartları (Streamlit UI)]
```

---

## 📂 Proje Dizin Yapısı

```
Financial-Analyst-ChatBot-Project/
├── data/                                 # 2020-2024 KAP PDF Raporları (Yıl bazlı klasörler)
├── vectorstore/
│   └── db_faiss/                         # Optimize edilmiş FAISS vektör indeksi
├── src/
│   ├── config.py                         # Merkezi ayarlar, model adları, şirket ve URL tanımları
│   ├── app.py                            # Kurumsal Streamlit Finans Terminali Arayüzü & Deep-Linking
│   ├── integration/                      # BIST-Trend-Forecaster Entegrasyon Modülü
│   │   ├── models.py                     # Standart Pydantic veri şemaları (FundamentalReport, KeyMetrics)
│   │   ├── report_generator.py           # RAG ve LLM tabanlı yapılandırılmış JSON rapor üreticisi
│   │   └── api.py                        # FastAPI REST uç noktaları (/api/v1/fundamental/{ticker})
│   ├── rag/
│   │   ├── embeddings.py                 # Gemini Embeddings sarmalayıcısı (429 Rate-Limit backoff destekli)
│   │   ├── parser.py                     # pypdfium2 tabanlı temiz metin çıkarıcı
│   │   ├── rewriter.py                   # Multi-turn takip sorusu çözücü
│   │   ├── ingestion.py                  # Chunking ve vektör veritabanı boru hattı
│   │   └── engine.py                     # Streaming RAG motoru ve MMR retriever
│   └── data_collection/
│       └── kap_downloader.py             # Otomatik KAP ve BIST 100 rapor indirme modülü
├── reports/                              # Üretilen standart {TICKER}_latest.json raporları
├── tests/                                # Entegrasyon ve şema birim testleri (pytest)
├── scripts/
│   └── index_recent.py                   # Raporları otomatik indeksleme betiği
├── requirements.txt                      # Optimize edilmiş Python bağımlılıkları
└── README.md                             # Proje dokümantasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler
- Python 3.10+
- [Google AI Studio](https://aistudio.google.com/)'dan alınmış Gemini API Anahtarı

### 2. Repoyu Klonlama ve Ortam Hazırlığı
```bash
git clone https://github.com/themuhammedguler/Financial-Analyst-ChatBot-Project.git
cd Financial-Analyst-ChatBot-Project

# Sanal ortam oluşturma ve etkinleştirme
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Gerekli bağımlılıkları yükleme
pip install -r requirements.txt
```

### 3. API Anahtarını Yapılandırma
Proje ana dizininde bir `.env` dosyası oluşturun ve API anahtarınızı tanımlayın:
```env
GOOGLE_API_KEY="AIzaSy..."
```

### 4. Raporları İndeksleme (İsteğe Bağlı)
Veritabanını sıfırdan yeniden oluşturmak veya yeni raporları eklemek için:
```bash
python -m src.rag.ingestion
```

### 5. Uygulamayı Başlatma
```bash
streamlit run src/app.py
```
Tarayıcınızda otomatik açılan ekrandan finansal analiz asistanını kullanmaya başlayabilirsiniz!

---

## 📊 BIST 100 Otomatik Rapor İndirme (CLI Kullanımı)

KAP üzerinden herhangi bir BIST 100 hissesine ait raporu komut satırından otomatik olarak indirebilirsiniz:
```bash
python -m src.data_collection.kap_downloader --ticker AKBNK --year 2024 --type faaliyet --url "https://www.kap.org.tr/tr/BildirimPdf/123456"
```
İndirilen rapor doğrudan ilgili yıl klasörüne kaydedilir ve arayüz üzerinden tek tıkla indekslenebilir.

---

## 🔗 BIST-Trend-Forecaster Entegrasyonu (Temel Analiz + RAG Servisi)

Bu proje, makine öğrenmesi ve kantitatif teknik analiz karar destek platformu olan **BIST-Trend-Forecaster** ile iki yönlü tam entegre çalışmaktadır. BIST-Trend-Forecaster teknik indikatörler, XGBoost trend tahminleri ve Kelly sermaye boyutlandırmasını sağlarken; bu platform KAP bildirimleri ve faaliyet raporları üzerinden **Temel Analiz, Niteliksel Duygu Skoru (Sentiment) ve RAG Zekasını** sağlar.

### 1. Streamlit URL Query Parametre Desteği (Deep-Linking)
Kullanıcı BIST-Trend-Forecaster üzerinden *"Temel Analiz & Chatbot'a Sor"* butonuna tıkladığında bu uygulamaya `?ticker=THYAO` parametresi ile yönlendirilir:
- **Otomatik Şirket Eşleme:** URL'deki `ticker` veya `symbol` parametresi yakalanır (`st.query_params`), şirket filtresi otomatik seçilir ve mavi entegrasyon rozeti görüntülenir.
- **Kişiselleştirilmiş Karşılama:** Hisseye özel hoş geldiniz mesajı üretilir (*"THYAO için son dönem bilanço ve KAP analizine hoş geldiniz..."*).
- **Hızlı Finansal Soru Çipleri (Prompt Chips):** İlgili hisseye özel tek tıkla sorulabilen dinamik soru şablonları sunulur (*Net Kâr, Temettü Kararları, Borçluluk Durumu, Gelecek Beklentileri*).

```
Örnek URL:
http://localhost:8501/?ticker=THYAO
http://localhost:8501/?symbol=AKBNK
```

### 2. Standart Yapılandırılmış Çıktı (Structured JSON Output & REST API)
BIST-Trend-Forecaster karar destek modellerinin temel analiz verilerini arka planda okuyabilmesi için standart `FundamentalReport` JSON şeması üretilir:

```json
{
  "ticker": "THYAO",
  "company_name": "Türk Hava Yolları",
  "report_date": "2026-11-08",
  "fundamental_summary": "Yolcu doluluk oranlarında güçlü artış ve kargo gelirlerindeki büyüme ile FAVÖK beklentileri aştı...",
  "sentiment_score": 0.75,
  "sentiment_label": "Pozitif",
  "key_metrics": {
    "pe_ratio": 4.8,
    "pb_ratio": 1.15,
    "net_profit_margin_trend": "Artış",
    "debt_to_equity": "Düşük/Yönetilebilir"
  },
  "catalysts": [
    "Yeni uçak siparişleri ve filo büyümesi",
    "Yaz sezonu güçlü rezervasyonları"
  ],
  "risk_factors": [
    "Havacılık yakıt maliyetleri oynaklığı",
    "Küresel makroekonomik yavaşlama riski"
  ]
}
```

#### Veri Erişim Yolları:
1. **Dosya Tabanlı Önbellek:** Üretilen raporlar `reports/{ticker}_latest.json` yolunda disk üzerinde kalıcı olarak saklanır.
2. **Streamlit UI İndirme:** Sol kenar çubuğundaki *"Temel Analiz JSON Raporu"* panelinden tek tıkla üretilir ve `.json` olarak indirilebilir.
3. **FastAPI REST Servisi:** BIST-Trend-Forecaster veya harici servislerin doğrudan HTTP GET isteği atabileceği yüksek performanslı API:
   ```bash
   # API Sunucusunu Başlatma
   python -m uvicorn src.integration.api:app --host 0.0.0.0 --port 8000 --reload
   ```
   - `GET /health` : Servis sağlık durumu kontrolü
   - `GET /api/v1/tickers` : Desteklenen BIST hisselerinin listesi
   - `GET /api/v1/fundamental/{ticker}` : İlgili hisse için standart temel analiz JSON çıktısı

### 3. Karşı Projeye Çift Yönlü Köprü (Cross-Reference Link)
- **Kenar Çubuğu Köprü Kartı:** Kenar çubuğunda seçili hisseye dinamik bağlanan `https://bist-trend-forecaster.streamlit.app/?ticker={active_ticker}` linki ile teknik tahmin kartı sunulur.
- **Sohbet Cevabı Köprüsü:** Chatbot'un ürettiği her temel analiz cevabının altında, kullanıcının tek tıkla teknik tahmin platformuna geçmesini sağlayan yönlendirme butonu yer alır:
  > *"📈 {ticker} hissesinin XGBoost trend tahminini, teknik indikatörlerini ve Kelly sermaye tahsisini inceleyin"*

### 4. Test ve Doğrulama
Entegrasyonun tüm bileşenlerini (Deep-linking, Pydantic şema doğrulaması, JSON çıktıları ve FastAPI uç noktaları) test etmek için:
```bash
pytest tests/test_integration.py -v
```
