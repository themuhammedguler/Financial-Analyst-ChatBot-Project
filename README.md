# RAG Tabanlı Finansal Analist Chatbot 📈

Bu proje, **Akbank GenAI Bootcamp** için geliştirilmiş, **Retrieval-Augmented Generation (RAG)** mimarisine dayalı, son teknoloji bir Finansal Analist Chatbot'udur. Uygulama; Akbank, Türk Hava Yolları ve Tüpraş gibi Türkiye'nin önde gelen şirketlerinin Kamuyu Aydınlatma Platformu'nda (KAP) yayınladığı yüzlerce sayfalık finansal raporları analiz eder ve kullanıcıların doğal dilde sorduğu sorulara, doğrudan raporlardan alınan kanıtlara dayalı, tutarlı cevaplar üretir.

<br>

![Finansal Analist Chatbot Arayüzü](https://raw.githubusercontent.com/themuhammedguler/Financial-Analyst-ChatBot-Project/main/png/chatbot.png)

---

## 🔗 Canlı Demo

Uygulamanın internet üzerinden erişilebilen canlı demosu aşağıdadır. Ücretsiz sunucuların uyku modundan çıkması nedeniyle uygulamanın ilk açılışı 1-2 dakika sürebilir.

### ➡️ **[Finansal Analist Chatbot'u Canlı Denemek İçin Buraya Tıklayın!](https://financial-analyst-chatbot-project.streamlit.app/)**

---

## 🎯 Projenin Amacı ve Çözdüğü Problem

Finansal analistler, yatırımcılar ve öğrenciler için şirketlerin faaliyet raporları ve finansal tabloları en değerli bilgi kaynaklarıdır. Ancak bu raporlar genellikle yüzlerce sayfadan oluşur; yapılandırılmamış metinler, karmaşık tablolar ve dipnotlar içerir. Bu durum, spesifik bir bilgiye ulaşmayı son derece zaman alıcı ve verimsiz bir hale getirir.

Bu proje, bu problemi çözmek amacıyla geliştirilmiştir:

-   ✅ **Hızlı Bilgi Erişimi:** Kullanıcıların saatlerce rapor taramak yerine, "Tüpraş'ın 2023 yılındaki sürdürülebilirlik yatırımları nelerdir?" gibi spesifik sorular sorarak saniyeler içinde cevap almasını sağlar.
-   ✅ **Veri Odaklı Cevaplar:** Büyük Dil Modelleri'nin (LLM) "halüsinasyon" (bilgi uydurma) eğilimini, cevapları yalnızca sağlanan raporlardaki verilere dayandırarak ortadan kaldırır.
-   ✅ **Verimlilik Artışı:** Finansal karar alma süreçlerinde ihtiyaç duyulan veri toplama ve analiz aşamasını dramatik bir şekilde hızlandırır.

---

## 📂 Veri Seti

Chatbot'un bilgi tabanı, Türkiye'nin farklı sektörlerdeki en büyük şirketlerinden bazılarının KAP'ta kamuya açıkladığı resmi dokümanlardan oluşmaktadır.

-   **Şirketler:** Akbank, Türk Hava Yolları, Tüpraş
-   **Doküman Türleri:** Yıllık Faaliyet Raporları, Konsolide Finansal Tablolar ve Dipnotlar
-   **Kapsanan Yıllar:** 2020 - 2024
-   **Veri Kaynağı:** [Kamuyu Aydınlatma Platformu (KAP)](https://www.kap.org.tr/)
-   **Format:** PDF

---

## 🛠️ Teknik Mimari ve Kullanılan Teknolojiler

Proje, modern ve güçlü bir **Retrieval-Augmented Generation (RAG)** akışı üzerine inşa edilmiştir. Bu mimari, LLM'lerin yaratıcılığını, harici bir bilgi tabanının doğruluğu ve güncelliği ile birleştirir.

### Çözüm Mimarisi

1.  **Veri Hazırlama (Ingestion - Çevrimdışı İşlem):**
    -   **Yükleme & Ayrıştırma:** PDF raporları, `unstructured` kütüphanesinin `"hi_res"` stratejisi kullanılarak sisteme yüklenir. Bu strateji, metinleri okumanın yanı sıra, Tesseract (OCR) ve Detectron2 (Görsel Analiz) gibi araçlarla dokümanların görsel yapısını (başlıklar, paragraflar, listeler ve **tablolar**) da anlar.
    -   **Parçalama (Chunking):** Ayrıştırılan dokümanlar, anlamsal bütünlüğü koruyacak şekilde daha küçük metin parçalarına (`chunks`) bölünür.
    -   **Gömme (Embedding):** Her bir metin parçası, Google'ın `embedding-001` modeli aracılığıyla sayısal bir vektöre dönüştürülür.
    -   **Depolama (Store):** Bu vektörler, verimli bir şekilde saklanmaları ve ışık hızında sorgulanabilmeleri için bir **FAISS** vektör veritabanına indekslenir.

2.  **Soru-Cevap Akışı (Inference - Gerçek Zamanlı İşlem):**
    -   **Sorgu (Query):** Kullanıcı, Streamlit arayüzü üzerinden sorusunu sorar.
    -   **Geri Getirme (Retrieve):** Kullanıcının sorusu da bir vektöre dönüştürülür ve FAISS veritabanında bu soruya anlamsal olarak en yakın metin parçaları bulunur.
    -   **Zenginleştirme (Augment):** Geri getirilen bu alakalı metin parçaları ("bağlam" olarak) ve kullanıcının orijinal sorusu, özenle hazırlanmış bir prompt şablonu ile birleştirilir.
    -   **Üretim (Generate):** Bu zenginleştirilmiş prompt, cevap üretmesi için **Google Gemini 1.5 Flash** modeline gönderilir. LLM, yalnızca kendisine sunulan bağlamdaki bilgileri kullanarak soruyu yanıtlar.

### Teknoloji Seti

-   **Model ve API:** Google Gemini 1.5 Flash, Google AI Platform
-   **RAG & LLM Orkestrasyonu:** LangChain
-   **Vektör Veritabanı:** FAISS (Facebook AI Similarity Search)
-   **Doküman İşleme:** `unstructured[local-inference]` (Tesseract, Poppler, Detectron2 ile)
-   **Web Arayüzü:** Streamlit
-   **Programlama Dili:** Python 3.10+

---

## 🚀 Projeyi Lokalde Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### Ön Gereksinimler

-   Python 3.10 veya üzeri
-   [Git](https://git-scm.com/)
-   **Tesseract OCR:** [Kurulum Talimatları](https://github.com/UB-Mannheim/tesseract/wiki) (Kurulum sırasında Türkçe dil paketini ve PATH'e ekleme seçeneğini işaretlediğinizden emin olun.)
-   **Poppler:** [Kurulum Talimatları](https://github.com/oschwartz10612/poppler-windows/releases/) (İndirdikten sonra `bin` klasörünü PATH'e eklemeniz gerekmektedir.)

### Kurulum Adımları

1.  **Repoyu Klonlayın:**
    ```bash
    git clone https://github.com/themuhammedguler/Financial-Analyst-ChatBot-Project.git
    cd Financial-Analyst-ChatBot-Project
    ```

2.  **Sanal Ortam Oluşturun ve Aktive Edin:**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # MacOS/Linux için:
    source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Not: `unstructured[local-inference]` kurulumu, bağımlılıkların büyüklüğü nedeniyle biraz zaman alabilir.)*

4.  **Google API Anahtarınızı Ayarlayın:**
    -   Proje ana dizininde `.env` adında bir dosya oluşturun.
    -   İçine [Google AI Studio](https://aistudio.google.com/)'dan aldığınız API anahtarınızı aşağıdaki gibi ekleyin:
      ```bash     
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
      ```

5.  **Veri Hazırlama (Sadece İlk Seferde):**
    -   Kendi PDF dosyalarınızı proje ana dizinindeki `data` klasörüne (veya alt klasörlerine) yerleştirin.
    -   `notebooks/01_data_ingestion.ipynb` notebook'unu çalıştırın. Bu işlem, donanımınıza ve veri setinizin büyüklüğüne bağlı olarak **çok uzun sürebilir.**
    -   İşlem tamamlandığında, projenizin ana dizininde `vectorstore` adında bir klasör oluşacaktır.

6.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run src/app.py
    ```
    Tarayıcınızda açılan web sayfasından chatbot'u kullanmaya başlayabilirsiniz!

---

## ✨ Elde Edilen Sonuçlar ve Gelecek Geliştirmeler

Bu proje, yapılandırılmamış PDF dokümanlarından akıllı bir şekilde bilgi çıkarabilen ve bunu kullanıcı dostu bir arayüzle sunabilen, uçtan uca fonksiyonel bir RAG uygulamasının başarılı bir kanıtıdır. Chatbot, özellikle metin tabanlı ve yarı yapısal bilgilere dayalı sorulara yüksek doğrulukla cevap verebilmektedir.

### Gelecek Geliştirmeler

-   **Gelişmiş Tablo Anlama:** Tabloları metin yerine yapısal veri (örn: Pandas DataFrame) olarak işleyerek "2022 ve 2023 yılları arasındaki ciro artış oranı nedir?" gibi karşılaştırmalı ve hesaplamalı sorulara cevap verebilme.
-   **Kullanıcı Geri Bildirim Mekanizması:** Cevapları değerlendirme (beğen/beğenme) özelliği ekleyerek modelin performansını izleme.
-   **Sohbet Geçmişi (Chat History):** Uygulamaya hafıza özelliği ekleyerek kullanıcıların takip soruları sormasına olanak tanıma.