# Akbank GenAI Bootcamp: RAG Tabanlı Finansal Analist Chatbot

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, Retrieval-Augmented Generation (RAG) mimarisine dayalı bir finansal analist chatbot'udur. Chatbot, belirtilen şirketlerin Kamuyu Aydınlatma Platformu'nda (KAP) yayınlanan finansal raporlarını kullanarak kullanıcıların sorularını yanıtlamaktadır.

---

## 📖 Projenin Amacı

Bu projenin temel amacı, büyük hacimli ve yapılandırılmamış finansal dokümanlar (yıllık faaliyet raporları, finansal tablolar vb.) içerisinden, doğal dil ile sorulan sorulara hızlı ve doğru cevaplar üretebilen bir yapay zeka asistanı geliştirmektir. Bu sayede finansal analistlerin, yatırımcıların veya öğrencilerin raporlar içinde manuel olarak bilgi arama zahmeti ortadan kaldırılmaktadır.

---

## 📂 Veri Seti Hakkında Bilgi

Chatbot'un bilgi kaynağı olarak Türkiye'nin önde gelen şirketlerinden olan **Akbank, Türk Hava Yolları ve Tüpraş**'ın KAP'ta yayınladığı resmi dokümanlar kullanılmıştır.

*   **Veri Kaynağı:** Kamuyu Aydınlatma Platformu (KAP)
*   **Doküman Türleri:** Yıllık Faaliyet Raporları ve Konsolide Finansal Tablolar
*   **Kapsanan Yıllar:** 2020, 2021, 2022, 2023, 2024
*   **Format:** PDF

Bu dokümanlar, şirketlerin finansal performansı, operasyonel faaliyetleri, stratejileri, riskleri ve gelecek hedefleri hakkında zengin ve güvenilir bilgiler içermektedir.

---

## 🛠️ Kullanılan Yöntemler ve Çözüm Mimarisi

Proje, **Retrieval-Augmented Generation (RAG)** mimarisi üzerine kurulmuştur. Bu mimari, Büyük Dil Modelleri'nin (LLM) kendi iç bilgisine ek olarak, dış ve güncel bir bilgi kaynağından faydalanmasını sağlar.

Projenin RAG akışı şu adımlardan oluşmaktadır:

1.  **Veri Yükleme ve Parçalama (Load & Chunk):** PDF formatındaki tüm finansal raporlar sisteme yüklenir ve yönetilebilir küçük metin parçalarına (chunks) ayrılır.
2.  **Gömme (Embedding):** Her bir metin parçası, Google'ın `embedding-001` modeli kullanılarak anlamsal olarak sayısal bir vektöre dönüştürülür.
3.  **Vektör Depolama (Vector Store):** Elde edilen bu vektörler, verimli bir şekilde saklanmaları ve hızlıca sorgulanabilmeleri için bir **FAISS** vektör veritabanına kaydedilir.
4.  **Sorgu ve Geri Getirme (Query & Retrieve):** Kullanıcı bir soru sorduğunda, bu soru da aynı embedding modeli ile bir vektöre çevrilir. Ardından FAISS veritabanında bu soru vektörüne en benzer (anlamsal olarak en yakın) metin parçaları bulunur.
5.  **Zenginleştirilmiş Üretim (Augmented Generation):** Kullanıcının orijinal sorusu ve veritabanından geri getirilen en alakalı metin parçaları, bir prompt şablonu ile birleştirilerek **Google Gemini Pro** modeline sunulur. LLM, bu bağlamı kullanarak soruyu yanıtlar ve böylece cevabın yalnızca sağlanan dokümanlara dayanması sağlanır.
6.  **Arayüz (UI):** Tüm bu süreç, kullanıcı dostu bir web arayüzü olan **Streamlit** üzerinden sunulur.

---

## 🚀 Kodun Çalışma Kılavuzu

Projeyi lokal makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [BU_REPOSITORININ_LINKI]
    cd financial-analyst-chatbot
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

4.  **API Anahtarını Ayarlayın:**
    *   Proje ana dizininde `.env` adında bir dosya oluşturun.
    *   İçine Google AI Studio'dan aldığınız API anahtarınızı aşağıdaki gibi ekleyin:
      ```
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
      ```

5.  **Uygulamayı Başlatın:**
    *   Bu repoda `vectorstore` dosyası hazır olarak bulunmaktadır, bu nedenle veri işleme notebook'unu tekrar çalıştırmanıza gerek yoktur.
    *   Aşağıdaki komut ile Streamlit uygulamasını başlatın:
      ```bash
      streamlit run src/app.py
      ```
    *   Tarayıcınızda açılan web sayfasından chatbot'u kullanmaya başlayabilirsiniz.

---

## ✨ Elde Edilen Sonuçlar

Proje sonucunda, seçilen finansal raporlar hakkında spesifik sorulara tutarlı ve doğru cevaplar üretebilen fonksiyonel bir chatbot geliştirilmiştir. Chatbot, cevaplarını dayandırdığı kaynak metinleri de sunarak şeffaflık ve doğrulanabilirlik sağlamaktadır.

![Chatbot Arayüz Görüntüsü]([İSTEĞE_BAĞLI_BURAYA_CHATBOT_EKRAN_GÖRÜNTÜSÜ_KOYABİLİRSİNİZ])

---

## 🔗 Web Linkiniz

[PROJEYİ STREAMLIT COMMUNITY CLOUD GİBİ BİR PLATFORMDA DEPLOY ETTİKTEN SONRA LİNKİ BURAYA EKLEYEBİLİRSİNİZ]