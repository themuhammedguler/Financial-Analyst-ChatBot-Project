# 📡 BIST-Trend-Forecaster için v2.0 Entegrasyon & Sürüm Bildirimi

**Kaynak Proje:** `Financial-Analyst-ChatBot-Project` (BIST Temel Analiz, Hibrit RAG & Kantitatif Karar Motoru)  
**Hedef Proje:** `BIST-Trend-Forecaster` (Teknik Analiz & Kantitatif Trend Tahmin Sistemi)  
**API Sürümü:** `v2.0.0` ([`src/integration/api.py`](../src/integration/api.py))

---

## 🚀 1. Bu Sürümde Eklenen Yeni Yetenekler Özeti

1. **Özel BIST Finans LLM & Çoklu Model Çıkarım Merkezi (`LLMProviderHub`):**
   - `Qwen2.5-7B-Instruct` (4-bit QLoRA / Unsloth) için **2.596 adet Chain-of-Thought (`<think>`) SFT** ve **2.049 adet DPO** veri kümesi oluşturuldu.
   - Google Colab Pro (ngrok köprüsü), yerel Ollama ve Google Gemini modelleri arasında canlı geçiş ve otomatik kurtarma (*self-healing*) desteği eklendi.
2. **Hibrit RAG + Aynı Sayfa Tablo Birleştirme (`_stitch_same_page_chunks`) & Sayısal Sadakat:**
   - KAP PDF faaliyet raporlarında iki parçaya bölünen tablolar (ör. *Filo Durumu*, *Özet Bilanço*, *Gelir Tablosu*) otomatik olarak birleştirilerek tablo alt toplamlarının (`Genel Toplam`) kaybolması önlendi.
   - Deterministik finansal motor çıktıları ve doğrulanmış BIST referans verileri doğrudan RAG bağlamına entegre edildi.
3. **Deterministik Finansal Hesaplama Motoru (`src/tools/financial_tools.py`):**
   - **DuPont 3-Aşamalı ROE Ayrıştırması:** Net Kâr Marjı × Aktif Devir Hızı × Finansal Kaldıraç.
   - **Altman Z-Skoru:** İflas dayanıklılığı ve Güvenli / Gri / Stres bölgesi sınıflandırması.
   - **İndirgenmiş Nakit Akımları (DCF):** 5 yıllık serbest nakit akışı projeksiyonu ve Gordon terminal değerleme modeli.
   - **Makroekonomik Şok & Tornado Simülatörü:** USD/TRY (`+%10`), TCMB politika faizi (`+250 bps`) ve Brent petrol (`+%15`) şoklarının net kâr üzerindeki bileşik etkisi.
   - **Kelly Kriteri Sermaye Tahsisi:** `f* = p - (1 - p) / b` formülüyle Half-Kelly portföy ağırlığı.
   - **Eşler Arası (Peer) Kıyaslama Matrisi:** İki BIST hissesinin yan yana DuPont ve Z-Skoru mukayesesi.
4. **Kullanıcıya (Tarayıcıya) Özel Model & Sunucu Hafızası (`src/user_storage.py`):**
   - Seçilen LLM sağlayıcısı ve özel Colab/ngrok URL'si tarayıcı çerezi + istemci parmak izi ile **yalnızca ilgili kullanıcı için** `F5` sonrasında hatırlanır; diğer ziyaretçilere sızmaz.

---

## 🔌 2. `BIST-Trend-Forecaster` İçin Yeni REST API Uç Noktaları (`v2.0.0`)

Mevcut `/api/v1/fundamental/{ticker}` ve `/api/v1/signal/{ticker}` uç noktalarına ek olarak 4 yeni uç nokta devreye alınmıştır:

| Metot | Uç Nokta (Endpoint) | Açıklama |
| :--- | :--- | :--- |
| `GET` | `/api/v1/quant/{ticker}` | Tek çağrıda DuPont ROE, Altman Z-Skoru, DCF Adil Değer, Temettü Güvenliği, Kelly Tahsisi ve KAP bildirimlerini döner. |
| `GET` | `/api/v1/macro-stress/{ticker}` | `usd_change_pct`, `rate_change_bps`, `oil_change_pct` parametreleriyle makro şok simülasyonu döner. |
| `GET` | `/api/v1/compare/{ticker_a}/{ticker_b}` | İki BIST hissesini DuPont marjları, kaldıraç ve Altman Z-Skoru bazında karşılaştırır ve kalite liderini belirler. |
| `POST` | `/api/v1/ask` | Doğal dildeki finansal soruları Hibrit RAG + Deterministik Motor ile kaynak belirterek yanıtlar. |

### Örnek Python İstemci Kodu (`BIST-Trend-Forecaster` İçinden Çağrı)

```python
import requests

CHATBOT_API_URL = "http://localhost:8000"

# 1. Tek Çağrıda Derin Kantitatif Karne (DuPont + Altman Z + DCF + Temettü + Kelly)
quant_data = requests.get(f"{CHATBOT_API_URL}/api/v1/quant/THYAO", timeout=10).json()
print("ROE %:", quant_data["dupont"]["roe_pct"])
print("Altman Z:", quant_data["altman_z"]["z_score"], quant_data["altman_z"]["zone"])
print("DCF Adil Değer (Milyon TL):", quant_data["dcf_valuation"]["estimated_fair_value_million_tl"])

# 2. Makroekonomik Stres Testi (Dolar +%10, Faiz +250 bps, Petrol +%15)
stress_data = requests.get(
    f"{CHATBOT_API_URL}/api/v1/macro-stress/EREGL",
    params={"usd_change_pct": 10.0, "rate_change_bps": 250.0, "oil_change_pct": 15.0},
    timeout=10,
).json()
print("Bileşik Net Kâr Etkisi %:", stress_data["total_impact_pct"])

# 3. İki Hisse Eşler Arası (Peer) Kıyaslama
peer_data = requests.get(f"{CHATBOT_API_URL}/api/v1/compare/THYAO/PGSUS", timeout=10).json()
print("Kalite Lideri:", peer_data["quality_leader"])
```
