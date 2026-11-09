import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

# Proje kök dizini
ROOT_DIR = Path(__file__).resolve().parent.parent

# .env dosyasını yükle
load_dotenv(ROOT_DIR / ".env")

# Google API Anahtarı
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# Dizin Yolları
DATA_DIR = ROOT_DIR / "data"
# FAISS Windows MSVC Türkçe karakter (Masaüstü) sorununu önlemek için göreceli yol kullanılır
VECTORSTORE_DIR = Path("vectorstore/db_faiss")
VECTORSTORE_PATH_STR = "vectorstore/db_faiss"

# Model Yapılandırması (2026 Üretim Standartları)
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"
ADVANCED_LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768

# RAG & Chunking Parametreleri
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
MMR_LAMBDA = 0.7

# Şirket ve Yıl Tanımlamaları (2020 - 2026 Dönemi)
SUPPORTED_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# BIST-Trend-Forecaster Entegrasyonu
FORECASTER_BASE_URL = "https://bist-trend-forecaster.streamlit.app"
REPORTS_DIR = ROOT_DIR / "reports"

# Ticker <-> Şirket Eşlemeleri (BIST 100 Tam Katalog)
TICKER_TO_COMPANY = {
    # Bankacılık & Finans
    "AKBNK": "Akbank",
    "GARAN": "Garanti BBVA",
    "ISCTR": "İş Bankası",
    "YKBNK": "Yapı Kredi",
    "HALKB": "Halkbank",
    "VAKBN": "Vakıfbank",
    "TSKB": "TSKB",
    "ALBRK": "Albaraka Türk",
    # Havacılık & Ulaştırma
    "THYAO": "Türk Hava Yolları",
    "PGSUS": "Pegasus",
    "TAVHL": "TAV Havalimanları",
    "CLEBI": "Çelebi Hava Servisi",
    # Enerji, Petrol & Rafineri
    "TUPRS": "Tüpraş",
    "PETKM": "Petkim",
    "ENKAI": "Enka İnşaat & Enerji",
    "ASTOR": "Astor Enerji",
    "EUPWR": "Europower Enerji",
    "CWENE": "CW Enerji",
    "AKFYE": "Akfen Yenilenebilir Enerji",
    "GWIND": "Galata Wind",
    "ODAS": "Odaş Elektrik",
    "ZOREN": "Zorlu Enerji",
    "ENJSA": "Enerjisa Enerji",
    "AKSEN": "Aksa Enerji",
    "ALARK": "Alarko Holding",
    "CANTE": "Çan2 Termik",
    "AYDEM": "Aydem Yenilenebilir Enerji",
    # Demir Çelik & Ağır Sanayi
    "EREGL": "Ereğli Demir Çelik",
    "KRDMD": "Kardemir",
    "KCAER": "Kocaer Çelik",
    "BRSAN": "Borusan Boru",
    # Savunma, Teknoloji & Telekomünikasyon
    "ASELS": "Aselsan",
    "MIATK": "Mia Teknoloji",
    "REEDR": "Reeder Teknoloji",
    "SDTTR": "SDT Uzay ve Savunma",
    "LOGO": "Logo Yazılım",
    "KFEIN": "Kafein Yazılım",
    "TCELL": "Turkcell",
    "TTKOM": "Türk Telekom",
    # Perakende, Gıda & İçecek
    "BIMAS": "BİM Mağazalar",
    "MGROS": "Migros",
    "SOKM": "Şok Marketler",
    "CCOLA": "Coca-Cola İçecek",
    "AEFES": "Anadolu Efes",
    "ULKER": "Ülker Bisküvi",
    "TABGD": "TAB Gıda",
    "EBEBK": "Ebebek Mağazacılık",
    # Otomotiv & Dayanıklı Tüketim
    "FROTO": "Ford Otomotiv",
    "TOASO": "Tofaş",
    "TTRAK": "Türk Traktör",
    "DOAS": "Doğuş Otomotiv",
    "OTKAR": "Otokar",
    "ARCLK": "Arçelik",
    "VESTL": "Vestel Elektronik",
    "VESBE": "Vestel Beyaz Eşya",
    # Holdingler & Yatırım
    "KCHOL": "Koç Holding",
    "SAHOL": "Sabancı Holding",
    "DOHOL": "Doğan Holding",
    "AGHOL": "AG Anadolu Grubu Holding",
    "GSDHO": "GSD Holding",
    "BERA": "Bera Holding",
    "TKFEN": "Tekfen Holding",
    # Cam, Çimento & Yapı
    "SISE": "Şişecam",
    "OYAKC": "Oyak Çimento",
    "CIMSA": "Çimsa",
    "BUCIM": "Bursa Çimento",
    "NUHCM": "Nuh Çimento",
    "BSOKE": "Batısöke Çimento",
    # Kimya, Gübre & Plastik
    "SASA": "SASA Polyester",
    "HEKTS": "Hektaş",
    "GUBRF": "Gübre Fabrikaları",
    "KORDS": "Kordsa",
    "KONTR": "Kontrolmatik Teknoloji",
    "GESAN": "Girişim Elektrik",
    "ALFAS": "Alfa Solar Enerji",
    "YEOTK": "YEO Teknoloji",
    # Madencilik & Kıymetli Madenler
    "KOZAL": "Koza Altın",
    "KOZAA": "Koza Madencilik",
    "IPEKE": "İpek Doğal Enerji",
    # Sağlık & İlaç
    "MPARK": "MLP Sağlık (Medical Park)",
    "GENIL": "Gen İlaç",
    "SELEC": "Selçuk Ecza Deposu",
    "ECZYT": "Eczacıbaşı Yatırım",
    # Gayrimenkul Yatırım Ortaklığı (GYO) & Sigorta
    "EKGYO": "Emlak Konut GYO",
    "ISGYO": "İş GYO",
    "TRGYO": "Torunlar GYO",
    "SNGYO": "Sinpaş GYO",
    "KLGYO": "Kiler GYO",
    "AGESA": "Agesa Hayat ve Emeklilik",
    "ANSGR": "Anadolu Sigorta",
    "TURSG": "Türkiye Sigorta",
    # İlaç, Otomotiv Yan Sanayi, Gıda & Diğer Sanayi
    "ECILC": "Eczacıbaşı İlaç",
    "EGEEN": "Ege Endüstri",
    "KONYA": "Konya Çimento",
    "BFREN": "Bosch Fren Sistemleri",
    "BIZIM": "Bizim Toptan",
    "AGROT": "Agrotech Yüksek Teknoloji",
    "GOLTS": "Göltaş Çimento",
    "PAPIL": "Papilon Savunma",
    "TUKAS": "Tukaş",
}

COMPANY_TO_TICKER = {v: k for k, v in TICKER_TO_COMPANY.items()}

# Hızlı arama için şirket alias sözlüğü
COMPANIES = {}
for ticker, comp_name in TICKER_TO_COMPANY.items():
    aliases = [ticker.lower(), comp_name.lower()]
    # Normalize common Turkish characters in alias
    latin_name = (
        comp_name.lower()
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    if latin_name not in aliases:
        aliases.append(latin_name)
    # Yaygın kısaltmalar ve halk ağzı kullanımları
    if "Türk Hava Yolları" in comp_name:
        aliases.append("thy")
    elif "Ereğli" in comp_name:
        aliases.append("erdemir")
        aliases.append("eregli")
    elif "Tüpraş" in comp_name:
        aliases.append("tupras")
    elif "Şişecam" in comp_name:
        aliases.append("sisecam")
    COMPANIES[comp_name] = aliases


def get_company_for_ticker(ticker: str) -> str:
    """Hisse sembolünü (ticker) standart şirket adına çevirir."""
    if not ticker:
        return "Tümü"
    t_clean = ticker.upper().strip()
    return TICKER_TO_COMPANY.get(t_clean, t_clean)


def get_ticker_for_company(company_name: str) -> str:
    """Şirket adını BIST hisse sembolüne (ticker) çevirir."""
    if not company_name:
        return "THYAO"
    c_clean = company_name.strip()
    if c_clean in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[c_clean]
    # Kısmi eşleşme
    c_lower = c_clean.lower()
    if "akbank" in c_lower:
        return "AKBNK"
    elif "hava yolları" in c_lower or "thy" in c_lower:
        return "THYAO"
    elif "tüpraş" in c_lower or "tupras" in c_lower:
        return "TUPRS"
    return "BIST"


def extract_text_from_content(content: Any) -> str:
    """
    LangChain / Gemini content nesnesini (str, list[str], list[dict])
    her durumda güvenle düz string metne dönüştürür ('list' object has no attribute 'strip' hatasını önler).
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)

