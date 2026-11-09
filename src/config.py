import os
from pathlib import Path
from dotenv import load_dotenv

# Proje kök dizini
ROOT_DIR = Path(__file__).resolve().parent.parent

# .env dosyasını yükle
load_dotenv(ROOT_DIR / ".env")

# Google API Anahtarı
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Dizin Yolları
DATA_DIR = ROOT_DIR / "data"
# FAISS Windows MSVC Türkçe karakter (Masaüstü) sorununu önlemek için göreceli yol kullanılır
VECTORSTORE_DIR = Path("vectorstore/db_faiss")
VECTORSTORE_PATH_STR = "vectorstore/db_faiss"

# Model Yapılandırması
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
ADVANCED_LLM_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768

# RAG & Chunking Parametreleri
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
MMR_LAMBDA = 0.7

# Şirket ve Yıl Tanımlamaları
COMPANIES = {
    "Akbank": ["akbank", "akb"],
    "Türk Hava Yolları": ["türk hava yolları", "turk hava yollari", "thy", "thya"],
    "Tüpraş": ["tüpraş", "tupras"]
}

SUPPORTED_YEARS = [2020, 2021, 2022, 2023, 2024]

# BIST-Trend-Forecaster Entegrasyonu
FORECASTER_BASE_URL = "https://bist-trend-forecaster.streamlit.app"
REPORTS_DIR = ROOT_DIR / "reports"

# Ticker <-> Şirket Eşlemeleri (BIST 100 Entegrasyonu)
TICKER_TO_COMPANY = {
    "AKBNK": "Akbank",
    "THYAO": "Türk Hava Yolları",
    "TUPRS": "Tüpraş",
    "EREGL": "Ereğli Demir Çelik",
    "SISE": "Şişecam",
    "ASELS": "Aselsan",
    "BIMAS": "BİM Mağazalar",
    "FROTO": "Ford Otomotiv",
    "GARAN": "Garanti BBVA",
    "ISCTR": "İş Bankası",
    "KCHOL": "Koç Holding",
    "SAHOL": "Sabancı Holding",
    "YKBNK": "Yapı Kredi",
    "PETKM": "Petkim",
    "PGSUS": "Pegasus",
    "TCELL": "Turkcell",
    "TTKOM": "Türk Telekom",
    "ENKAI": "Enka İnşaat",
    "ARCLK": "Arçelik",
    "TOASO": "Tofaş",
}

COMPANY_TO_TICKER = {v: k for k, v in TICKER_TO_COMPANY.items()}


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

