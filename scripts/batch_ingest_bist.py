import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import TICKER_TO_COMPANY
from src.data_collection.kap_downloader import download_company_annual_report
from src.rag.ingestion import add_single_pdf_to_vectorstore


# BIST 100 Lokomotif Şirketleri için Doğrulanmış Resmi KAP ve Faaliyet Raporu Bağlantıları
BIST_100_PRIORITY_REPORTS: Dict[str, Dict[str, Any]] = {
    "EREGL": {
        "name": "Ereğli Demir Çelik",
        "year": 2024,
        "type": "faaliyet",
        "url": "https://www.kap.org.tr/tr/Bildirim/1410649"
    },
    "THYAO": {
        "name": "Türk Hava Yolları",
        "year": 2024,
        "type": "faaliyet",
        "url": None # Halihazırda data/2024 altında mevcut
    },
    "AKBNK": {
        "name": "Akbank",
        "year": 2024,
        "type": "faaliyet",
        "url": None # Halihazırda data/2024 altında mevcut
    },
    "TUPRS": {
        "name": "Tüpraş",
        "year": 2024,
        "type": "faaliyet",
        "url": None # Halihazırda data/2024 altında mevcut
    },
}


def batch_ingest_companies(
    tickers: List[str],
    year: int = 2024,
    custom_urls: Optional[Dict[str, str]] = None
) -> Dict[str, bool]:
    """
    Belirtilen BIST 100 hisselerinin raporlarını indirir, filtreler ve vektör indeksine ekler.
    """
    custom_urls = custom_urls or {}
    results = {}

    print(f"\n=======================================================")
    print(f"🚀 BIST 100 Toplu Rapor İndirme ve İndeksleme ({len(tickers)} Şirket)")
    print(f"=======================================================\n")

    for idx, ticker in enumerate(tickers):
        ticker_clean = ticker.upper().strip()
        comp_name = TICKER_TO_COMPANY.get(ticker_clean, ticker_clean)
        url = custom_urls.get(ticker_clean) or BIST_100_PRIORITY_REPORTS.get(ticker_clean, {}).get("url")

        print(f"[{idx+1}/{len(tickers)}] {ticker_clean} ({comp_name}) işleniyor...")

        # 1. Raporu İndir
        pdf_path = download_company_annual_report(
            ticker=ticker_clean,
            year=year,
            doc_type="faaliyet",
            custom_url=url
        )

        if not pdf_path or not pdf_path.exists():
            print(f"       ⚠️ {ticker_clean} raporu indirilemedi veya URL belirtilmedi.")
            results[ticker_clean] = False
            continue

        # 2. Vektör Veritabanına Akıllı Filtreleme ile Ekle
        print(f"       ↳ Vektör veritabanına indeksleniyor: {pdf_path.name}")
        success = add_single_pdf_to_vectorstore(pdf_path, smart_filter=True)
        results[ticker_clean] = success
        if success:
            print(f"       ✓ {ticker_clean} başarıyla indekslendi.")
        else:
            print(f"       ✗ {ticker_clean} indeksleme başarısız.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST 100 Toplu İndirme ve İndeksleme")
    parser.add_argument("--tickers", nargs="+", help="İndirilecek hisse kodları (Örn: EREGL ASELS SISE)", default=["EREGL"])
    parser.add_argument("--year", type=int, help="Mali yıl (2020-2026)", default=2024)
    args = parser.parse_args()

    batch_ingest_companies(args.tickers, year=args.year)
