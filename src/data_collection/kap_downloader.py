import os
import sys
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# BIST 100 Şirketleri Kataloğu (Sembol, Şirket Adı, Sektör)
BIST_100_COMPANIES: Dict[str, Dict[str, str]] = {
    "AKBNK": {"name": "Akbank T.A.Ş.", "sector": "Bankacılık", "slug": "2413-akbank-t-a-s"},
    "THYAO": {"name": "Türk Hava Yolları A.O.", "sector": "Ulaştırma / Havacılık", "slug": "1107-turk-hava-yollari-a-o"},
    "TUPRS": {"name": "Tüpraş - Türkiye Petrol Rafinerileri A.Ş.", "sector": "Petrol & Enerji", "slug": "1105-tupras-turkiye-petrol-rafinerileri-a-s"},
    "EREGL": {"name": "Ereğli Demir ve Çelik Fabrikaları T.A.Ş.", "sector": "Demir Çelik", "slug": "1109-eregli-demir-ve-celik-fabrikalari-t-a-s"},
    "SISE": {"name": "Türkiye Şişe ve Cam Fabrikaları A.Ş.", "sector": "Cam & Sanayi", "slug": "1108-turkiye-sise-ve-cam-fabrikalari-a-s"},
    "ASELS": {"name": "Aselsan Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Savunma & Teknoloji", "slug": "1002-aselsan-elektronik-sanayi-ve-ticaret-a-s"},
    "BIMAS": {"name": "BİM Birleşik Mağazalar A.Ş.", "sector": "Perakende Ticaret", "slug": "1004-bim-birlesik-magazalar-a-s"},
    "FROTO": {"name": "Ford Otomotiv Sanayi A.Ş.", "sector": "Otomotiv", "slug": "1110-ford-otomotiv-sanayi-a-s"},
    "GARAN": {"name": "Türkiye Garanti Bankası A.Ş.", "sector": "Bankacılık", "slug": "1111-turkiye-garanti-bankasi-a-s"},
    "ISCTR": {"name": "Türkiye İş Bankası A.Ş.", "sector": "Bankacılık", "slug": "1112-turkiye-is-bankasi-a-s"},
    "KCHOL": {"name": "Koç Holding A.Ş.", "sector": "Holding & Yatırım", "slug": "1113-koc-holding-a-s"},
    "SAHOL": {"name": "Hacı Ömer Sabancı Holding A.Ş.", "sector": "Holding & Yatırım", "slug": "1114-haci-omer-sabanci-holding-a-s"},
    "YKBNK": {"name": "Yapı ve Kredi Bankası A.Ş.", "sector": "Bankacılık", "slug": "1115-yapi-ve-kredi-bankasi-a-s"},
    "PETKM": {"name": "Petkim Petrokimya Holding A.Ş.", "sector": "Petrokimya", "slug": "1116-petkim-petrokimya-holding-a-s"},
    "PGSUS": {"name": "Pegasus Hava Taşımacılığı A.Ş.", "sector": "Ulaştırma / Havacılık", "slug": "1710-pegasus-hava-tasimaciligi-a-s"},
    "TCELL": {"name": "Turkcell İletişim Hizmetleri A.Ş.", "sector": "Telekomünikasyon", "slug": "1117-turkcell-iletisim-hizmetleri-a-s"},
    "TTKOM": {"name": "Türk Telekomünikasyon A.Ş.", "sector": "Telekomünikasyon", "slug": "1118-turk-telekomunikasyon-a-s"},
    "ENKAI": {"name": "Enka İnşaat ve Sanayi A.Ş.", "sector": "İnşaat & Enerji", "slug": "1119-enka-insaat-ve-sanayi-a-s"},
    "ARCLK": {"name": "Arçelik A.Ş.", "sector": "Dayanıklı Tüketim", "slug": "1120-arcelik-a-s"},
    "VESTL": {"name": "Vestel Elektronik Sanayi ve Ticaret A.Ş.", "sector": "Elektronik & Beyaz Eşya", "slug": "1121-vestel-elektronik-sanayi-ve-ticaret-a-s"},
    "TOASO": {"name": "Tofaş Türk Otomobil Fabrikası A.Ş.", "sector": "Otomotiv", "slug": "1122-tofas-turk-otomobil-fabrikasi-a-s"},
    "KOZAL": {"name": "Koza Altın İşletmeleri A.Ş.", "sector": "Madencilik", "slug": "1123-koza-altin-isletmeleri-a-s"},
    "EKGYO": {"name": "Emlak Konut Gayrimenkul Yatırım Ortaklığı A.Ş.", "sector": "GYO", "slug": "1124-emlak-konut-gayrimenkul-yatirim-ortakligi-a-s"},
    "SOKM": {"name": "Şok Marketler Ticaret A.Ş.", "sector": "Perakende Ticaret", "slug": "1125-sok-marketler-ticaret-a-s"},
    "MGROS": {"name": "Migros Ticaret A.Ş.", "sector": "Perakende Ticaret", "slug": "1126-migros-ticaret-a-s"},
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def download_pdf_from_url(
    url: str,
    save_path: Path,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    min_size_kb: int = 10,
) -> bool:
    """
    Belirtilen URL'den PDF dosyasını güvenli bir şekilde indirir ve doğrular.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    req_headers = headers or DEFAULT_HEADERS

    try:
        response = requests.get(url, headers=req_headers, stream=True, timeout=timeout)
        if response.status_code != 200:
            print(f"[KAP Downloader] İndirme başarısız ({response.status_code}): {url}")
            return False

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)

        # Dosya boyutunu ve PDF başlığını doğrula
        file_size_kb = save_path.stat().st_size / 1024
        if file_size_kb < min_size_kb:
            print(f"[KAP Downloader] İndirilen dosya boyutu çok küçük ({file_size_kb:.1f} KB), siliniyor: {save_path.name}")
            save_path.unlink(missing_ok=True)
            return False

        # İlk baytları kontrol et (PDF sihirli baytı: %PDF)
        with open(save_path, "rb") as f:
            header_sample = f.read(512)

        pdf_offset = header_sample.find(b"%PDF")
        if pdf_offset == -1:
            print(f"[KAP Downloader] İndirilen içerik geçerli bir PDF değil, siliniyor.")
            save_path.unlink(missing_ok=True)
            return False
        elif pdf_offset > 0:
            # KAP Java Object Serialization stream (0xACED...) başlığını temizle
            print(f"[KAP Downloader] KAP Java akış başlığı ({pdf_offset} byte) temizleniyor...")
            with open(save_path, "rb") as f_in:
                full_bytes = f_in.read()
            with open(save_path, "wb") as f_out:
                f_out.write(full_bytes[pdf_offset:])
            file_size_kb = save_path.stat().st_size / 1024

        print(f"[KAP Downloader] Başarıyla indirildi: {save_path.name} ({file_size_kb / 1024:.2f} MB)")
        return True

    except Exception as e:
        print(f"[KAP Downloader] Hata ({url}): {e}")
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        return False


def resolve_kap_pdf_url(url_or_id: str) -> Optional[str]:
    """
    KAP bildirim ID'si veya URL'sinden doğrudan indirilebilir PDF bağlantısını çözer.
    Desteklenen formatlar:
    - 1410649 (Doğrudan Bildirim ID)
    - https://www.kap.org.tr/tr/Bildirim/1410649
    - https://www.kap.org.tr/tr/api/BildirimPdf/1410649
    - https://www.kap.org.tr/tr/api/file/download/...
    - Herhangi bir doğrudan .pdf bağlantısı
    """
    clean_target = url_or_id.strip()
    
    # Doğrudan PDF dosyası
    if clean_target.lower().endswith(".pdf"):
        return clean_target

    # file/download URL'si
    if "api/file/download/" in clean_target:
        return clean_target

    # Bildirim ID'sini yakala
    disclosure_id_match = re.search(r"(?:Bildirim(?:Pdf)?/|/tr/Bildirim/)(\d+)", clean_target)
    if not disclosure_id_match and clean_target.isdigit():
        disclosure_id = clean_target
    elif disclosure_id_match:
        disclosure_id = disclosure_id_match.group(1)
    else:
        disclosure_id = None

    if not disclosure_id:
        return clean_target

    # KAP bildirim sayfasını sorgulayarak ekli asıl faaliyet raporu PDF dosyasını ara
    disclosure_page_url = f"https://www.kap.org.tr/tr/Bildirim/{disclosure_id}"
    try:
        resp = requests.get(disclosure_page_url, headers=DEFAULT_HEADERS, timeout=15)
        if resp.status_code == 200:
            # Ekli dosya bağlantısı (api/file/download/...)
            file_match = re.search(r'href=[\'"]([^\'"]*api/file/download/[a-zA-Z0-9]+)[\'"]', resp.text)
            if file_match:
                found_url = file_match.group(1)
                if not found_url.startswith("http"):
                    found_url = "https://www.kap.org.tr" + found_url
                return found_url
    except Exception as e:
        print(f"[KAP Downloader] Bildirim sayfası ayrıştırılamadı ({disclosure_page_url}): {e}")

    # Ek dosya bulunamazsa standart KAP resmi bildirim PDF'ine yönlendir
    return f"https://www.kap.org.tr/tr/api/BildirimPdf/{disclosure_id}"


def download_kap_disclosure_pdf(disclosure_id: str, save_path: Path) -> bool:
    """
    KAP bildirim ID'si (disclosureIndex) üzerinden resmi PDF'i indirir.
    URL Formatı: https://www.kap.org.tr/tr/api/BildirimPdf/{disclosure_id}
    """
    url = resolve_kap_pdf_url(disclosure_id)
    if not url:
        return False
    return download_pdf_from_url(url, save_path)


def download_company_annual_report(
    ticker: str,
    year: int,
    doc_type: str = "faaliyet",
    output_dir: Optional[Path] = None,
    custom_url: Optional[str] = None
) -> Optional[Path]:
    """
    Belirli bir BIST şirketi ve yıl için faaliyet raporu veya finansal tablo indirir.
    Eğer custom_url verilmişse doğrudan oradan indirir, verilmemişse şirketin KAP bildirim
    bağlantılarını hedefler.
    """
    ticker = ticker.upper()
    comp_info = BIST_100_COMPANIES.get(ticker, {"name": ticker, "sector": "Genel"})
    
    if output_dir is None:
        from src.config import DATA_DIR
        output_dir = DATA_DIR / str(year)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    type_suffix = "Faaliyet_Raporu" if doc_type == "faaliyet" else "Konsolide_Finansal_Tablolar"
    file_name = f"{ticker}_{year}_{type_suffix}.pdf"
    target_path = output_dir / file_name

    if target_path.exists():
        print(f"[KAP Downloader] Dosya zaten mevcut: {target_path}")
        return target_path

    if custom_url:
        resolved_url = resolve_kap_pdf_url(custom_url) or custom_url
        success = download_pdf_from_url(resolved_url, target_path)
        return target_path if success else None

    # Ticker ve yıla göre KAP bildirim PDF'i indirme
    print(f"[KAP Downloader] {ticker} ({comp_info['name']}) için {year} yılı {type_suffix} indiriliyor...")
    # Not: Eğer kullanıcı doğrudan KAP bildirim URL'si verirse custom_url ile indirilebilir.
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAP ve BIST 100 Rapor İndirici")
    parser.add_argument("--ticker", type=str, help="Hisse Kodu (Örn: AKBNK, THYAO, TUPRS, EREGL)", default="AKBNK")
    parser.add_argument("--year", type=int, help="Yıl (2020-2024)", default=2024)
    parser.add_argument("--type", type=str, choices=["faaliyet", "finansal"], default="faaliyet")
    parser.add_argument("--url", type=str, help="Doğrudan PDF veya KAP BildirimPdf linki", default=None)
    args = parser.parse_args()

    res = download_company_annual_report(args.ticker, args.year, args.type, custom_url=args.url)
    if res:
        print(f"Rapor hazır: {res}")
    else:
        print("İndirme tamamlanamadı veya özel URL belirtilmedi.")
