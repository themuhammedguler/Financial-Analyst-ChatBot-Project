"""KAP PDF Raporları ve Finansal Tablolar Gelişmiş Ayrıştırıcı (Parser).

Bu modül:
1. pypdfium2 ile PDF sayfalarını yüksek hızda ve UTF-8 karakter doğruluğuyla çıkarır,
2. Finansal tabloların (Bilanço, Gelir Tablosu, Nakit Akış) ve Dipnotların yapısını korur,
3. Her sayfanın finansal bölümünü (section), içerdiği anahtar terimleri ve tablo varlığını tespit eder,
4. Vektör veritabanı için bağlamsal metadata (Contextual Metadata) üretir.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdfium2 as pdfium
from langchain_core.documents import Document

from src.config import TICKER_TO_COMPANY, COMPANIES


def infer_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """
    Dosya yolu ve adından şirket adı, yıl, borsa kodu ve doküman türünü çıkarır.
    """
    path_str = str(file_path).lower()
    name_str = file_path.name.lower()

    # 1. Yıl Çıkarımı (2020 - 2026 dönemi)
    year_match = re.search(r"(202[0-6])", path_str)
    year = int(year_match.group(1)) if year_match else 2026

    # 2. Şirket ve Ticker Çıkarımı
    company = "Bilinmeyen"
    ticker = "BIST"

    # Önce ticker bazlı doğrudan eşleme
    for t_sym, full_name in TICKER_TO_COMPANY.items():
        if t_sym.lower() in name_str:
            company = full_name
            ticker = t_sym
            break

    # Alternatif isimlerden arama
    if company == "Bilinmeyen":
        for comp_name, aliases in COMPANIES.items():
            if any(alias in name_str for alias in aliases):
                company = comp_name
                # Ticker'ı bul
                for t_sym, full_name in TICKER_TO_COMPANY.items():
                    if full_name == comp_name:
                        ticker = t_sym
                        break
                break

    # Genel yedek eşlemeler
    if company == "Bilinmeyen":
        if "akbank" in name_str or "akb" in name_str:
            company, ticker = "Akbank", "AKBNK"
        elif "thy" in name_str or "hava yolları" in name_str or "hava yollari" in name_str:
            company, ticker = "Türk Hava Yolları", "THYAO"
        elif "tupras" in name_str or "tüpraş" in name_str:
            company, ticker = "Tüpraş", "TUPRS"
        elif "eregli" in name_str or "erdemir" in name_str:
            company, ticker = "Ereğli Demir Çelik", "EREGL"

    # 3. Doküman Türü Çıkarımı
    if "faaliyet" in name_str or "frat" in name_str or "entegre" in name_str:
        doc_type = "Faaliyet Raporu"
    elif "finansal" in name_str or "tablo" in name_str or "spk" in name_str or "aralık" in name_str:
        doc_type = "Finansal Tablolar ve Dipnotlar"
    else:
        doc_type = "Finansal Rapor"

    return {
        "company": company,
        "ticker": ticker,
        "year": year,
        "doc_type": doc_type,
        "file_name": file_path.name,
        "source": str(file_path),
    }


def detect_financial_section(text: str) -> str:
    """
    Sayfa metnindeki başlıklardan finansal bölümü tespit eder:
    Bilanço, Gelir Tablosu, Nakit Akış, Finansal Dipnotlar, Yönetim Raporu vb.
    """
    t_lower = text.lower()
    
    # 1. Bilanço / Finansal Durum
    if any(k in t_lower for k in ["finansal durum tablosu", "konsolide bilanço", "varlıklar", "dönen varlıklar", "duran varlıklar", "kısa vadeli yükümlülükler"]):
        if "dipnot" not in t_lower[:120]:
            return "Bilanço (Finansal Durum Tablosu)"

    # 2. Gelir Tablosu / Kâr-Zarar
    if any(k in t_lower for k in ["kâr veya zarar tablosu", "gelir tablosu", "kapsamlı gelir tablosu", "esas faaliyet kârı", "dönem kârı (zararı)"]):
        if "dipnot" not in t_lower[:120]:
            return "Gelir Tablosu (Kâr/Zarar)"

    # 3. Nakit Akış Tablosu
    if any(k in t_lower for k in ["nakit akış tablosu", "nakit akışları", "işletme faaliyetlerinden nakit"]):
        return "Nakit Akış Tablosu"

    # 4. Özkaynak Değişim Tablosu
    if any(k in t_lower for k in ["özkaynak değişim tablosu", "özkaynaklar"]):
        if "tablo" in t_lower:
            return "Özkaynak Değişim Tablosu"

    # 5. Finansal Dipnotlar
    dipnot_match = re.search(r"(?:dipnot|not)\s*([0-9]{1,2})", t_lower)
    if dipnot_match or "dipnotlar" in t_lower or "finansal tablolara ilişkin notlar" in t_lower:
        note_num = dipnot_match.group(1) if dipnot_match else ""
        return f"Finansal Dipnotlar (Not {note_num})" if note_num else "Finansal Dipnotlar"

    # 6. Yönetim & Operasyonel Raporlama
    if any(k in t_lower for k in ["yönetim kurulu başkanı", "genel müdür mesajı", "stratejik hedefler", "operasyonel görünüm", "filo gelişimi", "rafineri marjı"]):
        return "Yönetim & Operasyonel Rapor"

    # 7. Sürdürülebilirlik & ESG
    if any(k in t_lower for k in ["sürdürülebilirlik", "emisyon", "karbon ayak izi", "çevresel"]):
        return "Sürdürülebilirlik & ESG"

    return "Finansal Genel Metin"


def extract_key_financial_terms(text: str) -> List[str]:
    """Sayfada geçen kritik finansal terimleri ve rasyoları etiketler."""
    t_lower = text.lower()
    terms = []
    
    watchlist = {
        "Net Kâr": ["net kâr", "net kar", "dönem net kârı", "dönem net karı"],
        "FAVÖK/EBITDA": ["favök", "ebitda", "faaliyet kârı"],
        "Hasılat/Satış": ["hasılat", "satış gelirleri", "net satışlar", "gelirler"],
        "Özkaynak": ["özkaynak", "özsermaye", "ödenmiş sermaye"],
        "Borç/Kaldıraç": ["finansal borç", "net borç", "kısa vadeli borç", "kredi borcu"],
        "Temettü": ["temettü", "kâr payı", "kar payı"],
        "Operasyonel": ["filo", "yolcu doluluk", "rafineri marjı", "kapasite kullanım", "üretim tonajı"],
        "Yatırımlar": ["maddi duran varlık", "yatırım harcaması", "capex", "bağlı ortaklık"],
    }
    
    for label, keywords in watchlist.items():
        if any(kw in t_lower for kw in keywords):
            terms.append(label)
            
    return terms


def clean_text_preserving_tables(text: str) -> str:
    """
    Finansal tabloların sütun yapısını ve satır bütünlüğünü bozmadan metni temizler.
    Sayısal değerlerin ve para birimlerinin birbirine yapışmasını engeller.
    """
    if not text:
        return ""
    # Sıfır genişlikli boşlukları ve kontrol karakterlerini kaldır
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # Çoklu satır başlarını makul seviyeye indir (max 2 yeni satır)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    
    # Satır sonlarındaki gereksiz boşlukları temizle
    lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in text.split("\n")]
    
    # Boş satırları filtrele ama paragraf yapısını koru
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
            
    return "\n".join(cleaned_lines).strip()


def parse_pdf_file(file_path: Path) -> List[Document]:
    """
    Tek bir PDF dosyasını pypdfium2 ile sayfa sayfa okur.
    Her sayfa için zengin finansal metadata ve bölüm tespiti üretir.
    """
    metadata_base = infer_metadata_from_path(file_path)
    documents = []

    try:
        doc = pdfium.PdfDocument(str(file_path))
        num_pages = len(doc)

        for page_idx in range(num_pages):
            page = doc[page_idx]
            text_page = page.get_textpage()
            raw_text = text_page.get_text_range()
            cleaned = clean_text_preserving_tables(raw_text)

            # Anlamsız derecede kısa (boş sayfa, sadece logo/reklam kapağı vb.) metinleri atla
            if len(cleaned) < 40:
                continue

            section = detect_financial_section(cleaned)
            key_terms = extract_key_financial_terms(cleaned)
            has_table = bool(re.search(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?", cleaned) and len(re.findall(r"\d+", cleaned)) > 5)

            page_metadata = {
                **metadata_base,
                "page": page_idx + 1,
                "total_pages": num_pages,
                "section": section,
                "key_terms": ", ".join(key_terms),
                "has_table": has_table
            }

            documents.append(Document(page_content=cleaned, metadata=page_metadata))

    except Exception as e:
        print(f"Hata ({file_path.name}): {e}")

    return documents
