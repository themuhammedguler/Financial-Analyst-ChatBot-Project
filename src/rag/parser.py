import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdfium2 as pdfium
from langchain_core.documents import Document


def infer_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """
    Dosya yolu ve adından şirket adı, yıl ve doküman türünü çıkarır.
    """
    path_str = str(file_path).lower()
    name_str = file_path.name.lower()
    
    # 1. Yıl Çıkarımı
    year_match = re.search(r"(2020|2021|2022|2023|2024)", path_str)
    year = int(year_match.group(1)) if year_match else 2024
    
    # 2. Şirket Çıkarımı
    from src.config import TICKER_TO_COMPANY, COMPANIES

    company = "Bilinmeyen"
    # Önce ticker bazlı eşleme (EREGL, ASELS, THYAO vb.)
    for ticker, full_name in TICKER_TO_COMPANY.items():
        if ticker.lower() in name_str:
            company = full_name
            break

    # Ticker eşleşmediyse şirket alternatif isimlerinden ara
    if company == "Bilinmeyen":
        for comp_name, aliases in COMPANIES.items():
            if any(alias in name_str for alias in aliases):
                company = comp_name
                break
            
    if company == "Bilinmeyen":
        if "akbank" in name_str or "akb" in name_str:
            company = "Akbank"
        elif "thy" in name_str or "hava yolları" in name_str or "hava yollari" in name_str:
            company = "Türk Hava Yolları"
        elif "tupras" in name_str or "tüpraş" in name_str:
            company = "Tüpraş"
        elif "eregli" in name_str or "erdemir" in name_str:
            company = "Ereğli Demir Çelik"
        
    # 3. Doküman Türü Çıkarımı
    if "faaliyet" in name_str or "frat" in name_str or "entegre" in name_str:
        doc_type = "Faaliyet Raporu"
    elif "finansal" in name_str or "tablo" in name_str or "spk" in name_str or "aralık" in name_str:
        doc_type = "Finansal Tablolar ve Dipnotlar"
    else:
        doc_type = "Finansal Rapor"

    return {
        "company": company,
        "year": year,
        "doc_type": doc_type,
        "file_name": file_path.name,
        "source": str(file_path),
    }


def clean_text(text: str) -> str:
    """Metindeki gereksiz boşlukları, özel karakter bozulmalarını temizler."""
    if not text:
        return ""
    # Sıfır genişlikli boşlukları ve kontrol karakterlerini kaldır
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Birden fazla boşluğu teke indir
    text = re.sub(r"[ \t]+", " ", text)
    # İkiden fazla yeni satırı ikiye indir
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_pdf_file(file_path: Path) -> List[Document]:
    """
    Tek bir PDF dosyasını pypdfium2 ile ultra hızlı ve temiz bir şekilde okur,
    her sayfa için Document nesnesi üretir.
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
            cleaned = clean_text(raw_text)
            
            # Anlamsız derecede kısa (örn: sadece kapak görseli/boş sayfa) metinleri atla
            if len(cleaned) < 40:
                continue
                
            page_metadata = {
                **metadata_base,
                "page": page_idx + 1,  # 1-tabanlı sayfa numarası
                "total_pages": num_pages
            }
            
            documents.append(Document(page_content=cleaned, metadata=page_metadata))
            
    except Exception as e:
        print(f"Hata ({file_path.name}): {e}")

    return documents
